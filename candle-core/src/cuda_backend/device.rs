use crate::backend::{BackendDevice, BackendStorage};
use crate::{CpuStorage, CpuStorageRef, DType, Layout, Result, Shape};
pub use candle_kernels as kernels;
pub use cudarc;
use cudarc::driver::CudaFunction;
use float8::F8E4M3;
use half::{bf16, f16};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use super::{CudaError, CudaStorage, CudaStorageSlice, WrapErr};

// ── CUDA Graph capture / replay ────────────────────────────────────────────
//
// CUDA Graphs capture the exact sequence of kernel launches for one training
// step and replay them as a single `cuGraphLaunch`, eliminating all per-kernel
// CPU launch overhead.  For GNN steps with 700+ kernels this can yield an
// additional 5-10× speedup on top of other optimisations.
//
// ### Requirements for the captured region
//
// 1. **Fixed tensor shapes every step** — graph capture records concrete device
//    pointers and launch configs.  If N (nodes) or E (edges) changes, the
//    cached graph must be invalidated and re-captured.  Use padding to a fixed
//    maximum to satisfy this.
//
// 2. **All ops on one CUDA stream** — capture records launches on the stream
//    that `begin_capture` was called on.  Do not issue work on other streams
//    inside the captured region.
//
// 3. **No CPU–GPU synchronisation inside the region** — `to_scalar()`,
//    `synchronize()`, and any blocking copy will break capture or produce
//    incorrect results.  Move any scalar reads to outside the captured region.
//
// ### Typical usage in a GNN training loop
//
// ```rust,no_run
// // Compute a key that encodes the shapes for this step.
// let graph_key = (num_nodes as u64) << 32 | num_edges as u64;
//
// // First call: executes f() normally while capturing; subsequent calls
// // with the same key replay the cached graph instead.
// device.with_cuda_graph(graph_key, || {
//     // forward + loss + backward + optimiser step
//     Ok(())
// })?;
// ```

/// A compiled CUDA graph that can be replayed cheaply on a fixed-shape
/// computation.
///
/// Obtained from [`CudaDevice::end_capture`].  Replayed with
/// [`CudaGraph::launch`].  Both the graph definition (`CUgraph`) and the
/// executable instance (`CUgraphExec`) are destroyed when this value is
/// dropped.
///
/// ### Thread safety
/// CUDA graph objects are **not** internally synchronised.  Access to a single
/// `CudaGraph` must be serialised externally (the `graph_cache` on
/// `CudaDevice` is protected by an `RwLock`).
pub struct CudaGraph {
    cu_graph: cudarc::driver::sys::CUgraph,
    cu_graph_exec: cudarc::driver::sys::CUgraphExec,
    /// The stream the graph was captured on — needed for `launch` and `upload`.
    cu_stream: cudarc::driver::sys::CUstream,
}

// Raw CUDA pointers are not automatically Send.  We assert Send here because:
// - CUgraphExec is only accessed under the graph_cache RwLock (serialised).
// - All graph API calls (launch, upload, destroy) are thread-safe per CUDA docs
//   when calls are serialised — which our RwLock guarantees.
unsafe impl Send for CudaGraph {}
unsafe impl Sync for CudaGraph {}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        if !self.cu_graph_exec.is_null() {
            let exec = std::mem::replace(&mut self.cu_graph_exec, std::ptr::null_mut());
            // Ignore errors on drop.
            let _ = unsafe { cudarc::driver::result::graph::exec_destroy(exec) };
        }
        if !self.cu_graph.is_null() {
            let graph = std::mem::replace(&mut self.cu_graph, std::ptr::null_mut());
            let _ = unsafe { cudarc::driver::result::graph::destroy(graph) };
        }
    }
}

impl CudaGraph {
    /// Replay the captured graph on the device's stream.
    ///
    /// All previously captured kernel launches execute as a single
    /// `cuGraphLaunch`, with no per-kernel CPU overhead.
    pub fn launch(&self) -> Result<()> {
        unsafe { cudarc::driver::result::graph::launch(self.cu_graph_exec, self.cu_stream) }
            .map_err(crate::Error::wrap)
    }

    /// Pre-upload graph resources to the device so the first [`launch`] incurs
    /// no setup cost.
    ///
    /// Optional but recommended when the graph is captured ahead of the hot
    /// path.
    pub fn upload(&self) -> Result<()> {
        unsafe { cudarc::driver::result::graph::upload(self.cu_graph_exec, self.cu_stream) }
            .map_err(crate::Error::wrap)
    }
}

/// Unique identifier for cuda devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

struct CudaRng(cudarc::curand::CudaRng);
unsafe impl Send for CudaRng {}

pub struct ModuleStore {
    mdls: [Option<Arc<cudarc::driver::CudaModule>>; kernels::ALL_IDS.len()],
}

#[derive(Clone)]
pub struct CudaDevice {
    id: DeviceId,
    context: Arc<cudarc::driver::CudaContext>,
    modules: Arc<std::sync::RwLock<ModuleStore>>,
    custom_modules: Arc<std::sync::RwLock<HashMap<String, Arc<cudarc::driver::CudaModule>>>>,
    stream: Arc<cudarc::driver::CudaStream>,
    pub(crate) blas: Arc<cudarc::cublas::CudaBlas>,
    curand: Arc<Mutex<CudaRng>>,
    seed_value: Arc<RwLock<u64>>,
    /// Cached compiled CUDA graphs keyed by a caller-supplied shape hash.
    ///
    /// The key encodes whatever makes two steps structurally identical (e.g.
    /// `(num_nodes as u64) << 32 | num_edges as u64`).  When shapes change
    /// the old entry is evicted and the next call re-captures.
    graph_cache: Arc<RwLock<HashMap<u64, Arc<CudaGraph>>>>,
}

impl std::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CudaDevice({:?})", self.id)
    }
}

impl CudaDevice {
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn alloc<T: cudarc::driver::DeviceRepr>(
        &self,
        len: usize,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        self.stream.alloc::<T>(len).w()
    }

    /// Allocates `len` elements using stream-ordered (`cuMemAllocAsync`) semantics.
    ///
    /// Prefer this over [`Self::alloc`] for intermediate tensors that are created and freed
    /// within a single training step: the CUDA stream-ordered allocator recycles memory
    /// without device-wide synchronisation, eliminating the ~500 µs stall that
    /// `cuMemAlloc_v2` imposes per allocation.
    ///
    /// Falls back to the standard [`Self::alloc`] path (which itself dispatches on
    /// [`cudarc::driver::CudaContext::has_async_alloc`]) when the device does not support
    /// memory pools (pre-CUDA 11.2 or Compute Capability < 5.2), so it is always safe to call.
    ///
    /// # Safety
    /// The allocated memory is uninitialised — callers must write before reading.
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn alloc_async<T: cudarc::driver::DeviceRepr>(
        &self,
        len: usize,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        if self.context.has_async_alloc() {
            // cuMemAllocAsync: stream-ordered, ~10 µs vs ~500 µs for cuMemAlloc_v2.
            self.stream.alloc_async::<T>(len).w()
        } else {
            // Device does not support memory pools — fall back to synchronous allocation.
            self.stream.alloc::<T>(len).w()
        }
    }

    /// Frees a [`cudarc::driver::CudaSlice`] with stream-ordered semantics on this device's
    /// default stream.
    ///
    /// In most cases simply dropping the slice is sufficient — `Drop for CudaSlice` issues
    /// `cuMemFreeAsync` automatically on devices with `has_async_alloc`. This method is
    /// provided for situations where you want to explicitly order the free before subsequent
    /// work on the stream without waiting for the Rust drop glue.
    ///
    /// # Safety
    /// See [`cudarc::driver::CudaStream::free_async`] — ownership of `slice` is consumed.
    pub unsafe fn free_async<T>(&self, slice: cudarc::driver::CudaSlice<T>) -> Result<()> {
        self.stream.free_async(slice).w()
    }

    pub fn alloc_zeros<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        self.stream.alloc_zeros::<T>(len).w()
    }

    pub fn memcpy_htod<
        T: cudarc::driver::DeviceRepr,
        Src: cudarc::driver::HostSlice<T> + ?Sized,
        Dst: cudarc::driver::DevicePtrMut<T>,
    >(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<()> {
        self.stream.memcpy_htod(src, dst).w()
    }

    pub fn clone_dtoh<T: cudarc::driver::DeviceRepr, Src: cudarc::driver::DevicePtr<T>>(
        &self,
        src: &Src,
    ) -> Result<Vec<T>> {
        self.stream.clone_dtoh(src).w()
    }

    pub fn memcpy_dtod<
        T,
        Src: cudarc::driver::DevicePtr<T>,
        Dst: cudarc::driver::DevicePtrMut<T>,
    >(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<()> {
        self.stream.memcpy_dtod(src, dst).w()
    }

    pub fn memcpy_dtoh<
        T: cudarc::driver::DeviceRepr,
        Src: cudarc::driver::DevicePtr<T>,
        Dst: cudarc::driver::HostSlice<T>,
    >(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<()> {
        self.stream.memcpy_dtoh(src, dst).w()
    }

    pub fn clone_htod<T: cudarc::driver::DeviceRepr, Src: cudarc::driver::HostSlice<T> + ?Sized>(
        &self,
        src: &Src,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        self.stream.clone_htod(src).w()
    }
}

pub struct CudaFunc {
    func: CudaFunction,
    stream: Arc<cudarc::driver::CudaStream>,
}

impl std::ops::Deref for CudaFunc {
    type Target = CudaFunction;

    fn deref(&self) -> &Self::Target {
        &self.func
    }
}

impl CudaFunc {
    pub fn into_cuda_function(self) -> CudaFunction {
        self.func
    }
}

#[macro_export]
macro_rules! builder_arg {
    ($b:ident, $($arg:expr),*) => {
        $(
            let __arg = $arg;
            $b.arg(&__arg);
        )*
    };
}

impl CudaFunc {
    pub fn builder(&self) -> cudarc::driver::LaunchArgs<'_> {
        self.stream.launch_builder(&self.func)
    }
}

impl CudaDevice {
    pub fn cuda_stream(&self) -> Arc<cudarc::driver::CudaStream> {
        self.stream.clone()
    }

    /// When turned on, all cuda tensors **created after calling this function** will
    /// not track uses via cuda events.
    ///
    /// # Safety
    ///
    /// It is up to the user to ensure proper synchronization between multiple streams:
    /// - Ensure that no tensor is freed before a use on another stream is finished.
    /// - Ensure that a tensor is not used on another stream before allocation on the
    ///   allocating stream finishes.
    /// - Ensure that a tensor is not written two concurrently by multiple streams.
    pub unsafe fn disable_event_tracking(&self) {
        self.context.disable_event_tracking()
    }

    pub fn is_event_tracking(&self) -> bool {
        self.context.is_event_tracking()
    }

    #[cfg(all(feature = "ug", not(target_arch = "wasm32")))]
    pub fn compile(
        &self,
        func_name: &'static str,
        kernel: candle_ug::lang::ssa::Kernel,
    ) -> Result<CudaFunc> {
        let mut buf = vec![];
        candle_ug::cuda::code_gen::gen(&mut buf, func_name, &kernel)?;
        let cuda_code = String::from_utf8(buf)?;
        let opts = cudarc::nvrtc::CompileOptions {
            use_fast_math: Some(true),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::safe::compile_ptx_with_opts(cuda_code, opts).w()?;
        let module = self.context.load_module(ptx).w()?;
        let func = module.load_function(func_name).w()?;
        Ok(CudaFunc {
            func,
            stream: self.stream.clone(),
        })
    }

    pub fn id(&self) -> DeviceId {
        self.id
    }

    pub fn get_or_load_custom_func(
        &self,
        fn_name: &str,
        module_name: &str,
        ptx: &str,
    ) -> Result<CudaFunc> {
        let ms = self.custom_modules.read().unwrap();
        if let Some(mdl) = ms.get(module_name).as_ref() {
            let func = mdl.load_function(fn_name).w()?;
            return Ok(CudaFunc {
                func,
                stream: self.stream.clone(),
            });
        }
        drop(ms);
        let mut ms = self.custom_modules.write().unwrap();
        let cuda_module = self.context.load_module(ptx.into()).w()?;
        ms.insert(module_name.to_string(), cuda_module.clone());
        let func = cuda_module.load_function(fn_name).w()?;
        Ok(CudaFunc {
            func,
            stream: self.stream.clone(),
        })
    }

    pub fn get_or_load_func(&self, fn_name: &str, mdl: &kernels::Module) -> Result<CudaFunc> {
        let ms = self.modules.read().unwrap();
        if let Some(mdl) = ms.mdls[mdl.index()].as_ref() {
            let func = mdl.load_function(fn_name).w()?;
            return Ok(CudaFunc {
                func,
                stream: self.stream.clone(),
            });
        }
        drop(ms);
        let mut ms = self.modules.write().unwrap();
        let cuda_module = self.context.load_module(mdl.ptx().into()).w()?;
        ms.mdls[mdl.index()] = Some(cuda_module.clone());
        let func = cuda_module.load_function(fn_name).w()?;
        Ok(CudaFunc {
            func,
            stream: self.stream.clone(),
        })
    }

    pub fn cublas_handle(&self) -> Arc<cudarc::cublas::CudaBlas> {
        self.blas.clone()
    }

    // ── CUDA Graph API ────────────────────────────────────────────────────

    /// Begin capturing all CUDA operations issued on this device's stream.
    ///
    /// Every kernel launch, memory copy, and cuBLAS/cuDNN call made on the
    /// device stream after this point is recorded rather than executed.
    /// Call [`end_capture`][CudaDevice::end_capture] to finish recording and
    /// obtain a [`CudaGraph`] that can be replayed cheaply.
    ///
    /// ### Constraints during capture
    /// - No CPU–GPU synchronisation (`to_scalar`, `synchronize`, etc.)
    /// - All ops must target the same CUDA stream (this device's stream)
    /// - Tensor shapes must match exactly on every replay
    ///
    /// Uses `CU_STREAM_CAPTURE_MODE_GLOBAL`: operations on other streams that
    /// synchronise with this stream are included in the capture.
    pub fn begin_capture(&self) -> Result<()> {
        self.stream
            .begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
            )
            .map_err(crate::Error::wrap)
    }

    /// End stream capture and compile the recorded operations into a
    /// [`CudaGraph`].
    ///
    /// Returns `None` if the stream was not in capture mode (e.g. the capture
    /// produced an empty graph).  In practice this should always return
    /// `Some(graph)` if [`begin_capture`][CudaDevice::begin_capture] was
    /// called first.
    ///
    /// The returned graph can be launched repeatedly with
    /// [`CudaGraph::launch`] as long as tensor shapes remain the same.
    ///
    /// Instantiation uses flags = 0 (no `AUTO_FREE_ON_LAUNCH`, no
    /// `DEVICE_LAUNCH`) so the graph's internal memory persists across
    /// repeated replays.  We call `cuGraphInstantiateWithFlags` directly
    /// rather than the cudarc safe wrapper because that wrapper's
    /// `CUgraphInstantiate_flags` enum has no zero-variant.
    pub fn end_capture(&self) -> Result<Option<CudaGraph>> {
        let cu_stream = self.stream.cu_stream();

        let cu_graph =
            unsafe { cudarc::driver::result::stream::end_capture(cu_stream) }
                .map_err(crate::Error::wrap)?;
        if cu_graph.is_null() {
            return Ok(None);
        }

        let cu_graph_exec = unsafe {
            let mut exec = std::mem::MaybeUninit::uninit();
            // flags = 0: no AUTO_FREE_ON_LAUNCH — memory persists between replays.
            cudarc::driver::sys::cuGraphInstantiateWithFlags(exec.as_mut_ptr(), cu_graph, 0u64)
                .result()
                .map_err(crate::Error::wrap)?;
            exec.assume_init()
        };

        Ok(Some(CudaGraph {
            cu_graph,
            cu_graph_exec,
            cu_stream,
        }))
    }

    /// Capture and cache a CUDA graph for a closure, replaying it on
    /// subsequent calls with the same `cache_key`.
    ///
    /// On the **first call** for a given key the closure `f` is executed
    /// normally while the device stream is in capture mode.  The resulting
    /// compiled graph is stored in an internal cache.
    ///
    /// On **subsequent calls** with the same key the cached graph is replayed
    /// directly via `cuGraphLaunch`, bypassing all CPU kernel-launch overhead.
    ///
    /// To **invalidate** a cached graph (e.g. after a padding-shape change)
    /// call [`invalidate_cuda_graph`][CudaDevice::invalidate_cuda_graph] with
    /// the same key before the next step.
    ///
    /// ### `cache_key` convention
    /// Encode whatever makes two steps structurally identical.  For a GNN step
    /// the minimum is the padded node and edge counts:
    /// ```ignore
    /// let key = (num_nodes_padded as u64) << 32 | num_edges_padded as u64;
    /// device.with_cuda_graph(key, || { /* forward + backward + opt */ Ok(()) })?;
    /// ```
    ///
    /// ### Constraints inside `f`
    /// See [`begin_capture`][CudaDevice::begin_capture] for the full list.
    /// The short version: fixed shapes, one stream, no blocking CPU–GPU sync.
    ///
    /// # Safety
    /// The closure must produce **identical kernel launches** (same ops, same
    /// shapes, same device pointers relative to the captured addresses) on
    /// every invocation.  Breaking this contract causes silent wrong results.
    pub fn with_cuda_graph<F>(&self, cache_key: u64, f: F) -> Result<()>
    where
        F: FnOnce() -> Result<()>,
    {
        // Fast path: replay a previously compiled graph.
        {
            let cache = self.graph_cache.read().unwrap();
            if let Some(graph) = cache.get(&cache_key) {
                let graph = Arc::clone(graph);
                drop(cache);
                return graph.launch();
            }
        }

        // Slow path (first call for this key): capture while executing f().
        self.begin_capture()?;
        let result = f();
        // Always attempt end_capture so the stream is left in a clean state
        // even if f() failed.
        let graph = self.end_capture()?;
        result?;

        if let Some(graph) = graph {
            // Pre-upload the graph so the very first replay has no setup cost.
            graph.upload()?;
            let mut cache = self.graph_cache.write().unwrap();
            cache.insert(cache_key, Arc::new(graph));
        }
        Ok(())
    }

    /// Remove a cached CUDA graph so the next [`with_cuda_graph`] call
    /// re-captures.
    ///
    /// Call this whenever tensor shapes change (e.g. the padded graph size
    /// changes between curriculum stages).
    pub fn invalidate_cuda_graph(&self, cache_key: u64) {
        self.graph_cache.write().unwrap().remove(&cache_key);
    }

    /// Remove all cached CUDA graphs.
    pub fn clear_cuda_graph_cache(&self) {
        self.graph_cache.write().unwrap().clear();
    }
}

impl CudaDevice {
    pub fn new_with_stream(ordinal: usize) -> Result<Self> {
        let context = cudarc::driver::CudaContext::new(ordinal).w()?;
        let stream = context.new_stream().w()?;
        let blas = cudarc::cublas::CudaBlas::new(stream.clone()).w()?;
        let curand = cudarc::curand::CudaRng::new(299792458, stream.clone()).w()?;
        let module_store = ModuleStore {
            mdls: [const { None }; kernels::ALL_IDS.len()],
        };
        Ok(Self {
            id: DeviceId::new(),
            context,
            stream,
            blas: Arc::new(blas),
            curand: Arc::new(Mutex::new(CudaRng(curand))),
            modules: Arc::new(std::sync::RwLock::new(module_store)),
            custom_modules: Arc::new(std::sync::RwLock::new(HashMap::new())),
            seed_value: Arc::new(RwLock::new(299792458)),
            graph_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }
}

impl BackendDevice for CudaDevice {
    type Storage = CudaStorage;

    fn new(ordinal: usize) -> Result<Self> {
        let context = cudarc::driver::CudaContext::new(ordinal).w()?;
        let stream = context.default_stream();
        let blas = cudarc::cublas::CudaBlas::new(stream.clone()).w()?;
        let curand = cudarc::curand::CudaRng::new(299792458, stream.clone()).w()?;
        let module_store = ModuleStore {
            mdls: [const { None }; kernels::ALL_IDS.len()],
        };
        Ok(Self {
            id: DeviceId::new(),
            context,
            stream,
            blas: Arc::new(blas),
            curand: Arc::new(Mutex::new(CudaRng(curand))),
            modules: Arc::new(std::sync::RwLock::new(module_store)),
            custom_modules: Arc::new(std::sync::RwLock::new(HashMap::new())),
            seed_value: Arc::new(RwLock::new(299792458)),
            graph_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        // We do not call set_seed but instead create a new curand object. This ensures that the
        // state will be identical and the same random numbers will be generated.
        let mut curand = self.curand.lock().unwrap();
        curand.0 = cudarc::curand::CudaRng::new(seed, self.stream.clone()).w()?;
        *self.seed_value.write().unwrap() = seed;
        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        Ok(*self.seed_value.read().unwrap())
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Cuda {
            gpu_id: self.context.ordinal(),
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.id == rhs.id
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<CudaStorage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U8 => {
                let data = self.alloc_zeros::<u8>(elem_count)?;
                CudaStorageSlice::U8(data)
            }
            DType::U32 => {
                let data = self.alloc_zeros::<u32>(elem_count)?;
                CudaStorageSlice::U32(data)
            }
            DType::I16 => {
                let data = self.alloc_zeros::<i16>(elem_count)?;
                CudaStorageSlice::I16(data)
            }
            DType::I32 => {
                let data = self.alloc_zeros::<i32>(elem_count)?;
                CudaStorageSlice::I32(data)
            }
            DType::I64 => {
                let data = self.alloc_zeros::<i64>(elem_count)?;
                CudaStorageSlice::I64(data)
            }
            DType::BF16 => {
                let data = self.alloc_zeros::<bf16>(elem_count)?;
                CudaStorageSlice::BF16(data)
            }
            DType::F16 => {
                let data = self.alloc_zeros::<f16>(elem_count)?;
                CudaStorageSlice::F16(data)
            }
            DType::F32 => {
                let data = self.alloc_zeros::<f32>(elem_count)?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let data = self.alloc_zeros::<f64>(elem_count)?;
                CudaStorageSlice::F64(data)
            }
            DType::F8E4M3 => {
                let data = self.alloc_zeros::<F8E4M3>(elem_count)?;
                CudaStorageSlice::F8E4M3(data)
            }
            DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                return Err(
                    CudaError::InternalError("Dummy types not supported in CUDA backend").into(),
                )
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, up: f64) -> Result<CudaStorage> {
        let elem_count = shape.elem_count();
        let curand = self.curand.lock().unwrap();
        let slice = match dtype {
            // TODO: Add support for F16 and BF16 though this is likely to require some upstream
            // cudarc changes.
            DType::U8
            | DType::U32
            | DType::I16
            | DType::I32
            | DType::I64
            | DType::F16
            | DType::BF16 => Err(CudaError::UnsupportedDtype {
                dtype,
                op: "rand_uniform",
            })
            .w()?,
            DType::F32 => {
                let mut data = unsafe { self.alloc::<f32>(elem_count)? };
                curand.0.fill_with_uniform(&mut data).w()?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let mut data = unsafe { self.alloc::<f64>(elem_count)? };
                curand.0.fill_with_uniform(&mut data).w()?;
                CudaStorageSlice::F64(data)
            }
            DType::F8E4M3 | DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                Err(CudaError::UnsupportedDtype {
                    dtype,
                    op: "rand_uniform",
                })
                .w()?
            }
        };
        let slice = if lo == 0. && up == 1.0 {
            slice
        } else {
            use super::utils::Map1;
            let layout = Layout::contiguous(shape);
            super::Affine(up - lo, lo).map(&slice, self, &layout)?
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<CudaStorage> {
        // TODO: Add support for F16 and BF16 though this is likely to require some upstream
        // cudarc changes.
        let elem_count = shape.elem_count();
        let curand = self.curand.lock().unwrap();
        // curand can only generate an odd number of values.
        // https://github.com/huggingface/candle/issues/734
        let elem_count_round = if elem_count % 2 == 1 {
            elem_count + 1
        } else {
            elem_count
        };
        let slice = match dtype {
            DType::U8
            | DType::U32
            | DType::I16
            | DType::I32
            | DType::I64
            | DType::F16
            | DType::BF16 => Err(CudaError::UnsupportedDtype {
                dtype,
                op: "rand_normal",
            })
            .w()?,
            DType::F32 => {
                let mut data = unsafe { self.alloc::<f32>(elem_count_round)? };
                curand
                    .0
                    .fill_with_normal(&mut data, mean as f32, std as f32)
                    .w()?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let mut data = unsafe { self.alloc::<f64>(elem_count_round)? };
                curand.0.fill_with_normal(&mut data, mean, std).w()?;
                CudaStorageSlice::F64(data)
            }
            DType::F8E4M3 | DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                Err(CudaError::UnsupportedDtype {
                    dtype,
                    op: "rand_normal",
                })
                .w()?
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U8 => {
                let data = self.alloc::<u8>(elem_count)?;
                CudaStorageSlice::U8(data)
            }
            DType::U32 => {
                let data = self.alloc::<u32>(elem_count)?;
                CudaStorageSlice::U32(data)
            }
            DType::I16 => {
                let data = self.alloc::<i16>(elem_count)?;
                CudaStorageSlice::I16(data)
            }
            DType::I32 => {
                let data = self.alloc::<i32>(elem_count)?;
                CudaStorageSlice::I32(data)
            }
            DType::I64 => {
                let data = self.alloc::<i64>(elem_count)?;
                CudaStorageSlice::I64(data)
            }
            DType::BF16 => {
                let data = self.alloc::<bf16>(elem_count)?;
                CudaStorageSlice::BF16(data)
            }
            DType::F16 => {
                let data = self.alloc::<f16>(elem_count)?;
                CudaStorageSlice::F16(data)
            }
            DType::F32 => {
                let data = self.alloc::<f32>(elem_count)?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let data = self.alloc::<f64>(elem_count)?;
                CudaStorageSlice::F64(data)
            }
            DType::F8E4M3 => {
                let data = self.alloc::<F8E4M3>(elem_count)?;
                CudaStorageSlice::F8E4M3(data)
            }
            DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                return Err(
                    CudaError::InternalError("Dummy types not supported in CUDA backend").into(),
                )
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_slice<T: crate::WithDType>(&self, s: &[T]) -> Result<Self::Storage> {
        let slice = match T::cpu_storage_ref(s) {
            CpuStorageRef::U8(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::U8(data)
            }
            CpuStorageRef::U32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::U32(data)
            }
            CpuStorageRef::I16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I16(data)
            }
            CpuStorageRef::I32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I32(data)
            }
            CpuStorageRef::I64(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I64(data)
            }
            CpuStorageRef::BF16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::BF16(data)
            }
            CpuStorageRef::F16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F16(data)
            }
            CpuStorageRef::F32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F32(data)
            }
            CpuStorageRef::F64(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F64(data)
            }
            CpuStorageRef::F8E4M3(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F8E4M3(data)
            }
            CpuStorageRef::F4(_)
            | CpuStorageRef::F6E2M3(_)
            | CpuStorageRef::F6E3M2(_)
            | CpuStorageRef::F8E8M0(_) => {
                return Err(CudaError::UnsupportedDtype {
                    dtype: T::DTYPE,
                    op: "storage_from_slice",
                }
                .into());
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<CudaStorage> {
        let slice = match storage {
            CpuStorage::U8(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::U8(data)
            }
            CpuStorage::U32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::U32(data)
            }
            CpuStorage::I16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I16(data)
            }
            CpuStorage::I32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I32(data)
            }
            CpuStorage::I64(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I64(data)
            }
            CpuStorage::BF16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::BF16(data)
            }
            CpuStorage::F16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F16(data)
            }
            CpuStorage::F32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F32(data)
            }
            CpuStorage::F64(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F64(data)
            }
            CpuStorage::F8E4M3(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F8E4M3(data)
            }
            CpuStorage::F4(_)
            | CpuStorage::F6E2M3(_)
            | CpuStorage::F6E3M2(_)
            | CpuStorage::F8E8M0(_) => {
                return Err(CudaError::UnsupportedDtype {
                    dtype: storage.dtype(),
                    op: "storage_from_cpu_storage",
                }
                .into());
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<CudaStorage> {
        let slice = match storage {
            CpuStorage::U8(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::U8(data)
            }
            CpuStorage::U32(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::U32(data)
            }
            CpuStorage::I16(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::I16(data)
            }
            CpuStorage::I32(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::I32(data)
            }
            CpuStorage::I64(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::I64(data)
            }
            CpuStorage::BF16(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::BF16(data)
            }
            CpuStorage::F16(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::F16(data)
            }
            CpuStorage::F32(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::F32(data)
            }
            CpuStorage::F64(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::F64(data)
            }
            CpuStorage::F8E4M3(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::F8E4M3(data)
            }
            CpuStorage::F4(_)
            | CpuStorage::F6E2M3(_)
            | CpuStorage::F6E3M2(_)
            | CpuStorage::F8E8M0(_) => {
                return Err(CudaError::UnsupportedDtype {
                    dtype: storage.dtype(),
                    op: "storage_from_cpu_storage_owned",
                }
                .into());
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn synchronize(&self) -> Result<()> {
        self.stream.synchronize().map_err(crate::Error::wrap)?;
        Ok(())
    }
}
