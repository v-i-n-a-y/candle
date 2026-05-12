/// cuSPARSE-backed sparse matrix operations for GNN message passing.
///
/// `CsrMatrix` holds a graph adjacency in Compressed Sparse Row (CSR) format and
/// exposes a single entry-point — [`CsrMatrix::spmm`] — that computes the aggregation
/// `out[N, D] = A[N, N] × feat[N, D]` using NVIDIA's cuSPARSE generic SpMM API
/// (`cusparseSpMM`).
///
/// # Why CSR / SpMM?
///
/// GNN message passing is dominated by the scatter step:
/// ```text
/// for every edge (u → v): out[v] += feat[u]
/// ```
/// This is mathematically identical to a sparse-matrix dense-matrix multiply
/// `out = A × feat` where `A[v, u] = 1` for edge `(u → v)`.  cuSPARSE's
/// `cusparseSpMM` dispatches to highly-tuned CUDA kernels (e.g. tiling, vectorised
/// loads, warp-level reductions) that are typically **2–5× faster** than our
/// `index_select + gnn_scatter_add` path for large graphs (|E| ≫ |N|).
///
/// For our E=144 k edges, N≈14 k nodes, D=128 feature workload:
/// * `index_select + gnn_scatter_add`: ~0.8 ms/step  (custom atomicAdd kernel)
/// * `cusparseSpMM` (CSR_ALG2, f32):   ~0.3 ms/step  (expected; depends on GPU)
///
/// # Layout invariant
///
/// `cusparseSpMM` requires the input dense matrix B (`feat`) in **column-major**
/// (Fortran) order when using `CUSPARSE_SPMM_CSR_ALG2`, OR row-major with
/// `CUSPARSE_SPMM_CSR_ALG1`. We use row-major (`CUSPARSE_ORDER_ROW`) for both B
/// and C to match candle's default tensor layout, which forces `CUSPARSE_SPMM_CSR_ALG1`.
///
/// # Feature gate
///
/// The entire module is compiled only when `candle-core` is built with
/// `features = ["cuda"]` AND the `cudarc` workspace dep includes `"cusparse"`.
/// Both are satisfied on the Perseus training machines (CUDA 12.x).
use std::mem::MaybeUninit;

use cudarc::cusparse::sys as sp;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, DeviceSlice};

use crate::cuda_backend::{CudaDevice, CudaStorage, CudaStorageSlice, WrapErr};
use crate::{Layout, Result};

// ---------------------------------------------------------------------------
// Helper: map cusparseStatus_t → candle Error
// ---------------------------------------------------------------------------

fn wrap_sp(status: sp::cusparseStatus_t) -> Result<()> {
    status
        .result()
        .map_err(|e| crate::Error::Msg(format!("cuSPARSE error: {:?}", e)).bt())
}

// ---------------------------------------------------------------------------
// CsrMatrix
// ---------------------------------------------------------------------------

/// Sparse adjacency matrix in CSR format, ready for cuSPARSE SpMM.
///
/// The matrix is unweighted (all values = 1.0): for GNN aggregation we only
/// need the sparsity pattern of the graph, not per-edge weights.
///
/// **Lifetime note:** the cuSPARSE handle is created lazily per `spmm` call
/// rather than stored here, because `cusparseHandle_t` is not `Send + Sync`
/// and candle's `CudaDevice` already owns thread-safe CUDA context/stream state.
pub struct CsrMatrix {
    /// Number of rows = number of destination nodes = N.
    pub n_rows: usize,
    /// Number of cols = number of source nodes = N (square for self-aggregation).
    pub n_cols: usize,
    /// Number of non-zeros = number of edges = E.
    pub nnz: usize,
    /// CSR row pointer array: `row_ptr[i]` = index into `col_idx` of the first
    /// edge whose destination is node `i`.  Length = N+1.
    pub row_ptr: CudaSlice<i32>,
    /// Column indices of non-zero entries (= source node IDs).  Length = E.
    pub col_idx: CudaSlice<i32>,
    /// Non-zero values — all 1.0f32 for unweighted graphs.  Length = E.
    pub values: CudaSlice<f32>,
}

impl CsrMatrix {
    /// Build a `CsrMatrix` from a COO `edge_index` tensor of shape `[2, E]`.
    ///
    /// Row 0 of `edge_index` = source node IDs (COO row indices in the
    /// *transposed* sense: A[dst, src] = 1, so the *column* of A).
    /// Row 1 of `edge_index` = destination node IDs (row of A).
    ///
    /// The function:
    /// 1. Copies `edge_index` to CPU (small: O(E) ints).
    /// 2. Computes the CSR row pointer via exclusive prefix sum of in-degree counts.
    /// 3. Fills `col_idx` with source node IDs sorted by destination (required by
    ///    CSR; we assume `edge_index` rows are already ordered by destination, which
    ///    is the convention in candle's Perseus pipeline — if not, call
    ///    `cusparseXcsrsort` on the result before `spmm`).
    /// 4. Uploads all three arrays to the device.
    ///
    /// # Arguments
    /// * `edge_index` — `[2, E]` integer tensor on any device.  Will be moved to
    ///   CPU for the COO→CSR conversion, then the CSR arrays are uploaded to
    ///   `device`.
    /// * `n_nodes` — total number of nodes `N` (needed for isolated nodes with no
    ///   incoming edges).
    /// * `device` — target CUDA device.
    ///
    /// # TODO (cuSPARSE path, requires CUDA at build time)
    /// Replace the CPU conversion with an on-device call to
    /// `cusparseXcoo2csr(handle, coo_row_ptr, nnz, n_rows, csr_row_ptr, ZERO_BASED)`
    /// once the cusparse feature is unconditionally enabled, to avoid the D→H→D round-trip.
    pub fn from_edge_index(
        edge_index: &crate::Tensor,
        n_nodes: usize,
        device: &CudaDevice,
    ) -> Result<Self> {
        use crate::IndexOp;

        let ei_shape = edge_index.shape().dims().to_vec();
        if ei_shape.len() != 2 || ei_shape[0] != 2 {
            crate::bail!(
                "CsrMatrix::from_edge_index: edge_index must be [2, E], got {:?}",
                ei_shape
            );
        }
        let e = ei_shape[1];

        // --- Step 1: pull edge_index to CPU as i32 --------------------------
        // We work with i32 because cuSPARSE's legacy xcoo2csr / xcsrsort use i32.
        let src_row = edge_index.i(0)?.to_dtype(crate::DType::I32)?.to_vec1::<i32>()?;
        let dst_row = edge_index.i(1)?.to_dtype(crate::DType::I32)?.to_vec1::<i32>()?;

        // --- Step 2: COO → CSR row pointer via exclusive prefix sum ---------
        //
        // CSR row_ptr[i] = number of edges with dst < i  (exclusive prefix sum
        // of in-degree).  row_ptr has length N+1.
        //
        // We treat A[dst, src] = 1, so "row" = destination, "col" = source.
        let mut in_degree = vec![0i32; n_nodes];
        for &dst in &dst_row {
            let d = dst as usize;
            if d >= n_nodes {
                crate::bail!(
                    "CsrMatrix::from_edge_index: dst index {} out of range [0, {})",
                    d,
                    n_nodes
                );
            }
            in_degree[d] += 1;
        }

        // Exclusive prefix sum → row_ptr[0..=N].
        let mut row_ptr_cpu = vec![0i32; n_nodes + 1];
        for i in 0..n_nodes {
            row_ptr_cpu[i + 1] = row_ptr_cpu[i] + in_degree[i];
        }
        debug_assert_eq!(row_ptr_cpu[n_nodes] as usize, e);

        // --- Step 3: fill col_idx sorted by destination ---------------------
        //
        // We scatter source IDs into col_idx at positions determined by row_ptr.
        // Use a mutable write-cursor per row (copy of row_ptr for safe indexing).
        let mut col_idx_cpu = vec![0i32; e];
        let mut write_cursor = row_ptr_cpu[..n_nodes].to_vec(); // length N
        for k in 0..e {
            let dst = dst_row[k] as usize;
            let pos = write_cursor[dst] as usize;
            col_idx_cpu[pos] = src_row[k];
            write_cursor[dst] += 1;
        }

        // All values = 1.0 (unweighted graph).
        let values_cpu = vec![1.0f32; e];

        // --- Step 4: upload to device ---------------------------------------
        let row_ptr = device.clone_htod(&row_ptr_cpu[..])?;
        let col_idx = device.clone_htod(&col_idx_cpu[..])?;
        let values = device.clone_htod(&values_cpu[..])?;

        Ok(Self {
            n_rows: n_nodes,
            n_cols: n_nodes,
            nnz: e,
            row_ptr,
            col_idx,
            values,
        })
    }

    /// SpMM: `out = A × feat`, where A is this CSR matrix and `feat` is `[N, D]`.
    ///
    /// Returns `[N, D]` output tensor (f32 only; extend for f16/bf16 as needed).
    ///
    /// # cuSPARSE generic API sequence
    ///
    /// The call sequence for `cusparseSpMM` (CUDA >= 11.0 generic API) is:
    ///
    /// ```text
    /// 1. cusparseCreate(&handle)
    /// 2. cusparseCreateCsr(&mat_a, n_rows, n_cols, nnz,
    ///        row_ptr, col_idx, values,
    ///        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
    ///        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F)
    /// 3. cusparseCreateDnMat(&mat_b, n_cols, D, /*ld=*/D, feat_ptr,
    ///        CUDA_R_32F, CUSPARSE_ORDER_ROW)
    /// 4. cusparseCreateDnMat(&mat_c, n_rows, D, /*ld=*/D, out_ptr,
    ///        CUDA_R_32F, CUSPARSE_ORDER_ROW)
    /// 5. cusparseSpMM_bufferSize(..., &buf_sz)
    /// 6. cudaMalloc(&work_buf, buf_sz)
    /// 7. cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    ///        CUSPARSE_OPERATION_NON_TRANSPOSE,
    ///        &alpha=1.0, mat_a, mat_b, &beta=0.0, mat_c,
    ///        CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, work_buf)
    /// 8. cusparseDestroySpMat(mat_a)
    /// 9. cusparseDestroyDnMat(mat_b / mat_c)
    /// 10. cusparseDestroy(handle)
    /// ```
    ///
    /// Steps 1–10 are mapped to cudarc calls below.  Because cudarc exposes only
    /// `cusparse::result::*` (no safe wrapper for `cusparseSpMM`), we call
    /// `cusparse::sys::*` directly via raw FFI, mirroring how `cublas::result` is
    /// used in `candle-core/src/cuda_backend/mod.rs`.
    ///
    /// # TODO (requires CUDA toolchain to compile / test)
    ///
    /// The body below is marked `unimplemented!()` on non-CUDA builds.  On CUDA
    /// builds it falls through to the raw-FFI block that is guarded by
    /// `#[cfg(feature = "cuda")]`.  When CUDA is available, remove the
    /// `unimplemented!()` guard and replace with the FFI sequence shown above.
    pub fn spmm(
        &self,
        feat: &CudaStorage,
        feat_layout: &Layout,
        device: &CudaDevice,
    ) -> Result<CudaStorage> {
        let feat_dims = feat_layout.dims();
        if feat_dims.len() != 2 {
            crate::bail!(
                "CsrMatrix::spmm: feat must be [N, D], got {:?}",
                feat_dims
            );
        }
        let n = feat_dims[0];
        let d = feat_dims[1];
        if n != self.n_cols {
            crate::bail!(
                "CsrMatrix::spmm: feat rows {} != matrix cols {}",
                n,
                self.n_cols
            );
        }

        let feat_f32 = match &feat.slice {
            CudaStorageSlice::F32(s) => s,
            _ => crate::bail!("CsrMatrix::spmm: only f32 feat is supported (got {:?}); \
                               cast feat to f32 before calling gnn_spmm", feat.dtype()),
        };

        // Require contiguous feat layout for the raw pointer passed to cuSPARSE.
        let (feat_o1, feat_o2) = feat_layout.contiguous_offsets().ok_or_else(|| {
            crate::Error::RequiresContiguous { op: "gnn-spmm" }.bt()
        })?;
        let feat_slice = feat_f32.slice(feat_o1..feat_o2);

        // Allocate output [N_out, D] where N_out = self.n_rows.
        let out_elems = self.n_rows * d;
        let mut out: CudaSlice<f32> = device.alloc_zeros(out_elems)?;

        // -----------------------------------------------------------------
        // Raw cuSPARSE FFI call
        //
        // This block is only reachable at runtime on CUDA builds; it is
        // guarded at compile time by the `cuda` feature propagated from
        // candle-core's Cargo.toml.  On non-CUDA builds `CudaStorage` itself
        // doesn't exist, so the entire `sparse` module is dead code.
        // -----------------------------------------------------------------
        unsafe {
            spmm_f32_raw(
                &self.row_ptr,
                &self.col_idx,
                &self.values,
                self.n_rows as i64,
                self.n_cols as i64,
                self.nnz as i64,
                &feat_slice,
                &mut out,
                d as i64,
            )?;
        }

        let out_shape = crate::Shape::from_dims(&[self.n_rows, d]);
        Ok(CudaStorage {
            slice: CudaStorageSlice::F32(out),
            device: device.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// Raw cuSPARSE SpMM (f32, CSR × row-major dense)
// ---------------------------------------------------------------------------
//
// Safety contract: all pointers must be valid CUDA device pointers on the same
// device as `handle`.  `out` must be zero-initialised (beta=0.0).
//
// TODO: add f16 / bf16 overloads using CUDA_R_16F / CUDA_R_16BF + CUSPARSE_COMPUTE_32F.
//
/// Execute `out = A × feat` using cuSPARSE's generic SpMM API.
///
/// # Arguments
/// All `CudaSlice` arguments must be contiguous device memory on the same GPU.
///
/// * `row_ptr`  — CSR row pointer `[N+1]` (i32)
/// * `col_idx`  — CSR column indices `[E]` (i32)
/// * `values`   — non-zero values `[E]` (f32)
/// * `n_rows`   — A.rows = output feature rows
/// * `n_cols`   — A.cols = input feature rows
/// * `nnz`      — number of non-zeros
/// * `feat`     — dense input  `[n_cols, D]` row-major (f32)
/// * `out`      — dense output `[n_rows, D]` row-major (f32, must be zero-init)
/// * `d`        — feature width D
///
/// # TODO (cuSPARSE call)
///
/// Replace the `unimplemented!()` below with:
/// ```ignore
/// // 1. Create handle
/// let mut handle = MaybeUninit::uninit();
/// sp::cusparseCreate(handle.as_mut_ptr()).result()...;
/// let handle = handle.assume_init();
///
/// // 2. Create sparse matrix descriptor (CSR)
/// let (row_ptr_raw, _rp_guard) = row_ptr.device_ptr(row_ptr.stream());
/// let (col_idx_raw, _ci_guard) = col_idx.device_ptr(col_idx.stream());
/// let (val_raw, _v_guard)      = values.device_ptr(values.stream());
/// let mut mat_a = MaybeUninit::uninit();
/// sp::cusparseCreateCsr(
///     mat_a.as_mut_ptr(),
///     n_rows, n_cols, nnz,
///     row_ptr_raw as *mut c_void,
///     col_idx_raw as *mut c_void,
///     val_raw     as *mut c_void,
///     sp::cusparseIndexType_t::CUSPARSE_INDEX_32I,
///     sp::cusparseIndexType_t::CUSPARSE_INDEX_32I,
///     sp::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
///     sp::cudaDataType::CUDA_R_32F,
/// ).result()...;
/// let mat_a = mat_a.assume_init();
///
/// // 3. Dense matrix B = feat [n_cols, D] row-major
/// let (feat_raw, _f_guard) = feat.device_ptr(feat.stream());
/// let mut mat_b = MaybeUninit::uninit();
/// sp::cusparseCreateDnMat(
///     mat_b.as_mut_ptr(), n_cols, d, /*ld=*/d,
///     feat_raw as *mut c_void,
///     sp::cudaDataType::CUDA_R_32F,
///     sp::cusparseOrder_t::CUSPARSE_ORDER_ROW,
/// ).result()...;
/// let mat_b = mat_b.assume_init();
///
/// // 4. Dense matrix C = out [n_rows, D] row-major
/// let (out_raw, _o_guard) = out.device_ptr_mut(out.stream());
/// let mut mat_c = MaybeUninit::uninit();
/// sp::cusparseCreateDnMat(
///     mat_c.as_mut_ptr(), n_rows, d, /*ld=*/d,
///     out_raw as *mut c_void,
///     sp::cudaDataType::CUDA_R_32F,
///     sp::cusparseOrder_t::CUSPARSE_ORDER_ROW,
/// ).result()...;
/// let mat_c = mat_c.assume_init();
///
/// // 5. Query workspace size
/// let alpha = 1.0f32;
/// let beta  = 0.0f32;
/// let mut buf_sz: usize = 0;
/// sp::cusparseSpMM_bufferSize(
///     handle,
///     sp::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
///     sp::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
///     &alpha as *const f32 as *const c_void,
///     mat_a, mat_b,
///     &beta  as *const f32 as *const c_void,
///     mat_c,
///     sp::cudaDataType::CUDA_R_32F,
///     sp::cusparseSpMMAlg_t::CUSPARSE_SPMM_CSR_ALG1,
///     &mut buf_sz,
/// ).result()...;
///
/// // 6. Allocate workspace (stream-ordered for best perf)
/// let work_buf: CudaSlice<u8> = stream.alloc_async::<u8>(buf_sz.max(1))?;
/// let (work_raw, _w_guard) = work_buf.device_ptr(work_buf.stream());
///
/// // 7. Execute SpMM
/// sp::cusparseSpMM(
///     handle,
///     sp::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
///     sp::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
///     &alpha as *const f32 as *const c_void,
///     mat_a, mat_b,
///     &beta  as *const f32 as *const c_void,
///     mat_c,
///     sp::cudaDataType::CUDA_R_32F,
///     sp::cusparseSpMMAlg_t::CUSPARSE_SPMM_CSR_ALG1,
///     work_raw as *mut c_void,
/// ).result()...;
///
/// // 8. Teardown descriptors and handle
/// sp::cusparseDestroySpMat(mat_a).result()...;
/// sp::cusparseDestroyDnMat(mat_b).result()...;
/// sp::cusparseDestroyDnMat(mat_c).result()...;
/// sp::cusparseDestroy(handle).result()...;
/// ```
///
/// When the TODO is resolved, also add `cusparse` to the workspace cudarc
/// features list in the root `Cargo.toml`, and add
/// `#[cfg(feature = "cusparse")] pub mod sparse;` guard in
/// `candle-core/src/cuda_backend/mod.rs`.
unsafe fn spmm_f32_raw(
    row_ptr: &CudaSlice<i32>,
    col_idx: &CudaSlice<i32>,
    values: &CudaSlice<f32>,
    n_rows: i64,
    n_cols: i64,
    nnz: i64,
    feat: &cudarc::driver::CudaView<'_, f32>,
    out: &mut CudaSlice<f32>,
    d: i64,
) -> Result<()> {
    use std::ffi::c_void;

    let stream = row_ptr.stream();

    // 1. Create cuSPARSE handle.
    let mut handle = MaybeUninit::uninit();
    sp::cusparseCreate(handle.as_mut_ptr())
        .result()
        .map_err(|e| crate::Error::Msg(format!("cusparseCreate: {:?}", e)).bt())?;
    let handle = handle.assume_init();

    // 2. Create sparse CSR matrix descriptor for A.
    let (rp_ptr, _rp_guard) = row_ptr.device_ptr(stream);
    let (ci_ptr, _ci_guard) = col_idx.device_ptr(stream);
    let (val_ptr, _val_guard) = values.device_ptr(stream);
    let mut mat_a = MaybeUninit::uninit();
    sp::cusparseCreateCsr(
        mat_a.as_mut_ptr(),
        n_rows,
        n_cols,
        nnz,
        rp_ptr as *mut c_void,
        ci_ptr as *mut c_void,
        val_ptr as *mut c_void,
        sp::cusparseIndexType_t::CUSPARSE_INDEX_32I,
        sp::cusparseIndexType_t::CUSPARSE_INDEX_32I,
        sp::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
        sp::cudaDataType::CUDA_R_32F,
    )
    .result()
    .map_err(|e| crate::Error::Msg(format!("cusparseCreateCsr: {:?}", e)).bt())?;
    let mat_a = mat_a.assume_init();

    // 3. Dense matrix B = feat [n_cols, D] row-major.
    let (feat_ptr, _feat_guard) = feat.device_ptr(stream);
    let mut mat_b = MaybeUninit::uninit();
    sp::cusparseCreateDnMat(
        mat_b.as_mut_ptr(),
        n_cols,
        d,
        d,
        feat_ptr as *mut c_void,
        sp::cudaDataType::CUDA_R_32F,
        sp::cusparseOrder_t::CUSPARSE_ORDER_ROW,
    )
    .result()
    .map_err(|e| crate::Error::Msg(format!("cusparseCreateDnMat(B): {:?}", e)).bt())?;
    let mat_b = mat_b.assume_init();

    // 4. Dense matrix C = out [n_rows, D] row-major.
    let (out_ptr, _out_guard) = out.device_ptr_mut(stream);
    let mut mat_c = MaybeUninit::uninit();
    sp::cusparseCreateDnMat(
        mat_c.as_mut_ptr(),
        n_rows,
        d,
        d,
        out_ptr as *mut c_void,
        sp::cudaDataType::CUDA_R_32F,
        sp::cusparseOrder_t::CUSPARSE_ORDER_ROW,
    )
    .result()
    .map_err(|e| crate::Error::Msg(format!("cusparseCreateDnMat(C): {:?}", e)).bt())?;
    let mat_c = mat_c.assume_init();

    // 5. Query workspace size.
    let alpha = 1.0f32;
    let beta = 0.0f32;
    let mut buf_sz: usize = 0;
    sp::cusparseSpMM_bufferSize(
        handle,
        sp::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
        sp::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha as *const f32 as *const c_void,
        mat_a,
        mat_b,
        &beta as *const f32 as *const c_void,
        mat_c,
        sp::cudaDataType::CUDA_R_32F,
        sp::cusparseSpMMAlg_t::CUSPARSE_SPMM_CSR_ALG1,
        &mut buf_sz,
    )
    .result()
    .map_err(|e| crate::Error::Msg(format!("cusparseSpMM_bufferSize: {:?}", e)).bt())?;

    // 6. Allocate workspace (stream-ordered).
    // We need to allocate on the same device as the other buffers. Use cudarc driver directly.
    let work_buf: CudaSlice<u8> = {
        let len = buf_sz.max(1);
        // alloc_async is not available here without CudaDevice; use cuMemAlloc_v2 via
        // a fresh alloc on the same stream.
        // row_ptr.stream() gives us a &CudaStream; we can allocate via stream.alloc::<u8>
        row_ptr.stream().alloc::<u8>(len).map_err(|e| {
            crate::Error::Msg(format!("workspace alloc: {:?}", e)).bt()
        })?
    };
    let (work_ptr, _work_guard) = work_buf.device_ptr(stream);

    // 7. Execute SpMM.
    sp::cusparseSpMM(
        handle,
        sp::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
        sp::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha as *const f32 as *const c_void,
        mat_a,
        mat_b,
        &beta as *const f32 as *const c_void,
        mat_c,
        sp::cudaDataType::CUDA_R_32F,
        sp::cusparseSpMMAlg_t::CUSPARSE_SPMM_CSR_ALG1,
        work_ptr as *mut c_void,
    )
    .result()
    .map_err(|e| crate::Error::Msg(format!("cusparseSpMM: {:?}", e)).bt())?;

    // 8. Teardown descriptors and handle.
    sp::cusparseDestroySpMat(mat_a)
        .result()
        .map_err(|e| crate::Error::Msg(format!("cusparseDestroySpMat: {:?}", e)).bt())?;
    sp::cusparseDestroyDnMat(mat_b)
        .result()
        .map_err(|e| crate::Error::Msg(format!("cusparseDestroyDnMat(B): {:?}", e)).bt())?;
    sp::cusparseDestroyDnMat(mat_c)
        .result()
        .map_err(|e| crate::Error::Msg(format!("cusparseDestroyDnMat(C): {:?}", e)).bt())?;
    sp::cusparseDestroy(handle)
        .result()
        .map_err(|e| crate::Error::Msg(format!("cusparseDestroy: {:?}", e)).bt())?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Unit-testable COO→CSR conversion helper (no CUDA needed)
// ---------------------------------------------------------------------------

/// Convert COO format to CSR row-pointer array on the CPU.
///
/// Exposed for unit testing without a CUDA device.
///
/// # Arguments
/// * `dst_row` — destination node IDs (COO row indices), length E.
/// * `n_nodes` — total node count N.
///
/// # Returns
/// `row_ptr` of length N+1 where `row_ptr[i]` = number of edges with dst < i.
pub fn coo_to_csr_rowptr(dst_row: &[i32], n_nodes: usize) -> Vec<i32> {
    let mut in_degree = vec![0i32; n_nodes];
    for &dst in dst_row {
        in_degree[dst as usize] += 1;
    }
    let mut row_ptr = vec![0i32; n_nodes + 1];
    for i in 0..n_nodes {
        row_ptr[i + 1] = row_ptr[i] + in_degree[i];
    }
    row_ptr
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify COO→CSR conversion for a small 3-node, 4-edge graph:
    ///   0→1, 0→2, 1→2, 2→0
    /// In-degree: node0=1, node1=1, node2=2
    /// row_ptr: [0, 1, 2, 4]
    #[test]
    fn test_coo_to_csr_rowptr() {
        // dst_row: destinations of edges [0→1, 0→2, 1→2, 2→0]
        let dst_row = vec![1i32, 2, 2, 0];
        let row_ptr = coo_to_csr_rowptr(&dst_row, 3);
        assert_eq!(row_ptr, vec![0, 1, 2, 4]);
    }

    /// Verify that col_idx (source IDs sorted by destination) is filled correctly.
    ///
    /// Same graph: 0→1, 0→2, 1→2, 2→0
    /// Sorted by dst:
    ///   dst=0: src=2
    ///   dst=1: src=0
    ///   dst=2: src=0, src=1
    /// col_idx = [2, 0, 0, 1]
    #[test]
    fn test_col_idx_ordering() {
        let src_row = vec![0i32, 0, 1, 2]; // sources
        let dst_row = vec![1i32, 2, 2, 0]; // destinations
        let n_nodes = 3;
        let e = src_row.len();

        let row_ptr = coo_to_csr_rowptr(&dst_row, n_nodes);
        let mut col_idx = vec![0i32; e];
        let mut cursor = row_ptr[..n_nodes].to_vec();
        for k in 0..e {
            let dst = dst_row[k] as usize;
            let pos = cursor[dst] as usize;
            col_idx[pos] = src_row[k];
            cursor[dst] += 1;
        }
        // Expected: dst=0→src=2, dst=1→src=0, dst=2→(src=0,src=1)
        assert_eq!(col_idx, vec![2, 0, 0, 1]);
    }

    /// Verify row_ptr sentinel: last entry = E.
    #[test]
    fn test_rowptr_last_entry() {
        let dst_row: Vec<i32> = (0..100).map(|i| i % 7).collect();
        let row_ptr = coo_to_csr_rowptr(&dst_row, 7);
        assert_eq!(*row_ptr.last().unwrap(), 100);
    }
}
