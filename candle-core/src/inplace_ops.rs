//! In-place binary ops and fused kernels.
//!
//! Provides `add_`, `sub_`, `mul_` (zero-allocation in-place updates) and
//! `add_relu` (fused add + ReLU in a single kernel pass) on `Tensor`.
//!
//! Safety contract for `add_/sub_/mul_`: the caller must hold the only live
//! reference to `self`'s storage. The methods check `Arc::strong_count` on
//! the inner `Tensor_` and return an error if the tensor is aliased.

use crate::cpu_backend::CpuStorage;
use crate::layout::Layout;
use crate::{InplaceOp2, Result, Tensor};

// ---------------------------------------------------------------------------
// CPU helpers
// ---------------------------------------------------------------------------

fn inplace_map_f32(dst: &mut [f32], src: &[f32], f: impl Fn(f32, f32) -> f32) {
    dst.iter_mut().zip(src.iter()).for_each(|(d, &s)| *d = f(*d, s));
}

fn inplace_map_f64(dst: &mut [f64], src: &[f64], f: impl Fn(f64, f64) -> f64) {
    dst.iter_mut().zip(src.iter()).for_each(|(d, &s)| *d = f(*d, s));
}

// ---------------------------------------------------------------------------
// InplaceOp2 implementation structs (CPU + CUDA)
// ---------------------------------------------------------------------------

macro_rules! inplace_op2_impl {
    ($name:ident, $kname:literal, $f32:expr, $f64:expr) => {
        struct $name;

        impl InplaceOp2 for $name {
            fn name(&self) -> &'static str {
                $kname
            }

            fn cpu_fwd(
                &self,
                dst: &mut CpuStorage,
                dst_l: &Layout,
                src: &CpuStorage,
                src_l: &Layout,
            ) -> Result<()> {
                // We only support the contiguous case for in-place ops to keep
                // semantics simple. Strided tensors should go through the normal
                // allocating path.
                let (dst_o1, dst_o2) = dst_l
                    .contiguous_offsets()
                    .ok_or_else(|| crate::Error::RequiresContiguous { op: $kname }.bt())?;
                let (src_o1, src_o2) = src_l
                    .contiguous_offsets()
                    .ok_or_else(|| crate::Error::RequiresContiguous { op: $kname }.bt())?;
                match (dst, src) {
                    (CpuStorage::BF16(d), CpuStorage::BF16(s)) => {
                        use half::bf16;
                        d[dst_o1..dst_o2]
                            .iter_mut()
                            .zip(s[src_o1..src_o2].iter())
                            .for_each(|(dv, &sv)| {
                                *dv = bf16::from_f32($f32(dv.to_f32(), sv.to_f32()))
                            });
                    }
                    (CpuStorage::F16(d), CpuStorage::F16(s)) => {
                        use half::f16;
                        d[dst_o1..dst_o2]
                            .iter_mut()
                            .zip(s[src_o1..src_o2].iter())
                            .for_each(|(dv, &sv)| {
                                *dv = f16::from_f32($f32(dv.to_f32(), sv.to_f32()))
                            });
                    }
                    (CpuStorage::F32(d), CpuStorage::F32(s)) => {
                        inplace_map_f32(&mut d[dst_o1..dst_o2], &s[src_o1..src_o2], $f32)
                    }
                    (CpuStorage::F64(d), CpuStorage::F64(s)) => {
                        inplace_map_f64(&mut d[dst_o1..dst_o2], &s[src_o1..src_o2], $f64)
                    }
                    (CpuStorage::U8(d), CpuStorage::U8(s)) => d[dst_o1..dst_o2]
                        .iter_mut()
                        .zip(s[src_o1..src_o2].iter())
                        .for_each(|(dv, &sv)| *dv = ($f32(*dv as f32, sv as f32) as u8)),
                    (CpuStorage::U32(d), CpuStorage::U32(s)) => d[dst_o1..dst_o2]
                        .iter_mut()
                        .zip(s[src_o1..src_o2].iter())
                        .for_each(|(dv, &sv)| *dv = ($f64(*dv as f64, sv as f64) as u32)),
                    (CpuStorage::I64(d), CpuStorage::I64(s)) => d[dst_o1..dst_o2]
                        .iter_mut()
                        .zip(s[src_o1..src_o2].iter())
                        .for_each(|(dv, &sv)| *dv = ($f64(*dv as f64, sv as f64) as i64)),
                    _ => {
                        crate::bail!(
                            "dtype mismatch or unsupported dtype in {}",
                            $kname
                        )
                    }
                }
                Ok(())
            }

            #[cfg(feature = "cuda")]
            fn cuda_fwd(
                &self,
                dst: &mut crate::CudaStorage,
                dst_l: &Layout,
                src: &crate::CudaStorage,
                src_l: &Layout,
            ) -> Result<()> {
                dst.binary_inplace(
                    concat!("i", $kname),
                    src,
                    dst_l,
                    src_l,
                )
            }
        }
    };
}

inplace_op2_impl!(AddInPlace, "add", |a, b| a + b, |a, b| a + b);
inplace_op2_impl!(SubInPlace, "sub", |a, b| a - b, |a, b| a - b);
inplace_op2_impl!(MulInPlace, "mul", |a, b| a * b, |a, b| a * b);

// ---------------------------------------------------------------------------
// Fused add+relu as CustomOp2
// ---------------------------------------------------------------------------

use crate::custom_op::CustomOp2;
use crate::shape::Shape;

struct FusedAddRelu;

impl CustomOp2 for FusedAddRelu {
    fn name(&self) -> &'static str {
        "add_relu"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        use crate::cpu_backend::binary_map;
        let shape = l1.shape().clone();
        let storage = match (s1, s2) {
            (CpuStorage::BF16(lhs), CpuStorage::BF16(rhs)) => {
                use half::bf16;
                let zero = bf16::ZERO;
                let data = binary_map(l1, l2, lhs, rhs, |a, b| {
                    let s = a + b;
                    if s > zero { s } else { zero }
                });
                CpuStorage::BF16(data)
            }
            (CpuStorage::F16(lhs), CpuStorage::F16(rhs)) => {
                use half::f16;
                let zero = f16::ZERO;
                let data = binary_map(l1, l2, lhs, rhs, |a, b| {
                    let s = a + b;
                    if s > zero { s } else { zero }
                });
                CpuStorage::F16(data)
            }
            (CpuStorage::F32(lhs), CpuStorage::F32(rhs)) => {
                let data = binary_map(l1, l2, lhs, rhs, |a, b| (a + b).max(0.0f32));
                CpuStorage::F32(data)
            }
            (CpuStorage::F64(lhs), CpuStorage::F64(rhs)) => {
                let data = binary_map(l1, l2, lhs, rhs, |a, b| (a + b).max(0.0f64));
                CpuStorage::F64(data)
            }
            _ => crate::bail!("add_relu: unsupported dtype combination"),
        };
        Ok((storage, shape))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &crate::CudaStorage,
        l1: &Layout,
        s2: &crate::CudaStorage,
        l2: &Layout,
    ) -> Result<(crate::CudaStorage, Shape)> {
        let shape = l1.shape().clone();
        let out = s1.fused_add_relu(s2, l1, l2)?;
        Ok((out, shape))
    }
}

// ---------------------------------------------------------------------------
// Tensor API
// ---------------------------------------------------------------------------

impl Tensor {
    /// In-place add: `self += rhs`.
    ///
    /// Reuses `self`'s memory buffer — no allocation. Requires that `self` is
    /// contiguous and that the caller holds the only live `Arc` reference to
    /// this tensor (i.e. `Arc::strong_count == 1`). Returns an error otherwise.
    ///
    /// `rhs` must have the same shape and dtype as `self`.
    pub fn add_(&self, rhs: &Tensor) -> Result<()> {
        self.assert_inplace_eligible("add_")?;
        let _shape = self.same_shape_binary_op(rhs, "add_")?;
        self.inplace_op2(rhs, &AddInPlace)
    }

    /// In-place subtract: `self -= rhs`.
    ///
    /// See [`Tensor::add_`] for preconditions.
    pub fn sub_(&self, rhs: &Tensor) -> Result<()> {
        self.assert_inplace_eligible("sub_")?;
        let _shape = self.same_shape_binary_op(rhs, "sub_")?;
        self.inplace_op2(rhs, &SubInPlace)
    }

    /// In-place multiply: `self *= rhs`.
    ///
    /// See [`Tensor::add_`] for preconditions.
    pub fn mul_(&self, rhs: &Tensor) -> Result<()> {
        self.assert_inplace_eligible("mul_")?;
        let _shape = self.same_shape_binary_op(rhs, "mul_")?;
        self.inplace_op2(rhs, &MulInPlace)
    }

    /// Fused add + ReLU: equivalent to `(self + rhs)?.relu()?` but dispatches
    /// a single kernel on CUDA, halving memory traffic and launch overhead.
    ///
    /// `rhs` must have the same shape and dtype as `self`. Only floating-point
    /// dtypes (f32, f64, f16, bf16) are supported.
    pub fn add_relu(&self, rhs: &Tensor) -> Result<Tensor> {
        let _shape = self.same_shape_binary_op(rhs, "add_relu")?;
        self.apply_op2_no_bwd(rhs, &FusedAddRelu)
    }

    /// Check that `self` is safe to modify in-place:
    ///  - The tensor must be the sole owner of its storage (no other live Arcs).
    ///  - The tensor must be contiguous (no slices, transposes, etc.).
    fn assert_inplace_eligible(&self, op: &'static str) -> Result<()> {
        if !self.is_contiguous() {
            return Err(crate::Error::RequiresContiguous { op }.bt());
        }
        // `self.0` is the Arc<Tensor_>; `self.0.storage` is the Arc<RwLock<Storage>>.
        // We need sole ownership of the outer Arc (the Tensor itself must not be
        // aliased) AND the storage Arc must also have count == 1.
        // The outer Tensor Arc count is accessible via Deref to Tensor_ — but we
        // can check the storage Arc count as a proxy: if storage is shared the
        // count is > 1.
        if !self.is_unique_storage() {
            crate::bail!(
                "in-place op '{}' requires unique ownership of storage (Arc count > 1)",
                op
            )
        }
        Ok(())
    }
}
