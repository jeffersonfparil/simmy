use crate::linalg::operations::{MatrixOps, Params, TensorOps};
use crate::linalg::tensor::GpuTensor;
use anyhow::{Result, ensure};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::Buffer;
use wgpu::util::{BufferInitDescriptor, DeviceExt};

/// Matrix Multiplication Parameters
///
/// The kernel computes:
///
/// ```text
/// A × B = C
///
/// A: n × p
/// B: p × k
/// C: n × k
/// ```
///
/// In addition to the matrix dimensions, this structure contains the
/// storage layout of each tensor, allowing the kernel to operate on
/// arbitrary matrix views rather than requiring contiguous storage.
///
/// * `n` is the number of rows in `A` and `C`.
/// * `p` is the shared contraction dimension.
/// * `k` is the number of columns in `B` and `C`.
///
/// * `a_offset` is the starting element of `A` within its backing storage.
/// * `a_row_stride` is the storage stride between rows of `A`.
/// * `a_col_stride` is the storage stride between columns of `A`.
///
/// * `b_offset` is the starting element of `B` within its backing storage.
/// * `b_row_stride` is the storage stride between rows of `B`.
/// * `b_col_stride` is the storage stride between columns of `B`.
///
/// * `c_offset` is the starting element of `C` within its backing storage.
/// * `c_row_stride` is the storage stride between rows of `C`.
/// * `c_col_stride` is the storage stride between columns of `C`.
///
/// The struct is marked `#[repr(C)]` and derives `Pod` and `Zeroable`
/// so that it can be safely transferred directly to GPU memory and
/// interpreted by WGSL shaders.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MatMulParams {
    pub n: u32,
    pub p: u32,
    pub k: u32,

    pub a_offset: u32,
    pub a_row_stride: u32,
    pub a_col_stride: u32,

    pub b_offset: u32,
    pub b_row_stride: u32,
    pub b_col_stride: u32,

    pub c_offset: u32,
    pub c_row_stride: u32,
    pub c_col_stride: u32,
}

impl MatrixOps<'_> {
    /// Perform GPU-accelerated matrix multiplication.
    ///
    /// Computes:
    ///
    /// ```text
    /// A × B = C
    ///
    /// A: n × p
    /// B: p × k
    /// C: n × k
    /// ```
    ///
    /// Both input tensors must be rank-2 matrices and must have compatible
    /// inner dimensions.
    ///
    /// The operation is executed using a WGSL compute kernel and returns a
    /// newly allocated tensor containing the result.
    ///
    /// * `a` is the left-hand matrix.
    /// * `b` is the right-hand matrix.
    ///
    /// # Returns
    /// Returns a tensor with shape:
    /// ```text
    /// [a.shape[0], b.shape[1]]
    /// ```
    ///
    /// # Errors
    /// Returns an error if:
    /// * `a` is not rank 2.
    /// * `b` is not rank 2.
    /// * The matrices have incompatible dimensions.
    pub fn multiply(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        ensure!(a.shape.len() == 2, "A must be rank 2!");
        ensure!(b.shape.len() == 2, "B must be rank 2!");
        ensure!(
            a.shape[1] == b.shape[0],
            "Incomaptible shapes: A {:?} x B {:?}!",
            a.shape,
            b.shape
        );
        let n = a.shape[0];
        let p = a.shape[1];
        let k = b.shape[1];
        let params = MatMulParams {
            n,
            p,
            k,
            a_offset: a.offset,
            a_row_stride: a.strides[0],
            a_col_stride: a.strides[1],
            b_offset: b.offset,
            b_row_stride: b.strides[0],
            b_col_stride: b.strides[1],
            c_offset: 0,
            c_row_stride: k,
            c_col_stride: 1,
        };
        let kernel_source: &str = include_str!("wgsl/matmul.wgsl");
        let c_buffer =
            self.execute_binary_kernel(Params::Multiplication(params), kernel_source, a, b)?;
        GpuTensor::from_buffer(Arc::new(c_buffer), vec![n, k], None, None)
    }
}

// TODO: implement...
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct TensorMulParams {}

impl TensorOps<'_> {
    pub fn multiply(&self, _a: &GpuTensor, _b: &GpuTensor) -> Result<GpuTensor> {
        todo!(
            "Implement for tensors of arbitrary ranks (as long as they are compatible) and along whichever dimension to sum over!"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::context::GpuContext;
    fn context() -> GpuContext {
        pollster::block_on(GpuContext::new()).expect("Failed to create GPU context")
    }
    #[test]
    fn matmul_returns_expected_output_shape() -> Result<()> {
        let ctx = context();
        let a = GpuTensor::from_f32(
            &ctx,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            None,
            None,
        )?;
        let b = GpuTensor::from_f32(
            &ctx,
            &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            vec![3, 2],
            None,
            None,
        )?;
        let ops = MatrixOps { ctx: &ctx };
        let c = ops.multiply(&a, &b).expect("Matrix multiplication failed");
        assert_eq!(c.shape, vec![2, 2]);
        Ok(())
    }
    #[test]
    fn matmul_rejects_rank_1_a() -> Result<()> {
        let ctx = context();
        let a = GpuTensor::from_f32(&ctx, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6], None, None)?;
        let b = GpuTensor::from_f32(&ctx, &[1.0; 6], vec![3, 2], None, None)?;
        let ops = MatrixOps { ctx: &ctx };
        assert!(ops.multiply(&a, &b).is_err());
        Ok(())
    }
    #[test]
    fn matmul_rejects_rank_1_b() -> Result<()> {
        let ctx = context();
        let a = GpuTensor::from_f32(&ctx, &[1.0; 6], vec![2, 3], None, None)?;
        let b = GpuTensor::from_f32(&ctx, &[1.0; 6], vec![6], None, None)?;
        let ops = MatrixOps { ctx: &ctx };
        assert!(ops.multiply(&a, &b).is_err());
        Ok(())
    }
    #[test]
    fn matmul_rejects_incompatible_shapes() -> Result<()> {
        let ctx = context();
        let a = GpuTensor::from_f32(&ctx, &[1.0; 6], vec![2, 3], None, None)?;
        let b = GpuTensor::from_f32(&ctx, &[1.0; 8], vec![4, 2], None, None)?;
        let ops = MatrixOps { ctx: &ctx };
        let result = ops.multiply(&a, &b);
        assert!(result.is_err());
        Ok(())
    }
    #[test]
    fn matmul_accepts_square_matrices() -> Result<()> {
        let ctx = context();
        let a = GpuTensor::from_f32(&ctx, &[1.0; 16], vec![4, 4], None, None)?;
        let b = GpuTensor::from_f32(&ctx, &[1.0; 16], vec![4, 4], None, None)?;
        let ops = MatrixOps { ctx: &ctx };
        let c = ops.multiply(&a, &b).expect("Matrix multiplication failed");
        assert_eq!(c.shape, vec![4, 4]);
        Ok(())
    }
    #[test]
    fn matmul_accepts_non_square_matrices() -> Result<()> {
        let ctx = context();
        let a = GpuTensor::from_f32(&ctx, &[1.0; 15], vec![5, 3], None, None)?;
        let b = GpuTensor::from_f32(&ctx, &[1.0; 21], vec![3, 7], None, None)?;
        let ops = MatrixOps { ctx: &ctx };
        let c = ops.multiply(&a, &b).expect("Matrix multiplication failed");
        assert_eq!(c.shape, vec![5, 7]);
        Ok(())
    }
}
