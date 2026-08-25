use crate::linalg::operations::{MatrixOps, MatrixParams, TensorOps};
use crate::linalg::tensor::GpuTensor;
use anyhow::{Result, ensure};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

/// Matrix Addition Parameters
///
/// The kernel computes:
///
/// ```text
/// A + B = C
///
/// A: n × p
/// B: n × p
/// C: n × p
/// ```
///
/// In addition to the matrix dimensions, this structure contains the
/// storage layout of each tensor, allowing the kernel to operate on
/// arbitrary matrix views rather than requiring contiguous storage.
///
/// * `n` is the number of rows in `A`, `B`, and `C`.
/// * `p` is the number of columns in `A`, `B`, and `C`.
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
pub struct MatrixAddParams {
    pub n: u32,
    pub p: u32,

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

const MAX_RANK: usize = 8;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct TensorAddParams {
    /// Number of tensor dimensions.
    pub rank: u32,
    /// Total number of logical tensor elements.
    pub n_elements: u32,
    /// Shape of the logical tensor.
    pub shape: [u32; MAX_RANK],
    /// Storage layout of A.
    pub a_offset: u32,
    pub a_strides: [u32; MAX_RANK],
    /// Storage layout of B.
    pub b_offset: u32,
    pub b_strides: [u32; MAX_RANK],
    /// Storage layout of C.
    pub c_offset: u32,
    pub c_strides: [u32; MAX_RANK],
}

impl MatrixOps<'_> {
    /// Perform GPU-accelerated matrix addition.
    ///
    /// Computes:
    ///
    /// ```text
    /// A + B = C
    ///
    /// A: n × p
    /// B: n × p
    /// C: n × p
    /// ```
    ///
    /// Both input tensors must be rank-2 matrices with identical shapes.
    ///
    /// The operation is executed using a WGSL compute kernel and returns a
    /// newly allocated tensor containing the result.
    ///
    /// The input tensors may represent arbitrary matrix views. The kernel
    /// uses tensor metadata, including strides and offsets, to determine
    /// how elements are accessed within the underlying storage.
    ///
    /// * `a` is the left-hand matrix.
    /// * `b` is the right-hand matrix.
    ///
    /// # Returns
    /// Returns a tensor with shape:
    ///
    /// ```text
    /// [a.shape[0], a.shape[1]]
    /// ```
    ///
    /// # Errors
    /// Returns an error if:
    /// * `a` is not rank 2.
    /// * `b` is not rank 2.
    /// * The matrices have different shapes.
    pub fn add(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        ensure!(a.shape.len() == 2, "A must be rank 2!");
        ensure!(b.shape.len() == 2, "B must be rank 2!");
        ensure!(
            a.shape == b.shape,
            "Incomaptible sahpes: A {:?} x B {:?}!",
            a.shape,
            b.shape
        );
        let n = a.shape[0];
        let p = a.shape[1];
        let params = MatrixAddParams {
            n,
            p,
            a_offset: a.offset,
            a_row_stride: a.strides[0],
            a_col_stride: a.strides[1],
            b_offset: b.offset,
            b_row_stride: b.strides[0],
            b_col_stride: b.strides[1],
            c_offset: 0,
            c_row_stride: p,
            c_col_stride: 1,
        };
        let kernel_source: &str = include_str!("wgsl/matadd.wgsl");
        let c_buffer =
            self.execute_binary_kernel(MatrixParams::Addition(params), kernel_source, a, b)?;
        GpuTensor::from_buffer(Arc::new(c_buffer), vec![n, p], None, None)
    }
}

impl TensorOps<'_> {
    pub fn add(&self, a: &GpuTensor, b: &GpuTensor) -> Result<()> {
        ensure!(
            a.shape == b.shape,
            "Incompatible shapes: {:?} + {:?}",
            a.shape,
            b.shape
        );

        ensure!(
            a.shape.len() <= MAX_RANK,
            "Tensor rank exceeds MAX_RANK ({})",
            MAX_RANK
        );

        let mut shape = [0u32; MAX_RANK];
        let mut a_strides = [0u32; MAX_RANK];
        let mut b_strides = [0u32; MAX_RANK];
        let mut c_strides = [0u32; MAX_RANK];

        for i in 0..a.shape.len() {
            shape[i] = a.shape[i];
            a_strides[i] = a.strides[i];
            b_strides[i] = b.strides[i];
        }

        let mut stride = 1u32;

        for i in (0..a.shape.len()).rev() {
            c_strides[i] = stride;
            stride *= a.shape[i];
        }

        let params = TensorAddParams {
            rank: a.shape.len() as u32,
            n_elements: a.shape.iter().product(),

            shape,

            a_offset: a.offset,
            a_strides,

            b_offset: b.offset,
            b_strides,

            c_offset: 0,
            c_strides,
        };
        Ok(())
    }
}

#[cfg(test)]
mod add_tests {
    use super::*;
    use crate::linalg::context::GpuContext;
    fn context() -> GpuContext {
        pollster::block_on(GpuContext::new()).expect("Failed to create GPU context")
    }
    #[test]
    fn add_returns_correct_result() -> Result<()> {
        let ctx = context();
        let ops = MatrixOps { ctx: &ctx };
        let a = GpuTensor::from_f32(&ctx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2], None, None)?;
        let b = GpuTensor::from_f32(&ctx, &[10.0, 20.0, 30.0, 40.0], vec![2, 2], None, None)?;
        let c = ops.add(&a, &b)?;
        assert_eq!(c.to_vec_f32(&ctx)?, vec![11.0, 22.0, 33.0, 44.0]);
        Ok(())
    }
    #[test]
    fn add_returns_correct_shape() -> Result<()> {
        let ctx = context();
        let ops = MatrixOps { ctx: &ctx };
        let a = GpuTensor::from_f32(&ctx, &[1.0; 6], vec![2, 3], None, None)?;
        let b = GpuTensor::from_f32(&ctx, &[2.0; 6], vec![2, 3], None, None)?;
        let c = ops.add(&a, &b)?;
        assert_eq!(c.shape, vec![2, 3]);
        Ok(())
    }
    #[test]
    fn add_rejects_rank_1_input() {
        let ctx = context();
        let ops = MatrixOps { ctx: &ctx };
        let a = GpuTensor::from_f32(&ctx, &[1.0, 2.0, 3.0], vec![3], None, None).unwrap();
        let b = GpuTensor::from_f32(&ctx, &[4.0, 5.0, 6.0], vec![3], None, None).unwrap();
        assert!(ops.add(&a, &b).is_err());
    }
    #[test]
    fn add_rejects_different_shapes() {
        let ctx = context();
        let ops = MatrixOps { ctx: &ctx };
        let a = GpuTensor::from_f32(&ctx, &[1.0; 6], vec![2, 3], None, None).unwrap();
        let b = GpuTensor::from_f32(&ctx, &[1.0; 6], vec![3, 2], None, None).unwrap();
        assert!(ops.add(&a, &b).is_err());
    }
    #[test]
    fn add_works_with_transposed_views() -> Result<()> {
        let ctx = context();
        let ops = MatrixOps { ctx: &ctx };
        let a = GpuTensor::from_f32(&ctx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2], None, None)?;
        let b = GpuTensor::from_f32(&ctx, &[10.0, 20.0, 30.0, 40.0], vec![2, 2], None, None)?;
        let at = a.transpose(None)?;
        let bt = b.transpose(None)?;
        let c = ops.add(&at, &bt)?;
        assert_eq!(c.to_vec_f32(&ctx)?, vec![11.0, 33.0, 22.0, 44.0]);
        assert_eq!(c.shape, vec![2, 2]);
        Ok(())
    }
    #[test]
    fn add_preserves_expected_output_layout() -> Result<()> {
        let ctx = context();
        let ops = MatrixOps { ctx: &ctx };
        let a = GpuTensor::from_f32(&ctx, &[1.0; 6], vec![2, 3], None, None)?;
        let b = GpuTensor::from_f32(&ctx, &[2.0; 6], vec![2, 3], None, None)?;
        let c = ops.add(&a, &b)?;
        assert_eq!(c.shape, vec![2, 3]);
        assert_eq!(c.strides, vec![3, 1]);
        assert_eq!(c.offset, 0);
        Ok(())
    }
}
