use crate::linalg::operations::{MatrixOps, MatrixParams, TensorOps, TensorParams};
use crate::linalg::tensor::GpuTensor;
use anyhow::{Result, ensure};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

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
pub struct MatrixMulParams {
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

pub const MAX_RANK: usize = 8;

/// Tensor Contraction Parameters
///
/// Describes an arbitrary-rank tensor contraction:
///
/// ```text
/// C = contract(A, B)
/// ```
///
/// where one or more axes of `A` are summed against corresponding
/// axes of `B`.
///
/// Matrix multiplication is a special case:
///
/// ```text
/// A[m, k]
/// B[k, n]
///
/// C[m, n]
///
/// a_contract_axes = [1]
/// b_contract_axes = [0]
/// ```
///
/// The result tensor is assumed to be laid out as:
///
/// ```text
/// C = [A free axes] + [B free axes]
/// ```
///
/// where "free axes" are axes that do not participate in the
/// contraction.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct TensorMulParams {
    pub a_rank: u32,
    pub b_rank: u32,
    pub c_rank: u32,

    pub contraction_rank: u32,

    pub c_elements: u32,

    pub a_shape: [u32; MAX_RANK],
    pub b_shape: [u32; MAX_RANK],
    pub c_shape: [u32; MAX_RANK],

    pub a_offset: u32,
    pub a_strides: [u32; MAX_RANK],

    pub b_offset: u32,
    pub b_strides: [u32; MAX_RANK],

    pub c_offset: u32,
    pub c_strides: [u32; MAX_RANK],

    pub a_contract_axes: [u32; MAX_RANK],
    pub b_contract_axes: [u32; MAX_RANK],
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
        let params = MatrixMulParams {
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
            self.execute_binary_kernel(MatrixParams::Multiplication(params), kernel_source, a, b)?;
        GpuTensor::from_buffer(Arc::new(c_buffer), vec![n, k], None, None)
    }
}

impl TensorOps<'_> {
    pub fn contract(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        a_contract_axes: &[usize],
        b_contract_axes: &[usize],
    ) -> Result<GpuTensor> {
        ensure!(
            a_contract_axes.len() == b_contract_axes.len(),
            "Contraction axis count mismatch"
        );
        let contraction_rank = a_contract_axes.len();
        ensure!(a.shape.len() <= MAX_RANK, "A rank exceeds MAX_RANK");
        ensure!(b.shape.len() <= MAX_RANK, "B rank exceeds MAX_RANK");
        for i in 0..contraction_rank {
            let a_axis = a_contract_axes[i];
            let b_axis = b_contract_axes[i];
            ensure!(
                a.shape[a_axis] == b.shape[b_axis],
                "Contracted dimensions must match: \
                A axis {} ({}) != B axis {} ({})",
                a_axis,
                a.shape[a_axis],
                b_axis,
                b.shape[b_axis],
            );
        }
        // Build output shape:
        // C = A free axes + B free axes
        let mut c_shape = Vec::new();
        for axis in 0..a.shape.len() {
            if !a_contract_axes.contains(&axis) {
                c_shape.push(a.shape[axis]);
            }
        }
        for axis in 0..b.shape.len() {
            if !b_contract_axes.contains(&axis) {
                c_shape.push(b.shape[axis]);
            }
        }
        ensure!(c_shape.len() <= MAX_RANK, "Result rank exceeds MAX_RANK");
        // Convert shapes
        let mut a_shape = [0u32; MAX_RANK];
        let mut b_shape = [0u32; MAX_RANK];
        let mut c_shape_arr = [0u32; MAX_RANK];
        a_shape[..a.shape.len()].copy_from_slice(&a.shape[..]);
        b_shape[..b.shape.len()].copy_from_slice(&b.shape[..]);
        c_shape_arr[..c_shape.len()].copy_from_slice(&c_shape[..]);
        // Convert strides
        let mut a_strides = [0u32; MAX_RANK];
        let mut b_strides = [0u32; MAX_RANK];
        let mut c_strides = [0u32; MAX_RANK];
        a_strides[..a.strides.len()].copy_from_slice(&a.strides[..]);
        b_strides[..b.strides.len()].copy_from_slice(&b.strides[..]);
        // Create contiguous output strides
        let mut stride = 1u32;
        for i in (0..c_shape.len()).rev() {
            c_strides[i] = stride;
            stride *= c_shape[i];
        }
        // Contract axes
        let mut a_contract = [0u32; MAX_RANK];
        let mut b_contract = [0u32; MAX_RANK];
        for i in 0..contraction_rank {
            a_contract[i] = a_contract_axes[i] as u32;
            b_contract[i] = b_contract_axes[i] as u32;
        }
        let c_elements = c_shape.iter().copied().product::<u32>();
        let params = TensorMulParams {
            a_rank: a.shape.len() as u32,
            b_rank: b.shape.len() as u32,
            c_rank: c_shape.len() as u32,
            contraction_rank: contraction_rank as u32,
            c_elements,
            a_shape,
            b_shape,
            c_shape: c_shape_arr,
            a_offset: a.offset,
            a_strides,
            b_offset: b.offset,
            b_strides,
            c_offset: 0,
            c_strides,
            a_contract_axes: a_contract,
            b_contract_axes: b_contract,
        };
        let buffer = self.execute_binary_kernel(
            TensorParams::Multiplication(params),
            include_str!("wgsl/tenmul.wgsl"),
            a,
            b,
        )?;
        GpuTensor::from_buffer(Arc::new(buffer), c_shape, None, None)
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

    ///////////////////////////////////////////
    // Tensor Contraction
    ///////////////////////////////////////////

    #[test]
    fn contract_matrix_multiplication_2x2() -> Result<()> {
        let ctx = context();

        let ops = TensorOps { ctx: &ctx };

        let a = GpuTensor::from_f32(&ctx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2], None, None)?;

        let b = GpuTensor::from_f32(&ctx, &[5.0, 6.0, 7.0, 8.0], vec![2, 2], None, None)?;

        let c = ops.contract(&a, &b, &[1], &[0])?;

        assert_eq!(c.shape, vec![2, 2]);

        assert_eq!(c.to_vec_f32(&ctx)?, vec![19.0, 22.0, 43.0, 50.0,]);

        Ok(())
    }

    #[test]
    fn contract_matrix_multiplication_rectangular() -> Result<()> {
        let ctx = context();

        let ops = TensorOps { ctx: &ctx };

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

        let c = ops.contract(&a, &b, &[1], &[0])?;

        assert_eq!(c.shape, vec![2, 2]);

        assert_eq!(c.to_vec_f32(&ctx)?, vec![58.0, 64.0, 139.0, 154.0,]);

        Ok(())
    }

    #[test]
    fn contract_rejects_axis_count_mismatch() {
        let ctx = context();

        let ops = TensorOps { ctx: &ctx };

        let a = GpuTensor::from_f32(&ctx, &[1.0; 24], vec![2, 3, 4], None, None).unwrap();

        let b = GpuTensor::from_f32(&ctx, &[1.0; 24], vec![3, 4, 2], None, None).unwrap();

        assert!(ops.contract(&a, &b, &[1, 2], &[0],).is_err());
    }

    #[test]
    fn contract_rejects_incompatible_contract_dimensions() {
        let ctx = context();

        let ops = TensorOps { ctx: &ctx };

        let a = GpuTensor::from_f32(&ctx, &[1.0; 24], vec![2, 3, 4], None, None).unwrap();

        let b = GpuTensor::from_f32(&ctx, &[1.0; 40], vec![5, 4, 2], None, None).unwrap();

        assert!(ops.contract(&a, &b, &[1], &[0],).is_err());
    }

    #[test]
    fn contract_returns_correct_output_shape() -> Result<()> {
        let ctx = context();

        let ops = TensorOps { ctx: &ctx };

        let a = GpuTensor::from_f32(&ctx, &[1.0; 24], vec![2, 3, 4], None, None)?;

        let b = GpuTensor::from_f32(&ctx, &[1.0; 20], vec![4, 5], None, None)?;

        let c = ops.contract(&a, &b, &[2], &[0])?;

        //
        // A free axes: [2,3]
        // B free axes: [5]
        //
        assert_eq!(c.shape, vec![2, 3, 5]);

        Ok(())
    }

    #[test]
    fn contract_preserves_contiguous_output_layout() -> Result<()> {
        let ctx = context();

        let ops = TensorOps { ctx: &ctx };

        let a = GpuTensor::from_f32(&ctx, &[1.0; 24], vec![2, 3, 4], None, None)?;

        let b = GpuTensor::from_f32(&ctx, &[1.0; 20], vec![4, 5], None, None)?;

        let c = ops.contract(&a, &b, &[2], &[0])?;

        assert_eq!(c.shape, vec![2, 3, 5]);
        assert_eq!(c.strides, vec![15, 5, 1]);
        assert_eq!(c.offset, 0);

        Ok(())
    }
}
