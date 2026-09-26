use crate::linalg::kernel::{GpuKernel, Params};
use crate::linalg::operations::Operation;
use crate::linalg::params::{BinaryMatrixParams, ContractMatrixParams, UnaryMatrixParams};
use crate::linalg::tensor::GpuTensor;
use anyhow::{Result, ensure};

impl GpuTensor {
    pub fn params_unary_matrix(&self, op: Operation) -> Result<Params> {
        ensure!(self.shape.len() == 2, "A must be rank 2!");
        let params = UnaryMatrixParams {
            // Number of rows in `A`, and `C`.
            n: self.shape[0],
            // Number of columns in `A`, and `C`.
            p: self.shape[1],
            // Starting element of `A` within its backing storage.
            a_offset: self.offset,
            // Storage stride between rows of `A`.
            a_row_stride: self.strides[0],
            // Storage stride between columns of `A`.
            a_col_stride: self.strides[1],
            // Starting element of `C` within its backing storage.
            c_offset: 0,
            // Storage stride between rows of `C`.
            c_row_stride: self.shape[1],
            // Storage stride between columns of `C`.
            c_col_stride: 1,
            // Mathematical operation: op(A) -> C (see operations.rs & wgsl/opcodes.wgsl).
            op: op.unary_opcode()?,
        };
        Ok(Params::UnaryMatrix(params))
    }

    pub fn params_binary_matrix(&self, b: &Self, op: Operation) -> Result<Params> {
        ensure!(self.shape.len() == 2, "A must be rank 2!");
        ensure!(b.shape.len() == 2, "B must be rank 2!");
        ensure!(self.shape == b.shape, "A and B mush have the same shape!");
        let params = BinaryMatrixParams {
            // Number of rows in `A`, `B`, and `C`.
            n: self.shape[0],
            // Number of columns in `A`, `B`, and `C`.
            p: self.shape[1],
            // Starting element of `A` within its backing storage.
            a_offset: self.offset,
            // Storage stride between rows of `A`.
            a_row_stride: self.strides[0],
            // Storage stride between columns of `A`.
            a_col_stride: self.strides[1],
            // Starting element of `B` within its backing storage.
            b_offset: b.offset,
            // Storage stride between rows of `B`.
            b_row_stride: b.strides[0],
            // Storage stride between columns of `B`.
            b_col_stride: b.strides[1],
            // Starting element of `C` within its backing storage.
            c_offset: 0,
            // Storage stride between rows of `C`.
            c_row_stride: self.shape[1],
            // Storage stride between columns of `C`.
            c_col_stride: 1,
            // Mathematical operation: op(A, B) -> C (see operations.rs & wgsl/opcodes.wgsl).
            op: op.binary_opcode()?,
        };
        Ok(Params::BinaryMatrix(params))
    }

    pub fn params_contract_matrix(
        &self,
        b: &Self,
        op_pairwise: Operation,
        op_reduction: Operation,
    ) -> Result<Params> {
        ensure!(self.shape.len() == 2, "A must be rank 2!");
        ensure!(b.shape.len() == 2, "B must be rank 2!");
        ensure!(
            self.shape[1] == b.shape[0],
            "Incomaptible shapes: A {:?} x B {:?}!",
            self.shape,
            b.shape
        );
        let params = ContractMatrixParams {
            // Number of rows in `A` and `C`.
            n: self.shape[0],
            // Shared contraction dimension.
            p: self.shape[1],
            // Number of columns in `B` and `C`.
            k: b.shape[1],
            // Starting element of `A` within its backing storage.
            a_offset: self.offset,
            // Storage stride between rows of `A`.
            a_row_stride: self.strides[0],
            // Storage stride between columns of `A`.
            a_col_stride: self.strides[1],
            // Starting element of `B` within its backing storage.
            b_offset: b.offset,
            // Storage stride between rows of `B`.
            b_row_stride: b.strides[0],
            // Storage stride between columns of `B`.
            b_col_stride: b.strides[1],
            // Starting element of `C` within its backing storage.
            c_offset: 0,
            // Storage stride between rows of `C`.
            c_row_stride: b.shape[1],
            // Storage stride between columns of `C`.
            c_col_stride: 1,
            // Pairwise operation (see operations.rs & wgsl/opcodes.wgsl)
            op_pairwise: op_pairwise.contract_pairwise_opcode()?,
            // Reduction operation (see operations.rs & wgsl/opcodes.wgsl)
            op_reduction: op_reduction.contract_reduction_opcode()?,
        };
        Ok(Params::ContractMatrix(params))
    }
}

impl GpuKernel<'_> {
    // Unary
    pub fn abs_matrix(&self, a: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_unary_matrix(Operation::ABS)?, a, None)
    }
    pub fn neg_matrix(&self, a: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_unary_matrix(Operation::NEG)?, a, None)
    }
    pub fn sqrt_matrix(&self, a: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_unary_matrix(Operation::SQRT)?, a, None)
    }
    pub fn exp_matrix(&self, a: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_unary_matrix(Operation::EXP)?, a, None)
    }
    pub fn log_matrix(&self, a: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_unary_matrix(Operation::LOG)?, a, None)
    }
    pub fn sin_matrix(&self, a: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_unary_matrix(Operation::SIN)?, a, None)
    }
    pub fn cos_matrix(&self, a: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_unary_matrix(Operation::COS)?, a, None)
    }

    // Binary
    pub fn add_matrix(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary_matrix(b, Operation::ADD)?, a, Some(b))
    }
    pub fn sub_matrix(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary_matrix(b, Operation::SUB)?, a, Some(b))
    }
    pub fn mul_matrix(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary_matrix(b, Operation::MUL)?, a, Some(b))
    }
    pub fn div_matrix(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary_matrix(b, Operation::DIV)?, a, Some(b))
    }
    pub fn min_matrix(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary_matrix(b, Operation::MIN)?, a, Some(b))
    }
    pub fn max_matrix(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary_matrix(b, Operation::MAX)?, a, Some(b))
    }
    pub fn pow_matrix(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary_matrix(b, Operation::POW)?, a, Some(b))
    }
    pub fn atan2_matrix(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary_matrix(b, Operation::ATAN2)?, a, Some(b))
    }
    pub fn eq_matrix(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary_matrix(b, Operation::EQ)?, a, Some(b))
    }
    pub fn ne_matrix(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary_matrix(b, Operation::NE)?, a, Some(b))
    }
    pub fn lt_matrix(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary_matrix(b, Operation::LT)?, a, Some(b))
    }
    pub fn le_matrix(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary_matrix(b, Operation::LE)?, a, Some(b))
    }
    pub fn gt_matrix(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary_matrix(b, Operation::GT)?, a, Some(b))
    }
    pub fn ge_matrix(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary_matrix(b, Operation::GE)?, a, Some(b))
    }

    // Matrix multiplication (MUL --> ADD)
    pub fn matmul(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        let params = a.params_contract_matrix(b, Operation::MUL, Operation::ADD)?;
        self.execute_kernel(params, a, Some(b))
    }
    // Hadamard sum (ADD --> ADD)
    pub fn hadamard_sum_matrix(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        let params = a.params_contract_matrix(b, Operation::ADD, Operation::ADD)?;
        self.execute_kernel(params, a, Some(b))
    }
    // Min-Plus Algebra (ADD --> MIN)
    pub fn min_plus_matrix(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        let params = a.params_contract_matrix(b, Operation::ADD, Operation::MIN)?;
        self.execute_kernel(params, a, Some(b))
    }
    // Max-Plus Algebra (ADD --> MAX)
    pub fn max_plus_matrix(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        let params = a.params_contract_matrix(b, Operation::ADD, Operation::MAX)?;
        self.execute_kernel(params, a, Some(b))
    }
    // Min-Mul (MUL --> MIN)
    pub fn min_mul_matrix(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        let params = a.params_contract_matrix(b, Operation::MUL, Operation::MIN)?;
        self.execute_kernel(params, a, Some(b))
    }
    // Max-Mul (MUL --> MAX)
    pub fn max_mul_matrix(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        let params = a.params_contract_matrix(b, Operation::MUL, Operation::MAX)?;
        self.execute_kernel(params, a, Some(b))
    }
    // Boolean matrix multiplication (AND --> OR)
    pub fn matmul_bool(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        let params = a.params_contract_matrix(b, Operation::AND, Operation::OR)?;
        self.execute_kernel(params, a, Some(b))
    }
    // Equality Counting (EQ --> ADD)
    pub fn eqcount_matrix(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        let params = a.params_contract_matrix(b, Operation::EQ, Operation::ADD)?;
        self.execute_kernel(params, a, Some(b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::context::GpuContext;
    use crate::linalg::operations::Operation;

    fn context() -> GpuContext {
        pollster::block_on(GpuContext::new()).expect("Failed to create GPU context")
    }

    fn vector(ctx: &GpuContext, n: usize) -> Result<GpuTensor> {
        let data: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
        GpuTensor::from_f32(ctx, &data, &[n as u32], None, None)
    }

    fn matrix(ctx: &GpuContext, rows: usize, cols: usize) -> Result<GpuTensor> {
        let data: Vec<f32> = (0..rows * cols).map(|i| (i + 1) as f32).collect();
        GpuTensor::from_f32(ctx, &data, &[rows as u32, cols as u32], None, None)
    }
    ////////////////////////////////////////////
    // Unary parameter builder
    ////////////////////////////////////////////
    #[test]
    fn builds_unary_matrix_params() -> Result<()> {
        let ctx = context();
        let a = matrix(&ctx, 3, 4)?;
        let params = a.params_unary_matrix(Operation::ABS)?;
        match params {
            Params::UnaryMatrix(p) => {
                assert_eq!(p.n, 3);
                assert_eq!(p.p, 4);
                assert_eq!(p.a_offset, 0);
                assert_eq!(p.a_row_stride, 4);
                assert_eq!(p.a_col_stride, 1);
                assert_eq!(p.c_offset, 0);
                assert_eq!(p.c_row_stride, 4);
                assert_eq!(p.c_col_stride, 1);
                assert_eq!(p.op, 0); // ABS
            }
            _ => panic!("Expected Params::UnaryMatrix"),
        }
        Ok(())
    }
    #[test]
    fn unary_matrix_requires_rank_2() -> Result<()> {
        let ctx = context();
        let a = vector(&ctx, 4)?;
        assert!(a.params_unary_matrix(Operation::ABS).is_err());
        Ok(())
    }
    #[test]
    fn unary_matrix_propagates_invalid_operation() -> Result<()> {
        let ctx = context();
        let a = matrix(&ctx, 3, 4)?;
        assert!(a.params_unary_matrix(Operation::ADD).is_err());
        Ok(())
    }
    ////////////////////////////////////////////
    // Binary parameter builder
    ////////////////////////////////////////////
    #[test]
    fn builds_binary_matrix_params() -> Result<()> {
        let ctx = context();
        let a = matrix(&ctx, 3, 4)?;
        let b = matrix(&ctx, 3, 4)?;
        let params = a.params_binary_matrix(&b, Operation::ADD)?;
        match params {
            Params::BinaryMatrix(p) => {
                assert_eq!(p.n, 3);
                assert_eq!(p.p, 4);
                assert_eq!(p.a_offset, 0);
                assert_eq!(p.b_offset, 0);
                assert_eq!(p.a_row_stride, 4);
                assert_eq!(p.a_col_stride, 1);
                assert_eq!(p.b_row_stride, 4);
                assert_eq!(p.b_col_stride, 1);
                assert_eq!(p.c_row_stride, 4);
                assert_eq!(p.c_col_stride, 1);
                assert_eq!(p.op, 0); // ADD
            }
            _ => panic!("Expected Params::BinaryMatrix"),
        }
        Ok(())
    }
    #[test]
    fn binary_matrix_requires_a_rank_2() -> Result<()> {
        let ctx = context();
        let a = vector(&ctx, 4)?;
        let b = matrix(&ctx, 4, 1)?;
        assert!(a.params_binary_matrix(&b, Operation::ADD).is_err());
        Ok(())
    }
    #[test]
    fn binary_matrix_requires_b_rank_2() -> Result<()> {
        let ctx = context();
        let a = matrix(&ctx, 4, 1)?;
        let b = vector(&ctx, 4)?;
        assert!(a.params_binary_matrix(&b, Operation::ADD).is_err());
        Ok(())
    }
    #[test]
    fn binary_matrix_requires_same_shape() -> Result<()> {
        let ctx = context();
        let a = matrix(&ctx, 3, 4)?;
        let b = matrix(&ctx, 4, 3)?;
        assert!(a.params_binary_matrix(&b, Operation::ADD).is_err());
        Ok(())
    }
    #[test]
    fn binary_matrix_propagates_invalid_operation() -> Result<()> {
        let ctx = context();
        let a = matrix(&ctx, 3, 4)?;
        let b = matrix(&ctx, 3, 4)?;
        assert!(a.params_binary_matrix(&b, Operation::ABS).is_err());
        Ok(())
    }
    #[test]
    fn binary_matrix_uses_correct_opcode() -> Result<()> {
        let ctx = context();
        let a = matrix(&ctx, 3, 4)?;
        let b = matrix(&ctx, 3, 4)?;
        let params = a.params_binary_matrix(&b, Operation::EQ)?;
        match params {
            Params::BinaryMatrix(p) => {
                assert_eq!(p.op, 8); // OP_EQ
            }
            _ => panic!("Expected Params::BinaryMatrix"),
        }
        Ok(())
    }
    ////////////////////////////////////////////
    // Contract parameter builder
    ////////////////////////////////////////////
    #[test]
    fn builds_contract_matrix_params() -> Result<()> {
        let ctx = context();
        let a = matrix(&ctx, 3, 5)?;
        let b = matrix(&ctx, 5, 7)?;
        let params = a.params_contract_matrix(&b, Operation::MUL, Operation::ADD)?;
        match params {
            Params::ContractMatrix(p) => {
                assert_eq!(p.n, 3);
                assert_eq!(p.p, 5);
                assert_eq!(p.k, 7);
                assert_eq!(p.a_offset, 0);
                assert_eq!(p.b_offset, 0);
                assert_eq!(p.a_row_stride, 5);
                assert_eq!(p.a_col_stride, 1);
                assert_eq!(p.b_row_stride, 7);
                assert_eq!(p.b_col_stride, 1);
                assert_eq!(p.c_row_stride, 7);
                assert_eq!(p.c_col_stride, 1);
                assert_eq!(p.op_pairwise, 2); // MUL
                assert_eq!(p.op_reduction, 0); // ADD
            }
            _ => panic!("Expected Params::ContractMatrix"),
        }
        Ok(())
    }
    #[test]
    fn contract_matrix_requires_a_rank_2() -> Result<()> {
        let ctx = context();
        let a = vector(&ctx, 5)?;
        let b = matrix(&ctx, 5, 7)?;
        assert!(
            a.params_contract_matrix(&b, Operation::MUL, Operation::ADD,)
                .is_err()
        );
        Ok(())
    }
    #[test]
    fn contract_matrix_requires_b_rank_2() -> Result<()> {
        let ctx = context();
        let a = matrix(&ctx, 3, 5)?;
        let b = vector(&ctx, 5)?;
        assert!(
            a.params_contract_matrix(&b, Operation::MUL, Operation::ADD,)
                .is_err()
        );
        Ok(())
    }
    #[test]
    fn contract_matrix_requires_compatible_shapes() -> Result<()> {
        let ctx = context();
        let a = matrix(&ctx, 3, 5)?;
        let b = matrix(&ctx, 6, 7)?;
        assert!(
            a.params_contract_matrix(&b, Operation::MUL, Operation::ADD,)
                .is_err()
        );
        Ok(())
    }
    #[test]
    fn contract_matrix_rejects_invalid_pairwise_operation() -> Result<()> {
        let ctx = context();
        let a = matrix(&ctx, 3, 5)?;
        let b = matrix(&ctx, 5, 7)?;
        assert!(
            a.params_contract_matrix(&b, Operation::POW, Operation::ADD,)
                .is_err()
        );
        Ok(())
    }
    #[test]
    fn contract_matrix_rejects_invalid_reduction_operation() -> Result<()> {
        let ctx = context();
        let a = matrix(&ctx, 3, 5)?;
        let b = matrix(&ctx, 5, 7)?;
        assert!(
            a.params_contract_matrix(&b, Operation::MUL, Operation::DIV,)
                .is_err()
        );
        Ok(())
    }
    ////////////////////////////////////////////
    // Semiring regression tests
    ////////////////////////////////////////////
    #[test]
    fn matmul_semiring_opcodes_match_expectations() -> Result<()> {
        let ctx = context();
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 3, 4)?;
        let params = a.params_contract_matrix(&b, Operation::MUL, Operation::ADD)?;
        match params {
            Params::ContractMatrix(p) => {
                assert_eq!(p.op_pairwise, 2);
                assert_eq!(p.op_reduction, 0);
            }
            _ => panic!("Expected Params::ContractMatrix"),
        }
        Ok(())
    }
    #[test]
    fn min_plus_matrix_semiring_opcodes_match_expectations() -> Result<()> {
        let ctx = context();
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 3, 4)?;
        let params = a.params_contract_matrix(&b, Operation::ADD, Operation::MIN)?;
        match params {
            Params::ContractMatrix(p) => {
                assert_eq!(p.op_pairwise, 0);
                assert_eq!(p.op_reduction, 2);
            }
            _ => panic!("Expected Params::ContractMatrix"),
        }
        Ok(())
    }
    #[test]
    fn max_plus_matrix_semiring_opcodes_match_expectations() -> Result<()> {
        let ctx = context();
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 3, 4)?;
        let params = a.params_contract_matrix(&b, Operation::ADD, Operation::MAX)?;
        match params {
            Params::ContractMatrix(p) => {
                assert_eq!(p.op_pairwise, 0);
                assert_eq!(p.op_reduction, 3);
            }
            _ => panic!("Expected Params::ContractMatrix"),
        }
        Ok(())
    }
    #[test]
    fn min_mul_matrix_semiring_opcodes_match_expectations() -> Result<()> {
        let ctx = context();
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 3, 4)?;
        let params = a.params_contract_matrix(&b, Operation::MUL, Operation::MIN)?;
        match params {
            Params::ContractMatrix(p) => {
                assert_eq!(p.op_pairwise, 2);
                assert_eq!(p.op_reduction, 2);
            }
            _ => panic!("Expected Params::ContractMatrix"),
        }
        Ok(())
    }
    #[test]
    fn max_mul_matrix_semiring_opcodes_match_expectations() -> Result<()> {
        let ctx = context();
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 3, 4)?;
        let params = a.params_contract_matrix(&b, Operation::MUL, Operation::MAX)?;
        match params {
            Params::ContractMatrix(p) => {
                assert_eq!(p.op_pairwise, 2);
                assert_eq!(p.op_reduction, 3);
            }
            _ => panic!("Expected Params::ContractMatrix"),
        }
        Ok(())
    }
    #[test]
    fn boolean_matmul_semiring_opcodes_match_expectations() -> Result<()> {
        let ctx = context();
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 3, 4)?;
        let params = a.params_contract_matrix(&b, Operation::AND, Operation::OR)?;
        match params {
            Params::ContractMatrix(p) => {
                assert_eq!(p.op_pairwise, 12);
                assert_eq!(p.op_reduction, 5);
            }
            _ => panic!("Expected Params::ContractMatrix"),
        }
        Ok(())
    }
    #[test]
    fn eqcount_matrix_semiring_opcodes_match_expectations() -> Result<()> {
        let ctx = context();
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 3, 4)?;
        let params = a.params_contract_matrix(&b, Operation::EQ, Operation::ADD)?;
        match params {
            Params::ContractMatrix(p) => {
                assert_eq!(p.op_pairwise, 6);
                assert_eq!(p.op_reduction, 0);
            }
            _ => panic!("Expected Params::ContractMatrix"),
        }
        Ok(())
    }
    ////////////////////////////////////////////
    // GpuKernel wrapper methods
    ////////////////////////////////////////////
    #[test]
    fn kernel_abs_matrix() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let c = kernel.abs_matrix(&a)?;
        assert_eq!(c.shape, vec![2, 3]);
        Ok(())
    }
    #[test]
    fn kernel_neg_matrix() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let c = kernel.neg_matrix(&a)?;
        assert_eq!(c.shape, vec![2, 3]);
        Ok(())
    }
    #[test]
    fn kernel_sqrt_matrix() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let c = kernel.sqrt_matrix(&a)?;
        assert_eq!(c.shape, vec![2, 3]);
        Ok(())
    }
    #[test]
    fn kernel_exp_matrix() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let c = kernel.exp_matrix(&a)?;
        assert_eq!(c.shape, vec![2, 3]);
        Ok(())
    }
    #[test]
    fn kernel_log_matrix() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let c = kernel.log_matrix(&a)?;
        assert_eq!(c.shape, vec![2, 3]);
        Ok(())
    }
    #[test]
    fn kernel_sin_matrix() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let c = kernel.sin_matrix(&a)?;
        assert_eq!(c.shape, vec![2, 3]);
        Ok(())
    }
    #[test]
    fn kernel_cos_matrix() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let c = kernel.cos_matrix(&a)?;
        assert_eq!(c.shape, vec![2, 3]);
        Ok(())
    }
    #[test]
    fn kernel_add_matrix() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 2, 3)?;
        let c = kernel.add_matrix(&a, &b)?;
        assert_eq!(c.shape, vec![2, 3]);
        Ok(())
    }
    #[test]
    fn kernel_sub_matrix() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 2, 3)?;
        let c = kernel.sub_matrix(&a, &b)?;
        assert_eq!(c.shape, vec![2, 3]);
        Ok(())
    }
    #[test]
    fn kernel_mul_matrix() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 2, 3)?;
        let c = kernel.mul_matrix(&a, &b)?;
        assert_eq!(c.shape, vec![2, 3]);
        Ok(())
    }
    #[test]
    fn kernel_div_matrix() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 2, 3)?;
        let c = kernel.div_matrix(&a, &b)?;
        assert_eq!(c.shape, vec![2, 3]);
        Ok(())
    }
    #[test]
    fn kernel_min_matrix() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 2, 3)?;
        let c = kernel.min_matrix(&a, &b)?;
        assert_eq!(c.shape, vec![2, 3]);
        Ok(())
    }
    #[test]
    fn kernel_max_matrix() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 2, 3)?;
        let c = kernel.max_matrix(&a, &b)?;
        assert_eq!(c.shape, vec![2, 3]);
        Ok(())
    }
    #[test]
    fn kernel_pow_matrix() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 2, 3)?;
        let c = kernel.pow_matrix(&a, &b)?;
        assert_eq!(c.shape, vec![2, 3]);
        Ok(())
    }
    #[test]
    fn kernel_atan2_matrix() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 2, 3)?;
        let c = kernel.atan2_matrix(&a, &b)?;
        assert_eq!(c.shape, vec![2, 3]);
        Ok(())
    }
    #[test]
    fn kernel_comparison_matrix_ops() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 2, 3)?;
        assert_eq!(kernel.eq_matrix(&a, &b)?.shape, vec![2, 3]);
        assert_eq!(kernel.ne_matrix(&a, &b)?.shape, vec![2, 3]);
        assert_eq!(kernel.lt_matrix(&a, &b)?.shape, vec![2, 3]);
        assert_eq!(kernel.le_matrix(&a, &b)?.shape, vec![2, 3]);
        assert_eq!(kernel.gt_matrix(&a, &b)?.shape, vec![2, 3]);
        assert_eq!(kernel.ge_matrix(&a, &b)?.shape, vec![2, 3]);
        Ok(())
    }
    #[test]
    fn kernel_binary_matrix_shape_mismatch_fails() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 3, 2)?;
        assert!(kernel.add_matrix(&a, &b).is_err());
        assert!(kernel.mul_matrix(&a, &b).is_err());
        assert!(kernel.eq_matrix(&a, &b).is_err());
        Ok(())
    }
    #[test]
    fn kernel_matmul() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 3, 4)?;
        let c = kernel.matmul(&a, &b)?;
        assert_eq!(c.shape, vec![2, 4]);
        Ok(())
    }
    #[test]
    fn kernel_hadamard_sum_matrix() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 3, 4)?;
        let c = kernel.hadamard_sum_matrix(&a, &b)?;
        assert_eq!(c.shape, vec![2, 4]);
        Ok(())
    }
    #[test]
    fn kernel_min_plus_matrix() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 3, 4)?;
        let c = kernel.min_plus_matrix(&a, &b)?;
        assert_eq!(c.shape, vec![2, 4]);
        Ok(())
    }
    #[test]
    fn kernel_max_plus_matrix() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 3, 4)?;
        let c = kernel.max_plus_matrix(&a, &b)?;
        assert_eq!(c.shape, vec![2, 4]);
        Ok(())
    }
    #[test]
    fn kernel_min_mul_matrix() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 3, 4)?;
        let c = kernel.min_mul_matrix(&a, &b)?;
        assert_eq!(c.shape, vec![2, 4]);
        Ok(())
    }
    #[test]
    fn kernel_max_mul_matrix() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 3, 4)?;
        let c = kernel.max_mul_matrix(&a, &b)?;
        assert_eq!(c.shape, vec![2, 4]);
        Ok(())
    }
    #[test]
    fn kernel_matmul_bool() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 3, 4)?;
        let c = kernel.matmul_bool(&a, &b)?;
        assert_eq!(c.shape, vec![2, 4]);
        Ok(())
    }
    #[test]
    fn kernel_eqcount_matrix() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 3, 4)?;
        let c = kernel.eqcount_matrix(&a, &b)?;
        assert_eq!(c.shape, vec![2, 4]);
        Ok(())
    }
    #[test]
    fn kernel_matmul_dimension_mismatch_fails() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 5, 4)?;
        assert!(kernel.matmul(&a, &b).is_err());
        Ok(())
    }
    #[test]
    fn kernel_hadamard_sum_matrix_dimension_mismatch_fails() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 5, 4)?;
        assert!(kernel.hadamard_sum_matrix(&a, &b).is_err());
        Ok(())
    }
    #[test]
    fn kernel_min_plus_matrix_dimension_mismatch_fails() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 5, 4)?;
        assert!(kernel.min_plus_matrix(&a, &b).is_err());
        Ok(())
    }
    #[test]
    fn kernel_max_plus_matrix_dimension_mismatch_fails() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 5, 4)?;
        assert!(kernel.max_plus_matrix(&a, &b).is_err());
        Ok(())
    }
    #[test]
    fn kernel_min_mul_matrix_dimension_mismatch_fails() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 5, 4)?;
        assert!(kernel.min_mul_matrix(&a, &b).is_err());
        Ok(())
    }
    #[test]
    fn kernel_max_mul_matrix_dimension_mismatch_fails() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 5, 4)?;
        assert!(kernel.max_mul_matrix(&a, &b).is_err());
        Ok(())
    }
    #[test]
    fn kernel_matmul_bool_dimension_mismatch_fails() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 5, 4)?;
        assert!(kernel.matmul_bool(&a, &b).is_err());
        Ok(())
    }
    #[test]
    fn kernel_eqcount_matrix_dimension_mismatch_fails() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 5, 4)?;
        assert!(kernel.eqcount_matrix(&a, &b).is_err());
        Ok(())
    }
}
