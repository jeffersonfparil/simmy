use crate::linalg::kernel::{GpuKernel, Params};
use crate::linalg::params::{BinaryMatrixParams, ContractMatrixParams, UnaryMatrixParams};
use crate::linalg::tensor::GpuTensor;
use crate::linalg::operations::Operation;
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
            // Mathematical operation: op(A) -> C.
            //     - ABS = 0
            //     - NEG = 1
            //     - SQRT = 2
            //     - EXP = 3
            //     - LOG = 4
            //     - SIN = 5
            //     - COS = 6
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
            // Mathematical operation: op(A, B) -> C.
            //     - ADD = 0;
            //     - SUB = 1;
            //     - MUL = 2;
            //     - DIV = 3;
            //     - MIN = 4;
            //     - MAX = 5;
            //     - POW = 6;
            //     - ATAN2 = 7;
            //     - EQ = 8;
            //     - NE = 9;
            //     - OP_LT = 10;
            //     - OP_LE = 11;
            //     - OP_GT = 12;
            //     - OP_GE = 13;
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
            // Pairwise operation
            //     - OP_PAIR_ADD = 0;
            //     - OP_PAIR_SUB = 1;
            //     - OP_PAIR_MUL = 2;
            //     - OP_PAIR_DIV = 3;
            //     - OP_PAIR_MIN = 4;
            //     - OP_PAIR_MAX = 5;
            op_pairwise: op_pairwise.contract_pairwise_opcode()?,
            // Reduction operation
            //     - OP_REDUCE_ADD = 0;
            //     - OP_REDUCE_MUL = 1;
            //     - OP_REDUCE_MIN = 2;
            //     - OP_REDUCE_MAX = 3;
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
    pub fn hadamard_sum(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        let params = a.params_contract_matrix(b, Operation::ADD, Operation::ADD)?;
        self.execute_kernel(params, a, Some(b))
    }
    // Min-Plus Algebra (ADD --> MIN)
    pub fn min_plus(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        let params = a.params_contract_matrix(b, Operation::ADD, Operation::MIN)?;
        self.execute_kernel(params, a, Some(b))
    }
    // Max-Plus Algebra (ADD --> MAX)
    pub fn max_plus(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        let params = a.params_contract_matrix(b, Operation::ADD, Operation::MAX)?;
        self.execute_kernel(params, a, Some(b))
    }
    // Min-Mul (MUL --> MIN)
    pub fn min_mul(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        let params = a.params_contract_matrix(b, Operation::MUL, Operation::MIN)?;
        self.execute_kernel(params, a, Some(b))
    }
    // Max-Mul (MUL --> MAX)
    pub fn max_mul(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        let params = a.params_contract_matrix(b, Operation::MUL, Operation::MAX)?;
        self.execute_kernel(params, a, Some(b))
    }
    // Boolean matrix multiplication (AND --> OR)
    pub fn matmul_bool(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        let params = a.params_contract_matrix(b, Operation::AND, Operation::OR)?;
        self.execute_kernel(params, a, Some(b))
    }
    // Equality Counting (EQ --> ADD)
    pub fn eqcount(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        let params = a.params_contract_matrix(b, Operation::EQ, Operation::ADD)?;
        self.execute_kernel(params, a, Some(b))
    }
}
