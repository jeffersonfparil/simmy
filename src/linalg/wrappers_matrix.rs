use crate::linalg::kernel::{GpuKernel, Params};
use crate::linalg::params::{BinaryMatrixParams, ContractMatrixParams, UnaryMatrixParams};
use crate::linalg::tensor::GpuTensor;
use anyhow::{Result, bail, ensure};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operations {
    ABS,
    NEG,
    SQRT,
    EXP,
    LOG,
    SIN,
    COS,

    ADD,
    SUB,
    MUL,
    DIV,
    MIN,
    MAX,
    POW,
    ATAN2,
    EQ,
    NE,
    LT,
    LE,
    GT,
    GE,
}

impl Operations {
    fn unary_opcode(self) -> Result<u32> {
        match self {
            Self::ABS => Ok(0),
            Self::NEG => Ok(1),
            Self::SQRT => Ok(2),
            Self::EXP => Ok(3),
            Self::LOG => Ok(4),
            Self::SIN => Ok(5),
            Self::COS => Ok(6),
            _ => bail!("Invalid unary matrix operation!"),
        }
    }
    fn binary_opcode(self) -> Result<u32> {
        match self {
            Self::ADD => Ok(0),
            Self::SUB => Ok(1),
            Self::MUL => Ok(2),
            Self::DIV => Ok(3),
            Self::MIN => Ok(4),
            Self::MAX => Ok(5),
            Self::POW => Ok(6),
            Self::ATAN2 => Ok(7),
            Self::EQ => Ok(8),
            Self::NE => Ok(9),
            Self::LT => Ok(10),
            Self::LE => Ok(11),
            Self::GT => Ok(12),
            Self::GE => Ok(13),
            _ => bail!("Invalid binary matrix operation!"),
        }
    }
    fn contract_pairwise_opcode(self) -> Result<u32> {
        match self {
            Self::ADD => Ok(0),
            Self::SUB => Ok(1),
            Self::MUL => Ok(2),
            Self::DIV => Ok(3),
            Self::MIN => Ok(4),
            Self::MAX => Ok(5),
            _ => bail!("Invalid contract matrix pairwise operation!"),
        }
    }
    fn contract_reduction_opcode(self) -> Result<u32> {
        match self {
            Self::ADD => Ok(0),
            Self::MUL => Ok(1),
            Self::MIN => Ok(2),
            Self::MAX => Ok(3),
            _ => bail!("Invalid contract matrix reduction operation!"),
        }
    }
}

impl GpuTensor {
    pub fn params_unary_matrix(&self, op: Operations) -> Result<Params> {
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
            //     - OP_ABS = 0
            //     - OP_NEG = 1
            //     - OP_SQRT = 2
            //     - OP_EXP = 3
            //     - OP_LOG = 4
            //     - OP_SIN = 5
            //     - OP_COS = 6
            op: op.unary_opcode()?,
        };
        Ok(Params::UnaryMatrix(params))
    }

    pub fn params_binary_matrix(&self, b: &Self, op: Operations) -> Result<Params> {
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
            //     - OP_ADD = 0;
            //     - OP_SUB = 1;
            //     - OP_MUL = 2;
            //     - OP_DIV = 3;
            //     - OP_MIN = 4;
            //     - OP_MAX = 5;
            //     - OP_POW = 6;
            //     - OP_ATAN2 = 7;
            //     - OP_EQ = 8;
            //     - OP_NE = 9;
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
        op_pairwise: Operations,
        op_reduction: Operations,
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
        self.execute_kernel(a.params_unary_matrix(Operations::ABS)?, a, None)
    }
    pub fn neg_matrix(&self, a: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_unary_matrix(Operations::NEG)?, a, None)
    }
    pub fn sqrt_matrix(&self, a: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_unary_matrix(Operations::SQRT)?, a, None)
    }
    pub fn exp_matrix(&self, a: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_unary_matrix(Operations::EXP)?, a, None)
    }
    pub fn log_matrix(&self, a: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_unary_matrix(Operations::LOG)?, a, None)
    }
    pub fn sin_matrix(&self, a: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_unary_matrix(Operations::SIN)?, a, None)
    }
    pub fn cos_matrix(&self, a: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_unary_matrix(Operations::COS)?, a, None)
    }

    // Binary
    pub fn add_matrix(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary_matrix(b, Operations::ADD)?, a, Some(b))
    }
    pub fn sub_matrix(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary_matrix(b, Operations::SUB)?, a, Some(b))
    }
    pub fn mul_matrix(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary_matrix(b, Operations::MUL)?, a, Some(b))
    }
    pub fn div_matrix(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary_matrix(b, Operations::DIV)?, a, Some(b))
    }
    pub fn min_matrix(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary_matrix(b, Operations::MIN)?, a, Some(b))
    }
    pub fn max_matrix(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary_matrix(b, Operations::MAX)?, a, Some(b))
    }
    pub fn pow_matrix(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary_matrix(b, Operations::POW)?, a, Some(b))
    }
    pub fn atan2_matrix(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary_matrix(b, Operations::ATAN2)?, a, Some(b))
    }
    pub fn eq_matrix(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary_matrix(b, Operations::EQ)?, a, Some(b))
    }
    pub fn ne_matrix(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary_matrix(b, Operations::NE)?, a, Some(b))
    }
    pub fn lt_matrix(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary_matrix(b, Operations::LT)?, a, Some(b))
    }
    pub fn le_matrix(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary_matrix(b, Operations::LE)?, a, Some(b))
    }
    pub fn gt_matrix(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary_matrix(b, Operations::GT)?, a, Some(b))
    }
    pub fn ge_matrix(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary_matrix(b, Operations::GE)?, a, Some(b))
    }

    // Contract
    pub fn matmul(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(
            a.params_contract_matrix(b, Operations::MUL, Operations::ADD)?,
            a,
            Some(b),
        )
    }
}
