use crate::linalg::kernel::{GpuKernel, Params};
use crate::linalg::operations::Operation;
use crate::linalg::params::{BinaryTensorParams, ContractTensorParams, UnaryTensorParams};
use crate::linalg::tensor::GpuTensor;
use anyhow::{Result, ensure};

const MAX_RANK: usize = 8;

impl GpuTensor {
    pub fn params_unary(&self, op: Operation) -> Result<Params> {
        ensure!(
            self.shape.len() <= MAX_RANK,
            "Tensor rank exceeds MAX_RANK ({})",
            MAX_RANK
        );
        let n_elements: u32 = self.shape.iter().product();
        let mut shape = [0u32; MAX_RANK];
        let mut self_strides = [0u32; MAX_RANK];
        let mut c_strides = [0u32; MAX_RANK];
        shape[..self.shape.len()].copy_from_slice(&self.shape[..]);
        self_strides[..self.shape.len()].copy_from_slice(&self.strides[..]);
        let mut stride = 1u32;
        for i in (0..self.shape.len()).rev() {
            c_strides[i] = stride;
            stride *= self.shape[i];
        }
        let params = UnaryTensorParams {
            // Number of tensor dimensions.
            rank: self.shape.len() as u32,
            // Total number of logical tensor elements.
            n_elements: n_elements,
            // Shape of the logical tensor.
            shape: shape,
            // Starting element of `A` within its backing storage.
            a_offset: self.offset,
            // Storage strides of `A` per dimension.
            a_strides: self_strides,
            // Starting element of `C` within its backing storage.
            c_offset: 0,
            // Storage strides of `C` per dimension.
            c_strides: c_strides,
            // Mathematical operation: op(A) -> C (see operations.rs & wgsl/opcodes.wgsl).
            op: op.unary_opcode()?,
        };
        Ok(Params::UnaryTensor(params))
    }

    pub fn params_binary(&self, b: &Self, op: Operation) -> Result<Params> {
        ensure!(
            self.shape == b.shape,
            "Incompatible shapes: {:?} and {:?}",
            self.shape,
            b.shape
        );
        ensure!(
            self.shape.len() <= MAX_RANK,
            "Tensor rank exceeds MAX_RANK ({})",
            MAX_RANK
        );
        let mut shape = [0u32; MAX_RANK];
        let mut a_strides = [0u32; MAX_RANK];
        let mut b_strides = [0u32; MAX_RANK];
        let mut c_strides = [0u32; MAX_RANK];
        shape[..self.shape.len()].copy_from_slice(&self.shape[..]);
        a_strides[..self.shape.len()].copy_from_slice(&self.strides[..]);
        b_strides[..self.shape.len()].copy_from_slice(&b.strides[..]);
        let mut stride = 1u32;
        for i in (0..self.shape.len()).rev() {
            c_strides[i] = stride;
            stride *= self.shape[i];
        }
        let params = BinaryTensorParams {
            // Number of tensor dimensions.
            rank: self.shape.len() as u32,
            // Total number of logical tensor elements.
            n_elements: self.shape.iter().product(),
            // Shape of the logical tensor.
            shape,
            // Starting element of `A` within its backing storage.
            a_offset: self.offset,
            // Storage strides of `A` per dimension.
            a_strides,
            // Starting element of `B` within its backing storage.
            b_offset: b.offset,
            // Storage strides of `B` per dimension.
            b_strides,
            // Starting element of `C` within its backing storage.
            c_offset: 0,
            // Storage strides of `C` per dimension.
            c_strides,
            // Mathematical operation: op(A, B) -> C (see operations.rs & wgsl/opcodes.wgsl).
            op: op.binary_opcode()?,
        };
        Ok(Params::BinaryTensor(params))
    }

    pub fn params_contract(
        &self,
        b: &Self,
        op_pairwise: Operation,
        op_reduction: Operation,
        contract_axes: Option<(Vec<usize>, Vec<usize>)>,
    ) -> Result<Params> {
        let (a_contract_axes, b_contract_axes) = match contract_axes {
            Some(x) => x,
            None => {
                ensure!(
                    !self.shape.is_empty(),
                    "Cannot infer contraction axes for a scalar tensor!"
                );
                (vec![self.shape.len() - 1], vec![0])
            }
        };
        ensure!(
            a_contract_axes.len() == b_contract_axes.len(),
            "Contraction axis count mismatch"
        );
        for i in 0..a_contract_axes.len() {
            ensure!(
                a_contract_axes[i] < self.shape.len(),
                "The contract axis of A is out of bounds!"
            );
            ensure!(
                b_contract_axes[i] < b.shape.len(),
                "The contract axis of B is out of bounds!"
            );
        }
        for i in 0..a_contract_axes.len() {
            for j in (i + 1)..a_contract_axes.len() {
                ensure!(
                    a_contract_axes[i] != a_contract_axes[j],
                    "Duplicate contraction axes ({} at {} and {}) are not allowed!",
                    a_contract_axes[i],
                    i,
                    j
                );
                ensure!(
                    b_contract_axes[i] != b_contract_axes[j],
                    "Duplicate contraction axes ({} at {} and {}) are not allowed!",
                    b_contract_axes[i],
                    i,
                    j
                );
            }
        }
        let contraction_rank = a_contract_axes.len();
        ensure!(self.shape.len() <= MAX_RANK, "A rank exceeds MAX_RANK");
        ensure!(b.shape.len() <= MAX_RANK, "B rank exceeds MAX_RANK");
        for i in 0..contraction_rank {
            let a_axis = a_contract_axes[i];
            let b_axis = b_contract_axes[i];
            ensure!(
                self.shape[a_axis] == b.shape[b_axis],
                "Contracted dimensions must match: \
                A axis {} ({}) != B axis {} ({})",
                a_axis,
                self.shape[a_axis],
                b_axis,
                b.shape[b_axis],
            );
        }
        // Build output shape:
        // C = A free axes + B free axes
        let mut c_shape = Vec::new();
        for axis in 0..self.shape.len() {
            if !a_contract_axes.contains(&axis) {
                c_shape.push(self.shape[axis]);
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
        a_shape[..self.shape.len()].copy_from_slice(&self.shape[..]);
        b_shape[..b.shape.len()].copy_from_slice(&b.shape[..]);
        c_shape_arr[..c_shape.len()].copy_from_slice(&c_shape[..]);
        // Convert strides
        let mut a_strides = [0u32; MAX_RANK];
        let mut b_strides = [0u32; MAX_RANK];
        let mut c_strides = [0u32; MAX_RANK];
        a_strides[..self.strides.len()].copy_from_slice(&self.strides[..]);
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
        let params = ContractTensorParams {
            // Number of dimensions in tensor A.
            a_rank: self.shape.len() as u32,
            // Number of dimensions in tensor B.
            b_rank: b.shape.len() as u32,
            // Number of dimensions in tensor C.
            c_rank: c_shape.len() as u32,
            // Number of contracted axis pairs.
            contraction_rank: contraction_rank as u32,
            // Total number of logical elements in the output tensor C.
            // i.e.: product(c_shape[..c_rank])
            c_elements,
            // Logical shape of tensor A.
            a_shape,
            // Logical shape of tensor B.
            b_shape,
            // Logical shape of tensor C.
            c_shape: c_shape_arr,
            // Starting element of `A` within its backing storage.
            a_offset: self.offset,
            // Storage strides of `A` per dimension.
            a_strides,
            // Starting element of `B` within its backing storage.
            b_offset: b.offset,
            // Storage strides of `B` per dimension.
            b_strides,
            // Starting element of `C` within its backing storage.
            c_offset: 0,
            // Storage strides of `C` per dimension.
            c_strides,
            // Contracted axes of tensor A.
            a_contract_axes: a_contract,
            // Contracted axes of tensor B.
            b_contract_axes: b_contract,
            // Pairwise operation (see operations.rs & wgsl/opcodes.wgsl).
            op_pairwise: op_pairwise.contract_pairwise_opcode()?,
            // Reduction operation (see operations.rs & wgsl/opcodes.wgsl).
            op_reduction: op_reduction.contract_reduction_opcode()?,
        };
        Ok(Params::ContractTensor(params))
    }
}

impl GpuKernel<'_> {
    // Unary
    pub fn abs(&self, a: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_unary(Operation::ABS)?, a, None)
    }
    pub fn neg(&self, a: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_unary(Operation::NEG)?, a, None)
    }
    pub fn sqrt(&self, a: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_unary(Operation::SQRT)?, a, None)
    }
    pub fn exp(&self, a: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_unary(Operation::EXP)?, a, None)
    }
    pub fn log(&self, a: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_unary(Operation::LOG)?, a, None)
    }
    pub fn sin(&self, a: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_unary(Operation::SIN)?, a, None)
    }
    pub fn cos(&self, a: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_unary(Operation::COS)?, a, None)
    }

    // Binary
    pub fn add(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary(b, Operation::ADD)?, a, Some(b))
    }
    pub fn sub(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary(b, Operation::SUB)?, a, Some(b))
    }
    pub fn mul(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary(b, Operation::MUL)?, a, Some(b))
    }
    pub fn div(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary(b, Operation::DIV)?, a, Some(b))
    }
    pub fn min(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary(b, Operation::MIN)?, a, Some(b))
    }
    pub fn max(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary(b, Operation::MAX)?, a, Some(b))
    }
    pub fn pow(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary(b, Operation::POW)?, a, Some(b))
    }
    pub fn atan2(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary(b, Operation::ATAN2)?, a, Some(b))
    }
    pub fn eq(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary(b, Operation::EQ)?, a, Some(b))
    }
    pub fn ne(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary(b, Operation::NE)?, a, Some(b))
    }
    pub fn lt(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary(b, Operation::LT)?, a, Some(b))
    }
    pub fn le(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary(b, Operation::LE)?, a, Some(b))
    }
    pub fn gt(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary(b, Operation::GT)?, a, Some(b))
    }
    pub fn ge(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        self.execute_kernel(a.params_binary(b, Operation::GE)?, a, Some(b))
    }

    // Matrix multiplication (MUL --> ADD)
    pub fn contract(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        let params = a.params_contract(b, Operation::MUL, Operation::ADD, None)?;
        self.execute_kernel(params, a, Some(b))
    }
    // Hadamard sum (ADD --> ADD)
    pub fn hadamard_sum(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        let params = a.params_contract(b, Operation::ADD, Operation::ADD, None)?;
        self.execute_kernel(params, a, Some(b))
    }
    // Min-Plus Algebra (ADD --> MIN)
    pub fn min_plus(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        let params = a.params_contract(b, Operation::ADD, Operation::MIN, None)?;
        self.execute_kernel(params, a, Some(b))
    }
    // Max-Plus Algebra (ADD --> MAX)
    pub fn max_plus(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        let params = a.params_contract(b, Operation::ADD, Operation::MAX, None)?;
        self.execute_kernel(params, a, Some(b))
    }
    // Min-Mul (MUL --> MIN)
    pub fn min_mul(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        let params = a.params_contract(b, Operation::MUL, Operation::MIN, None)?;
        self.execute_kernel(params, a, Some(b))
    }
    // Max-Mul (MUL --> MAX)
    pub fn max_mul(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        let params = a.params_contract(b, Operation::MUL, Operation::MAX, None)?;
        self.execute_kernel(params, a, Some(b))
    }
    // Boolean matrix multiplication (AND --> OR)
    pub fn contract_bool(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        let params = a.params_contract(b, Operation::AND, Operation::OR, None)?;
        self.execute_kernel(params, a, Some(b))
    }
    // Equality Counting (EQ --> ADD)
    pub fn eqcount(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        let params = a.params_contract(b, Operation::EQ, Operation::ADD, None)?;
        self.execute_kernel(params, a, Some(b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::context::GpuContext;
    use crate::linalg::kernel::Params;
    use crate::linalg::operations::Operation;
    use crate::linalg::tensor::GpuTensor;
    use anyhow::Result;

    fn context() -> GpuContext {
        pollster::block_on(GpuContext::new()).expect("Failed to create GPU context")
    }

    fn vector(ctx: &GpuContext, n: usize) -> Result<GpuTensor> {
        let data: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
        GpuTensor::from_f32(ctx, &data, vec![n as u32], None, None)
    }

    fn matrix(ctx: &GpuContext, rows: usize, cols: usize) -> Result<GpuTensor> {
        let data: Vec<f32> = (0..rows * cols).map(|i| (i + 1) as f32).collect();
        GpuTensor::from_f32(ctx, &data, vec![rows as u32, cols as u32], None, None)
    }

    fn tensor(ctx: &GpuContext, shape: &[u32]) -> Result<GpuTensor> {
        let n_elements: usize = shape.iter().copied().map(|x| x as usize).product();
        let data: Vec<f32> = (0..n_elements.max(1)).map(|i| (i + 1) as f32).collect();
        GpuTensor::from_f32(ctx, &data, shape.to_vec(), None, None)
    }
    ////////////////////////////////////////
    // Unary
    ////////////////////////////////////////
    #[test]
    fn unary_params_rank_and_element_count() -> Result<()> {
        let ctx = context();
        let a = matrix(&ctx, 2, 3)?;
        let params = a.params_unary(Operation::ABS)?;
        match params {
            Params::UnaryTensor(p) => {
                assert_eq!(p.rank, 2);
                assert_eq!(p.n_elements, 6);
                assert_eq!(&p.shape[..2], &[2, 3]);
            }
            _ => panic!("Expected UnaryTensor params"),
        }
        Ok(())
    }
    #[test]
    fn unary_output_strides_are_contiguous() -> Result<()> {
        let ctx = context();
        let a = tensor(&ctx, &[2, 3, 4])?;
        let params = a.params_unary(Operation::ABS)?;
        match params {
            Params::UnaryTensor(p) => {
                assert_eq!(&p.c_strides[..3], &[12, 4, 1]);
            }
            _ => panic!("Expected UnaryTensor params"),
        }
        Ok(())
    }
    ////////////////////////////////////////
    // Binary
    ////////////////////////////////////////
    #[test]
    fn binary_accepts_matching_shapes() -> Result<()> {
        let ctx = context();
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 2, 3)?;
        assert!(a.params_binary(&b, Operation::ADD).is_ok());
        Ok(())
    }
    #[test]
    fn binary_rejects_mismatched_shapes() -> Result<()> {
        let ctx = context();
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 3, 2)?;
        assert!(a.params_binary(&b, Operation::ADD).is_err());
        Ok(())
    }
    #[test]
    fn binary_params_rank_and_element_count() -> Result<()> {
        let ctx = context();
        let a = tensor(&ctx, &[2, 3, 4])?;
        let b = tensor(&ctx, &[2, 3, 4])?;
        let params = a.params_binary(&b, Operation::MUL)?;
        match params {
            Params::BinaryTensor(p) => {
                assert_eq!(p.rank, 3);
                assert_eq!(p.n_elements, 24);
                assert_eq!(&p.shape[..3], &[2, 3, 4]);
            }
            _ => panic!("Expected BinaryTensor params"),
        }
        Ok(())
    }
    #[test]
    fn binary_output_strides_are_contiguous() -> Result<()> {
        let ctx = context();
        let a = tensor(&ctx, &[2, 3, 4])?;
        let b = tensor(&ctx, &[2, 3, 4])?;
        let params = a.params_binary(&b, Operation::ADD)?;
        match params {
            Params::BinaryTensor(p) => {
                assert_eq!(&p.c_strides[..3], &[12, 4, 1]);
            }
            _ => panic!("Expected BinaryTensor params"),
        }
        Ok(())
    }
    ////////////////////////////////////////
    // Contraction validation
    ////////////////////////////////////////
    #[test]
    fn contract_default_matrix_multiplication_is_valid() -> Result<()> {
        let ctx = context();
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 3, 4)?;
        assert!(
            a.params_contract(&b, Operation::MUL, Operation::ADD, None)
                .is_ok()
        );
        Ok(())
    }
    #[test]
    fn contract_rejects_scalar_default_axes() -> Result<()> {
        let ctx = context();
        let a = tensor(&ctx, &[])?;
        let b = vector(&ctx, 4)?;
        assert!(
            a.params_contract(&b, Operation::MUL, Operation::ADD, None)
                .is_err()
        );
        Ok(())
    }
    #[test]
    fn contract_rejects_axis_count_mismatch() -> Result<()> {
        let ctx = context();
        let a = tensor(&ctx, &[2, 3, 4])?;
        let b = tensor(&ctx, &[2, 3, 4])?;
        assert!(
            a.params_contract(
                &b,
                Operation::MUL,
                Operation::ADD,
                Some((vec![0, 1], vec![0]))
            )
            .is_err()
        );
        Ok(())
    }
    #[test]
    fn contract_rejects_a_axis_out_of_bounds() -> Result<()> {
        let ctx = context();
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 3, 4)?;
        assert!(
            a.params_contract(
                &b,
                Operation::MUL,
                Operation::ADD,
                Some((vec![99], vec![0]))
            )
            .is_err()
        );
        Ok(())
    }
    #[test]
    fn contract_rejects_b_axis_out_of_bounds() -> Result<()> {
        let ctx = context();
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 3, 4)?;
        assert!(
            a.params_contract(
                &b,
                Operation::MUL,
                Operation::ADD,
                Some((vec![1], vec![99]))
            )
            .is_err()
        );
        Ok(())
    }
    #[test]
    fn contract_rejects_duplicate_a_axes() -> Result<()> {
        let ctx = context();
        let a = tensor(&ctx, &[2, 3, 4])?;
        let b = tensor(&ctx, &[3, 4, 5])?;
        assert!(
            a.params_contract(
                &b,
                Operation::MUL,
                Operation::ADD,
                Some((vec![1, 1], vec![0, 1]))
            )
            .is_err()
        );
        Ok(())
    }
    #[test]
    fn contract_rejects_duplicate_b_axes() -> Result<()> {
        let ctx = context();
        let a = tensor(&ctx, &[2, 3, 4])?;
        let b = tensor(&ctx, &[3, 4, 5])?;
        assert!(
            a.params_contract(
                &b,
                Operation::MUL,
                Operation::ADD,
                Some((vec![1, 2], vec![0, 0]))
            )
            .is_err()
        );
        Ok(())
    }
    #[test]
    fn contract_rejects_dimension_mismatch() -> Result<()> {
        let ctx = context();
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 4, 5)?;
        assert!(
            a.params_contract(&b, Operation::MUL, Operation::ADD, Some((vec![1], vec![0])))
                .is_err()
        );
        Ok(())
    }
    ////////////////////////////////////////
    // Contraction correctness
    ////////////////////////////////////////
    #[test]
    fn contract_accepts_multi_axis_contraction() -> Result<()> {
        let ctx = context();
        let a = tensor(&ctx, &[2, 3, 4, 5])?;
        let b = tensor(&ctx, &[4, 3, 7])?;
        assert!(
            a.params_contract(
                &b,
                Operation::MUL,
                Operation::ADD,
                Some((vec![2, 1], vec![0, 1]))
            )
            .is_ok()
        );
        Ok(())
    }
    #[test]
    fn contract_output_shape_is_correct() -> Result<()> {
        let ctx = context();
        let a = tensor(&ctx, &[2, 3, 4])?;
        let b = tensor(&ctx, &[4, 5])?;
        let params =
            a.params_contract(&b, Operation::MUL, Operation::ADD, Some((vec![2], vec![0])))?;
        match params {
            Params::ContractTensor(p) => {
                assert_eq!(p.c_rank, 3);
                assert_eq!(&p.c_shape[..3], &[2, 3, 5]);
                assert_eq!(p.c_elements, 30);
            }
            _ => panic!("Expected ContractTensor params"),
        }
        Ok(())
    }
    #[test]
    fn contract_output_strides_are_contiguous() -> Result<()> {
        let ctx = context();
        let a = tensor(&ctx, &[2, 3, 4])?;
        let b = tensor(&ctx, &[4, 5])?;
        let params =
            a.params_contract(&b, Operation::MUL, Operation::ADD, Some((vec![2], vec![0])))?;
        match params {
            Params::ContractTensor(p) => {
                assert_eq!(&p.c_strides[..3], &[15, 5, 1]);
            }
            _ => panic!("Expected ContractTensor params"),
        }
        Ok(())
    }
    #[test]
    fn contract_matrix_multiplication_output_shape() -> Result<()> {
        let ctx = context();
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 3, 4)?;
        let params = a.params_contract(&b, Operation::MUL, Operation::ADD, None)?;
        match params {
            Params::ContractTensor(p) => {
                assert_eq!(p.c_rank, 2);
                assert_eq!(&p.c_shape[..2], &[2, 4]);
                assert_eq!(p.c_elements, 8);
            }
            _ => panic!("Expected ContractTensor params"),
        }
        Ok(())
    }
    #[test]
    fn contract_preserves_contracted_axes() -> Result<()> {
        let ctx = context();
        let a = tensor(&ctx, &[2, 3, 4])?;
        let b = tensor(&ctx, &[5, 4, 6])?;
        let params =
            a.params_contract(&b, Operation::MUL, Operation::ADD, Some((vec![2], vec![1])))?;
        match params {
            Params::ContractTensor(p) => {
                assert_eq!(p.contraction_rank, 1);
                assert_eq!(p.a_contract_axes[0], 2);
                assert_eq!(p.b_contract_axes[0], 1);
            }
            _ => panic!("Expected ContractTensor params"),
        }
        Ok(())
    }
    #[test]
    fn contract_rejects_result_rank_exceeding_max_rank() -> Result<()> {
        let ctx = context();
        let a = tensor(&ctx, &[2, 2, 2, 2, 2])?;
        let b = tensor(&ctx, &[2, 2, 2, 2, 2])?;
        assert!(
            a.params_contract(
                &b,
                Operation::MUL,
                Operation::ADD,
                Some((vec![10], vec![10]))
            )
            .is_err()
        );
        Ok(())
    }
    ////////////////////////////////////////
    // GpuKernel wrapper methods
    ////////////////////////////////////////
    #[test]
    fn kernel_abs() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = vector(&ctx, 8)?;
        let c = kernel.abs(&a)?;
        assert_eq!(c.shape, vec![8]);
        Ok(())
    }
    #[test]
    fn kernel_neg() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = vector(&ctx, 8)?;
        let c = kernel.neg(&a)?;
        assert_eq!(c.shape, vec![8]);
        Ok(())
    }
    #[test]
    fn kernel_sqrt() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = vector(&ctx, 8)?;
        let c = kernel.sqrt(&a)?;
        assert_eq!(c.shape, vec![8]);
        Ok(())
    }
    #[test]
    fn kernel_exp() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = vector(&ctx, 8)?;
        let c = kernel.exp(&a)?;
        assert_eq!(c.shape, vec![8]);
        Ok(())
    }
    #[test]
    fn kernel_log() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = vector(&ctx, 8)?;
        let c = kernel.log(&a)?;
        assert_eq!(c.shape, vec![8]);
        Ok(())
    }
    #[test]
    fn kernel_sin() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = vector(&ctx, 8)?;
        let c = kernel.sin(&a)?;
        assert_eq!(c.shape, vec![8]);
        Ok(())
    }
    #[test]
    fn kernel_cos() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = vector(&ctx, 8)?;
        let c = kernel.cos(&a)?;
        assert_eq!(c.shape, vec![8]);
        Ok(())
    }
    #[test]
    fn kernel_add() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 2, 3)?;
        let c = kernel.add(&a, &b)?;
        assert_eq!(c.shape, vec![2, 3]);
        Ok(())
    }
    #[test]
    fn kernel_sub() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 2, 3)?;
        let c = kernel.sub(&a, &b)?;
        assert_eq!(c.shape, vec![2, 3]);
        Ok(())
    }
    #[test]
    fn kernel_mul() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 2, 3)?;
        let c = kernel.mul(&a, &b)?;
        assert_eq!(c.shape, vec![2, 3]);
        Ok(())
    }
    #[test]
    fn kernel_div() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 2, 3)?;
        let c = kernel.div(&a, &b)?;
        assert_eq!(c.shape, vec![2, 3]);
        Ok(())
    }
    #[test]
    fn kernel_min() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 2, 3)?;
        let c = kernel.min(&a, &b)?;
        assert_eq!(c.shape, vec![2, 3]);
        Ok(())
    }
    #[test]
    fn kernel_max() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 2, 3)?;
        let c = kernel.max(&a, &b)?;
        assert_eq!(c.shape, vec![2, 3]);
        Ok(())
    }
    #[test]
    fn kernel_pow() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 2, 3)?;
        let c = kernel.pow(&a, &b)?;
        assert_eq!(c.shape, vec![2, 3]);
        Ok(())
    }
    #[test]
    fn kernel_atan2() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 2, 3)?;
        let c = kernel.atan2(&a, &b)?;
        assert_eq!(c.shape, vec![2, 3]);
        Ok(())
    }
    #[test]
    fn kernel_comparison_ops() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 2, 3)?;
        assert_eq!(kernel.eq(&a, &b)?.shape, vec![2, 3]);
        assert_eq!(kernel.ne(&a, &b)?.shape, vec![2, 3]);
        assert_eq!(kernel.lt(&a, &b)?.shape, vec![2, 3]);
        assert_eq!(kernel.le(&a, &b)?.shape, vec![2, 3]);
        assert_eq!(kernel.gt(&a, &b)?.shape, vec![2, 3]);
        assert_eq!(kernel.ge(&a, &b)?.shape, vec![2, 3]);
        Ok(())
    }
    #[test]
    fn kernel_binary_shape_mismatch_fails() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 3, 2)?;
        assert!(kernel.add(&a, &b).is_err());
        assert!(kernel.mul(&a, &b).is_err());
        assert!(kernel.eq(&a, &b).is_err());
        Ok(())
    }
    #[test]
    fn kernel_contract() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 3, 4)?;
        let c = kernel.contract(&a, &b)?;
        assert_eq!(c.shape, vec![2, 4]);
        Ok(())
    }
    #[test]
    fn kernel_hadamard_sum() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 3, 4)?;
        let c = kernel.hadamard_sum(&a, &b)?;
        assert_eq!(c.shape, vec![2, 4]);
        Ok(())
    }
    #[test]
    fn kernel_min_plus() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 3, 4)?;
        let c = kernel.min_plus(&a, &b)?;
        assert_eq!(c.shape, vec![2, 4]);
        Ok(())
    }
    #[test]
    fn kernel_max_plus() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 3, 4)?;
        let c = kernel.max_plus(&a, &b)?;
        assert_eq!(c.shape, vec![2, 4]);
        Ok(())
    }
    #[test]
    fn kernel_min_mul() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 3, 4)?;
        let c = kernel.min_mul(&a, &b)?;
        assert_eq!(c.shape, vec![2, 4]);
        Ok(())
    }
    #[test]
    fn kernel_max_mul() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 3, 4)?;
        let c = kernel.max_mul(&a, &b)?;
        assert_eq!(c.shape, vec![2, 4]);
        Ok(())
    }
    #[test]
    fn kernel_contract_bool() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 3, 4)?;
        let c = kernel.contract_bool(&a, &b)?;
        assert_eq!(c.shape, vec![2, 4]);
        Ok(())
    }
    #[test]
    fn kernel_eqcount() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 3, 4)?;
        let c = kernel.eqcount(&a, &b)?;
        assert_eq!(c.shape, vec![2, 4]);
        Ok(())
    }
    #[test]
    fn kernel_contract_dimension_mismatch_fails() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 5, 4)?;
        assert!(kernel.contract(&a, &b).is_err());
        Ok(())
    }
    #[test]
    fn kernel_hadamard_sum_dimension_mismatch_fails() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 5, 4)?;
        assert!(kernel.hadamard_sum(&a, &b).is_err());
        Ok(())
    }
    #[test]
    fn kernel_min_plus_dimension_mismatch_fails() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 5, 4)?;
        assert!(kernel.min_plus(&a, &b).is_err());
        Ok(())
    }
    #[test]
    fn kernel_max_plus_dimension_mismatch_fails() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 5, 4)?;
        assert!(kernel.max_plus(&a, &b).is_err());
        Ok(())
    }
    #[test]
    fn kernel_min_mul_dimension_mismatch_fails() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 5, 4)?;
        assert!(kernel.min_mul(&a, &b).is_err());
        Ok(())
    }
    #[test]
    fn kernel_max_mul_dimension_mismatch_fails() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 5, 4)?;
        assert!(kernel.max_mul(&a, &b).is_err());
        Ok(())
    }
    #[test]
    fn kernel_contract_bool_dimension_mismatch_fails() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 5, 4)?;
        assert!(kernel.contract_bool(&a, &b).is_err());
        Ok(())
    }
    #[test]
    fn kernel_eqcount_dimension_mismatch_fails() -> Result<()> {
        let ctx = context();
        let kernel = GpuKernel::new(&ctx);
        let a = matrix(&ctx, 2, 3)?;
        let b = matrix(&ctx, 5, 4)?;
        assert!(kernel.eqcount(&a, &b).is_err());
        Ok(())
    }
}
