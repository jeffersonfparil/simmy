use bytemuck::{Pod, Zeroable};

const MAX_RANK: usize = 8;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct UnaryMatrixParams {
    /// Number of rows in `A`, and `C`.
    pub n: u32,
    /// Number of columns in `A`, and `C`.
    pub p: u32,
    /// Starting element of `A` within its backing storage.
    pub a_offset: u32,
    /// Storage stride between rows of `A`.
    pub a_row_stride: u32,
    /// Storage stride between columns of `A`.
    pub a_col_stride: u32,
    /// Starting element of `C` within its backing storage.
    pub c_offset: u32,
    /// Storage stride between rows of `C`.
    pub c_row_stride: u32,
    /// Storage stride between columns of `C`.
    pub c_col_stride: u32,
    /// Mathematical operation: op(A) -> C.
    ///     - OP_ABS = 0
    ///     - OP_NEG = 1
    ///     - OP_SQRT = 2
    ///     - OP_EXP = 3
    ///     - OP_LOG = 4
    ///     - OP_SIN = 5
    ///     - OP_COS = 6
    pub op: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct BinaryMatrixParams {
    /// Number of rows in `A`, `B`, and `C`.
    pub n: u32,
    /// Number of columns in `A`, `B`, and `C`.
    pub p: u32,
    /// Starting element of `A` within its backing storage.
    pub a_offset: u32,
    /// Storage stride between rows of `A`.
    pub a_row_stride: u32,
    /// Storage stride between columns of `A`.
    pub a_col_stride: u32,
    /// Starting element of `B` within its backing storage.
    pub b_offset: u32,
    /// Storage stride between rows of `B`.
    pub b_row_stride: u32,
    /// Storage stride between columns of `B`.
    pub b_col_stride: u32,
    /// Starting element of `C` within its backing storage.
    pub c_offset: u32,
    /// Storage stride between rows of `C`.
    pub c_row_stride: u32,
    /// Storage stride between columns of `C`.
    pub c_col_stride: u32,
    /// Mathematical operation: op(A, B) -> C.
    ///     - OP_ADD = 0;
    ///     - OP_SUB = 1;
    ///     - OP_MUL = 2;
    ///     - OP_DIV = 3;
    ///     - OP_MIN = 4;
    ///     - OP_MAX = 5;
    ///     - OP_POW = 6;
    ///     - OP_ATAN2 = 7;
    ///     - OP_EQ = 8;
    ///     - OP_NE = 9;
    ///     - OP_LT = 10;
    ///     - OP_LE = 11;
    ///     - OP_GT = 12;
    ///     - OP_GE = 13;
    pub op: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ContractMatrixParams {
    /// Number of rows in `A` and `C`.
    pub n: u32,
    /// Shared contraction dimension.
    pub p: u32,
    /// Number of columns in `B` and `C`.
    pub k: u32,
    /// Starting element of `A` within its backing storage.
    pub a_offset: u32,
    /// Storage stride between rows of `A`.
    pub a_row_stride: u32,
    /// Storage stride between columns of `A`.
    pub a_col_stride: u32,
    /// Starting element of `B` within its backing storage.
    pub b_offset: u32,
    /// Storage stride between rows of `B`.
    pub b_row_stride: u32,
    /// Storage stride between columns of `B`.
    pub b_col_stride: u32,
    /// Starting element of `C` within its backing storage.
    pub c_offset: u32,
    /// Storage stride between rows of `C`.
    pub c_row_stride: u32,
    /// Storage stride between columns of `C`.
    pub c_col_stride: u32,
    /// Pairwise operation
    ///     - OP_PAIR_ADD = 0;
    ///     - OP_PAIR_SUB = 1;
    ///     - OP_PAIR_MUL = 2;
    ///     - OP_PAIR_DIV = 3;
    ///     - OP_PAIR_MIN = 4;
    ///     - OP_PAIR_MAX = 5;
    pub op_pairwise: u32,
    /// Reduction operation
    ///     - OP_REDUCE_ADD = 0;
    ///     - OP_REDUCE_MUL = 1;
    ///     - OP_REDUCE_MIN = 2;
    ///     - OP_REDUCE_MAX = 3;
    pub op_reduction: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct UnaryTensorParams {
    /// Number of tensor dimensions.
    pub rank: u32,
    /// Total number of logical tensor elements.
    pub n_elements: u32,
    /// Shape of the logical tensor.
    pub shape: [u32; MAX_RANK],
    /// Starting element of `A` within its backing storage.
    pub a_offset: u32,
    /// Storage strides of `A` per dimension.
    pub a_strides: [u32; MAX_RANK],
    /// Starting element of `C` within its backing storage.
    pub c_offset: u32,
    /// Storage strides of `C` per dimension.
    pub c_strides: [u32; MAX_RANK],
    /// Mathematical operation: op(A) -> C.
    ///     - OP_ABS = 0
    ///     - OP_NEG = 1
    ///     - OP_SQRT = 2
    ///     - OP_EXP = 3
    ///     - OP_LOG = 4
    ///     - OP_SIN = 5
    ///     - OP_COS = 6
    pub op: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct BinaryTensorParams {
    /// Number of tensor dimensions.
    pub rank: u32,
    /// Total number of logical tensor elements.
    pub n_elements: u32,
    /// Shape of the logical tensor.
    pub shape: [u32; MAX_RANK],
    /// Starting element of `A` within its backing storage.
    pub a_offset: u32,
    /// Storage strides of `A` per dimension.
    pub a_strides: [u32; MAX_RANK],
    /// Starting element of `B` within its backing storage.
    pub b_offset: u32,
    /// Storage strides of `B` per dimension.
    pub b_strides: [u32; MAX_RANK],
    /// Starting element of `C` within its backing storage.
    pub c_offset: u32,
    /// Storage strides of `C` per dimension.
    pub c_strides: [u32; MAX_RANK],
    /// Mathematical operation: op(A, B) -> C.
    ///     - OP_ADD = 0;
    ///     - OP_SUB = 1;
    ///     - OP_MUL = 2;
    ///     - OP_DIV = 3;
    ///     - OP_MIN = 4;
    ///     - OP_MAX = 5;
    ///     - OP_POW = 6;
    ///     - OP_ATAN2 = 7;
    ///     - OP_EQ = 8;
    ///     - OP_NE = 9;
    ///     - OP_LT = 10;
    ///     - OP_LE = 11;
    ///     - OP_GT = 12;
    ///     - OP_GE = 13;
    pub op: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ContractTensorParams {
    // Number of dimensions in tensor A.
    pub a_rank: u32,
    // Number of dimensions in tensor B.
    pub b_rank: u32,
    // Number of dimensions in tensor C.
    pub c_rank: u32,
    // Number of contracted axis pairs.
    // Example:
    //     A[a,b,c,d]
    //     B[c,d,e,f]
    // Contracts:
    //     [c,d]
    // contraction_rank = 2
    pub contraction_rank: u32,
    // Total number of logical elements in the output tensor C.
    // i.e.: product(c_shape[..c_rank])
    pub c_elements: u32,
    // Logical shape of tensor A.
    pub a_shape: [u32; MAX_RANK],
    // Logical shape of tensor B.
    pub b_shape: [u32; MAX_RANK],
    // Logical shape of tensor C.
    pub c_shape: [u32; MAX_RANK],
    /// Starting element of `A` within its backing storage.
    pub a_offset: u32,
    /// Storage strides of `A` per dimension.
    pub a_strides: [u32; MAX_RANK],
    /// Starting element of `B` within its backing storage.
    pub b_offset: u32,
    /// Storage strides of `B` per dimension.
    pub b_strides: [u32; MAX_RANK],
    /// Starting element of `C` within its backing storage.
    pub c_offset: u32,
    /// Storage strides of `C` per dimension.
    pub c_strides: [u32; MAX_RANK],
    // Contracted axes of tensor A.
    // Example:
    //     A[a,b,c,d]
    //     B[c,d,e,f]
    // a_contract_axes = [2,3]
    pub a_contract_axes: [u32; MAX_RANK],
    // Contracted axes of tensor B.
    // Example:
    //     A[a,b,c,d]
    //     B[c,d,e,f]
    // b_contract_axes = [0,1]
    pub b_contract_axes: [u32; MAX_RANK],
    /// Pairwise operation
    ///     - OP_PAIR_ADD = 0;
    ///     - OP_PAIR_SUB = 1;
    ///     - OP_PAIR_MUL = 2;
    ///     - OP_PAIR_DIV = 3;
    ///     - OP_PAIR_MIN = 4;
    ///     - OP_PAIR_MAX = 5;
    pub op_pairwise: u32,
    /// Reduction operation
    ///     - OP_REDUCE_ADD = 0;
    ///     - OP_REDUCE_MUL = 1;
    ///     - OP_REDUCE_MIN = 2;
    ///     - OP_REDUCE_MAX = 3;
    pub op_reduction: u32,
}
