struct UnaryTensorParams {
    rank: u32,
    n_elements: u32,

    shape: array<u32, 8>,

    a_offset: u32,
    a_strides: array<u32, 8>,

    c_offset: u32,
    c_strides: array<u32, 8>,

    op: u32,
};

// Convert a logical tensor element index into an index within the
// underlying storage buffer.
//
// Tensor operations dispatch one thread per logical tensor element.
// For example, a tensor with shape:
//     [2, 3, 4]
// contains:
//     2 × 3 × 4 = 24
// logical elements, numbered:
//     0, 1, 2, ..., 23
//
// The GPU kernel therefore receives a flat (linear) index:
//     linear_idx
// but the tensor storage may be strided, transposed, or represent a
// view into another tensor. We therefore cannot use `linear_idx`
// directly to access the backing buffer.
//
// The algorithm proceeds in two steps:
// 1. Convert the linear index into tensor coordinates.
//    For shape [2, 3, 4]:
//        linear_idx = 17
//    corresponds to:
//        (1, 1, 1)
//    The coordinates are recovered from the last axis toward the first
//    using repeated modulus and integer division:
//        coord = idx % dimension_size
//        idx   = idx / dimension_size
//    This is analogous to extracting digits from a number in a mixed
//    radix system whose bases are given by the tensor shape.
//
// 2. Convert tensor coordinates into a storage index.
//    Given coordinates:
//        (i₀, i₁, ..., iₙ)
//    and strides:
//        (s₀, s₁, ..., sₙ)
//    the storage position is:
//        offset +
//        i₀·s₀ +
//        i₁·s₁ +
//        ...
//        iₙ·sₙ
//
//    This formulation supports:
//    * Contiguous tensors.
//    * Tensor views.
//    * Tensor slices.
//    * Tensor transposes.
//    without changing the kernel implementation. Only the shape,
//    strides, and offset metadata need to differ.
fn tensor_index(
    linear_idx: u32,
    offset: u32,
    shape: array<u32, 8>,
    strides: array<u32, 8>,
    rank: u32,
) -> u32 {
    if (rank == 0u) {
        return offset;
    }
    // Working copy of the logical element index.
    var idx = linear_idx;
    // Initialize the storage position to the tensor's starting offset
    // within the backing buffer.
    var storage_idx = offset;
    // Recover tensor coordinates from the fastest-varying axis to the
    // slowest-varying axis.
    // For shape [2, 3, 4] and linear_idx = 17:
    //     axis = 2  -> coord = 1
    //     axis = 1  -> coord = 1
    //     axis = 0  -> coord = 1
    // yielding coordinates (1, 1, 1).
    for (var axis = i32(rank) - 1; axis >= 0; axis--) {
        let i = u32(axis);
        // Coordinate of the current tensor dimension.
        let coord = idx % shape[i];
        // Remove the coordinate just extracted, preparing the index
        // for the next (slower-varying) dimension.
        idx = idx / shape[i];
        // Accumulate the corresponding contribution to the storage
        // index using the tensor stride for this dimension.
        storage_idx += coord * strides[i];
    }
    return storage_idx;
}

@group(0) @binding(0)
var<storage, read> A: array<f32>;

@group(0) @binding(1)
var<storage, read_write> C: array<f32>;

@group(0) @binding(2)
var<storage> params: UnaryTensorParams;

@compute
@workgroup_size(256)
fn main(
    @builtin(global_invocation_id)
    gid: vec3<u32>,
) {
    let linear_idx = gid.x;
    if (linear_idx >= params.n_elements) {
        return;
    }
    let a_idx = tensor_index(
        linear_idx,
        params.a_offset,
        params.shape,
        params.a_strides,
        params.rank,
    );
    let c_idx = tensor_index(
        linear_idx,
        params.c_offset,
        params.shape,
        params.c_strides,
        params.rank,
    );
    let a = A[a_idx];
    var result = a;
    switch(params.op) {
        case OP_ABS: {
            result = abs(a);
        }
        case OP_NEG: {
            result = -a;
        }
        case OP_SQRT: {
            result = sqrt(a); // NaN if a < 0
        }
        case OP_EXP: {
            result = exp(a);
        }
        case OP_LOG: {
            result = log(a); // -inf if a == 0 and NaN if a < 0
        }
        case OP_SIN: {
            result = sin(a);
        }
        case OP_COS: {
            result = cos(a);
        }
        default: {
            return;
        }
    }
    C[c_idx] = result;
}