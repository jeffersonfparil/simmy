struct BinaryTensorParams {
    // Tensor rank
    rank: u32,

    // Total number of logical output elements.
    //
    // Example:
    //     shape = [1000, 64]
    //
    // then:
    //     n_elements = 64000
    //
    n_elements: u32,

    // Logical output shape.
    //
    // For broadcasting operations this is the shape of
    // the resulting tensor.
    //
    // Example:
    //
    //     A = [1000, 64]
    //     B = [1,    64]
    //     C = [1000, 64]
    //
    // shape = [1000, 64]
    //
    shape: array<u32, 8>,

    // Tensor A metadata
    a_offset: u32,
    a_strides: array<u32, 8>,

    // Tensor B metadata
    b_offset: u32,
    b_strides: array<u32, 8>,

    // Tensor C metadata
    c_offset: u32,
    c_strides: array<u32, 8>,

    op: u32,
};

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

    var idx = linear_idx;

    // Start at beginning of tensor view.
    var storage_idx = offset;

    // Recover coordinates from linear index.
    //
    // Example:
    //
    //     shape = [2, 3, 4]
    //     linear_idx = 17
    //
    // gives:
    //
    //     (1, 1, 1)
    //
    for (var axis = i32(rank) - 1; axis >= 0; axis--) {

        let i = u32(axis);

        let coord = idx % shape[i];

        idx = idx / shape[i];

        // IMPORTANT:
        //
        // Broadcasting is implemented through
        // zero-stride dimensions.
        //
        // Example:
        //
        //     tensor shape  = [1, 64]
        //     tensor stride = [0, 1]
        //
        // Logical coordinate:
        //
        //     (732, 17)
        //
        // contributes:
        //
        //     732 * 0 + 17 * 1
        //
        // therefore indexing:
        //
        //     (0, 17)
        //
        // This automatically implements
        // NumPy/PyTorch-style broadcasting
        // without any special-case logic.
        //
        storage_idx += coord * strides[i];
    }

    return storage_idx;
}

@group(0) @binding(0)
var<storage, read> A: array<f32>;

@group(0) @binding(1)
var<storage, read> B: array<f32>;

@group(0) @binding(2)
var<storage, read_write> C: array<f32>;

@group(0) @binding(3)
var<storage> params: BinaryTensorParams;

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

    let b_idx = tensor_index(
        linear_idx,
        params.b_offset,
        params.shape,
        params.b_strides,
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
    let b = B[b_idx];

    var result = a;

    switch(params.op) {

        case OP_ADD: {
            result = a + b;
        }

        case OP_SUB: {
            result = a - b;
        }

        case OP_MUL: {
            result = a * b;
        }

        case OP_DIV: {
            result = a / b;
        }

        case OP_MIN: {
            result = min(a, b);
        }

        case OP_MAX: {
            result = max(a, b);
        }

        case OP_POW: {
            result = pow(a, b);
        }

        case OP_ATAN2: {
            result = atan2(a, b);
        }

        case OP_EQ: {
            result = select(0.0, 1.0, a == b);
        }

        case OP_NE: {
            result = select(0.0, 1.0, a != b);
        }

        case OP_LT: {
            result = select(0.0, 1.0, a < b);
        }

        case OP_LE: {
            result = select(0.0, 1.0, a <= b);
        }

        case OP_GT: {
            result = select(0.0, 1.0, a > b);
        }

        case OP_GE: {
            result = select(0.0, 1.0, a >= b);
        }

        case OP_AND: {
            result = select(
                0.0,
                1.0,
                (a != 0.0) && (b != 0.0)
            );
        }

        case OP_OR: {
            result = select(
                0.0,
                1.0,
                (a != 0.0) || (b != 0.0)
            );
        }

        case OP_XOR: {
            result = select(
                0.0,
                1.0,
                (a != 0.0) != (b != 0.0)
            );
        }

        default: {
            return;
        }
    }

    C[c_idx] = result;
}