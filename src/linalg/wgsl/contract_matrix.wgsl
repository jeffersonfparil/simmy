struct ContractMatrixParams {
    // A %*% B = C
    n: u32, // nrows of A and C
    p: u32, // nrows of B and ncols of A
    k: u32, // ncols of B and C

    a_offset: u32,
    a_row_stride: u32,
    a_col_stride: u32,

    b_offset: u32,
    b_row_stride: u32,
    b_col_stride: u32,

    c_offset: u32,
    c_row_stride: u32,
    c_col_stride: u32,
    
    op_pairwise: u32,
    op_reduction: u32,
};

fn matrix_index(
    row: u32,
    col: u32,
    offset: u32,
    row_stride: u32,
    col_stride: u32
) -> u32 {
    return offset +
           row * row_stride +
           col * col_stride;
}

@group(0) @binding(0)
var<storage, read> A: array<f32>;

@group(0) @binding(1)
var<storage, read> B: array<f32>;

@group(0) @binding(2)
var<storage, read_write> C: array<f32>;

@group(0) @binding(3)
var<storage> params: ContractMatrixParams;

@compute
@workgroup_size(16,16,1)
fn main(
    @builtin(global_invocation_id)
    gid: vec3<u32>
) {
    let row = gid.y;
    let col = gid.x;
    if (row >= params.n || col >= params.k) {
        return;
    }
    var reduced = 0.0;
    switch(params.op_reduction) {
        case OP_REDUCE_ADD: {
            reduced = 0.0;
        }
        case OP_REDUCE_MUL: {
            reduced = 1.0;
        }
        case OP_REDUCE_MIN: {
            reduced = 3.4028235e+38;
        }
        case OP_REDUCE_MAX: {
            reduced = -3.4028235e+38;
        }
        default: {
            return;
        }
    }
    for (var p = 0u; p < params.p; p++) {
        let a_idx = matrix_index(
            row,
            p,
            params.a_offset,
            params.a_row_stride,
            params.a_col_stride
        );
        let b_idx = matrix_index(
            p,
            col,
            params.b_offset,
            params.b_row_stride,
            params.b_col_stride
        );
        let a = A[a_idx];
        let b = B[b_idx];
        var paired = a;
        switch(params.op_pairwise) {
            case OP_PAIR_ADD: {
                paired = a + b;
            }
            case OP_PAIR_SUB: {
                paired = a - b;
            }
            case OP_PAIR_MUL: {
                paired = a * b;
            }
            case OP_PAIR_DIV: {
                paired = a / b;
            }
            case OP_PAIR_MIN: {
                paired = min(a, b);
            }
            case OP_PAIR_MAX: {
                paired = max(a, b);
            }
            default: {
                return;
            }
        }
        switch(params.op_reduction) {
            case OP_REDUCE_ADD: {
                reduced += paired;
            }
            case OP_REDUCE_MUL: {
                reduced *= paired;
            }
            case OP_REDUCE_MIN: {
                reduced = min(reduced, paired);
            }
            case OP_REDUCE_MAX: {
                reduced = max(reduced, paired);
            }
            default: {
                return;
            }
        }
    }
    let c_idx = matrix_index(
        row,
        col,
        params.c_offset,
        params.c_row_stride,
        params.c_col_stride
    );
    C[c_idx] = reduced;
}