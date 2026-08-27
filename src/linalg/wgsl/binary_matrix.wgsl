struct BinaryMatrixParams {
    n: u32,
    p: u32,

    a_offset: u32,
    a_row_stride: u32,
    a_col_stride: u32,

    b_offset: u32,
    b_row_stride: u32,
    b_col_stride: u32,

    c_offset: u32,
    c_row_stride: u32,
    c_col_stride: u32,

    op: u32,
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
var<storage> params: BinaryMatrixParams;

@compute
@workgroup_size(16,16,1)
fn main(
    @builtin(global_invocation_id)
    gid: vec3<u32>
) {
    let row = gid.y;
    let col = gid.x;
    if (row >= params.n || col >= params.p) {
        return;
    }
    let a_idx = matrix_index(
        row,
        col,
        params.a_offset,
        params.a_row_stride,
        params.a_col_stride
    );
    let b_idx = matrix_index(
        row,
        col,
        params.b_offset,
        params.b_row_stride,
        params.b_col_stride
    );
    let c_idx = matrix_index(
        row,
        col,
        params.c_offset,
        params.c_row_stride,
        params.c_col_stride
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
        default: {
            return;
        }
    }
    C[c_idx] = result;
}