struct UnaryMatrixParams {
    // op(A) = C
    n: u32,
    p: u32,

    a_offset: u32,
    a_row_stride: u32,
    a_col_stride: u32,

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
var<storage, read_write> C: array<f32>;

@group(0) @binding(2)
var<storage> params: UnaryMatrixParams;

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
    let c_idx = matrix_index(
        row,
        col,
        params.c_offset,
        params.c_row_stride,
        params.c_col_stride
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
        case OP_NOT: {
            result = select(0.0, 1.0, a == 0.0);
        }
        default: {
            return;
        }
    }
    C[c_idx] = result;
}