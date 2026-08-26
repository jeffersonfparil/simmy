struct MatrixAddParams {
    // A + B = C
    n: u32, // nrows of A, B and C
    p: u32, // ncols of A, B and C

    a_offset: u32,
    a_row_stride: u32,
    a_col_stride: u32,

    b_offset: u32,
    b_row_stride: u32,
    b_col_stride: u32,

    c_offset: u32,
    c_row_stride: u32,
    c_col_stride: u32,
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
var<uniform> params: MatrixAddParams;

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
    C[c_idx] = A[a_idx] + B[b_idx];
}