///////////////////////////////////////////////////////////////////////////////
// Generic Tensor Contraction Kernel
//
// ELI5
// ----
// Matrix multiplication computes:
//
//     C[i,j] = Σk A[i,k] * B[k,j]
//
// For every output element C[i,j], we:
//
// 1. Pick a row from A.
// 2. Pick a column from B.
// 3. Multiply matching values.
// 4. Add (sum) the results.
//
// The shared dimension k disappears because it is summed over.
//
// Tensor contraction is exactly the same idea, but generalized from
// matrices (rank-2 tensors) to tensors of arbitrary rank.
//
// Example matrix multiplication:
//
//     A[m,k]
//     B[k,n]
//
//     C[m,n]
//
// We contract:
//
//     A axis 1
//     B axis 0
//
// Example tensor contraction:
//
//     A[a,b,c,d]
//     B[c,d,e,f]
//
//     C[a,b,e,f]
//
// We contract:
//
//     A axes [2,3]
//     B axes [0,1]
//
// The contracted dimensions (c,d) disappear from the result because
// we sum over them, just like k disappears during matrix multiplication.
//
// Core Principle
// --------------
// Every output element of C is computed independently:
//
//     C[free_axes]
//
// We:
//
// 1. Decode which logical element of C this thread owns.
// 2. Recover the corresponding coordinates in A and B for all
//    non-contracted ("free") axes.
// 3. Enumerate every coordinate in the contraction subspace.
// 4. Read matching elements from A and B.
// 5. Accumulate:
//
//        sum += A[...] * B[...]
//
// 6. Write the final sum into C.
//
// Unlike matrix multiplication, this kernel:
//
// * Supports arbitrary tensor rank.
// * Supports arbitrary contraction axes.
// * Supports tensor views.
// * Supports tensor slices.
// * Supports tensor transposes.
//
// because all indexing is performed through shape, stride, and offset
// metadata rather than assuming contiguous matrix storage.
//
// Output Axis Ordering
// --------------------
// The output tensor C is defined as:
//
//     C = [A free axes] + [B free axes]
//
// Example:
//
//     A[a,b,c]
//     B[c,d,e]
//
// Contract:
//
//     c
//
// Result:
//
//     C[a,b,d,e]
//
// The first dimensions of C come from the non-contracted axes of A,
// followed by the non-contracted axes of B.
///////////////////////////////////////////////////////////////////////////////

struct TensorMulParams {
    // Tensor ranks.
    a_rank: u32,
    b_rank: u32,
    c_rank: u32,

    // Number of contracted axis pairs.
    contraction_rank: u32,

    // Total logical elements in C.
    c_elements: u32,

    // Logical tensor shapes.
    a_shape: array<u32, 8>,
    b_shape: array<u32, 8>,
    c_shape: array<u32, 8>,

    // Storage layout of A.
    a_offset: u32,
    a_strides: array<u32, 8>,

    // Storage layout of B.
    b_offset: u32,
    b_strides: array<u32, 8>,

    // Storage layout of C.
    c_offset: u32,
    c_strides: array<u32, 8>,

    // Contracted axes.
    a_contract_axes: array<u32, 8>,
    b_contract_axes: array<u32, 8>,
};

///////////////////////////////////////////////////////////////////////////////
// Decode a linear tensor element index into tensor coordinates.
//
// Example:
//
//     shape = [2,3,4]
//
//     linear_idx = 17
//
// becomes:
//
//     coords = [1,1,1]
//
// Conceptually this performs a mixed-radix base conversion where each
// shape dimension acts as the base for that axis.
///////////////////////////////////////////////////////////////////////////////
fn tensor_coords(
    linear_idx: u32,
    rank: u32,
    shape: array<u32, 8>,
) -> array<u32, 8> {

    var coords: array<u32, 8>;

    var idx = linear_idx;

    // Recover coordinates from fastest-moving axis to
    // slowest-moving axis.
    for (var axis = i32(rank) - 1; axis >= 0; axis--) {
        let i = u32(axis);

        coords[i] = idx % shape[i];
        idx = idx / shape[i];
    }

    return coords;
}

///////////////////////////////////////////////////////////////////////////////
// Convert tensor coordinates into an index within a backing storage buffer.
//
// Given:
//
//     coords  = [i₀,i₁,...]
//
// and:
//
//     strides = [s₀,s₁,...]
//
// computes:
//
//     offset +
//     i₀*s₀ +
//     i₁*s₁ +
//     ...
//
// This enables arbitrary tensor views, slices, and transposes.
///////////////////////////////////////////////////////////////////////////////
fn storage_index(
    coords: array<u32, 8>,
    rank: u32,
    offset: u32,
    strides: array<u32, 8>,
) -> u32 {

    var idx = offset;

    for (var axis = 0u; axis < rank; axis++) {
        idx += coords[axis] * strides[axis];
    }

    return idx;
}

@group(0) @binding(0)
var<storage, read> A: array<f32>;

@group(0) @binding(1)
var<storage, read> B: array<f32>;

@group(0) @binding(2)
var<storage, read_write> C: array<f32>;

@group(0) @binding(3)
var<storage, read> params: TensorMulParams;

@compute
@workgroup_size(256)
fn main(
    @builtin(global_invocation_id)
    gid: vec3<u32>,
) {
    // One thread computes one logical element of C.
    let c_linear_idx = gid.x;

    if (c_linear_idx >= params.c_elements) {
        return;
    }

    ///////////////////////////////////////////////////////////////////////
    // Step 1.
    //
    // Determine which logical output element this thread owns.
    //
    // Convert:
    //
    //     c_linear_idx
    //
    // into:
    //
    //     c_coords
    //
    ///////////////////////////////////////////////////////////////////////
    var c_coords = tensor_coords(
        c_linear_idx,
        params.c_rank,
        params.c_shape,
    );

    ///////////////////////////////////////////////////////////////////////
    // Step 2.
    //
    // Build coordinate vectors for A and B. Contracted coordinates will
    // be filled later while iterating through the contraction space.
    ///////////////////////////////////////////////////////////////////////
    var a_coords: array<u32, 8>;
    var b_coords: array<u32, 8>;

    ///////////////////////////////////////////////////////////////////////
    // Step 3.
    //
    // Copy free (non-contracted) coordinates from C into A and B.
    //
    // Since:
    //
    //     C = [A free axes] + [B free axes]
    //
    // C's coordinates can be distributed back into A and B.
    ///////////////////////////////////////////////////////////////////////
    var c_axis = 0u;

    for (var a_axis = 0u; a_axis < params.a_rank; a_axis++) {

        var contracted = false;

        for (var i = 0u;
             i < params.contraction_rank;
             i++) {

            if (a_axis == params.a_contract_axes[i]) {
                contracted = true;
                break;
            }
        }

        if (!contracted) {
            a_coords[a_axis] = c_coords[c_axis];
            c_axis += 1u;
        }
    }

    for (var b_axis = 0u; b_axis < params.b_rank; b_axis++) {

        var contracted = false;

        for (var i = 0u;
             i < params.contraction_rank;
             i++) {

            if (b_axis == params.b_contract_axes[i]) {
                contracted = true;
                break;
            }
        }

        if (!contracted) {
            b_coords[b_axis] = c_coords[c_axis];
            c_axis += 1u;
        }
    }

    ///////////////////////////////////////////////////////////////////////
    // Step 4.
    //
    // Compute the total number of coordinates in the contraction space.
    //
    // Example:
    //
    //     contract axes sizes = [4, 5]
    //
    // then:
    //
    //     contract_elements = 20
    //
    ///////////////////////////////////////////////////////////////////////
    var contract_elements = 1u;

    for (var i = 0u;
         i < params.contraction_rank;
         i++) {

        let a_axis =
            params.a_contract_axes[i];

        contract_elements *=
            params.a_shape[a_axis];
    }

    ///////////////////////////////////////////////////////////////////////
    // Step 5.
    //
    // Enumerate every coordinate in the contraction subspace and compute:
    //
    //     Σ A[...] * B[...]
    //
    ///////////////////////////////////////////////////////////////////////
    var sum = 0.0;

    for (var contract_linear = 0u;
         contract_linear < contract_elements;
         contract_linear++) {

        ///////////////////////////////////////////////////////////////////
        // Decode one point in the contraction subspace.
        ///////////////////////////////////////////////////////////////////
        var tmp = contract_linear;

        for (var i = i32(params.contraction_rank) - 1;
             i >= 0;
             i--) {

            let contract_axis = u32(i);

            let a_axis =
                params.a_contract_axes[contract_axis];

            let b_axis =
                params.b_contract_axes[contract_axis];

            let extent =
                params.a_shape[a_axis];

            let coord =
                tmp % extent;

            tmp =
                tmp / extent;

            a_coords[a_axis] =
                coord;

            b_coords[b_axis] =
                coord;
        }

        ///////////////////////////////////////////////////////////////////
        // Convert tensor coordinates into backing-buffer locations.
        ///////////////////////////////////////////////////////////////////
        let a_idx = storage_index(
            a_coords,
            params.a_rank,
            params.a_offset,
            params.a_strides,
        );

        let b_idx = storage_index(
            b_coords,
            params.b_rank,
            params.b_offset,
            params.b_strides,
        );

        ///////////////////////////////////////////////////////////////////
        // Accumulate contribution from this contraction coordinate.
        ///////////////////////////////////////////////////////////////////
        sum += A[a_idx] * B[b_idx];
    }

    ///////////////////////////////////////////////////////////////////////
    // Step 6.
    //
    // Store the final result.
    ///////////////////////////////////////////////////////////////////////
    let c_idx = storage_index(
        c_coords,
        params.c_rank,
        params.c_offset,
        params.c_strides,
    );

    C[c_idx] = sum;
}
