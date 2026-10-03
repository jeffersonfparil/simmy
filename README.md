# simmy
Quantitative genetics and population breeding simulation tool (with [WGSL](https://en.wikipedia.org/wiki/WebGPU_Shading_Language)-based tensors and kernels)

## Quickstart

### Compile

```shell
cargo build --release
```

### Examples

```shell
# TODO: UI and CLI arguments still to be drafted!
```


# Development stuff

## Architecture

```shell
src/
├── main.rs
├── lib.rs
├── io.rs
└── linalg/
    ├── context.rs          # GpuContext initialisation (WGPU instance, adapter, device, queue)
    ├── tensor.rs           # GpuTensor memory model (shape, strides, offsets, buffer)
    ├── kernel.rs           # GpuKernel compute pipeline bindings
    ├── operations.rs       # Maths operations & WGSL opcodes mapping
    ├── params.rs           # Buffer parameter structs for shaders
    ├── transpose.rs        # Stride-based zero-copy tensor axis permutations
    ├── wrappers_matrix.rs  # Matrix-specific kernel dispatch wrappers
    ├── wrappers_tensor.rs  # Arbitrary-rank tensor kernel dispatch wrappers
    └── wgsl/
        ├── opcodes.wgsl            # Operation codes
        ├── unary_matrix.wgsl       # Element-wise unary matrix kernels
        ├── binary_matrix.wgsl      # Element-wise binary matrix kernels
        ├── contract_matrix.wgsl    # Matrix multiplication & other contraction operations
        ├── unary_tensor.wgsl       # Element-wise unary tensor kernels
        ├── binary_tensor.wgsl      # Element-wise binary tensor kernels
        └── contract_tensor.wgsl    # Arbitrary rank tensor contractions
```

## Testing

```shell
cd simmy/
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo run
# cargo test -- --show-output
# cargo test -- --test-threads=1
# cargo build --release
# cargo doc --open
cargo tree
```

## Roadmap

- [-] WGSL Tensors and operations
- [-] Basic structs
- [-] I/O (will need I/O for the full genome structs + confg.toml file or something...)
- [ ] Summary stats
- [ ] Breeding program builder and runner
- [ ] UI (CLI arguments parsing and main entry point)

# Licence

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).