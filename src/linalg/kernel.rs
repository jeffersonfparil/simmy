use crate::linalg::context::GpuContext;

#[derive(Debug)]
#[expect(dead_code)]
pub struct GpuKernel {
    pipeline: wgpu::ComputePipeline,
}

impl GpuKernel {
    pub fn new(ctx: &GpuContext, wgsl: &str) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &module,
                entry_point: None,
                compilation_options: Default::default(),
                cache: None,
            });
        Self { pipeline }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn context() -> GpuContext {
        pollster::block_on(GpuContext::new()).expect("Failed to create GPU context")
    }
    const SIMPLE_SHADER: &str = r#"
        @compute
        @workgroup_size(1)
        fn main() {
        }
    "#;
    #[test]
    fn creates_kernel_from_valid_wgsl() {
        let ctx = context();
        let _kernel = GpuKernel::new(&ctx, SIMPLE_SHADER);
    }
    #[test]
    fn creates_multiple_kernels() {
        let ctx = context();
        let _kernel1 = GpuKernel::new(&ctx, SIMPLE_SHADER);
        let _kernel2 = GpuKernel::new(&ctx, SIMPLE_SHADER);
    }
    #[test]
    fn creates_kernel_with_storage_buffer_binding() {
        let ctx = context();
        let shader = r#"
            @group(0) @binding(0)
            var<storage, read_write> data: array<f32>;
            @compute
            @workgroup_size(1)
            fn main(
                @builtin(global_invocation_id)
                gid: vec3<u32>
            ) {
                data[gid.x] *= 2.0;
            }
        "#;
        let _kernel = GpuKernel::new(&ctx, shader);
    }
    #[test]
    fn creates_kernel_with_uniform_binding() {
        let ctx = context();
        let shader = r#"
            struct Params {
                value: f32,
            };
            @group(0) @binding(0)
            var<uniform> params: Params;
            @compute
            @workgroup_size(1)
            fn main() {
                let _x = params.value;
            }
        "#;
        let _kernel = GpuKernel::new(&ctx, shader);
    }
}
