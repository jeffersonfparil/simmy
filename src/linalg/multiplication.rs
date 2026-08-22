use crate::linalg::operations::{MatrixOps, TensorOps};
use crate::linalg::tensor::GpuTensor;
use anyhow::{Result, ensure};
use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct MatMulParams {
    // Matrix multiplication dimensions:
    // A: n × p
    // B: p × k
    // C: n × k
    n: u32,
    p: u32,
    k: u32,

    a_offset: u32,
    a_row_stride: u32,
    a_col_stride: u32,

    b_offset: u32,
    b_row_stride: u32,
    b_col_stride: u32,

    c_offset: u32,
    c_row_stride: u32,
    c_col_stride: u32,

    // Padding added to satisfy uniform-buffer alignment requirements.
    // Depending on the WGSL layout and GPU backend this may be
    // required to ensure the Rust struct's memory layout matches
    // what the kernel expects. If size/alignment checks show it is
    // unnecessary, this field can be removed.
    padding: u32,
}

impl MatrixOps<'_> {
    pub fn multiply(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        ensure!(a.shape.len() == 2, "A must be rank 2!");
        ensure!(b.shape.len() == 2, "B must be rank 2!");
        ensure!(
            a.shape[1] == b.shape[0],
            "Incomaptible sahpes: A {:?} x B {:?}!",
            a.shape,
            b.shape
        );
        let kernel_source = include_str!("wgsl/matmul.wgsl");
        let kernel_module = self
            .ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("matmul-kernel"),
                source: wgpu::ShaderSource::Wgsl(kernel_source.into()),
            });
        let n = a.shape[0];
        let p = a.shape[1];
        let k = b.shape[1];
        let params = MatMulParams {
            n,
            p,
            k,
            a_offset: 0,
            a_row_stride: p,
            a_col_stride: 1,
            b_offset: 0,
            b_row_stride: k,
            b_col_stride: 1,
            c_offset: 0,
            c_row_stride: k,
            c_col_stride: 1,
            padding: 0,
        };
        let params_buffer = self.ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("matmul-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let c_elements = n * k;
        let c_buffer = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matmul-output"),
            size: (c_elements as u64) * (std::mem::size_of::<f32>() as u64),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pipeline = self
            .ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("matmul-pipeline"),
                layout: None,
                module: &kernel_module,
                entry_point: None,
                compilation_options: Default::default(),
                cache: None,
            });
        let bind_group = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("matmul-bind-group"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: a.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: c_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("matmul-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let dispatch_x = k.div_ceil(16);
            let dispatch_y = n.div_ceil(16);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }
        self.ctx.queue.submit(std::iter::once(encoder.finish()));
        Ok(GpuTensor {
            shape: vec![n, k],
            buffer: c_buffer,
        })
    }
}

// TODO: implement...
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct TensorMulParams {}

impl TensorOps<'_> {
    pub fn multiply(&self, _a: &GpuTensor, _b: &GpuTensor) -> Result<GpuTensor> {
        todo!(
            "Implement for tensors of arbitrary ranks (as long as they are compatible) and along whichever dimension to sum over!"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::context::GpuContext;
    fn context() -> GpuContext {
        pollster::block_on(GpuContext::new()).expect("Failed to create GPU context")
    }
    #[test]
    fn matmul_returns_expected_output_shape() -> Result<()> {
        let ctx = context();
        let a = GpuTensor::from_f32(&ctx, vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
        let b = GpuTensor::from_f32(&ctx, vec![3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0])?;
        let ops = MatrixOps { ctx: &ctx };
        let c = ops.multiply(&a, &b).expect("Matrix multiplication failed");
        assert_eq!(c.shape, vec![2, 2]);
        Ok(())
    }
    #[test]
    fn matmul_rejects_rank_1_a() -> Result<()> {
        let ctx = context();
        let a = GpuTensor::from_f32(&ctx, vec![6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
        let b = GpuTensor::from_f32(&ctx, vec![3, 2], &[1.0; 6])?;
        let ops = MatrixOps { ctx: &ctx };
        assert!(ops.multiply(&a, &b).is_err());
        Ok(())
    }
    #[test]
    fn matmul_rejects_rank_1_b() -> Result<()> {
        let ctx = context();
        let a = GpuTensor::from_f32(&ctx, vec![2, 3], &[1.0; 6])?;
        let b = GpuTensor::from_f32(&ctx, vec![6], &[1.0; 6])?;
        let ops = MatrixOps { ctx: &ctx };
        assert!(ops.multiply(&a, &b).is_err());
        Ok(())
    }
    #[test]
    fn matmul_rejects_incompatible_shapes() -> Result<()> {
        let ctx = context();
        let a = GpuTensor::from_f32(&ctx, vec![2, 3], &[1.0; 6])?;
        let b = GpuTensor::from_f32(&ctx, vec![4, 2], &[1.0; 8])?;
        let ops = MatrixOps { ctx: &ctx };
        let result = ops.multiply(&a, &b);
        assert!(result.is_err());
        Ok(())
    }
    #[test]
    fn matmul_accepts_square_matrices() -> Result<()> {
        let ctx = context();
        let a = GpuTensor::from_f32(&ctx, vec![4, 4], &[1.0; 16])?;
        let b = GpuTensor::from_f32(&ctx, vec![4, 4], &[1.0; 16])?;
        let ops = MatrixOps { ctx: &ctx };
        let c = ops.multiply(&a, &b).expect("Matrix multiplication failed");
        assert_eq!(c.shape, vec![4, 4]);
        Ok(())
    }
    #[test]
    fn matmul_accepts_non_square_matrices() -> Result<()> {
        let ctx = context();
        let a = GpuTensor::from_f32(&ctx, vec![5, 3], &[1.0; 15])?;
        let b = GpuTensor::from_f32(&ctx, vec![3, 7], &[1.0; 21])?;
        let ops = MatrixOps { ctx: &ctx };
        let c = ops.multiply(&a, &b).expect("Matrix multiplication failed");
        assert_eq!(c.shape, vec![5, 7]);
        Ok(())
    }
}
