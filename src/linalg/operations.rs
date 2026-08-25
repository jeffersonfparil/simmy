use crate::linalg::addition::{MatrixAddParams, TensorAddParams};
use crate::linalg::context::GpuContext;
use crate::linalg::multiplication::MatrixMulParams;
use crate::linalg::tensor::GpuTensor;
use anyhow::Result;
use wgpu::Buffer;
use wgpu::util::{BufferInitDescriptor, DeviceExt};

/// Matrix Operations
///
/// A `MatrixOps` borrows a `GpuContext`, allowing multiple matrix
/// operations to share the same GPU device, queue, and adapter.
///
/// Because the context is shared, multiple operations may be queued and
/// executed concurrently by the underlying GPU and driver, subject to
/// hardware and backend capabilities.
///
/// * `ctx` is the GPU context used to compile kernels, allocate
///   temporary resources, and submit work to the compute device.
pub struct MatrixOps<'a> {
    pub ctx: &'a GpuContext,
}

/// Tensor Operations
///
/// A `TensorOps`  borrows a `GpuContext`, allowing multiple tensor
/// operations to share the same device, queue, and adapter.
///
/// * `ctx` is the GPU context used to compile kernels, allocate
///   temporary resources, and submit work to the compute device.
///
/// Note that we will use this struct for future tensor or arbitrarily
/// ranked tensor operations --> beyond matrices.
pub struct TensorOps<'a> {
    pub ctx: &'a GpuContext,
}

/// Kernel Parameter Payload
///
/// Each variant contains the layout and execution parameters required
/// by a specific operation and is intended to be transferred directly
/// to GPU memory before kernel dispatch.
///
/// * `Addition` contains parameters for matrix addition kernels.
/// * `Multiplication` contains parameters for matrix multiplication
///   kernels.
/// * ... more to come ...
pub enum MatrixParams {
    Addition(MatrixAddParams),
    Multiplication(MatrixMulParams),
}

pub enum TensorParams {
    Addition(TensorAddParams),
    // Multiplication(TensorMulParams),
}

impl MatrixOps<'_> {
    /// Execute a binary matrix operation on the GPU, i.e. C = op(A, B)
    ///
    /// * `params` contains the operation-specific kernel parameters.
    /// * `kernel_source` is the WGSL source code implementing the kernel.
    /// * `a` is the left-hand input tensor.
    /// * `b` is the right-hand input tensor.
    ///
    /// A new output buffer is allocated and bound as the kernel result.
    /// The kernel is then dispatched using a two-dimensional workgroup
    /// grid derived from the output matrix dimensions.
    ///
    /// The returned buffer contains the operation result and may be wrapped
    /// in a `GpuTensor` to create the corresponding tensor view.
    ///
    /// # Returns
    /// Returns the GPU buffer containing the operation result.
    ///
    /// # Errors
    /// Returns an error if:
    /// * The operation parameters are invalid.
    /// * The kernel cannot be compiled.
    /// * GPU resource allocation fails.
    /// * The operation cannot be submitted to the compute device.
    pub fn execute_binary_kernel(
        &self,
        params: MatrixParams,
        kernel_source: &str,
        a: &GpuTensor,
        b: &GpuTensor,
    ) -> Result<Buffer> {
        let (params_buffer, n, p) = match params {
            MatrixParams::Addition(par) => (
                self.ctx.device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("params"),
                    contents: bytemuck::bytes_of(&par),
                    usage: wgpu::BufferUsages::UNIFORM,
                }),
                par.n,
                par.p,
            ),
            MatrixParams::Multiplication(par) => (
                self.ctx.device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("params"),
                    contents: bytemuck::bytes_of(&par),
                    usage: wgpu::BufferUsages::UNIFORM,
                }),
                par.n,
                par.k,
            ),
        };
        let c_elements = n * p;
        let c_buffer: Buffer = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: (c_elements as u64) * (std::mem::size_of::<f32>() as u64),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let kernel_module = self
            .ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("kernel"),
                source: wgpu::ShaderSource::Wgsl(kernel_source.into()),
            });
        let pipeline = self
            .ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("pipeline"),
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
                label: Some("bind-group"),
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
                label: Some("encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let dispatch_x = p.div_ceil(16);
            let dispatch_y = n.div_ceil(16);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }
        self.ctx.queue.submit(std::iter::once(encoder.finish()));
        Ok(c_buffer)
    }
}

impl TensorOps<'_> {
    pub fn execute_binary_kernel(
        &self,
        params: TensorParams,
        kernel_source: &str,
        a: &GpuTensor,
        b: &GpuTensor,
    ) -> Result<Buffer> {
        let (params_buffer, n) = match params {
            TensorParams::Addition(par) => (
                self.ctx.device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("params"),
                    contents: bytemuck::bytes_of(&par),
                    usage: wgpu::BufferUsages::STORAGE,
                }),
                par.n_elements,
            ),
        };
        let c_elements = n;
        let c_buffer: Buffer = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: (c_elements as u64) * (std::mem::size_of::<f32>() as u64),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let kernel_module = self
            .ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("kernel"),
                source: wgpu::ShaderSource::Wgsl(kernel_source.into()),
            });
        let pipeline = self
            .ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("pipeline"),
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
                label: Some("bind-group"),
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
                label: Some("encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let dispatch_x = n.div_ceil(256);
            pass.dispatch_workgroups(dispatch_x, 1, 1);
        }
        self.ctx.queue.submit(std::iter::once(encoder.finish()));
        Ok(c_buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    fn context() -> GpuContext {
        pollster::block_on(GpuContext::new()).expect("Failed to create GPU context")
    }
    ///////////////////////////////////////////
    // Matrices (i.e. tensors of rank=2)
    ///////////////////////////////////////////
    #[test]
    fn execute_binary_kernel_matmul_allocates_correct_output_size() -> Result<()> {
        let ctx = context();
        let ops = MatrixOps { ctx: &ctx };
        let a = GpuTensor::from_f32(
            &ctx,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            None,
            None,
        )?;
        let b = GpuTensor::from_f32(&ctx, &[0.0; 12], vec![3, 4], None, None)?;
        let params = MatrixMulParams {
            n: 2,
            p: 3,
            k: 4,
            a_offset: 0,
            a_row_stride: 3,
            a_col_stride: 1,
            b_offset: 0,
            b_row_stride: 4,
            b_col_stride: 1,
            c_offset: 0,
            c_row_stride: 4,
            c_col_stride: 1,
        };
        let buffer = ops.execute_binary_kernel(
            MatrixParams::Multiplication(params),
            include_str!("wgsl/matmul.wgsl"),
            &a,
            &b,
        )?;
        assert_eq!(buffer.size(), (2 * 4 * std::mem::size_of::<f32>()) as u64);
        Ok(())
    }
    #[test]
    fn execute_binary_kernel_add_allocates_correct_output_size() -> Result<()> {
        let ctx = context();
        let ops = MatrixOps { ctx: &ctx };
        let a = GpuTensor::from_f32(&ctx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2], None, None)?;
        let b = GpuTensor::from_f32(&ctx, &[5.0, 6.0, 7.0, 8.0], vec![2, 2], None, None)?;
        let params = MatrixAddParams {
            n: 2,
            p: 2,
            a_offset: 0,
            a_row_stride: 2,
            a_col_stride: 1,
            b_offset: 0,
            b_row_stride: 2,
            b_col_stride: 1,
            c_offset: 0,
            c_row_stride: 2,
            c_col_stride: 1,
        };
        let buffer = ops.execute_binary_kernel(
            MatrixParams::Addition(params),
            include_str!("wgsl/matadd.wgsl"),
            &a,
            &b,
        )?;
        assert_eq!(buffer.size(), (4 * std::mem::size_of::<f32>()) as u64);
        Ok(())
    }
    #[test]
    fn execute_binary_kernel_matmul_result_can_be_read() -> Result<()> {
        let ctx = context();
        let ops = MatrixOps { ctx: &ctx };
        let a = GpuTensor::from_f32(&ctx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2], None, None)?;
        let b = GpuTensor::from_f32(&ctx, &[5.0, 6.0, 7.0, 8.0], vec![2, 2], None, None)?;
        let params = MatrixMulParams {
            n: 2,
            p: 2,
            k: 2,
            a_offset: 0,
            a_row_stride: 2,
            a_col_stride: 1,
            b_offset: 0,
            b_row_stride: 2,
            b_col_stride: 1,
            c_offset: 0,
            c_row_stride: 2,
            c_col_stride: 1,
        };
        let buffer = ops.execute_binary_kernel(
            MatrixParams::Multiplication(params),
            include_str!("wgsl/matmul.wgsl"),
            &a,
            &b,
        )?;
        let tensor = GpuTensor::from_buffer(Arc::new(buffer), vec![2, 2], None, None)?;
        let values = tensor.to_vec_f32(&ctx)?;
        assert_eq!(values.len(), 4);
        assert_eq!(values, vec![19.0, 22.0, 43.0, 50.0]);
        Ok(())
    }
    #[test]
    fn params_layout_is_stable() {
        assert_eq!(std::mem::size_of::<MatrixMulParams>(), 48);
    }
    ///////////////////////////////////////////
    // Tensors of arbitraty ranks
    ///////////////////////////////////////////
    #[test]
    fn execute_binary_kernel_tensor_add_allocates_correct_output_size() -> Result<()> {
        let ctx = context();
        let ops = TensorOps { ctx: &ctx };
        let a = GpuTensor::from_f32(&ctx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2], None, None)?;
        let b = GpuTensor::from_f32(&ctx, &[5.0, 6.0, 7.0, 8.0], vec![2, 2], None, None)?;
        let params = TensorAddParams {
            rank: 2,
            n_elements: 4,
            shape: [2, 2, 0, 0, 0, 0, 0, 0],
            a_offset: 0,
            a_strides: [2, 1, 0, 0, 0, 0, 0, 0],
            b_offset: 0,
            b_strides: [2, 1, 0, 0, 0, 0, 0, 0],
            c_offset: 0,
            c_strides: [2, 1, 0, 0, 0, 0, 0, 0],
        };
        let buffer = ops.execute_binary_kernel(
            TensorParams::Addition(params),
            include_str!("wgsl/tenadd.wgsl"),
            &a,
            &b,
        )?;
        assert_eq!(buffer.size(), (4 * std::mem::size_of::<f32>()) as u64);
        Ok(())
    }
    #[test]
    fn execute_binary_kernel_tensor_add_2d_result_can_be_read() -> Result<()> {
        let ctx = context();
        let ops = TensorOps { ctx: &ctx };
        let a = GpuTensor::from_f32(&ctx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2], None, None)?;
        let b = GpuTensor::from_f32(&ctx, &[10.0, 20.0, 30.0, 40.0], vec![2, 2], None, None)?;
        let params = TensorAddParams {
            rank: 2,
            n_elements: 4,
            shape: [2, 2, 0, 0, 0, 0, 0, 0],
            a_offset: 0,
            a_strides: [2, 1, 0, 0, 0, 0, 0, 0],
            b_offset: 0,
            b_strides: [2, 1, 0, 0, 0, 0, 0, 0],
            c_offset: 0,
            c_strides: [2, 1, 0, 0, 0, 0, 0, 0],
        };
        let buffer = ops.execute_binary_kernel(
            TensorParams::Addition(params),
            include_str!("wgsl/tenadd.wgsl"),
            &a,
            &b,
        )?;
        let tensor = GpuTensor::from_buffer(Arc::new(buffer), vec![2, 2], None, None)?;
        let values = tensor.to_vec_f32(&ctx)?;
        assert_eq!(values.len(), 4);
        assert_eq!(values, vec![11.0, 22.0, 33.0, 44.0]);
        Ok(())
    }
}
