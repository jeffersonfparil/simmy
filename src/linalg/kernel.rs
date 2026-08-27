use crate::linalg::context::GpuContext;
use crate::linalg::params::{
    BinaryMatrixParams, BinaryTensorParams, ContractMatrixParams, ContractTensorParams,
    UnaryMatrixParams, UnaryTensorParams,
};
use crate::linalg::tensor::GpuTensor;
use anyhow::{Result, ensure};
use std::sync::Arc;
use wgpu::Buffer;
use wgpu::util::{BufferInitDescriptor, DeviceExt};

/// Kernels or Tensor Operations
///
/// A `GpuKernel`  borrows a `GpuContext`, allowing multiple tensor
/// operations to share the same device, queue, and adapter.
///
/// * `ctx` is the GPU context used to compile kernels, allocate
///   temporary resources, and submit work to the compute device.
///
/// Note that we will use this struct for future tensor or arbitrarily
/// ranked tensor operations --> beyond matrices.
pub struct GpuKernel<'a> {
    pub ctx: &'a GpuContext,
}

pub enum Params {
    UnaryMatrix(UnaryMatrixParams),
    BinaryMatrix(BinaryMatrixParams),
    ContractMatrix(ContractMatrixParams),
    UnaryTensor(UnaryTensorParams),
    BinaryTensor(BinaryTensorParams),
    ContractTensor(ContractTensorParams),
}

impl GpuKernel<'_> {
    pub fn execute_kernel(
        &self,
        params: Params,
        a: &GpuTensor,
        b: Option<&GpuTensor>,
    ) -> Result<GpuTensor> {
        if matches!(&params, Params::UnaryMatrix(..) | Params::UnaryTensor(..)) {
            ensure!(b.is_none(), "The b matrix should be None!");
        } else {
            ensure!(!b.is_none(), "The b matrix should be supplied!");
        }
        let (params_buffer, c_shape) = match &params {
            Params::UnaryMatrix(par) => (
                self.ctx.device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("params"),
                    contents: bytemuck::bytes_of(par),
                    usage: wgpu::BufferUsages::STORAGE,
                }),
                a.shape.clone(),
            ),
            Params::UnaryTensor(par) => (
                self.ctx.device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("params"),
                    contents: bytemuck::bytes_of(par),
                    usage: wgpu::BufferUsages::STORAGE,
                }),
                a.shape.clone(),
            ),
            Params::BinaryMatrix(par) => (
                self.ctx.device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("params"),
                    contents: bytemuck::bytes_of(par),
                    usage: wgpu::BufferUsages::STORAGE,
                }),
                a.shape.clone(),
            ),
            Params::BinaryTensor(par) => (
                self.ctx.device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("params"),
                    contents: bytemuck::bytes_of(par),
                    usage: wgpu::BufferUsages::STORAGE,
                }),
                a.shape.clone(),
            ),
            Params::ContractMatrix(par) => (
                self.ctx.device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("params"),
                    contents: bytemuck::bytes_of(par),
                    usage: wgpu::BufferUsages::STORAGE,
                }),
                vec![par.n, par.k],
            ),
            Params::ContractTensor(par) => (
                self.ctx.device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("params"),
                    contents: bytemuck::bytes_of(par),
                    usage: wgpu::BufferUsages::STORAGE,
                }),
                par.c_shape[..par.c_rank as usize].to_vec(),
            ),
        };
        let c_elements = c_shape.iter().product::<u32>();
        let c_buffer: Buffer = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: (c_elements as u64) * (std::mem::size_of::<f32>() as u64),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pipeline = match &params {
            Params::UnaryMatrix(_) => &self.ctx.unary_matrix_pipeline,
            Params::UnaryTensor(_) => &self.ctx.unary_tensor_pipeline,
            Params::BinaryMatrix(_) => &self.ctx.binary_matrix_pipeline,
            Params::BinaryTensor(_) => &self.ctx.binary_tensor_pipeline,
            Params::ContractMatrix(_) => &self.ctx.contract_matrix_pipeline,
            Params::ContractTensor(_) => &self.ctx.contract_tensor_pipeline,
        };
        let bind_group = match &params {
            Params::UnaryMatrix(_) | Params::UnaryTensor(_) => {
                self.ctx
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
                                resource: c_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: params_buffer.as_entire_binding(),
                            },
                        ],
                    })
            }
            _ => self
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
                            resource: b
                                .expect("Validated to exist for non-unary kernels!")
                                .buffer
                                .as_entire_binding(),
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
                }),
        };
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
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            if matches!(
                &params,
                Params::UnaryMatrix(_) | Params::BinaryMatrix(_) | Params::ContractMatrix(_)
            ) {
                let dispatch_x = c_shape[1].div_ceil(16);
                let dispatch_y = c_shape[0].div_ceil(16);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            } else {
                let dispatch_x = c_elements.div_ceil(256);
                pass.dispatch_workgroups(dispatch_x, 1, 1);
            }
        }
        self.ctx.queue.submit(std::iter::once(encoder.finish()));
        GpuTensor::from_buffer(Arc::new(c_buffer), c_shape, None, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::context::GpuContext;
    use crate::linalg::params::{BinaryMatrixParams, ContractMatrixParams, UnaryMatrixParams};
    use crate::linalg::tensor::GpuTensor;
    use anyhow::Result;
    use std::f64::consts::PI;

    // Unary operations
    const OP_ABS: u32 = 0;
    const OP_NEG: u32 = 1;
    const OP_SQRT: u32 = 2;
    const OP_EXP: u32 = 3;
    const OP_LOG: u32 = 4;
    const OP_SIN: u32 = 5;
    const OP_COS: u32 = 6;

    // Binary operations
    const OP_ADD: u32 = 0;
    const OP_SUB: u32 = 1;
    const OP_MUL: u32 = 2;
    const OP_DIV: u32 = 3;
    const OP_MIN: u32 = 4;
    const OP_MAX: u32 = 5;
    const OP_POW: u32 = 6;
    const OP_ATAN2: u32 = 7;
    const OP_EQ: u32 = 8;
    const OP_NE: u32 = 9;
    const OP_LT: u32 = 10;
    const OP_LE: u32 = 11;
    const OP_GT: u32 = 12;
    const OP_GE: u32 = 13;

    const OP_PAIR_ADD: u32 = 0;
    const OP_PAIR_SUB: u32 = 1;
    const OP_PAIR_MUL: u32 = 2;
    const OP_PAIR_DIV: u32 = 3;
    const OP_PAIR_MIN: u32 = 4;
    const OP_PAIR_MAX: u32 = 5;

    const OP_REDUCE_ADD: u32 = 0;
    const OP_REDUCE_MUL: u32 = 1;
    const OP_REDUCE_MIN: u32 = 2;
    const OP_REDUCE_MAX: u32 = 3;

    fn context() -> GpuContext {
        pollster::block_on(GpuContext::new()).expect("Failed to create GPU context")
    }

    fn ops<'a>(ctx: &'a GpuContext) -> GpuKernel<'a> {
        GpuKernel { ctx }
    }

    // Matrices

    fn unary_matrix_params(op: u32) -> Params {
        Params::UnaryMatrix(UnaryMatrixParams {
            n: 2,
            p: 2,
            a_offset: 0,
            a_row_stride: 2,
            a_col_stride: 1,
            c_offset: 0,
            c_row_stride: 2,
            c_col_stride: 1,
            op,
        })
    }

    fn binary_matrix_params(op: u32) -> Params {
        Params::BinaryMatrix(BinaryMatrixParams {
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
            op,
        })
    }

    fn contract_matrix_params(pairwise: u32, reduction: u32) -> Params {
        Params::ContractMatrix(ContractMatrixParams {
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
            op_pairwise: pairwise,
            op_reduction: reduction,
        })
    }

    #[test]
    fn unary_rejects_b_matrix() -> Result<()> {
        let ctx = context();
        let ops = ops(&ctx);
        let a = GpuTensor::from_f32(&ctx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2], None, None)?;
        let b = GpuTensor::from_f32(&ctx, &[5.0, 6.0, 7.0, 8.0], vec![2, 2], None, None)?;
        assert!(
            ops.execute_kernel(unary_matrix_params(OP_ABS), &a, Some(&b),)
                .is_err()
        );
        Ok(())
    }
    #[test]
    fn binary_requires_b_matrix() -> Result<()> {
        let ctx = context();
        let ops = ops(&ctx);
        let a = GpuTensor::from_f32(&ctx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2], None, None)?;
        assert!(
            ops.execute_kernel(binary_matrix_params(OP_ADD), &a, None,)
                .is_err()
        );
        Ok(())
    }
    #[test]
    fn contract_requires_b_matrix() -> Result<()> {
        let ctx = context();
        let ops = ops(&ctx);
        let a = GpuTensor::from_f32(&ctx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2], None, None)?;
        assert!(
            ops.execute_kernel(
                contract_matrix_params(OP_PAIR_MUL, OP_REDUCE_ADD,),
                &a,
                None,
            )
            .is_err()
        );
        Ok(())
    }
    #[test]
    fn unary_preserves_shape() -> Result<()> {
        let ctx = context();
        let ops = ops(&ctx);
        let a = GpuTensor::from_f32(&ctx, &[1.0, -2.0, 3.0, -4.0], vec![2, 2], None, None)?;
        let c = ops.execute_kernel(unary_matrix_params(OP_ABS), &a, None)?;
        assert_eq!(c.shape, vec![2, 2]);
        Ok(())
    }
    #[test]
    fn binary_preserves_shape() -> Result<()> {
        let ctx = context();
        let ops = ops(&ctx);
        let a = GpuTensor::from_f32(&ctx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2], None, None)?;
        let b = GpuTensor::from_f32(&ctx, &[5.0, 6.0, 7.0, 8.0], vec![2, 2], None, None)?;
        let c = ops.execute_kernel(binary_matrix_params(OP_ADD), &a, Some(&b))?;
        assert_eq!(c.shape, vec![2, 2]);
        Ok(())
    }
    #[test]
    fn contract_returns_n_by_k_shape() -> Result<()> {
        let ctx = context();
        let ops = ops(&ctx);
        let a = GpuTensor::from_f32(&ctx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2], None, None)?;
        let b = GpuTensor::from_f32(&ctx, &[5.0, 6.0, 7.0, 8.0], vec![2, 2], None, None)?;
        let c = ops.execute_kernel(
            contract_matrix_params(OP_PAIR_MUL, OP_REDUCE_ADD),
            &a,
            Some(&b),
        )?;
        assert_eq!(c.shape, vec![2, 2]);
        Ok(())
    }
    #[test]
    fn unary_abs() -> Result<()> {
        let ctx = context();
        let ops = ops(&ctx);
        let a = GpuTensor::from_f32(&ctx, &[-1.0, 2.0, -3.0, 4.0], vec![2, 2], None, None)?;
        let c = ops.execute_kernel(unary_matrix_params(OP_ABS), &a, None)?;
        assert_eq!(c.to_vec_f32(&ctx)?, vec![1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }
    #[test]
    fn unary_neg() -> Result<()> {
        let ctx = context();
        let ops = ops(&ctx);
        let a = GpuTensor::from_f32(&ctx, &[1.0, -2.0, 3.0, -4.0], vec![2, 2], None, None)?;
        let c = ops.execute_kernel(unary_matrix_params(OP_NEG), &a, None)?;
        assert_eq!(c.to_vec_f32(&ctx)?, vec![-1.0, 2.0, -3.0, 4.0]);
        Ok(())
    }
    #[test]
    fn unary_sqrt() -> Result<()> {
        let ctx = context();
        let ops = ops(&ctx);
        let a = GpuTensor::from_f32(&ctx, &[1.0, 4.0, 9.0, 16.0], vec![2, 2], None, None)?;
        let c = ops.execute_kernel(unary_matrix_params(OP_SQRT), &a, None)?;
        assert_eq!(c.to_vec_f32(&ctx)?, vec![1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }
    #[test]
    fn binary_add() -> Result<()> {
        let ctx = context();
        let ops = ops(&ctx);
        let a = GpuTensor::from_f32(&ctx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2], None, None)?;
        let b = GpuTensor::from_f32(&ctx, &[5.0, 6.0, 7.0, 8.0], vec![2, 2], None, None)?;
        let c = ops.execute_kernel(binary_matrix_params(OP_ADD), &a, Some(&b))?;
        assert_eq!(c.to_vec_f32(&ctx)?, vec![6.0, 8.0, 10.0, 12.0]);
        Ok(())
    }
    #[test]
    fn binary_sub() -> Result<()> {
        let ctx = context();
        let ops = ops(&ctx);
        let a = GpuTensor::from_f32(&ctx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2], None, None)?;
        let b = GpuTensor::from_f32(&ctx, &[5.0, 6.0, 7.0, 8.0], vec![2, 2], None, None)?;
        let c = ops.execute_kernel(binary_matrix_params(OP_SUB), &a, Some(&b))?;
        assert_eq!(c.to_vec_f32(&ctx)?, vec![-4.0, -4.0, -4.0, -4.0]);
        Ok(())
    }
    #[test]
    fn binary_mul() -> Result<()> {
        let ctx = context();
        let ops = ops(&ctx);
        let a = GpuTensor::from_f32(&ctx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2], None, None)?;
        let b = GpuTensor::from_f32(&ctx, &[5.0, 6.0, 7.0, 8.0], vec![2, 2], None, None)?;
        let c = ops.execute_kernel(binary_matrix_params(OP_MUL), &a, Some(&b))?;
        assert_eq!(c.to_vec_f32(&ctx)?, vec![5.0, 12.0, 21.0, 32.0]);
        Ok(())
    }
    #[test]
    fn binary_div() -> Result<()> {
        let ctx = context();
        let ops = ops(&ctx);
        let a = GpuTensor::from_f32(&ctx, &[8.0, 16.0, 18.0, 20.0], vec![2, 2], None, None)?;
        let b = GpuTensor::from_f32(&ctx, &[2.0, 4.0, 3.0, 5.0], vec![2, 2], None, None)?;
        let c = ops.execute_kernel(binary_matrix_params(OP_DIV), &a, Some(&b))?;
        assert_eq!(c.to_vec_f32(&ctx)?, vec![4.0, 4.0, 6.0, 4.0]);
        Ok(())
    }
    #[test]
    fn binary_min() -> Result<()> {
        let ctx = context();
        let ops = ops(&ctx);
        let a = GpuTensor::from_f32(&ctx, &[1.0, 6.0, 3.0, 8.0], vec![2, 2], None, None)?;
        let b = GpuTensor::from_f32(&ctx, &[5.0, 2.0, 7.0, 4.0], vec![2, 2], None, None)?;
        let c = ops.execute_kernel(binary_matrix_params(OP_MIN), &a, Some(&b))?;
        assert_eq!(c.to_vec_f32(&ctx)?, vec![1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }
    #[test]
    fn binary_max() -> Result<()> {
        let ctx = context();
        let ops = ops(&ctx);
        let a = GpuTensor::from_f32(&ctx, &[1.0, 6.0, 3.0, 8.0], vec![2, 2], None, None)?;
        let b = GpuTensor::from_f32(&ctx, &[5.0, 2.0, 7.0, 4.0], vec![2, 2], None, None)?;
        let c = ops.execute_kernel(binary_matrix_params(OP_MAX), &a, Some(&b))?;
        assert_eq!(c.to_vec_f32(&ctx)?, vec![5.0, 6.0, 7.0, 8.0]);
        Ok(())
    }
    #[test]
    fn binary_eq() -> Result<()> {
        let ctx = context();
        let ops = ops(&ctx);
        let a = GpuTensor::from_f32(&ctx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2], None, None)?;
        let b = GpuTensor::from_f32(&ctx, &[1.0, 0.0, 3.0, 9.0], vec![2, 2], None, None)?;
        let c = ops.execute_kernel(binary_matrix_params(OP_EQ), &a, Some(&b))?;
        assert_eq!(c.to_vec_f32(&ctx)?, vec![1.0, 0.0, 1.0, 0.0]);
        Ok(())
    }
    #[test]
    fn binary_atan2() -> Result<()> {
        let ctx = context();
        let ops = ops(&ctx);
        let pi = PI as f32;
        let a = GpuTensor::from_f32(&ctx, &[1.0, 0.0, 0.0, 1.0], vec![2, 2], None, None)?;
        let b = GpuTensor::from_f32(&ctx, &[0.0, 1.0, 1.0, 0.0], vec![2, 2], None, None)?;
        let c = ops.execute_kernel(binary_matrix_params(OP_ATAN2), &a, Some(&b))?;
        assert_eq!(c.to_vec_f32(&ctx)?, vec![pi / 2.0, 0.0, 0.0, pi / 2.0]);
        Ok(())
    }
    #[test]
    fn contract_standard_matrix_multiply() -> Result<()> {
        let ctx = context();
        let ops = ops(&ctx);
        let a = GpuTensor::from_f32(&ctx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2], None, None)?;
        let b = GpuTensor::from_f32(&ctx, &[5.0, 6.0, 7.0, 8.0], vec![2, 2], None, None)?;
        let c = ops.execute_kernel(
            contract_matrix_params(OP_PAIR_MUL, OP_REDUCE_ADD),
            &a,
            Some(&b),
        )?;
        assert_eq!(c.to_vec_f32(&ctx)?, vec![19.0, 22.0, 43.0, 50.0]);
        Ok(())
    }
    #[test]
    fn contract_min_plus() -> Result<()> {
        let ctx = context();
        let ops = ops(&ctx);
        let a = GpuTensor::from_f32(&ctx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2], None, None)?;
        let b = GpuTensor::from_f32(&ctx, &[10.0, 20.0, 30.0, 40.0], vec![2, 2], None, None)?;
        let c = ops.execute_kernel(
            contract_matrix_params(OP_PAIR_ADD, OP_REDUCE_MIN),
            &a,
            Some(&b),
        )?;
        assert_eq!(c.to_vec_f32(&ctx)?, vec![11.0, 21.0, 13.0, 23.0]);
        Ok(())
    }
    #[test]
    fn contract_max_plus() -> Result<()> {
        let ctx = context();
        let ops = ops(&ctx);
        let a = GpuTensor::from_f32(&ctx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2], None, None)?;
        let b = GpuTensor::from_f32(&ctx, &[10.0, 20.0, 30.0, 40.0], vec![2, 2], None, None)?;
        let c = ops.execute_kernel(
            contract_matrix_params(OP_PAIR_ADD, OP_REDUCE_MAX),
            &a,
            Some(&b),
        )?;
        assert_eq!(c.to_vec_f32(&ctx)?, vec![32.0, 42.0, 34.0, 44.0]);
        Ok(())
    }

    // Tensors

    const TENSOR_SHAPE_3D: &[u32] = &[2, 2, 2];
    const TENSOR_SHAPE_4D: &[u32] = &[2, 2, 2, 2];

    fn unary_tensor_params(op: u32) -> Params {
        Params::UnaryTensor(UnaryTensorParams {
            rank: 3,
            n_elements: 8,
            shape: [2, 2, 2, 0, 0, 0, 0, 0],
            a_offset: 0,
            a_strides: [4, 2, 1, 0, 0, 0, 0, 0],
            c_offset: 0,
            c_strides: [4, 2, 1, 0, 0, 0, 0, 0],
            op,
        })
    }

    fn binary_tensor_params(op: u32) -> Params {
        Params::BinaryTensor(BinaryTensorParams {
            rank: 3,
            n_elements: 8,
            shape: [2, 2, 2, 0, 0, 0, 0, 0],
            a_offset: 0,
            a_strides: [4, 2, 1, 0, 0, 0, 0, 0],
            b_offset: 0,
            b_strides: [4, 2, 1, 0, 0, 0, 0, 0],
            c_offset: 0,
            c_strides: [4, 2, 1, 0, 0, 0, 0, 0],
            op,
        })
    }

    fn contract_tensor_params(pairwise: u32, reduction: u32) -> Params {
        Params::ContractTensor(ContractTensorParams {
            a_rank: 2,
            b_rank: 2,
            c_rank: 2,
            contraction_rank: 1,
            c_elements: 4,
            a_shape: [2, 2, 0, 0, 0, 0, 0, 0],
            b_shape: [2, 2, 0, 0, 0, 0, 0, 0],
            c_shape: [2, 2, 0, 0, 0, 0, 0, 0],
            a_offset: 0,
            a_strides: [2, 1, 0, 0, 0, 0, 0, 0],
            b_offset: 0,
            b_strides: [2, 1, 0, 0, 0, 0, 0, 0],
            c_offset: 0,
            c_strides: [2, 1, 0, 0, 0, 0, 0, 0],
            a_contract_axes: [1, 0, 0, 0, 0, 0, 0, 0],
            b_contract_axes: [0, 0, 0, 0, 0, 0, 0, 0],
            op_pairwise: pairwise,
            op_reduction: reduction,
        })
    }

    #[test]
    fn unary_tensor_rejects_b_tensor() -> Result<()> {
        let ctx = context();
        let ops = ops(&ctx);
        let a = GpuTensor::from_f32(&ctx, &[1.0; 8], vec![2, 2, 2], None, None)?;
        let b = GpuTensor::from_f32(&ctx, &[2.0; 8], vec![2, 2, 2], None, None)?;
        assert!(
            ops.execute_kernel(unary_tensor_params(OP_ABS), &a, Some(&b),)
                .is_err()
        );
        Ok(())
    }
    #[test]
    fn unary_tensor_preserves_shape() -> Result<()> {
        let ctx = context();
        let ops = ops(&ctx);
        let a = GpuTensor::from_f32(&ctx, &[1.0; 8], vec![2, 2, 2], None, None)?;
        let c = ops.execute_kernel(unary_tensor_params(OP_ABS), &a, None)?;
        assert_eq!(c.shape, vec![2, 2, 2]);
        Ok(())
    }
    #[test]
    fn binary_tensor_preserves_shape() -> Result<()> {
        let ctx = context();
        let ops = ops(&ctx);
        let a = GpuTensor::from_f32(&ctx, &[1.0; 8], vec![2, 2, 2], None, None)?;
        let b = GpuTensor::from_f32(&ctx, &[2.0; 8], vec![2, 2, 2], None, None)?;
        let c = ops.execute_kernel(binary_tensor_params(OP_ADD), &a, Some(&b))?;
        assert_eq!(c.shape, vec![2, 2, 2]);
        Ok(())
    }
    #[test]
    fn unary_tensor_abs() -> Result<()> {
        let ctx = context();
        let ops = ops(&ctx);
        let a = GpuTensor::from_f32(
            &ctx,
            &[-1.0, -2.0, 3.0, 4.0, -5.0, 6.0, -7.0, 8.0],
            vec![2, 2, 2],
            None,
            None,
        )?;
        let c = ops.execute_kernel(unary_tensor_params(OP_ABS), &a, None)?;
        assert_eq!(
            c.to_vec_f32(&ctx)?,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
        Ok(())
    }
    #[test]
    fn unary_tensor_neg() -> Result<()> {
        let ctx = context();
        let ops = ops(&ctx);
        let a = GpuTensor::from_f32(
            &ctx,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 2, 2],
            None,
            None,
        )?;
        let c = ops.execute_kernel(unary_tensor_params(OP_NEG), &a, None)?;
        assert_eq!(
            c.to_vec_f32(&ctx)?,
            vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0]
        );
        Ok(())
    }
    #[test]
    fn binary_tensor_add() -> Result<()> {
        let ctx = context();
        let ops = ops(&ctx);
        let a = GpuTensor::from_f32(
            &ctx,
            &[1., 2., 3., 4., 5., 6., 7., 8.],
            vec![2, 2, 2],
            None,
            None,
        )?;
        let b = GpuTensor::from_f32(
            &ctx,
            &[1., 1., 1., 1., 1., 1., 1., 1.],
            vec![2, 2, 2],
            None,
            None,
        )?;
        let c = ops.execute_kernel(binary_tensor_params(OP_ADD), &a, Some(&b))?;
        assert_eq!(c.to_vec_f32(&ctx)?, vec![2., 3., 4., 5., 6., 7., 8., 9.]);
        Ok(())
    }
    #[test]
    fn binary_tensor_mul() -> Result<()> {
        let ctx = context();
        let ops = ops(&ctx);
        let a = GpuTensor::from_f32(
            &ctx,
            &[1., 2., 3., 4., 5., 6., 7., 8.],
            vec![2, 2, 2],
            None,
            None,
        )?;
        let b = GpuTensor::from_f32(
            &ctx,
            &[2., 2., 2., 2., 2., 2., 2., 2.],
            vec![2, 2, 2],
            None,
            None,
        )?;
        let c = ops.execute_kernel(binary_tensor_params(OP_MUL), &a, Some(&b))?;
        assert_eq!(
            c.to_vec_f32(&ctx)?,
            vec![2., 4., 6., 8., 10., 12., 14., 16.]
        );
        Ok(())
    }
    #[test]
    fn contract_tensor_preserves_output_shape() -> Result<()> {
        let ctx = context();
        let ops = ops(&ctx);
        let a = GpuTensor::from_f32(&ctx, &[1., 2., 3., 4.], vec![2, 2], None, None)?;
        let b = GpuTensor::from_f32(&ctx, &[5., 6., 7., 8.], vec![2, 2], None, None)?;
        let c = ops.execute_kernel(
            contract_tensor_params(OP_PAIR_MUL, OP_REDUCE_ADD),
            &a,
            Some(&b),
        )?;
        assert_eq!(c.shape, vec![2, 2]);
        Ok(())
    }
    #[test]
    fn contract_tensor_matches_matrix_multiply() -> Result<()> {
        let ctx = context();
        let ops = ops(&ctx);
        let a = GpuTensor::from_f32(&ctx, &[1., 2., 3., 4.], vec![2, 2], None, None)?;
        let b = GpuTensor::from_f32(&ctx, &[5., 6., 7., 8.], vec![2, 2], None, None)?;
        let c = ops.execute_kernel(
            contract_tensor_params(OP_PAIR_MUL, OP_REDUCE_ADD),
            &a,
            Some(&b),
        )?;
        assert_eq!(c.to_vec_f32(&ctx)?, vec![19.0, 22.0, 43.0, 50.0]);
        Ok(())
    }
}
