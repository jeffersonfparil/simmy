use crate::linalg::context::GpuContext;
use anyhow::{Result, ensure};
use bytemuck::cast_slice;
use std::fmt;
use std::sync::mpsc::channel;
use wgpu::util::{BufferInitDescriptor, DeviceExt};

#[derive(Debug)]
pub struct GpuTensor {
    pub shape: Vec<u32>,
    pub buffer: wgpu::Buffer,
}

impl GpuTensor {
    pub fn from_f32(ctx: &GpuContext, shape: Vec<u32>, data: &[f32]) -> Result<Self> {
        ensure!(
            data.len() == shape.iter().product::<u32>() as usize,
            "The shape and data are incompatible!"
        );
        let buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        Ok(Self { shape, buffer })
    }

    pub fn to_vec_f32(&self, ctx: &GpuContext) -> Result<Vec<f32>> {
        let size = self.buffer.size();
        let temp_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tensor-readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, // critical to be MAP_READ and not COPY_DST as in `from_f32`
            mapped_at_creation: false,
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tensor-readback"),
            });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &temp_buffer, 0, size);
        ctx.queue.submit([encoder.finish()]);
        let (tx, rx) = channel();
        temp_buffer.map_async(wgpu::MapMode::Read, .., move |result| {
            tx.send(result).unwrap();
        });
        ctx.device.poll(wgpu::PollType::wait_indefinitely())?;
        rx.recv()??;
        let mapped = temp_buffer.get_mapped_range(..)?;
        let values = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
        drop(mapped);
        temp_buffer.unmap();
        Ok(values)
    }
}

// TODO: implement printing
impl fmt::Display for GpuTensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "shape: {:?}\nbuffer: {:?}", self.shape, self.buffer)
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
    fn creates_1d_tensor() -> Result<()> {
        let ctx = context();
        let tensor = GpuTensor::from_f32(&ctx, vec![4], &vec![1.0f32, 2.0, 3.0, 4.0])?;
        println!("tensor: {}", tensor);
        assert_eq!(tensor.shape, vec![4]);
        assert_eq!(
            tensor.buffer.size(),
            (4 * std::mem::size_of::<f32>()) as u64
        );
        Ok(())
    }
    #[test]
    fn creates_2d_tensor() -> Result<()> {
        let ctx = context();
        let tensor = GpuTensor::from_f32(&ctx, vec![2, 3], &vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])?;
        assert_eq!(tensor.shape, vec![2, 3]);
        assert_eq!(
            tensor.buffer.size(),
            (6 * std::mem::size_of::<f32>()) as u64
        );
        Ok(())
    }
    #[test]
    fn creates_3d_tensor() -> Result<()> {
        let ctx = context();
        let tensor = GpuTensor::from_f32(&ctx, vec![2, 3, 4], &vec![0.0f32; 24])?;
        assert_eq!(tensor.shape, vec![2, 3, 4]);
        assert_eq!(
            tensor.buffer.size(),
            (24 * std::mem::size_of::<f32>()) as u64
        );
        Ok(())
    }
    #[test]
    fn creates_empty_tensor() -> Result<()> {
        let ctx = context();
        let tensor = GpuTensor::from_f32(&ctx, vec![0], &[])?;
        assert_eq!(tensor.shape, vec![0]);
        assert_eq!(tensor.buffer.size(), 0);
        Ok(())
    }
    #[test]
    fn round_trip_tensor_data() -> Result<()> {
        let ctx = context();
        let original = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = GpuTensor::from_f32(&ctx, vec![4], &original)?;
        let extracted = tensor.to_vec_f32(&ctx)?;
        assert_eq!(extracted, original);
        Ok(())
    }
}
