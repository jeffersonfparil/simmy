use crate::linalg::context::GpuContext;
use std::sync::Arc;
use anyhow::{Result, ensure};
use bytemuck::cast_slice;
use std::fmt;
use std::sync::mpsc::channel;
use wgpu::{Buffer, util::{BufferInitDescriptor, DeviceExt}};

/// GPU Tensor
///
/// * `shape` contains the size of each tensor dimension.
/// * `strides` contains the storage stride of each dimension.
/// * `offset` is the starting element within the underlying storage.
/// * `buffer` contains the backing tensor data in GPU memory.
///
/// Tensors use a strided storage model, allowing multiple tensor views
/// to reference the same underlying storage without copying data.
/// Operations such as slicing, transposition, reshaping, and tensor
/// contraction can therefore be implemented by modifying tensor
/// metadata rather than reallocating storage.
///
/// The backing buffer is reference counted via `Arc`, allowing multiple
/// tensors and tensor views to safely share ownership of the same GPU
/// allocation. Creating a tensor view therefore creates a new
/// `GpuTensor` instance with its own shape, strides, and offset while
/// reusing the existing GPU storage.
///
/// For example:
/// ```text
/// shape   = [2, 3, 4]
/// strides = [12, 4, 1]
/// offset  = 0
/// ```
/// describes a contiguous row-major tensor.
///
/// Note that the backing buffer may contain more data than is directly
/// accessible through this tensor. This enables tensor views and
/// sub-tensors to share storage with one another while avoiding
#[derive(Debug)]
pub struct GpuTensor {
    pub shape: Vec<u32>,
    pub strides: Vec<u32>,
    pub offset: u32,
    pub buffer: Arc<wgpu::Buffer>,
}

/// Parse and validate tensor layout parameters.
///
/// * `n` is the number of elements available in the backing storage.
/// * `shape` defines the tensor dimensions.
/// * `strides` defines the storage stride of each dimension.
/// * `offset` is the starting element within the backing storage.
///
/// If `strides` is not supplied, contiguous row-major strides are
/// generated automatically.
/// 
/// If `offset` is not supplied, an offset of `0` is used.
///
/// For example:
/// ```text
/// shape   = [2, 3, 4]
/// strides = [12, 4, 1]
/// ```
///
/// The function validates:
/// * The tensor rank matches the stride rank.
/// * The backing storage is large enough for the requested tensor view.
/// * The offset lies within the backing storage.
///
/// Empty tensors (those containing a dimension of size zero) are
/// permitted and require no storage.
///
/// # Returns
/// Returns the normalized `(shape, strides, offset)` tuple.
///
/// # Errors
/// Returns an error if:
/// * `shape.len() != strides.len()`.
/// * The backing storage is too small for the specified layout
fn parse_tensor_params(n: u32, shape: Vec<u32>, strides: Option<Vec<u32>>, offset: Option<u32>) -> Result<(Vec<u32>, Vec<u32>, u32)> {
    let n: u32 = n.max(1);
    let strides: Vec<u32> = strides.unwrap_or_else(|| {
        let mut products: Vec<u32> = Vec::with_capacity(shape.len());
        let mut stride = 1;
        for &d in shape.iter().rev() {
            products.push(stride);
            stride *= d;
        }
        products.reverse();
        products
    });
    ensure!(
        shape.len() == strides.len(),
        "The shape and strides are incompatible!"
    );
    let required_len = if shape.iter().any(|&x| x == 0) {
        0
    } else {
        shape.iter()
            .zip(strides.iter())
            .map(|(&shape, &stride)| (shape - 1) * stride)
            .sum::<u32>() + 1
    };
    ensure!(
        n >= required_len,
        "The shape and strides are incompatible with the data!"
    );
    let offset: u32 = offset.unwrap_or(0);
    ensure!(
        n > offset,
        "The offset must range from 0 to {}", n - 1
    );
    Ok((
        shape,
        strides,
        offset,
    ))
}

impl GpuTensor {
    /// Construct a tensor from host memory.
    ///
    /// * `ctx` provides access to the GPU device.
    /// * `data` is the backing tensor storage to upload.
    /// * `shape` defines the tensor dimensions.
    /// * `strides` defines the storage stride of each dimension.
    /// * `offset` is the starting element within the backing storage.
    ///
    /// If `strides` is not provided, contiguous row-major strides are
    /// generated automatically.
    ///
    /// If `offset` is not provided, an offset of `0` is used.
    ///
    /// # Errors
    /// Returns an error if:
    /// * The shape and strides have different ranks.
    /// * The backing storage is too small for the specified layout.
    /// * The offset falls outside the backing storage.
    pub fn from_f32(ctx: &GpuContext, data: &[f32], shape: Vec<u32>, strides: Option<Vec<u32>>, offset: Option<u32>) -> Result<Self> {
        let (shape, strides, offset) = parse_tensor_params(data.len() as u32, shape, strides, offset)?;
        let buffer: Arc<Buffer> = Arc::new(
            ctx.device.create_buffer_init(&BufferInitDescriptor {
                label: None,
                contents: cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
        }));
        Ok(Self { shape, strides, offset, buffer })
    }

    /// Construct a tensor view from an existing GPU buffer.
    ///
    /// * `buffer` is the backing GPU storage.
    /// * `shape` defines the tensor dimensions.
    /// * `strides` defines the storage stride of each dimension.
    /// * `offset` is the starting element within the backing storage.
    ///
    /// If `strides` is not provided, contiguous row-major strides are
    /// generated automatically.
    ///
    /// If `offset` is not provided, an offset of `0` is used.
    ///
    /// The buffer is assumed to contain `f32` elements.
    ///
    /// # Errors
    /// Returns an error if:
    /// * The shape and strides have different ranks.
    /// * The buffer is too small for the specified layout.
    /// * The offset falls outside the buffer.
    pub fn from_buffer(buffer: Arc<Buffer>, shape: Vec<u32>, strides: Option<Vec<u32>>, offset: Option<u32>) -> Result<Self> {
        let n: u32 = ((buffer.size() as usize) / std::mem::size_of::<f32>()) as u32;
        let (shape, strides, offset) = parse_tensor_params(n, shape, strides, offset)?;
        Ok(Self { shape, strides, offset, buffer })
    }

    /// Copy tensor data from GPU memory into a host vector.
    ///
    ///
    /// * `self` is the tensor whose backing storage will be read.
    /// * `ctx` provides access to the GPU device and queue.
    ///
    /// # Returns
    /// Returns the contents of the backing GPU buffer as a vector of `f32`
    /// values.
    ///
    /// # Errors
    /// Returns an error if:
    /// * The temporary readback buffer cannot be mapped.
    /// * GPU execution fails during readback.
    /// * The mapped data cannot be interpreted as `f32` values.
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

/// Displays the tensor metadata, i.e.:
/// * `Shape` - the dimensions of the tensor.
/// * `Strides` - the storage stride of each dimension.
/// * `Offset` - the starting element within the backing storage.
/// * `Buffer Size` - the size of the underlying GPU buffer in bytes.
impl fmt::Display for GpuTensor {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        writeln!(f, "GPU Tensor")?;
        writeln!(f, "\t- Shape:   {:?}", self.shape)?;
        writeln!(f, "\t- Strides: {:?}", self.strides)?;
        writeln!(f, "\t- Offset:  {}", self.offset)?;
        write!(f,"\t- Buffer Size: {} bytes",self.buffer.size())
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
        let tensor = GpuTensor::from_f32(&ctx, &[1.0f32, 2.0, 3.0, 4.0], vec![4], None, None)?;
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
        let tensor = GpuTensor::from_f32(&ctx, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], None, None)?;
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
        let tensor = GpuTensor::from_f32(&ctx, &[0.0f32; 24], vec![2, 3, 4], None, None)?;
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
        let tensor = GpuTensor::from_f32(&ctx, &[], vec![0], None, None)?;
        assert_eq!(tensor.shape, vec![0]);
        assert_eq!(tensor.buffer.size(), 0);
        Ok(())
    }
    #[test]
    fn creates_tensor_from_buffer() -> Result<()> {
        let ctx = context();
        let original = vec![
            1.0f32, 2.0, 3.0, 4.0,
            5.0, 6.0,
        ];
        let source = GpuTensor::from_f32(
            &ctx,
            &original,
            vec![2, 3],
            None,
            None,
        )?;
        let tensor = GpuTensor::from_buffer(
            source.buffer,
            vec![2, 3],
            None,
            None,
        )?;
        assert_eq!(tensor.shape, vec![2, 3]);
        assert_eq!(tensor.strides, vec![3, 1]);
        assert_eq!(tensor.offset, 0);
        Ok(())
    }
    #[test]
    fn from_buffer_preserves_custom_strides() -> Result<()> {
        let ctx = context();
        let source = GpuTensor::from_f32(
            &ctx,
            &[0.0f32; 24],
            vec![2, 3, 4],
            None,
            None,
        )?;

        let tensor = GpuTensor::from_buffer(
            source.buffer,
            vec![2, 3, 4],
            Some(vec![12, 4, 1]),
            None,
        )?;
        assert_eq!(tensor.shape, vec![2, 3, 4]);
        assert_eq!(tensor.strides, vec![12, 4, 1]);
        Ok(())
    }
    #[test]
    fn from_buffer_preserves_offset() -> Result<()> {
        let ctx = context();
        let source = GpuTensor::from_f32(
            &ctx,
            &[0.0f32; 32],
            vec![32],
            None,
            None,
        )?;
        let tensor = GpuTensor::from_buffer(
            source.buffer,
            vec![4],
            None,
            Some(8),
        )?;
        assert_eq!(tensor.offset, 8);
        Ok(())
    }
    #[test]
    fn from_buffer_generates_default_strides() -> Result<()> {
        let ctx = context();
        let source = GpuTensor::from_f32(
            &ctx,
            &[0.0f32; 24],
            vec![2, 3, 4],
            None,
            None,
        )?;
        let tensor = GpuTensor::from_buffer(
            source.buffer,
            vec![2, 3, 4],
            None,
            None,
        )?;
        assert_eq!(tensor.strides, vec![12, 4, 1]);
        Ok(())
    }
    #[test]
    fn from_buffer_rejects_shape_stride_rank_mismatch() {
        let ctx = context();
        let source = GpuTensor::from_f32(
            &ctx,
            &[0.0f32; 24],
            vec![2, 3, 4],
            None,
            None,
        )
        .unwrap();
        let result = GpuTensor::from_buffer(
            source.buffer,
            vec![2, 3, 4],
            Some(vec![12, 4]),
            None,
        );
        assert!(result.is_err());
    }
    #[test]
    fn from_buffer_rejects_excessive_offset() {
        let ctx = context();
        let source = GpuTensor::from_f32(
            &ctx,
            &[0.0f32; 16],
            vec![16],
            None,
            None,
        )
        .unwrap();
        let result = GpuTensor::from_buffer(
            source.buffer,
            vec![4],
            None,
            Some(100),
        );
        assert!(result.is_err());
    }
    #[test]
    fn round_trip_tensor_data() -> Result<()> {
        let ctx = context();
        let original = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = GpuTensor::from_f32(&ctx, &original, vec![4], None, None)?;
        let extracted = tensor.to_vec_f32(&ctx)?;
        assert_eq!(extracted, original);
        Ok(())
    }
    #[test]
    fn from_buffer_round_trip_data() -> Result<()> {
        let ctx = context();
        let original: Vec<f32> =
            (0..32).map(|x| x as f32).collect();
        let source = GpuTensor::from_f32(
            &ctx,
            &original,
            vec![32],
            None,
            None,
        )?;
        let tensor = GpuTensor::from_buffer(
            source.buffer,
            vec![32],
            None,
            None,
        )?;
        let extracted = tensor.to_vec_f32(&ctx)?;
        assert_eq!(extracted, original);
        Ok(())
    }
}
