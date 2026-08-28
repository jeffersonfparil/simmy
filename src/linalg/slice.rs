use crate::linalg::tensor::GpuTensor;
use anyhow::{Result, ensure};
use std::sync::Arc;

impl GpuTensor {
    pub fn slice_mut(&mut self, ranges: &[(usize, usize)]) -> Result<()> {
        let rank = self.shape.len();
        ensure!(
            ranges.len() == rank,
            "Number of slice ranges ({}) must match the tensor rank ({})",
            ranges.len(),
            rank
        );
        for (i, &(start, end)) in ranges.iter().enumerate() {
            ensure!(
                start <= end,
                "The slice range should be (start <= end), i.e. {} is not less than {}.",
                start,
                end
            );
            ensure!(
                end <= self.shape[i] as usize,
                "The range is out-of-bounds, i.e. ranges[{}][1] = {} and shape[{}] = {}",
                i,
                end,
                i,
                self.shape[i]
            );
        }
        let mut new_offset = self.offset;
        for (i, &(start, end)) in ranges.iter().enumerate() {
            // Math: The offset shifts by the start coordinate multiplied by the stride of that dimension.
            // new_offset = old_offset + sum_i (start_i * stride_i)
            new_offset += (start as u32) * self.strides[i];
            // The new shape of this dimension is the length of the slice range.
            self.shape[i] = (end - start) as u32;
        }
        // Strides remain completely unchanged because the step sizes along each dimension
        // in memory are preserved [1].
        self.offset = new_offset;
        Ok(())
    }

    pub fn slice_view(&self, ranges: &[(usize, usize)]) -> Result<Self> {
        let mut view = Self {
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            buffer: Arc::clone(&self.buffer),
        };
        view.slice_mut(ranges)?;
        Ok(view)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::context::GpuContext;
    use crate::linalg::tensor::GpuTensor;
    use anyhow::Result;

    fn context() -> GpuContext {
        pollster::block_on(GpuContext::new()).expect("Failed to create GPU context")
    }

    ///////////////////////
    // 1D SLICING
    ///////////////////////
    #[test]
    fn test_slice_mut_1d() -> Result<()> {
        let ctx = context();
        let data = vec![10.0f32, 11.0, 12.0, 13.0, 14.0];
        // Correct shape for 1D tensor with 5 elements
        let mut tensor = GpuTensor::from_f32(&ctx, &data, vec![5], None, None)?;
        tensor.slice_mut(&[(1, 4)])?;
        assert_eq!(tensor.shape, vec![3]); // 4 - 1 = 3
        assert_eq!(tensor.strides, vec![1]); // row-major 1D stride
        assert_eq!(tensor.offset, 1); // start * stride = 1 * 1
        Ok(())
    }
    ///////////////////////
    // 2D SLICING
    ///////////////////////
    #[test]
    fn test_slice_mut_2d() -> Result<()> {
        let ctx = context();
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        // Correct shape: 3 rows × 4 columns
        let mut tensor = GpuTensor::from_f32(&ctx, &data, vec![3, 4], None, None)?;
        // Slice rows [1,3) and cols [1,3)
        tensor.slice_mut(&[(1, 3), (1, 3)])?;
        assert_eq!(tensor.shape, vec![2, 2]); // (3-1, 3-1)
        assert_eq!(tensor.strides, vec![4, 1]); // row-major: row stride = 4, col stride = 1
        assert_eq!(tensor.offset, 5); // 1*4 + 1*1 = 5
        Ok(())
    }
    ///////////////////////
    // 3D SLICING
    ///////////////////////
    #[test]
    fn test_slice_mut_3d() -> Result<()> {
        let ctx = context();
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        // Correct shape: 2 × 3 × 4
        let mut tensor = GpuTensor::from_f32(&ctx, &data, vec![2, 3, 4], None, None)?;
        // Slice: axis0 [1,2), axis1 [0,2), axis2 [1,4)
        tensor.slice_mut(&[(1, 2), (0, 2), (1, 4)])?;
        assert_eq!(tensor.shape, vec![1, 2, 3]); // (2-1, 2-0, 4-1)
        assert_eq!(tensor.strides, vec![12, 4, 1]); // row-major: [3*4, 4, 1]
        // offset = 1*12 + 0*4 + 1*1 = 13
        assert_eq!(tensor.offset, 13);
        Ok(())
    }
    ///////////////////////
    // VIEW SHARING
    ///////////////////////
    #[test]
    fn test_slice_view_shares_buffer() -> Result<()> {
        let ctx = context();
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let parent = GpuTensor::from_f32(&ctx, &data, vec![3, 4], None, None)?;
        let view = parent.slice_view(&[(1, 3), (1, 3)])?;
        // Parent unchanged
        assert_eq!(parent.shape, vec![3, 4]);
        assert_eq!(parent.offset, 0);
        // View sliced correctly
        assert_eq!(view.shape, vec![2, 2]);
        assert_eq!(view.offset, 5);
        // Shared buffer
        assert_eq!(parent.buffer.size(), view.buffer.size());
        Ok(())
    }
    ///////////////////////
    // VALIDATION
    ///////////////////////
    #[test]
    fn test_slice_validation_checks() {
        let ctx = context();
        let data = vec![0.0f32; 12];
        let mut tensor = GpuTensor::from_f32(&ctx, &data, vec![3, 4], None, None).unwrap();
        // Rank mismatch
        assert!(tensor.slice_mut(&[(0, 2)]).is_err());
        // start > end
        assert!(tensor.slice_mut(&[(2, 1), (0, 2)]).is_err());
        // end > shape[i]
        assert!(tensor.slice_mut(&[(0, 4), (0, 5)]).is_err());
        // Valid boundary slice
        assert!(tensor.slice_mut(&[(0, 3), (0, 4)]).is_ok());
    }
}
