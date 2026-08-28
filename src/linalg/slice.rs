use crate::linalg::tensor::GpuTensor;
use anyhow::{ensure, Result};
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
            ensure!(start <= end, "The slice range should be (start <= end), i.e. {} is not less than {}.", start, end);
            ensure!(end <= self.shape[i] as usize, "The range is out-of-bounds, i.e. ranges[{}][1] = {} and shape[{}] = {}", i, end, i, self.shape[i]);
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
