use crate::linalg::tensor::GpuTensor;
use anyhow::{Result, ensure};
use std::sync::Arc;

impl GpuTensor {
    /// Transponse a tensor
    ///
    /// * `perm` defines the permutation of tensor axes.
    ///   For example:
    ///
    /// ```text
    /// [0, 1]    -> [1, 0]
    /// [0, 1, 2] -> [2, 0, 1]
    /// ```
    ///
    /// If `perm` is not provided, the tensor axes are reversed:
    ///
    /// ```text
    /// [0, 1]    -> [1, 0]
    /// [0, 1, 2] -> [2, 1, 0]
    /// ```
    ///
    /// This operation does not copy any data. Instead, a new `GpuTensor`
    /// is created with permuted `shape` and `strides` metadata while
    /// sharing ownership of the same backing GPU buffer.
    ///
    /// # Returns
    /// Returns a new tensor view with permuted axes.
    ///
    /// # Errors
    /// Returns an error if:
    /// * The permutation rank differs from the tensor rank.
    /// * A permutation axis is out of bounds.
    /// * A permutation axis appears more than once.
    pub fn transpose(&self, perm: Option<Vec<usize>>) -> Result<Self> {
        let perm: Vec<usize> = perm.unwrap_or_else(|| {
            let mut x = (0..self.shape.len()).collect::<Vec<usize>>();
            x.reverse();
            x
        });
        ensure!(
            perm.len() == self.shape.len(),
            "Shape of the tensor ({:?}) and permutations argument ({:?}) are not compatible!",
            self.shape,
            perm
        );
        let mut seen: Vec<bool> = vec![false; perm.len()];
        for &axis in perm.iter() {
            ensure!(
                axis < self.shape.len(),
                "Axis {} is out of bounds for rank {} tensor!",
                axis,
                self.shape.len()
            );
            ensure!(
                !seen[axis],
                "Axis {} appears multiple times in the permutation!",
                axis
            );
            seen[axis] = true;
        }
        let new_shape: Vec<u32> = perm.iter().map(|&i| self.shape[i]).collect();
        let new_strides: Vec<u32> = perm.iter().map(|&i| self.strides[i]).collect();
        Ok(Self {
            buffer: Arc::clone(&self.buffer),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        })
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
    fn transpose_2d_default() -> Result<()> {
        let ctx = context();
        let tensor = GpuTensor::from_f32(
            &ctx,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            None,
            None,
        )?;
        let t = tensor.transpose(None)?;
        assert_eq!(t.shape, vec![3, 2]);
        assert_eq!(t.strides, vec![1, 3]);
        assert_eq!(t.offset, tensor.offset);
        Ok(())
    }
    #[test]
    fn transpose_3d_default() -> Result<()> {
        let ctx = context();
        let tensor = GpuTensor::from_f32(&ctx, &[0.0f32; 24], vec![2, 3, 4], None, None)?;
        let t = tensor.transpose(None)?;
        assert_eq!(t.shape, vec![4, 3, 2]);
        assert_eq!(t.strides, vec![1, 4, 12]);
        Ok(())
    }
    #[test]
    fn transpose_custom_permutation() -> Result<()> {
        let ctx = context();
        let tensor = GpuTensor::from_f32(&ctx, &[0.0f32; 24], vec![2, 3, 4], None, None)?;
        let t = tensor.transpose(Some(vec![2, 0, 1]))?;
        assert_eq!(t.shape, vec![4, 2, 3]);
        assert_eq!(t.strides, vec![1, 12, 4]);
        Ok(())
    }
    #[test]
    fn transpose_preserves_offset() -> Result<()> {
        let ctx = context();
        let tensor = GpuTensor::from_f32(&ctx, &[0.0f32; 32], vec![4, 4], None, Some(8))?;
        let t = tensor.transpose(None)?;
        assert_eq!(t.offset, 8);
        Ok(())
    }
    #[test]
    fn transpose_shares_buffer() -> Result<()> {
        let ctx = context();
        let tensor = GpuTensor::from_f32(&ctx, &[1.0f32, 2.0, 3.0, 4.0], vec![2, 2], None, None)?;
        let t = tensor.transpose(None)?;
        assert_eq!(tensor.buffer.size(), t.buffer.size());
        Ok(())
    }
    #[test]
    fn transpose_rejects_rank_mismatch() {
        let ctx = context();
        let tensor = GpuTensor::from_f32(&ctx, &[0.0f32; 24], vec![2, 3, 4], None, None).unwrap();
        let result = tensor.transpose(Some(vec![1, 0]));
        assert!(result.is_err());
    }
    #[test]
    fn transpose_rejects_duplicate_axis() {
        let ctx = context();
        let tensor = GpuTensor::from_f32(&ctx, &[0.0f32; 24], vec![2, 3, 4], None, None).unwrap();
        let result = tensor.transpose(Some(vec![0, 0, 2]));
        assert!(result.is_err());
    }
    #[test]
    fn transpose_rejects_out_of_bounds_axis() {
        let ctx = context();
        let tensor = GpuTensor::from_f32(&ctx, &[0.0f32; 24], vec![2, 3, 4], None, None).unwrap();
        let result = tensor.transpose(Some(vec![0, 1, 3]));
        assert!(result.is_err());
    }
    #[test]
    fn transpose_twice_returns_original_layout() -> Result<()> {
        let ctx = context();
        let tensor = GpuTensor::from_f32(&ctx, &[0.0f32; 24], vec![2, 3, 4], None, None)?;
        let t1 = tensor.transpose(Some(vec![2, 1, 0]))?;
        let t2 = t1.transpose(Some(vec![2, 1, 0]))?;
        assert_eq!(t2.shape, tensor.shape);
        assert_eq!(t2.strides, tensor.strides);
        Ok(())
    }
}
