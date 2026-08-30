use crate::linalg::tensor::GpuTensor;
use anyhow::{Result, ensure};
use std::sync::Arc;

impl GpuTensor {
    /// Transpose the tensor in place according to a permutation of axes.
    ///
    /// ### Tensor Engine Context
    /// This operation reorders the logical dimensions of the tensor by
    /// permuting both its `shape` and `strides`. Because the tensor uses a
    /// strided storage model, transposition is performed entirely through
    /// metadata updates without reallocating or copying GPU memory.
    ///
    /// If `perm` is `None`, the tensor is reversed along all axes:
    /// ```text
    /// shape = [d0, d1, d2] → perm = [2, 1, 0]
    /// ```
    ///
    /// ### Validation
    /// * The permutation must contain exactly one entry for each axis.
    /// * All axes must be unique and within bounds.
    ///
    /// ### Effects
    /// * `shape[i]` becomes `shape[perm[i]]`
    /// * `strides[i]` becomes `strides[perm[i]]`
    /// * `offset` and `buffer` remain unchanged
    ///
    /// ### Notes
    /// This method mutates the tensor in place. Use `transpose_view` to
    /// create a zero‑copy transposed view instead.
    pub fn transpose_mut(&mut self, perm: Option<&[usize]>) -> Result<()> {
        let perm: Vec<usize> = match perm {
            Some(x) => x.to_vec(),
            None => {
                let mut x = (0..self.shape.len()).collect::<Vec<usize>>();
                x.reverse();
                x
            }
        };
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
        self.shape = perm.iter().map(|&i| self.shape[i]).collect();
        self.strides = perm.iter().map(|&i| self.strides[i]).collect();
        Ok(())
    }

    /// Create a zero‑copy transposed tensor view.
    ///
    /// ### Tensor Engine Context
    /// This method constructs a new `GpuTensor` that shares the same GPU
    /// buffer as the parent tensor but whose `shape` and `strides` have
    /// been permuted according to the provided axis order.
    ///
    /// The underlying GPU memory is never duplicated; only metadata is
    /// updated. This enables efficient creation of transposed views for
    /// slicing, contraction, and other tensor operations.
    ///
    /// ### Behavior
    /// * The returned tensor has identical `offset` and `buffer`.
    /// * `shape` and `strides` are permuted according to `perm`.
    /// * If `perm` is `None`, axes are reversed.
    ///
    /// ### Notes
    /// This method does not modify the original tensor. It is the
    /// functional (non‑mutating) counterpart to `transpose_mut`.
    pub fn transpose_view(&self, perm: Option<&[usize]>) -> Result<Self> {
        let mut view = Self {
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            buffer: Arc::clone(&self.buffer),
        };
        view.transpose_mut(perm)?;
        Ok(view)
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
    fn transpose_mut_reverses_axes_by_default() -> Result<()> {
        let ctx = context();
        // 2×3×4 tensor
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let mut t = GpuTensor::from_f32(&ctx, &data, &[2, 3, 4], None, None)?;
        t.transpose_mut(None)?;
        assert_eq!(t.shape, &[4, 3, 2]);
        assert_eq!(t.strides, &[1, 4, 12]); // reversed original strides
        Ok(())
    }
    #[test]
    fn transpose_mut_applies_custom_permutation() -> Result<()> {
        let ctx = context();
        // shape = [2, 3, 4]
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let mut t = GpuTensor::from_f32(&ctx, &data, &[2, 3, 4], None, None)?;
        // permute axes: [1, 2, 0]
        t.transpose_mut(Some(&[1, 2, 0]))?;
        assert_eq!(t.shape, &[3, 4, 2]);
        assert_eq!(t.strides, &[4, 1, 12]);
        Ok(())
    }
    #[test]
    fn transpose_mut_rejects_invalid_rank() -> Result<()> {
        let ctx = context();
        let data: Vec<f32> = (0..6).map(|x| x as f32).collect();
        let mut t = GpuTensor::from_f32(&ctx, &data, &[2, 3], None, None)?;
        // wrong length permutation
        assert!(t.transpose_mut(Some(&[0, 1, 2])).is_err());
        Ok(())
    }
    #[test]
    fn transpose_mut_rejects_duplicate_axes() -> Result<()> {
        let ctx = context();
        let data: Vec<f32> = (0..6).map(|x| x as f32).collect();
        let mut t = GpuTensor::from_f32(&ctx, &data, &[2, 3], None, None)?;
        assert!(t.transpose_mut(Some(&[0, 0])).is_err());
        Ok(())
    }
    #[test]
    fn transpose_view_produces_correct_metadata() -> Result<()> {
        let ctx = context();
        let data: Vec<f32> = (0..6).map(|x| x as f32).collect();
        let t = GpuTensor::from_f32(&ctx, &data, &[2, 3], None, None)?;
        let v = t.transpose_view(Some(&[1, 0]))?;
        // parent unchanged
        assert_eq!(t.shape, &[2, 3]);
        assert_eq!(t.strides, &[3, 1]);
        // view transposed
        assert_eq!(v.shape, &[3, 2]);
        assert_eq!(v.strides, &[1, 3]);
        // shared buffer
        assert_eq!(t.buffer.size(), v.buffer.size());
        Ok(())
    }
    #[test]
    fn transpose_view_preserves_offset() -> Result<()> {
        let ctx = context();
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let mut t = GpuTensor::from_f32(&ctx, &data, &[3, 4], None, None)?;
        // slice first to create non-zero offset
        t.slice_mut(&[(1, 3), (1, 4)])?;
        let offset_before = t.offset;
        let v = t.transpose_view(None)?;
        assert_eq!(v.offset, offset_before);
        assert_eq!(v.shape, &[3, 2]); // reversed axes after slice
        Ok(())
    }
}
