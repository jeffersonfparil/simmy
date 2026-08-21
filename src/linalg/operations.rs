use crate::linalg::context::GpuContext;

pub struct MatrixOps<'a> {
    pub ctx: &'a GpuContext,
}

pub struct TensorOps<'a> {
    pub ctx: &'a GpuContext,
}
