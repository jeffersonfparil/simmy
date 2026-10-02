use crate::linalg::context::GpuContext;
use anyhow::{Result, bail};
use std::borrow::Cow;
use std::fs::exists;
use wgpu::ComputePipeline;

pub fn pipeline(ctx: &GpuContext) -> Result<ComputePipeline> {
    match exists("meiosis.wgsl") {
        Ok(_) => (),
        Err(_) => bail!("The \"compute_pipelines/meiosis.wgsl\" kernel source does not exist!"),
    };
    let device = &ctx.device;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Meiosis Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("meiosis.wgsl"))),
    });
    Ok(
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Meiosis Pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        }),
    )
}
