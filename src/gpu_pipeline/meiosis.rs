use crate::compute_pipelines::meiosis;
use crate::entry::*;
use crate::genotype::*;
use crate::linalg::context::GpuContext;
use crate::linalg::tensor::GpuTensor;
use crate::phenotype::*;
use anyhow::{Result, bail, ensure};
use bytemuck::{Pod, Zeroable};
use rand::prelude::IndexedRandom;
use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng};
use std::borrow::Cow;
use std::fmt;
use std::fs::exists;
use wgpu::ComputePipeline;
use wgpu::util::DeviceExt;

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
