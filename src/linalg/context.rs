use anyhow::{Context, Result};
use std::fmt;
use wgpu::ComputePipeline;

/// GPU Context
///
/// Owns the GPU runtime state required in building and manipulating tensors including kernel operations.
/// This is intended to be created once and then shared by tensors, kernels, and other ternsor operations.
///
/// * `instance` discovers the GPU backend and adapters.
/// * `adapter` represents the selected physical GPU.
/// * `device` is the logical connection used to create GPU resources.
/// * `queue` is used to submit work to the GPU.
///
/// Note that this can represent a CPU, if a dedicate physical GPU hardware does not exist, i.e. CPU-backed Vulkan implementation is used.
#[derive(Debug)]
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,

    pub unary_matrix_pipeline: wgpu::ComputePipeline,
    pub binary_matrix_pipeline: wgpu::ComputePipeline,
    pub contract_matrix_pipeline: wgpu::ComputePipeline,

    pub unary_tensor_pipeline: wgpu::ComputePipeline,
    pub binary_tensor_pipeline: wgpu::ComputePipeline,
    pub contract_tensor_pipeline: wgpu::ComputePipeline,
}

/// Print GPU context information
///
/// * `Name` is the adapter name reported by the backend.
/// * `Type` indicates whether the adapter is a CPU, integrated GPU,
///   discrete GPU, virtual GPU, or other device.
/// * `Backend` identifies the WGPU backend in use (e.g. Vulkan,
///   Metal, DirectX, or OpenGL).
/// * `Vendor` is the hardware vendor identifier.
/// * `Driver` contains the driver name and additional driver details.
impl fmt::Display for GpuContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let info = self.adapter.get_info();
        writeln!(f, "GPU Hardware Information:")?;
        writeln!(f, "  Name:    {}", info.name)?;
        writeln!(f, "  Type:    {:?}", info.device_type)?;
        writeln!(f, "  Backend: {:?}", info.backend)?;
        writeln!(f, "  Vendor:  {}", info.vendor)?;
        write!(f, "  Driver:  {} ({})", info.driver, info.driver_info)
    }
}

impl GpuContext {
    /// Initialize a GPU compute context.
    ///
    /// Creates and configures all GPU state required by the tensor engine.
    ///
    /// Initialization proceeds in several stages:
    ///
    /// 1. Create a WGPU instance.
    ///
    ///    The instance is responsible for discovering and managing available
    ///    graphics and compute backends (such as Vulkan, Metal, DirectX, or
    ///    OpenGL-compatible implementations).
    ///
    /// 2. Select a compute adapter.
    ///
    ///    The adapter request prefers high-performance hardware:
    ///
    ///    ```text
    ///    PowerPreference::HighPerformance
    ///    ```
    ///
    ///    Depending on the host platform, the selected adapter may be:
    ///
    ///    * A discrete GPU.
    ///    * An integrated GPU.
    ///    * A virtual GPU.
    ///    * A software implementation such as Vulkan Lavapipe.
    ///
    ///    The exact adapter selected is determined by WGPU and the available
    ///    system hardware.
    ///
    /// 3. Create a logical device and command queue.
    ///
    ///    The device is responsible for:
    ///
    ///    * Buffer allocation.
    ///    * Shader execution.
    ///    * Pipeline creation.
    ///    * Resource binding.
    ///
    ///    The queue is used to submit command buffers for execution.
    ///
    /// 4. Compile all built-in WGSL compute kernels.
    ///
    ///    The following kernels are embedded directly into the binary using
    ///    `include_str!` and compiled during initialization:
    ///
    ///    Matrix kernels:
    ///
    ///    * `unary_matrix.wgsl`
    ///      Element-wise unary matrix operations.
    ///
    ///    * `binary_matrix.wgsl`
    ///      Element-wise binary matrix operations.
    ///
    ///    * `contract_matrix.wgsl`
    ///      Matrix contraction operations, including matrix multiplication.
    ///
    ///    Tensor kernels:
    ///
    ///    * `unary_tensor.wgsl`
    ///      Element-wise unary tensor operations on arbitrary-rank tensors.
    ///
    ///    * `binary_tensor.wgsl`
    ///      Element-wise binary tensor operations on arbitrary-rank tensors.
    ///
    ///    * `contract_tensor.wgsl`
    ///      General tensor contractions supporting:
    ///      - Arbitrary tensor rank.
    ///      - Arbitrary contraction axes.
    ///      - Tensor views.
    ///      - Tensor slices.
    ///      - Tensor transposes.
    ///      - Custom pairwise and reduction operators.
    ///
    /// 5. Create compute pipelines.
    ///
    ///    Each shader module is converted into a compute pipeline and stored
    ///    inside the context for reuse. Pipelines are created once during
    ///    initialization to avoid repeated shader compilation during tensor
    ///    execution.
    ///
    /// The resulting context owns:
    ///
    /// * The WGPU instance.
    /// * The selected adapter.
    /// * The logical device.
    /// * The command queue.
    /// * All built-in compute pipelines.
    ///
    /// These resources are subsequently used to:
    ///
    /// * Allocate GPU-backed tensors.
    /// * Upload and download tensor data.
    /// * Execute matrix operations.
    /// * Execute tensor operations.
    /// * Dispatch compute workloads.
    /// * Manage GPU resources throughout the lifetime of the application.
    ///
    /// # Performance
    ///
    /// Shader compilation and pipeline creation occur during context
    /// initialization rather than during the first operation dispatch. This
    /// front-loads startup cost while minimizing runtime latency during tensor
    /// execution.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * No suitable compute adapter can be found.
    /// * The selected adapter cannot create a logical device.
    /// * The command queue cannot be created.
    /// * Any embedded WGSL shader fails validation or compilation.
    /// * Any compute pipeline fails to be created.
    /// * The underlying graphics or compute backend fails to initialize.
    ///
    /// # Notes
    ///
    /// The context is intended to be created once and reused for the lifetime
    /// of a workload. Creating multiple contexts may incur additional shader
    /// compilation, pipeline creation, and device initialization overhead.
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        // Request for the high performance adapter which should detect a dedicated GPU if they exist
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default() // use the defaults for the other fields
            })
            .await
            .context("No adapter")?;
        // let (device, queue) = adapter.request_device(&Default::default()).await?;
        let supported = adapter.limits();
        let required_limits = wgpu::Limits {
            max_color_attachments: supported.max_color_attachments,
            ..wgpu::Limits::downlevel_defaults()
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("device"),
                required_features: wgpu::Features::empty(),
                required_limits,
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                trace: wgpu::Trace::default(),
            })
            .await?;
        let opcode_source: &str = include_str!("wgsl/opcodes.wgsl");
        let kernel_sources: Vec<&str> = vec![
            include_str!("wgsl/unary_matrix.wgsl"),
            include_str!("wgsl/binary_matrix.wgsl"),
            include_str!("wgsl/contract_matrix.wgsl"),
            include_str!("wgsl/unary_tensor.wgsl"),
            include_str!("wgsl/binary_tensor.wgsl"),
            include_str!("wgsl/contract_tensor.wgsl"),
        ];
        let mut pipelines: Vec<ComputePipeline> = Vec::with_capacity(kernel_sources.len());
        for kernel_source in kernel_sources {
            let source = format!("{}\n{}", opcode_source, kernel_source);
            let kernel_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("kernel"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("pipeline"),
                layout: None,
                module: &kernel_module,
                entry_point: None,
                compilation_options: Default::default(),
                cache: None,
            });
            pipelines.push(pipeline);
        }
        let unary_matrix_pipeline = pipelines.remove(0);
        let binary_matrix_pipeline = pipelines.remove(0);
        let contract_matrix_pipeline = pipelines.remove(0);
        let unary_tensor_pipeline = pipelines.remove(0);
        let binary_tensor_pipeline = pipelines.remove(0);
        let contract_tensor_pipeline = pipelines.remove(0);
        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            unary_matrix_pipeline,
            binary_matrix_pipeline,
            contract_matrix_pipeline,
            unary_tensor_pipeline,
            binary_tensor_pipeline,
            contract_tensor_pipeline,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn context() -> GpuContext {
        pollster::block_on(GpuContext::new()).expect("Failed to create GPU context")
    }
    #[test]
    fn creates_gpu_context() {
        let ctx = context();
        println!("ctx: {}", ctx);
        let info = ctx.adapter.get_info();
        assert!(!info.name.is_empty(), "Adapter name should not be empty");
    }
    #[test]
    fn exposes_valid_adapter_info() {
        let ctx = context();
        let info = ctx.adapter.get_info();
        println!("Adapter: {:?}", info);
        assert!(!info.name.is_empty());
    }
    #[test]
    fn exposes_non_zero_compute_limits() {
        let ctx = context();
        let limits = ctx.device.limits();
        assert!(limits.max_compute_workgroup_size_x > 0);
        assert!(limits.max_compute_workgroup_size_y > 0);
        assert!(limits.max_compute_workgroup_size_z > 0);
        assert!(limits.max_compute_invocations_per_workgroup > 0);
    }
    #[test]
    fn can_create_buffer() {
        let ctx = context();
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test-buffer"),
            size: 1024,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        assert_eq!(buffer.size(), 1024);
    }

    #[test]
    fn can_create_command_encoder() {
        let ctx = context();
        let _encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("test-encoder"),
            });
    }
    #[test]
    fn can_submit_empty_command_buffer() {
        let ctx = context();
        let encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("test-encoder"),
            });
        ctx.queue.submit(std::iter::once(encoder.finish()));
        ctx.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("Device poll failed");
    }
    #[test]
    fn device_exposes_features() {
        let ctx = context();
        let _features = ctx.device.features();
    }
}
