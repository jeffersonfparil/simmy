use anyhow::{Context, Result};
use std::fmt;

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
}

/// Print GPU context information
///
/// * `Name` is the adapter name reported by the backend.
/// * `Type` indicates whether the adapter is a CPU, integrated GPU,
/// discrete GPU, virtual GPU, or other device.
/// * `Backend` identifies the WGPU backend in use (e.g. Vulkan,
/// Metal, DirectX, or OpenGL).
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
    /// Initialize a GPU context.
    ///
    /// Discovers the available compute backends, selects an adapter,
    /// creates a logical device, and obtains a command queue for
    /// submitting work.
    ///
    /// The adapter request is configured to prefer high-performance
    /// hardware when available. Depending on the system configuration,
    /// the selected adapter may still be:
    /// * A discrete GPU
    /// * An integrated GPU
    /// * A virtual GPU
    /// * A CPU-backed implementation such as Vulkan Lavapipe
    ///
    /// The resulting context owns all WGPU state required to allocate
    /// tensors, compile kernels, and execute tensor operations.
    ///
    /// # Errors
    /// Returns an error if:
    /// * No suitable adapter can be found.
    /// * A logical device cannot be created from the selected adapter.
    /// * The underlying graphics or compute backend fails to initialize.
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
        let (device, queue) = adapter.request_device(&Default::default()).await?;
        Ok(Self {
            instance,
            adapter,
            device,
            queue,
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
