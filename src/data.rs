use std::f64::consts::TAU;

use candle_core::{DType, Device, IndexOp, Tensor};
use anyhow::Result;

#[derive(Clone, Copy, Debug)]
pub struct LinearDynamicalSystemConfig {
    pub timesteps: u32,
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
}

pub struct LinearDynamicalSystem {
    pub config: LinearDynamicalSystemConfig,
    pub a: Tensor,
    pub b: Tensor,
    pub c: Tensor,
    pub d: Tensor,

    pub x: Tensor,
    pub y: Tensor,
    pub w: Tensor,
    pub v: Tensor
}

impl LinearDynamicalSystem {    
    pub fn generate_input_batch(&self, batch_size: usize, device: &Device) -> Result<Tensor> {
        // Generate linear progressions
        let x = Tensor::from_iter(0..self.config.timesteps, device)?
            .to_dtype(DType::F32)?
            .unsqueeze(0)?
            .unsqueeze(candle_core::D::Minus1)?
            .repeat((batch_size, 1, self.config.input_dim))?;
    
        // Frequency shift
        let x = x * Tensor::rand(1.0, 5.0, (batch_size, 1, self.config.input_dim), device)?;
        let x = x?;

        // Phase shift
        let x = x + Tensor::rand(0.0, TAU, (batch_size, 1, self.config.input_dim), device)?;
        let x = x?;

        // Apply sine
        let x = x.sin()?;
        
        Ok(x)
    }

    pub fn generate_output_batch(&self, u: Tensor) -> anyhow::Result<(Tensor, Tensor, Tensor)> {
        let batch_size = u.shape().dim(0)?;
        let device = u.device();
        let mut x = Tensor::zeros((batch_size, self.config.timesteps as usize, self.config.hidden_dim), DType::F32, device)?;
        let mut y = Tensor::zeros((batch_size, self.config.timesteps as usize, self.config.output_dim), DType::F32, device)?;
        let mut w = Tensor::randn(0.0f32, 1.0f32, (batch_size, self.config.timesteps as usize, self.config.hidden_dim), device)?;
        let mut v = Tensor::randn(0.0f32, 1.0f32, (batch_size, self.config.timesteps as usize, self.config.output_dim), device)?;

        let mut x_col = x.i((0, .., ..))?;
        x_col = (x_col + Tensor::randn(0.0f32, 1.0f32, (self.config.timesteps as usize, self.config.hidden_dim), device)?)?;

        for t in 0..self.config.timesteps {

        }

        todo!()
    }
}