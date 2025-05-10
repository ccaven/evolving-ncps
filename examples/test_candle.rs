use candle_core::{DType, Device, IndexOp, Tensor};

fn main() -> anyhow::Result<()> {

    let device = Device::cuda_if_available(0)?;

    let batch_size = 2;
    let timesteps = 2;
    let hidden_dim = 2;

    let mut x = Tensor::zeros((batch_size, timesteps, hidden_dim), DType::F32, &device)?;

    let mut x_first_row = x.i((0, .., ..))?;
    
    x.i((.., 0, 0))?.add(&Tensor::rand(0f32, 1f32, (batch_size), &device)?)?;

    x_first_row = (x_first_row + 2.0)?;

    println!("{:?}", x_first_row.to_vec2::<f32>().unwrap());
    println!("{:?}", x.to_vec3::<f32>().unwrap());

    Ok(())
}