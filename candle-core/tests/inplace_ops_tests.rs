use candle_core::{test_device, Device, Result, Tensor};

fn add_inplace(device: &Device) -> Result<()> {
    let a = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], device)?;
    let b = Tensor::new(&[10.0f32, 20.0, 30.0, 40.0], device)?;
    a.add_(&b)?;
    let vals = a.to_vec1::<f32>()?;
    assert_eq!(vals, &[11.0, 22.0, 33.0, 44.0]);
    Ok(())
}

fn sub_inplace(device: &Device) -> Result<()> {
    let a = Tensor::new(&[10.0f32, 20.0, 30.0, 40.0], device)?;
    let b = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], device)?;
    a.sub_(&b)?;
    let vals = a.to_vec1::<f32>()?;
    assert_eq!(vals, &[9.0, 18.0, 27.0, 36.0]);
    Ok(())
}

fn mul_inplace(device: &Device) -> Result<()> {
    let a = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], device)?;
    let b = Tensor::new(&[2.0f32, 3.0, 4.0, 5.0], device)?;
    a.mul_(&b)?;
    let vals = a.to_vec1::<f32>()?;
    assert_eq!(vals, &[2.0, 6.0, 12.0, 20.0]);
    Ok(())
}

fn add_inplace_2d(device: &Device) -> Result<()> {
    let a = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], device)?;
    let b = Tensor::new(&[[10.0f32, 20.0], [30.0, 40.0]], device)?;
    a.add_(&b)?;
    let vals = a.to_vec2::<f32>()?;
    assert_eq!(vals, vec![vec![11.0, 22.0], vec![33.0, 44.0]]);
    Ok(())
}

fn add_inplace_u32(device: &Device) -> Result<()> {
    let a = Tensor::new(&[1u32, 2, 3, 4], device)?;
    let b = Tensor::new(&[10u32, 20, 30, 40], device)?;
    a.add_(&b)?;
    let vals = a.to_vec1::<u32>()?;
    assert_eq!(vals, &[11, 22, 33, 44]);
    Ok(())
}

fn add_inplace_aliased_fails(device: &Device) -> Result<()> {
    let a = Tensor::new(&[1.0f32, 2.0, 3.0], device)?;
    let a2 = a.clone();
    // a and a2 share storage — add_ must fail.
    let result = a.add_(&a2);
    assert!(
        result.is_err(),
        "add_ on aliased tensor should return an error"
    );
    Ok(())
}

fn add_relu_basic(device: &Device) -> Result<()> {
    // [-1, 0, 1] + [-1, 0, 1] = [-2, 0, 2], relu => [0, 0, 2]
    let a = Tensor::new(&[-1.0f32, 0.0, 1.0], device)?;
    let b = Tensor::new(&[-1.0f32, 0.0, 1.0], device)?;
    let c = a.add_relu(&b)?;
    let vals = c.to_vec1::<f32>()?;
    assert_eq!(vals, &[0.0, 0.0, 2.0]);
    Ok(())
}

fn add_relu_2d(device: &Device) -> Result<()> {
    let a = Tensor::new(&[[-2.0f32, 3.0], [1.0, -4.0]], device)?;
    let b = Tensor::new(&[[1.0f32, -1.0], [-2.0, 5.0]], device)?;
    let c = a.add_relu(&b)?;
    // sums: [-1, 2, -1, 1], relu => [0, 2, 0, 1]
    let vals = c.to_vec2::<f32>()?;
    assert_eq!(vals, vec![vec![0.0, 2.0], vec![0.0, 1.0]]);
    Ok(())
}

fn add_relu_matches_naive(device: &Device) -> Result<()> {
    let a = Tensor::randn(0.0f32, 1.0, (16, 16), device)?;
    let b = Tensor::randn(0.0f32, 1.0, (16, 16), device)?;
    let fused = a.add_relu(&b)?;
    let naive = a.add(&b)?.relu()?;
    // Check values match within floating-point tolerance.
    let fused_v = fused.flatten_all()?.to_vec1::<f32>()?;
    let naive_v = naive.flatten_all()?.to_vec1::<f32>()?;
    for (f, n) in fused_v.iter().zip(naive_v.iter()) {
        assert!((f - n).abs() < 1e-5, "mismatch: {f} vs {n}");
    }
    Ok(())
}

fn add_inplace_matches_add(device: &Device) -> Result<()> {
    let vals_a = vec![1.0f32, -2.0, 3.14, -0.5, 100.0, 0.0];
    let vals_b = vec![0.5f32, 1.0, -1.0, 2.0, -50.0, 7.0];
    let a_inplace = Tensor::new(vals_a.as_slice(), device)?;
    let b = Tensor::new(vals_b.as_slice(), device)?;
    a_inplace.add_(&b)?;

    let a_alloc = Tensor::new(vals_a.as_slice(), device)?;
    let expected = a_alloc.add(&b)?;

    let got = a_inplace.to_vec1::<f32>()?;
    let exp = expected.to_vec1::<f32>()?;
    assert_eq!(got, exp);
    Ok(())
}

fn sub_inplace_matches_sub(device: &Device) -> Result<()> {
    let vals_a = vec![5.0f32, -2.0, 10.0, 0.0];
    let vals_b = vec![1.0f32, 1.0, 3.0, -1.0];
    let a_inplace = Tensor::new(vals_a.as_slice(), device)?;
    let b = Tensor::new(vals_b.as_slice(), device)?;
    a_inplace.sub_(&b)?;

    let a_alloc = Tensor::new(vals_a.as_slice(), device)?;
    let expected = a_alloc.sub(&b)?;

    let got = a_inplace.to_vec1::<f32>()?;
    let exp = expected.to_vec1::<f32>()?;
    assert_eq!(got, exp);
    Ok(())
}

fn mul_inplace_matches_mul(device: &Device) -> Result<()> {
    let vals_a = vec![2.0f32, -1.0, 0.5, 4.0];
    let vals_b = vec![3.0f32, 2.0, -2.0, 0.25];
    let a_inplace = Tensor::new(vals_a.as_slice(), device)?;
    let b = Tensor::new(vals_b.as_slice(), device)?;
    a_inplace.mul_(&b)?;

    let a_alloc = Tensor::new(vals_a.as_slice(), device)?;
    let expected = a_alloc.mul(&b)?;

    let got = a_inplace.to_vec1::<f32>()?;
    let exp = expected.to_vec1::<f32>()?;
    assert_eq!(got, exp);
    Ok(())
}

test_device!(add_inplace, add_inplace_cpu, add_inplace_gpu, add_inplace_metal);
test_device!(sub_inplace, sub_inplace_cpu, sub_inplace_gpu, sub_inplace_metal);
test_device!(mul_inplace, mul_inplace_cpu, mul_inplace_gpu, mul_inplace_metal);
test_device!(
    add_inplace_2d,
    add_inplace_2d_cpu,
    add_inplace_2d_gpu,
    add_inplace_2d_metal
);
test_device!(
    add_inplace_u32,
    add_inplace_u32_cpu,
    add_inplace_u32_gpu,
    add_inplace_u32_metal
);
test_device!(
    add_relu_basic,
    add_relu_basic_cpu,
    add_relu_basic_gpu,
    add_relu_basic_metal
);
test_device!(
    add_relu_2d,
    add_relu_2d_cpu,
    add_relu_2d_gpu,
    add_relu_2d_metal
);
test_device!(
    add_relu_matches_naive,
    add_relu_matches_naive_cpu,
    add_relu_matches_naive_gpu,
    add_relu_matches_naive_metal
);
test_device!(
    add_inplace_matches_add,
    add_inplace_matches_add_cpu,
    add_inplace_matches_add_gpu,
    add_inplace_matches_add_metal
);
test_device!(
    sub_inplace_matches_sub,
    sub_inplace_matches_sub_cpu,
    sub_inplace_matches_sub_gpu,
    sub_inplace_matches_sub_metal
);
test_device!(
    mul_inplace_matches_mul,
    mul_inplace_matches_mul_cpu,
    mul_inplace_matches_mul_gpu,
    mul_inplace_matches_mul_metal
);

#[test]
fn add_inplace_aliased_fails_cpu() -> Result<()> {
    add_inplace_aliased_fails(&Device::Cpu)
}
