fn main() {
    for rows in [8usize, 16, 64] {
        let h = 2624;
        let u: Vec<f32> = (0..rows * h * 2)
            .map(|i| (i % 97) as f32 * 0.08 - 3.84)
            .collect();
        let mut y = vec![0.; rows * h];
        let mut samples = Vec::new();
        for i in 0..20 {
            let start = std::time::Instant::now();
            cpu_kernels::geglu_into(&u, h, rows, &mut y);
            if i >= 5 {
                samples.push(start.elapsed().as_secs_f64() * 1000.);
            }
        }
        samples.sort_by(f64::total_cmp);
        println!("rows={rows} host_geglu_ms={}", samples[7]);
    }
    for rows in [8usize, 16, 64] {
        let d = 1024;
        let x: Vec<f32> = (0..rows * d)
            .map(|i| (i % 97) as f32 * 0.08 - 3.84)
            .collect();
        let w = vec![1.; d];
        let b = vec![0.1; d];
        let mut y = vec![0.; x.len()];
        let mut samples = Vec::new();
        for i in 0..20 {
            let start = std::time::Instant::now();
            cpu_kernels::layer_norm_feature_first_into(&x, d, rows, &w, Some(&b), 1e-5, &mut y);
            if i >= 5 {
                samples.push(start.elapsed().as_secs_f64() * 1000.);
            }
        }
        samples.sort_by(f64::total_cmp);
        println!("rows={rows} host_norm_ms={}", samples[7]);
    }
}
