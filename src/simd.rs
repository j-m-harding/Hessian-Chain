// ============================================================
// simd.rs — SIMD-friendly 벡터 연산
//   Rust 컴파일러 auto-vectorization 유도:
//   - row-major contiguous 접근
//   - 단순 루프 + no branch → SIMD load/store
//   - εHv 계산, momentum, project clamp 벡터화
// ============================================================

/// x += α·y  (AXPY, auto-vectorized)
#[inline]
pub fn axpy(x: &mut [f64], alpha: f64, y: &[f64]) {
    debug_assert_eq!(x.len(), y.len());
    x.iter_mut().zip(y.iter()).for_each(|(xi, &yi)| *xi += alpha * yi);
}

/// dot(x, y) — auto-vectorized
#[inline]
pub fn dot(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum()
}

/// x = α·x + β·y  (in-place scale-add, vectorized)
#[inline]
pub fn scale_add(x: &mut [f64], alpha: f64, y: &[f64], beta: f64) {
    debug_assert_eq!(x.len(), y.len());
    x.iter_mut().zip(y.iter()).for_each(|(xi, &yi)| *xi = alpha * *xi + beta * yi);
}

/// element-wise clamp [lo, hi] (vectorized)
#[inline]
pub fn clamp_vec(x: &mut [f64], lo: f64, hi: f64) {
    x.iter_mut().for_each(|v| *v = v.clamp(lo, hi));
}

/// ε·Hv: out[k] = (grad_p[k] - grad[k]) / ε  (vectorized)
#[inline]
pub fn eps_hv(out: &mut [f64], grad_p: &[f64], grad: &[f64], eps: f64) {
    debug_assert_eq!(out.len(), grad_p.len());
    debug_assert_eq!(out.len(), grad.len());
    let inv_eps = 1.0 / eps;
    out.iter_mut()
       .zip(grad_p.iter().zip(grad.iter()))
       .for_each(|(o, (&gp, &g))| *o = (gp - g) * inv_eps);
}

/// 각 행(row)의 L1 norm으로 simplex projection
/// data: [n * m] row-major
#[inline]
pub fn row_simplex_project(data: &mut [f64], n: usize, m: usize) {
    for i in 0..n {
        let row = &mut data[i * m..(i + 1) * m];
        // clamp 음수 → 0
        row.iter_mut().for_each(|v| { if *v < 0.0 { *v = 0.0; } });
        let s: f64 = row.iter().sum();
        if s > 1e-12 {
            let inv_s = 1.0 / s;
            row.iter_mut().for_each(|v| *v *= inv_s);
        } else {
            // 전부 0이면 균등 배분
            let uniform = 1.0 / m as f64;
            row.iter_mut().for_each(|v| *v = uniform);
        }
    }
}

/// argmax per row → discrete shard index
#[inline]
pub fn row_argmax(data: &[f64], n: usize, m: usize) -> Vec<usize> {
    (0..n).map(|i| {
        let row = &data[i * m..(i + 1) * m];
        row.iter()
           .enumerate()
           .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
           .map(|(idx, _)| idx)
           .unwrap_or(0)
    }).collect()
}
