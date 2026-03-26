// ============================================================
// optimizer.rs — Hessian-Free MI Scheduler
//   - soft assignment x[n][m]  (n=tx수, m=샤드수)
//   - Hessian-vector product:  Hv ≈ (∇L(x+εv) − ∇L(x)) / ε
//   - Momentum:                v  = β·v + grad
//   - Conjugate Gradient:      Δx = H⁻¹·v
//   - Project + clamp → discrete shard assignment
// ============================================================

use std::collections::HashMap;
use crate::types::{TxId, ShardId, NUM_SHARDS, EPSILON, BETA, CG_ITER, Transaction};

/// n × m 행렬 (row-major, contiguous)
/// x[i][p] = TX i가 shard p에 배정될 "soft 확률"
pub struct SoftAssignment {
    pub n: usize,          // TX 수
    pub m: usize,          // shard 수 (= NUM_SHARDS)
    pub data: Vec<f64>,    // [n * m]
}

impl SoftAssignment {
    pub fn new(n: usize) -> Self {
        let m = NUM_SHARDS;
        // 균등 초기화: 1/m
        let init = 1.0 / m as f64;
        Self { n, m, data: vec![init; n * m] }
    }

    #[inline(always)]
    pub fn get(&self, i: usize, p: usize) -> f64 {
        self.data[i * self.m + p]
    }

    #[inline(always)]
    pub fn set(&mut self, i: usize, p: usize, val: f64) {
        self.data[i * self.m + p] = val;
    }
}

/// 충돌 손실 함수 L(x):
///   충돌 = 두 TX가 같은 주소를 write 하면서 같은 shard에 있을 때
///   L = Σ_{i<j, conflict(i,j)} Σ_p x[i,p] * x[j,p]
pub struct ConflictLoss<'a> {
    pub txs: &'a [Transaction],
    /// conflict_pairs[i] = i와 충돌하는 TX 인덱스 목록
    pub conflict_pairs: Vec<Vec<usize>>,
}

impl<'a> ConflictLoss<'a> {
    pub fn build(txs: &'a [Transaction]) -> Self {
        let n = txs.len();
        let mut conflict_pairs = vec![vec![]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                if Self::conflicts(&txs[i], &txs[j]) {
                    conflict_pairs[i].push(j);
                    conflict_pairs[j].push(i);
                }
            }
        }
        Self { txs, conflict_pairs }
    }

    fn conflicts(a: &Transaction, b: &Transaction) -> bool {
        // write-write 또는 write-read 겹침 → 충돌
        for wa in &a.write_set {
            if b.write_set.contains(wa) || b.read_set.contains(wa) {
                return true;
            }
        }
        for wb in &b.write_set {
            if a.read_set.contains(wb) {
                return true;
            }
        }
        false
    }

    /// ∇L(x) 계산, grad: &mut Vec<f64> [n*m]
    pub fn gradient(&self, x: &SoftAssignment, grad: &mut Vec<f64>) {
        let n = x.n;
        let m = x.m;
        grad.iter_mut().for_each(|g| *g = 0.0);

        for i in 0..n {
            for &j in &self.conflict_pairs[i] {
                if j <= i { continue; } // 한 번만 계산
                for p in 0..m {
                    let xi = x.get(i, p);
                    let xj = x.get(j, p);
                    // ∂L/∂x[i,p] += x[j,p]
                    grad[i * m + p] += xj;
                    // ∂L/∂x[j,p] += x[i,p]
                    grad[j * m + p] += xi;
                }
            }
        }
    }
}

/// Hessian-Free Optimizer
pub struct HessianFreeOptimizer {
    pub n: usize,
    pub m: usize,
    /// momentum 벡터 v[n*m]
    pub momentum: Vec<f64>,
    /// 사전 할당 버퍼
    buf_grad:   Vec<f64>,
    buf_grad_p: Vec<f64>,  // ∇L(x + ε·v_cg)
    buf_hv:     Vec<f64>,  // Hessian-vector product
}

impl HessianFreeOptimizer {
    pub fn new(n: usize) -> Self {
        let m = NUM_SHARDS;
        let sz = n * m;
        Self {
            n, m,
            momentum:   vec![0.0; sz],
            buf_grad:   vec![0.0; sz],
            buf_grad_p: vec![0.0; sz],
            buf_hv:     vec![0.0; sz],
        }
    }

    /// Hv ≈ (∇L(x + ε·v) − ∇L(x)) / ε   (matrix-free)
    fn hessian_vec(
        &mut self,
        loss: &ConflictLoss,
        x: &SoftAssignment,
        v: &[f64],          // [n*m]
        hv_out: &mut Vec<f64>,
    ) {
        let sz = self.n * self.m;

        // x_p = x + ε·v
        let mut x_p = SoftAssignment { n: self.n, m: self.m, data: x.data.clone() };
        for k in 0..sz {
            x_p.data[k] += EPSILON * v[k];
        }

        // ∇L(x)
        loss.gradient(x, &mut self.buf_grad);
        // ∇L(x + ε·v)
        loss.gradient(&x_p, &mut self.buf_grad_p);

        // Hv = (∇L(x+εv) − ∇L(x)) / ε
        for k in 0..sz {
            hv_out[k] = (self.buf_grad_p[k] - self.buf_grad[k]) / EPSILON;
        }
    }

    /// Conjugate Gradient: H·Δx ≈ −v  → Δx
    /// 반환: Δx [n*m]
    fn conjugate_gradient(
        &mut self,
        loss: &ConflictLoss,
        x: &SoftAssignment,
        rhs: &[f64],  // = −momentum (우변)
    ) -> Vec<f64> {
        let sz = self.n * self.m;
        let mut delta = vec![0.0f64; sz]; // 초기 해 x0 = 0
        let mut r = rhs.to_vec();          // r = rhs − H·0 = rhs
        let mut p = r.clone();
        let mut rs_old: f64 = r.iter().map(|x| x * x).sum();

        for _ in 0..CG_ITER {
            if rs_old < 1e-12 { break; }

            // Hp = H · p
            let mut hp = vec![0.0f64; sz];
            self.hessian_vec(loss, x, &p, &mut hp);

            // α = rs_old / (p^T H p)
            let pthp: f64 = p.iter().zip(hp.iter()).map(|(a, b)| a * b).sum();
            if pthp.abs() < 1e-15 { break; }
            let alpha = rs_old / pthp;

            // delta += α·p
            for k in 0..sz { delta[k] += alpha * p[k]; }
            // r -= α·Hp
            for k in 0..sz { r[k] -= alpha * hp[k]; }

            let rs_new: f64 = r.iter().map(|x| x * x).sum();
            let beta_cg = rs_new / rs_old;
            // p = r + β_cg·p
            for k in 0..sz { p[k] = r[k] + beta_cg * p[k]; }
            rs_old = rs_new;
        }
        delta
    }

    /// 한 스텝 실행: momentum update → CG → x 업데이트 → project
    pub fn step(&mut self, loss: &ConflictLoss, x: &mut SoftAssignment) {
        let sz = self.n * self.m;

        // ∇L(x)
        loss.gradient(x, &mut self.buf_grad);

        // momentum: v = β·v + grad
        for k in 0..sz {
            self.momentum[k] = BETA * self.momentum[k] + self.buf_grad[k];
        }

        // rhs = −v  (descent direction)
        let rhs: Vec<f64> = self.momentum.iter().map(|&v| -v).collect();

        // CG → Δx
        let delta = self.conjugate_gradient(loss, x, &rhs);

        // x += Δx, then project onto simplex per row, clamp [0,1]
        for i in 0..self.n {
            for p in 0..self.m {
                let k = i * self.m + p;
                x.data[k] = (x.data[k] + delta[k]).clamp(0.0, 1.0);
            }
            // simplex projection: 각 행의 합 = 1
            let row_sum: f64 = (0..self.m).map(|p| x.data[i * self.m + p]).sum();
            if row_sum > 1e-12 {
                for p in 0..self.m {
                    x.data[i * self.m + p] /= row_sum;
                }
            }
        }
    }

    /// soft x → discrete shard assignment (argmax per row)
    pub fn round(x: &SoftAssignment, tx_ids: &[TxId]) -> HashMap<TxId, ShardId> {
        tx_ids.iter().enumerate().map(|(i, &tx_id)| {
            let shard = (0..x.m)
                .max_by(|&a, &b| x.get(i, a).partial_cmp(&x.get(i, b)).unwrap())
                .unwrap_or(0);
            (tx_id, shard)
        }).collect()
    }
}
