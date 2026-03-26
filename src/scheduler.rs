// ============================================================
// scheduler.rs — Block-Level DAG Scheduler
//   전체 파이프라인 오케스트레이션:
//   1. DAG 위상 정렬 → 레이어 분할
//   2. HF-MI Optimizer → soft x[i,p] → discrete 배정
//   3. 레이어별 병렬 실행 (MVCC)
//   4. 글로벌 Merkle root 계산
// ============================================================

use std::collections::HashMap;
use crate::types::{TxId, ShardId, Transaction, DagNode};
use crate::dag::DependencyDag;
use crate::optimizer::{SoftAssignment, ConflictLoss, HessianFreeOptimizer};
use crate::mvcc::ShardedMvccStatedb;
use crate::executor::ParallelExecutor;

/// 스케줄러 설정
pub struct SchedulerConfig {
    /// optimizer 스텝 횟수
    pub opt_steps: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self { opt_steps: 5 }
    }
}

/// 블록 처리 결과
pub struct BlockResult {
    pub global_root:     [u8; 32],
    pub total_committed: usize,
    pub total_aborted:   usize,
    pub conflict_rate:   f64,
}

/// Block-Level DAG Scheduler
pub struct DagScheduler {
    pub db:     ShardedMvccStatedb,
    pub config: SchedulerConfig,
}

impl DagScheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        Self { db: ShardedMvccStatedb::new(), config }
    }

    /// 하나의 블록 처리
    pub fn process_block(
        &mut self,
        txs: Vec<Transaction>,
        dag_nodes: Vec<DagNode>,
    ) -> BlockResult {
        let n = txs.len();
        if n == 0 {
            return BlockResult {
                global_root: [0u8; 32],
                total_committed: 0,
                total_aborted: 0,
                conflict_rate: 0.0,
            };
        }

        // ── Step 1: DAG 위상 정렬 ──────────────────────────────────
        let dag = DependencyDag::new(dag_nodes);
        let layers = dag.topological_layers();
        println!("[Scheduler] {} TX → {} 레이어", n, layers.len());

        // ── Step 2: HF-MI Optimizer ────────────────────────────────
        let tx_ids: Vec<TxId> = txs.iter().map(|t| t.id).collect();
        let loss = ConflictLoss::build(&txs);

        // 충돌 쌍 수 계산 (통계용)
        let conflict_count: usize = loss.conflict_pairs.iter().map(|v| v.len()).sum::<usize>() / 2;
        let conflict_rate = if n > 1 {
            conflict_count as f64 / ((n * (n - 1) / 2) as f64)
        } else { 0.0 };
        println!("[Optimizer] 충돌 쌍: {} / 충돌률: {:.4}%", conflict_count, conflict_rate * 100.0);

        let mut x = SoftAssignment::new(n);
        let mut opt = HessianFreeOptimizer::new(n);

        for step in 0..self.config.opt_steps {
            opt.step(&loss, &mut x);
            if step % 2 == 0 {
                // 현재 손실 계산 (통계)
                let mut grad = vec![0.0f64; n * x.m];
                loss.gradient(&x, &mut grad);
                let loss_val: f64 = grad.iter().map(|g| g.abs()).sum::<f64>() / (n * x.m) as f64;
                println!("[Optimizer] step={} avg_grad_norm={:.6}", step, loss_val);
            }
        }

        // soft → discrete
        let assignment: HashMap<TxId, ShardId> = HessianFreeOptimizer::round(&x, &tx_ids);

        // 샤드 분포 출력
        let mut shard_count = vec![0usize; 8];
        for &s in assignment.values() { shard_count[s] += 1; }
        println!("[Scheduler] 샤드 분포: {:?}", shard_count);

        // ── Step 3: 레이어별 병렬 실행 ────────────────────────────
        let executor = ParallelExecutor::new(&self.db);
        let tx_map: HashMap<TxId, &Transaction> =
            txs.iter().map(|t| (t.id, t)).collect();

        let mut total_committed = 0;
        let mut total_aborted = 0;

        for (layer_idx, layer) in layers.iter().enumerate() {
            // 이 레이어의 TX 목록
            let layer_txs: Vec<Transaction> = layer.iter()
                .filter_map(|id| tx_map.get(id).map(|t| (*t).clone()))
                .collect();

            let result = executor.execute_layer(&layer_txs, &assignment);
            println!(
                "[Executor] 레이어 {}: committed={} aborted={}",
                layer_idx, result.total_committed, result.total_aborted
            );
            total_committed += result.total_committed;
            total_aborted += result.total_aborted;
        }

        // ── Step 4: 글로벌 Merkle root ────────────────────────────
        let global_root = self.db.global_root();
        println!("[Commit] global_root: {}", hex_str(&global_root));

        BlockResult { global_root, total_committed, total_aborted, conflict_rate }
    }
}

fn hex_str(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect::<Vec<_>>().join("")
}
