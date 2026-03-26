// ============================================================
// executor.rs — Parallel TX Executor
//   - Rayon 기반 병렬 샤드 실행
//   - per-thread ThreadBuffer (false sharing 방지)
//   - 스냅샷 읽기 → 로컬 실행 → 샤드 커밋
// ============================================================

use std::collections::HashMap;
use rayon::prelude::*;
use crate::types::{TxId, ShardId, Transaction, NUM_SHARDS};
use crate::mvcc::{ShardedMvccStatedb, ThreadBuffer};

/// 샤드별 TX 목록
pub type ShardBatch = Vec<Vec<Transaction>>;

/// DAG 레이어 단위로 병렬 실행
pub struct ParallelExecutor<'a> {
    pub db: &'a ShardedMvccStatedb,
}

impl<'a> ParallelExecutor<'a> {
    pub fn new(db: &'a ShardedMvccStatedb) -> Self {
        Self { db }
    }

    /// 한 레이어의 모든 TX를 샤드별로 병렬 실행
    /// assignment: tx_id → shard_id
    pub fn execute_layer(
        &self,
        txs: &[Transaction],
        assignment: &HashMap<TxId, ShardId>,
    ) -> ExecutionResult {
        // 샤드별 TX 분류
        let mut shard_batches: Vec<Vec<&Transaction>> = vec![vec![]; NUM_SHARDS];
        for tx in txs {
            if let Some(&shard) = assignment.get(&tx.id) {
                shard_batches[shard].push(tx);
            }
        }

        // 각 샤드 병렬 실행 (Rayon)
        let results: Vec<ShardResult> = shard_batches
            .par_iter()
            .enumerate()
            .map(|(shard_id, shard_txs)| {
                self.execute_shard(shard_id, shard_txs)
            })
            .collect();

        // 결과 취합
        let mut total_committed = 0;
        let mut total_aborted = 0;
        for r in &results {
            total_committed += r.committed;
            total_aborted += r.aborted;
        }

        ExecutionResult { total_committed, total_aborted, shard_results: results }
    }

    fn execute_shard(&self, shard_id: ShardId, txs: &[&Transaction]) -> ShardResult {
        let mut buf = ThreadBuffer::new();
        let mut committed = 0usize;
        let mut aborted = 0usize;

        let snapshot_ver = self.db.snapshot_version(shard_id);

        for tx in txs {
            buf.clear();

            // 스냅샷 읽기
            for addr in &tx.read_set {
                let val = self.db.read(shard_id, addr, snapshot_ver).unwrap_or(0);
                buf.read_set.insert(*addr, val);
            }

            // TX 실행 (간단한 transfer 시뮬레이션)
            let success = self.apply_tx(tx, &mut buf, snapshot_ver, shard_id);

            if success {
                // 커밋: 로컬 write_set → 샤드 DB
                self.db.commit(shard_id, &buf);
                committed += 1;
            } else {
                aborted += 1;
            }
        }

        ShardResult { shard_id, committed, aborted }
    }

    fn apply_tx(
        &self,
        tx: &Transaction,
        buf: &mut ThreadBuffer,
        snapshot_ver: u64,
        shard_id: ShardId,
    ) -> bool {
        // 잔액 차감 시뮬레이션
        let from_bal = self.db.read(shard_id, &tx.from, snapshot_ver).unwrap_or(1000);
        if from_bal < tx.value {
            return false; // 잔액 부족 → abort
        }
        let to_bal = self.db.read(shard_id, &tx.to, snapshot_ver).unwrap_or(0);

        buf.write_set.insert(tx.from, from_bal - tx.value);
        buf.write_set.insert(tx.to, to_bal + tx.value);
        true
    }
}

#[derive(Debug)]
pub struct ShardResult {
    pub shard_id:  ShardId,
    pub committed: usize,
    pub aborted:   usize,
}

#[derive(Debug)]
pub struct ExecutionResult {
    pub total_committed:  usize,
    pub total_aborted:    usize,
    pub shard_results:    Vec<ShardResult>,
}
