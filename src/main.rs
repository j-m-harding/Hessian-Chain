// ============================================================
// main.rs — Block-Level DAG Scheduler 데모
// ============================================================

mod types;
mod dag;
mod optimizer;
mod mvcc;
mod simd;
mod executor;
mod scheduler;

use rand::Rng;
use crate::types::{Transaction, DagNode, Address};
use crate::scheduler::{DagScheduler, SchedulerConfig};

fn rand_addr(rng: &mut impl Rng) -> Address {
    let mut a = [0u8; 20];
    rng.fill(&mut a);
    a
}

fn main() {
    println!("════════════════════════════════════════════");
    println!(" Block-Level DAG Scheduler  (Rust)");
    println!("════════════════════════════════════════════\n");

    let mut rng = rand::thread_rng();

    // ── TX 배치 생성 ──────────────────────────────────────────
    let num_tx = 64usize;
    let hot_addrs: Vec<Address> = (0..8).map(|_| rand_addr(&mut rng)).collect();

    let txs: Vec<Transaction> = (0..num_tx).map(|i| {
        // 20% 확률로 hot address 사용 → 의도적 충돌 생성
        let use_hot = rng.gen_bool(0.20);
        let from = if use_hot { hot_addrs[rng.gen_range(0..hot_addrs.len())] }
                   else       { rand_addr(&mut rng) };
        let to   = if use_hot { hot_addrs[rng.gen_range(0..hot_addrs.len())] }
                   else       { rand_addr(&mut rng) };

        Transaction {
            id: i as u64,
            from,
            to,
            value: rng.gen_range(1..=100),
            read_set:  vec![from],
            write_set: vec![from, to],
        }
    }).collect();

    // ── DAG 생성 (랜덤 의존 관계) ─────────────────────────────
    let dag_nodes: Vec<DagNode> = (0..num_tx).map(|i| {
        let deps: Vec<u64> = if i == 0 { vec![] }
        else {
            // 각 TX는 앞 TX 중 0~2개에 의존
            let num_deps = rng.gen_range(0..=2.min(i));
            (0..num_deps)
                .map(|_| rng.gen_range(0..i) as u64)
                .collect()
        };
        DagNode { tx_id: i as u64, deps }
    }).collect();

    // ── 스케줄러 실행 ─────────────────────────────────────────
    let config = SchedulerConfig { opt_steps: 5 };
    let mut scheduler = DagScheduler::new(config);

    let result = scheduler.process_block(txs, dag_nodes);

    // ── 결과 출력 ─────────────────────────────────────────────
    println!("\n════════════════════════════════════════════");
    println!(" 블록 처리 완료");
    println!("════════════════════════════════════════════");
    println!(" committed:    {}", result.total_committed);
    println!(" aborted:      {}", result.total_aborted);
    println!(" conflict_rate: {:.4}%", result.conflict_rate * 100.0);
    println!(
        " global_root:  {}",
        result.global_root.iter().map(|b| format!("{:02x}", b)).collect::<String>()
    );

    println!("\n[OK] EVM-compatible StateDB 업데이트 완료.");
}
