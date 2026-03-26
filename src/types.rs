// ============================================================
// types.rs — 공통 타입 및 상수
// ============================================================

pub type TxId = u64;
pub type ShardId = usize;
pub type Address = [u8; 20];
pub type Hash = [u8; 32];

pub const NUM_SHARDS: usize = 8;
pub const EPSILON: f64 = 1e-5;
pub const BETA: f64 = 0.9;       // momentum 계수
pub const CG_ITER: usize = 20;   // Conjugate Gradient 최대 반복
pub const CACHE_LINE: usize = 64; // 캐시 라인 크기 (bytes)

/// 단일 트랜잭션
#[derive(Clone, Debug)]
pub struct Transaction {
    pub id: TxId,
    pub from: Address,
    pub to: Address,
    pub value: u64,
    pub read_set: Vec<Address>,
    pub write_set: Vec<Address>,
}

/// DAG 노드: tx_id + 선행 의존 목록
#[derive(Clone, Debug)]
pub struct DagNode {
    pub tx_id: TxId,
    pub deps: Vec<TxId>, // 이 tx가 의존하는 tx 목록
}

/// 샤드 배정 결과
#[derive(Clone, Debug)]
pub struct ShardAssignment {
    pub tx_id: TxId,
    pub shard: ShardId,
}
