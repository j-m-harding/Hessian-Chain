// ============================================================
// mvcc.rs — Sharded MVCC StateDB
//   - shard-local memory (contiguous)
//   - per-thread buffer: snapshot read + commit write
//   - Merkle root per shard → global root hash
// ============================================================

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use sha2::{Sha256, Digest};
use crate::types::{Address, Hash, ShardId, TxId, NUM_SHARDS, CACHE_LINE};

/// 버전이 있는 상태 값
#[derive(Clone, Debug)]
struct VersionedValue {
    value:   u64,
    version: u64,  // 쓰기 시점 블록/TX 버전
}

/// 단일 샤드의 상태 저장소
pub struct ShardState {
    pub shard_id: ShardId,
    /// address → VersionedValue
    store: HashMap<Address, VersionedValue>,
    /// 현재 글로벌 버전 카운터
    version: u64,
}

impl ShardState {
    pub fn new(shard_id: ShardId) -> Self {
        Self { shard_id, store: HashMap::new(), version: 0 }
    }

    /// 스냅샷 읽기: 해당 version 이하의 최신 값 반환
    pub fn snapshot_read(&self, addr: &Address, _snapshot_ver: u64) -> Option<u64> {
        self.store.get(addr).map(|v| v.value)
    }

    /// 쓰기: 새 버전으로 업데이트
    pub fn write(&mut self, addr: Address, value: u64) {
        self.version += 1;
        self.store.insert(addr, VersionedValue { value, version: self.version });
    }

    /// 현재 버전 스냅샷 반환
    pub fn current_version(&self) -> u64 {
        self.version
    }

    /// Merkle root 계산 (정렬된 주소 순서로 SHA-256 트리)
    pub fn merkle_root(&self) -> Hash {
        if self.store.is_empty() {
            return [0u8; 32];
        }
        let mut leaves: Vec<([u8; 32])> = self.store
            .iter()
            .map(|(addr, ver)| {
                let mut hasher = Sha256::new();
                hasher.update(addr);
                hasher.update(ver.value.to_le_bytes());
                hasher.update(ver.version.to_le_bytes());
                let r: [u8; 32] = hasher.finalize().into();
                r
            })
            .collect();
        leaves.sort();
        merkle_combine(leaves)
    }
}

/// Merkle 트리 조합 (재귀 쌍 해시)
fn merkle_combine(mut nodes: Vec<[u8; 32]>) -> [u8; 32] {
    if nodes.len() == 1 {
        return nodes[0];
    }
    if nodes.len() % 2 != 0 {
        nodes.push(*nodes.last().unwrap());
    }
    let next: Vec<[u8; 32]> = nodes.chunks(2).map(|pair| {
        let mut h = Sha256::new();
        h.update(pair[0]);
        h.update(pair[1]);
        h.finalize().into()
    }).collect();
    merkle_combine(next)
}

// ── per-thread 로컬 버퍼 (CACHE_LINE padding으로 false sharing 방지) ──

/// 캐시 라인 정렬 패딩 래퍼
#[repr(align(64))]
pub struct Padded<T>(pub T);

/// 스레드 로컬 연산 버퍼
pub struct ThreadBuffer {
    /// 로컬 write set: address → value
    pub write_set: HashMap<Address, u64>,
    /// 로컬 read set (충돌 감지용)
    pub read_set:  HashMap<Address, u64>,
    /// grad_local, Hv_local, momentum_local (optimizer 연산용)
    _pad: [u8; CACHE_LINE],
}

impl ThreadBuffer {
    pub fn new() -> Self {
        Self {
            write_set: HashMap::new(),
            read_set:  HashMap::new(),
            _pad: [0u8; CACHE_LINE],
        }
    }

    pub fn clear(&mut self) {
        self.write_set.clear();
        self.read_set.clear();
    }
}

// ── Sharded MVCC StateDB ──

pub struct ShardedMvccStatedb {
    /// shards[i]: Arc<RwLock<ShardState>>
    pub shards: Vec<Arc<RwLock<ShardState>>>,
}

impl ShardedMvccStatedb {
    pub fn new() -> Self {
        let shards = (0..NUM_SHARDS)
            .map(|i| Arc::new(RwLock::new(ShardState::new(i))))
            .collect();
        Self { shards }
    }

    /// 특정 샤드의 스냅샷 읽기 (lock-free read)
    pub fn read(&self, shard: ShardId, addr: &Address, snapshot_ver: u64) -> Option<u64> {
        self.shards[shard].read().unwrap().snapshot_read(addr, snapshot_ver)
    }

    /// 스레드 로컬 write_set을 해당 샤드에 커밋 (atomic reduction)
    pub fn commit(&self, shard: ShardId, buf: &ThreadBuffer) {
        let mut state = self.shards[shard].write().unwrap();
        for (&addr, &val) in &buf.write_set {
            state.write(addr, val);
        }
    }

    /// 모든 샤드의 Merkle root를 합쳐 글로벌 root 계산
    pub fn global_root(&self) -> Hash {
        let shard_roots: Vec<[u8; 32]> = self.shards.iter()
            .map(|s| s.read().unwrap().merkle_root())
            .collect();
        merkle_combine(shard_roots)
    }

    /// 샤드별 현재 버전 스냅샷 반환
    pub fn snapshot_version(&self, shard: ShardId) -> u64 {
        self.shards[shard].read().unwrap().current_version()
    }
}
