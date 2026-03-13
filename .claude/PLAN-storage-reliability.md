# Plan: Storage Reliability Fix (TASK-01, Issue 1)

Source: docs/TASK-01-openclaw-skill-issues.md, docs/openclaw-e2e-transcript.md

## Problem

The server keeps running after the data directory (~/.hebbs/data) is deleted. Reads return empty silently. Writes crash with I/O error. No proactive detection.

## Root cause analysis

Three gaps in the storage layer:

1. **No pre-flight validation** - `RocksDbBackend::open()` in `crates/hebbs-storage/src/rocksdb_backend.rs` passes the path to RocksDB without checking directory exists/writable
2. **No runtime monitoring** - After startup, nothing checks data dir is still there. Health endpoints only run on-demand.
3. **Background ops swallow errors** - `decay.rs` (~line 395) uses `let _ = storage.write_batch(...)`. `engine.rs` (~line 2810) prints to stderr but continues. Both hide storage failures.

## Fix: 4 parts

### Part 1: Directory validation at startup
**File:** `crates/hebbs-storage/src/rocksdb_backend.rs` (in `open()`)

- Before calling RocksDB open, verify parent directory exists and is writable
- Create explicitly if `create_if_missing` is true, so we control the error
- Fail fast: "data directory does not exist" or "data directory is not writable"
- No RocksDB cryptic errors reaching the user

### Part 2: Periodic health check in server
**File:** `crates/hebbs-server/src/server.rs`

- Background task every 30 seconds:
  - Check `data_dir` exists on disk (`std::fs::metadata`)
  - Attempt lightweight read (`engine.count()`)
  - On failure: set server-wide health flag to unhealthy, log at ERROR level
- Readiness endpoint already returns 503 on count failure; health flag complements by catching problems proactively
- Server goes unhealthy within 30s of data dir loss, not on next client request

### Part 3: Stop swallowing write errors
**Files:** `crates/hebbs-core/src/decay.rs`, `crates/hebbs-core/src/engine.rs`

Two changes:
1. `decay.rs` ~line 395: `let _ = storage.write_batch(&update_ops)` - log at `error!` level, set health flag unhealthy
2. `engine.rs` ~line 2810: recall reinforcement stderr print - change to `error!` log, set health flag

Both remain non-fatal (decay and reinforcement are best-effort). But failures become visible.

### Part 4: Update skill health check
**File:** `hebbs-skill/hebbs/SKILL.md` ("Before every command" section)

Current check only catches connection errors:
```
hebbs-cli recall "test" --format json 2>&1
```

Add HTTP readiness check that catches storage-alive-but-broken state:
```
curl -sf http://localhost:6381/v1/health/ready
```

Or add `hebbs-cli health` command if it does not exist.

## What this does NOT fix

- **Data recovery** - deleted data is gone. RocksDB does not replicate. Plan makes failure loud, not invisible.
- **Root cause of deletion** - unknown why ~/.hebbs vanished (macOS cleanup? user error? another process?). Plan makes HEBBS resilient to it, not immune.

## Implementation order

1. Part 1 (startup validation) - smallest change, biggest safety net
2. Part 3 (stop swallowing errors) - small, fixes silent corruption
3. Part 2 (periodic health check) - medium, catches runtime failures
4. Part 4 (skill update) - trivial, improves agent behavior immediately

## Key files reference

- `crates/hebbs-storage/src/rocksdb_backend.rs` - RocksDB open, write_batch, iterators
- `crates/hebbs-storage/src/error.rs` - StorageError enum (Io variant)
- `crates/hebbs-server/src/server.rs` - Server init, no periodic health
- `crates/hebbs-server/src/rest.rs` - Health endpoints (lines 1238-1254)
- `crates/hebbs-server/src/grpc/health_service.rs` - gRPC health
- `crates/hebbs-core/src/engine.rs` - Engine ops, reinforcement error swallowed (~2810)
- `crates/hebbs-core/src/decay.rs` - Decay write error swallowed (~395)
- `crates/hebbs-storage/src/traits.rs` - StorageBackend trait
