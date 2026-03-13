# HEBBS Documentation Examples

Runnable companion code for the [HEBBS documentation](https://docs.hebbs.ai) cookbooks and SDK guides.

## Prerequisites

1. A running HEBBS server on `localhost:6380` (gRPC) / `localhost:6381` (HTTP)
2. Python >= 3.10
3. The HEBBS Python SDK

## Setup

```bash
pip install -r requirements.txt
```

## Running Examples

Each directory contains a self-contained example. Start the HEBBS server first, then run:

```bash
# Quickstart
python quickstart/quickstart.py

# Multi-strategy recall comparison
python multi-strategy-recall/multi_strategy.py

# Entity-scoped memory isolation
python entity-scoped/entity_scoped.py

# GDPR compliance (forget + verify)
python gdpr-compliance/gdpr_forget.py

# Real-time subscribe
python realtime-subscribe/subscribe_demo.py

# Background learning (reflect + insights)
python background-learning/reflect_demo.py

# Causal chain traversal
python causal-chains/causal_demo.py
```

## Monitoring Stack

The `monitoring-stack/` directory contains a docker-compose setup for HEBBS + Prometheus + Grafana:

```bash
cd monitoring-stack
docker-compose up
```

## Rust Example

The `rust-quickstart/` directory contains a Cargo project:

```bash
cd rust-quickstart
cargo run
```

## Verifying All Examples

From the repo root:

```bash
./scripts/verify-all.sh
```
