# Validation Results

## Automated validation in the development environment

- Tests passed: **6**
- Branch-aware coverage: **95.54%**
- Ruff lint checks: **passed**
- Python compilation: **passed**
- Docker Compose YAML parsing: **passed**

## Upgrade completed

Amazon Bedrock generation, LLM-as-judge metrics, reference correctness, persistent run history, API tests.

## External integration status

Bedrock adapter validated with test double; run a live Bedrock evaluation before claiming live scores/cost.

Do not add throughput, latency, recovery time, cloud cost, or live model-quality numbers to a resume
until the corresponding integration or benchmark script has been run on the machine used for the
claim and the raw result has been committed under `docs/` or `benchmark-results/`.
