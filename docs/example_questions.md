# Example architecture questions

Good reviews ask a concrete decision question rather than "review my architecture".

## Text-to-SQL

```text
Should schema selection, query planning, and SQL repair remain deterministic, or should any of them become bounded agentic stages? Compare execution accuracy, token cost, p95 latency, debuggability, and failure containment.
```

```text
How should this Text-to-SQL system support multiple tenants without leaking schema metadata, retrieved examples, query history, or database results across tenants?
```

```text
Should successful-query retrieval be kept as a separate retrieval stage or merged into the schema/example retrieval index? Define an experiment to decide.
```

## KPI / analytics agents

```text
Should KPI monitoring use scheduled deterministic workers with an LLM interpretation step, or an autonomous agent loop? Evaluate reliability, alert noise, cost, reproducibility, and incident recovery.
```

```text
What should be the boundary between dbt-owned metric definitions, Text-to-SQL, Python analysis, and LLM-generated narrative insights?
```

## General software architecture

```text
Should this service remain a modular monolith or be split into services? Identify the concrete scale or organizational triggers that would justify the split.
```

```text
Which state must be persisted, which can be recomputed, and where are the current consistency and concurrency risks?
```

```text
What are the top three architecture changes required before exposing this system to untrusted external users?
```
