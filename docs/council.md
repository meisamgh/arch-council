# ArchCouncil 0.8: practical discussion

`review` now defaults to two agents (Proposer and Critic), up to two rounds, optional research,
and one final decision call. `--agents 3` adds a specialist. The earlier three-proposal weighted
arbitration pipeline remains available with `--mode legacy`.

```text
Independent positions → claim-specific critique → optional research → requester responds
                     → revisions/concessions → stop evaluation → final decision
```

Python assigns claim/challenge/request IDs and authorship. Only an owner can revise a claim;
only the challenger can close an objection with a reason. Changing a claim invalidates its
previous endorsements and reopens objections. Unknown claim or evidence references are rejected.
The complete ledger survives between turns. Each prompt contains a compact index and details
for two active claims; raw outputs are stored separately in SQLite.

Stop reasons are `converged`, `max_rounds`, or `budget_exhausted`. Convergence requires all
other participating agents to support each claim, no open high-severity challenges, no incomplete
research request, and no material change in that round. A round limit is not evidence of agreement.
Experiments are proposed plans, not executed or validated results. The final report includes the
complete ledger, dissent history, experiments, source URLs, coverage gaps, and budget counters.

## Cost controls and rounds

Start with **two rounds**. One round is useful for a quick critique; three rounds allow another
response when important disputes remain. This is a cost-conscious default, not a guarantee of
decision quality. The coordinator can stop early.

| Agents | One round | Two rounds | Three rounds |
|---|---:|---:|---:|
| 2 | 5 calls | 7 calls | 9 calls |
| 3 | 7 calls | 10 calls | 13 calls |

Counts include independent positions, discussion turns, and one final synthesis, before retries,
repairs, or evidence follow-ups. Each requesting agent can receive one immediate evidence
follow-up per turn. Research never runs merely because another round started.

Defaults: 12 total HTTP attempts, 16,000 input characters per call, 160,000 cumulative input
characters, 1,200 output tokens per call, 3 research queries and 4 source inspections per review.
Counters persist across resume. An exhausted budget returns a partial report and requires an
explicitly larger allowance to continue. Prompt size is bounded; it is not guaranteed identical
each round. Characters are not tokens, and these counters do not claim an exact monetary cost.

Structured turns are compacted deterministically, so **no summarizer call is required**.
`--summarizer-model` and `--disable-summarizer` affect legacy mode only.
`--arbiter-model` optionally changes the single final decision model in council mode.

```bash
arch-council review \
  --repo "/Users/meisam/Documents/Feature Engineer Agent" \
  --readme-only --agents 2 --rounds 2 \
  --model-a justwoker/claude-opus-4-8 \
  --model-b justwoker/claude-opus-4-8 \
  --research --research-provider searxng \
  --research-queries 3 --research-results 2 --evidence-inspections 4 \
  --max-total-calls 12 --max-prompt-chars 16000 --max-output-tokens 1200 \
  --call-delay-seconds 10 --review-id features-council-001 \
  --question "Improve this feature-discovery architecture. Challenge specific claims, request papers or repository evidence when needed, preserve dissent, and propose tests for novelty, leakage, diversity and out-of-sample value."
```

For a topic without a repository:

Council questions may contain up to 12,000 characters. Instructions remain intact in every
turn; they are not chunked or summarized. The complete prompt (instructions, README and
discussion state) must still fit `--max-prompt-chars`. Longer questions therefore leave
less space for evidence and discussion; use `--max-prompt-chars 24000` when appropriate.

```bash
arch-council review --topic "How should we evaluate novel feature hypotheses?" --rounds 2
```

Model defaults resolve from `.env`; explicit provider/model aliases override them. Provider keys
are kept separate and never substituted from a different provider. Availability still depends on
your provider account; offline tests do not establish live model access.

## Evidence and recovery

Research requests identify the claim, question, source family and decision purpose. The requester
receives results before the floor moves to the next agent. Failed inspections retain labeled search
snippets and partial coverage. GitHub inspection reads up to two selected text/code/test files at a
pinned commit, without execution. arXiv inspection retrieves bounded HTML excerpts when available;
it is not a full-paper review. Supported official documentation hosts can be inspected; other
sites retain search snippets. Each source inspection is bounded in time and downloaded bytes.

Each validated agent turn, raw model response, research result and final decision is checkpointed.
Resume checks the topic, context hash, models, research configuration and prompt version, then
reuses completed operations. A local process lock prevents concurrent workers for the same review.
An interrupted HTTP request whose response was never saved cannot be recovered from the gateway.

Intermediate turns keep `position` at 800 characters maximum; long-form submission requests
apply to final synthesis, not to this field. Invalid turns get up to two repair calls using
field-specific errors and a bounded copy of the rejected candidate. Repairs share the review
budget. Validated turns are never silently truncated, and failed raw candidates remain auditable.

```bash
arch-council resume --review-id features-council-001
arch-council resume --review-id features-council-001 --status-only
# Explicitly raise the cumulative attempt allowance after budget exhaustion:
arch-council resume --review-id features-council-001 --max-total-calls 16
```

Reports are written to `reports/` (or `--output-dir`) and persisted in SQLite. API credentials are
not stored in the launch manifest. Offline integration tests cover challenge persistence, ownership,
research/critique/resume, partial evidence, bounded prompts, transport retries, repair accounting,
and topic-only CLI execution. Real provider/network availability is not exercised by those tests.

## README summaries

`review --readme-only` supports both council and legacy modes:

- `--readme-summary auto` (default): preserve the complete context up to 8,000 characters;
  above that, generate a reusable structured brief before proposals.
- `--readme-summary always`: generate a brief even for a short README.
- `--readme-summary never`: use the original text. Recommended for wording-level reviews
  of short interview submissions. In council mode, an oversized prompt stops with a
  partial budget-limited report instead of silently dropping the source.
- `--summarizer-model groq/qwen/qwen3.8-27b`: default configured alias; choose another
  provider/model explicitly if needed. Live availability of this alias is not verified.
- `--summary-max-tokens 1200`: output allowance per summary call (256–2000).

Example using JustWoker for every model:

```bash
arch-council review --repo /path/to/project --readme-only \
  --agents 3 --rounds 2 --readme-summary auto \
  --summarizer-model justwoker/claude-opus-4-8 --summary-max-tokens 1200 \
  --model-a justwoker/claude-opus-4-8 \
  --model-b justwoker/claude-opus-4-8 \
  --model-c justwoker/claude-opus-4-8 \
  --arbiter-model justwoker/claude-opus-4-8 \
  --max-total-calls 20 --call-delay-seconds 10 \
  --question "Challenge the design, preserve disagreements, and recommend concrete improvements."
```

The brief records objectives, design, constraints, assumptions, evidence, unanswered
questions, and exact source excerpts. It labels README statements as unverified claims.
Pydantic validates its structure; exact quotes are checked against the source. This is
not proof of semantic completeness: summarization remains lossy. Use `never` for exact
wording review. Agents currently do not have an on-demand original-section retrieval tool.

Original source, raw responses and validated chunk summaries remain in SQLite. The cache
key includes source content, model, output allowance and summary prompt version. Unchanged
documents reuse the brief across reviews. Long documents use 3,000-character sequential
chunks, carrying a bounded brief forward, not accumulated transcripts. An uncached source
therefore needs approximately one summary call per chunk, plus at most one repair per chunk.
This can cost more than no summary for a short review; `auto` avoids it for short inputs.

Council summary calls and transport/repair attempts share the existing persisted call and
input-character budgets. Resume reuses completed chunks and rejects changed source/configuration.
Legacy summary calls share its existing call allowance; legacy resume/accounting limitations
are unchanged. `--disable-summarizer` controls legacy **debate** summaries only; use
`--readme-summary never` to disable the README brief. `chat` is unchanged.

If `--max-context-chars` would truncate the README, review now fails before summarization:
increase that source-loading limit to cover the entire file. The downstream council prompt
limit still applies independently. Its former silent 2,000-character excerpt is removed.
