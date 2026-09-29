# ArchCouncil

ArchCouncil is a production-oriented Python CLI for bounded, evidence-grounded
architecture and design reviews. Multiple LLM roles discuss a question through a
structured claim ledger, challenge one another, optionally request external evidence,
and produce a final recommendation with dissent and validation experiments preserved.

It supports reviews such as feature-engineering designs, data and ML architectures,
customer-discovery proposals, and implementation choices before a prototype is built.
It is a decision-support tool: model-generated recommendations are not production
validation, and downloaded research code is never executed.

## Workflow

```text
Question + bounded repository context
                 |
                 v
        independent proposals
                 |
                 v
       shared claims and objections
                 |
                 v
       targeted research on request
                 |
                 v
          bounded debate rounds
                 |
                 v
        convergence or round limit
                 |
                 v
          final decision report
```

The council stores structured state instead of repeatedly passing full transcripts. Each
turn can contain a position, claims, challenges, revisions, resolutions, evidence requests,
reviews, and proposed experiments. Claims and objections receive stable IDs, so later agents
can respond to specific reasoning.

## Installation

Requires Python 3.11 or newer.

```bash
git clone https://github.com/meisamgh/arch-council.git
cd arch-council
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
```

Add credentials for the provider you intend to use. Never commit `.env` or real API keys.
Provider availability and model names depend on the configured gateway account.

## Commands

Initialize an immutable, hashed README project profile:

```bash
arch-council init --repo "/path/to/project"
```

Run the default two-agent council, or add a third specialist:

```bash
arch-council review \
  --repo "/path/to/project" \
  --readme-only \
  --agents 3 \
  --rounds 2 \
  --model-a justwoker/claude-opus-4-8 \
  --model-b justwoker/claude-opus-4-8 \
  --model-c justwoker/claude-opus-4-8 \
  --arbiter-model justwoker/claude-opus-4-8 \
  --max-total-calls 18 \
  --max-prompt-chars 16000 \
  --max-output-tokens 1200 \
  --question "Evaluate the architecture, identify important risks, and recommend the smallest trustworthy next step."
```

`--readme-only` shares only the root README. Use the full repository or `--diff-base` when
implementation evidence is needed. For exact wording reviews of short documents, use:

```bash
arch-council review \
  --repo "/path/to/project" \
  --readme-only \
  --readme-summary never \
  --question "Review this submission for completeness, clarity, and practical feasibility."
```

Large READMEs use `--readme-summary auto` by default. The summary is structured, validated,
cached, and reused. Select its model with `--summarizer-model` and its output limit with
`--summary-max-tokens`. Use `--readme-summary never` when exact wording matters.

Resume an interrupted review or inspect its status:

```bash
arch-council resume --review-id REVIEW_ID
arch-council resume --review-id REVIEW_ID --status-only
arch-council resume --review-id REVIEW_ID --max-total-calls 24
arch-council history
```

Completed operations are reused. Changing the question, source context, models, or review
configuration requires a new review ID so incompatible evidence cannot be mixed.

Interactive single-agent chat is also available:

```bash
arch-council chat --repo "/path/to/project" --readme-only \
  --model justwoker/claude-opus-4-8
```

## Discussion Roles

The council has roles, not necessarily different underlying models:

| Role | Responsibility |
| --- | --- |
| Proposer | Build a practical initial position and claims |
| Critic | Test assumptions, feasibility, cost, and evidence |
| Specialist | Add an alternative perspective or missing risks |
| Decision writer | Produce the final recommendation and preserve dissent |

Three roles may use the same model with different instructions, or explicit provider/model
aliases can be assigned per role. The roles interact through structured state rather than
generating unrelated essays.

## Research

Research is disabled by default. When enabled, an agent must request evidence for an existing
claim and explain why it could change the decision.

```bash
arch-council review \
  --repo "/path/to/project" --readme-only --research \
  --research-provider searxng --research-queries 3 \
  --research-results 2 --evidence-inspections 4 \
  --question "Compare this design with relevant papers, official documentation, and open-source systems."
```

Research is bounded and selective. SearXNG or Tavily can provide search results; GitHub
inspections pin a commit and read a small number of files; paper inspections use bounded
excerpts; official documentation is preferred for API and library behavior; failed sources
are recorded as unavailable or partial evidence; downloaded code is not executed.

For a short interview or customer-discovery submission, research is usually unnecessary.
It is more useful when a claim depends on a particular paper, repository, API behavior, or
established method.

## Cost and Token Controls

Reviews enforce maximum physical LLM attempts, prompt characters per call, cumulative input
characters, output tokens per call, research queries, and source inspections. They also use
deterministic structured state, cached summaries and research, SQLite checkpoints, and early
convergence when important objections are resolved.

Two rounds are the normal starting point. One round gives a quick critique; three rounds are
useful for unresolved high-impact disagreements. More rounds increase cost and do not guarantee
a better decision. Character and call limits are not exact monetary estimates because gateways
may add hidden context or report provider-specific usage.

## Provider Configuration

Provider-specific aliases keep credentials separate. A JustWoker configuration is:

```env
ARCH_COUNCIL_PROVIDER=justwoker
JUSTWOKER_API_KEY=replace_me
JUSTWOKER_BASE_URL=https://api.justwoker.icu
ARCHITECT_A_MODEL=claude-opus-4-8
ARCHITECT_B_MODEL=claude-opus-4-8
ARCHITECT_C_MODEL=claude-opus-4-8
```

The JustWoker client uses the Anthropic-compatible `/v1/messages` endpoint. Other configured
providers include SeekAI, VyceAI, AgentRouter, Gemini, Groq, True-SOTA, and SearXNG research.
Use an explicit alias such as `justwoker/claude-opus-4-8` when provider selection must be clear.

## Project Structure

```text
src/arch_council/
├── cli.py                 # init, review, resume, history, and chat commands
├── config.py              # environment-based provider configuration
├── agent.py               # interactive single-agent chat
├── client.py              # provider clients, retries, repairs, and routing
├── context.py             # README, repository, and Git-diff context builders
├── council.py             # structured claim ledger and bounded council runtime
├── council_cli.py         # council orchestration and report persistence
├── readme_summary.py      # cached, validated README briefs
├── debate.py              # legacy multi-stage debate workflow
├── governance.py          # claim and feature governance helpers
├── prompts.py             # legacy prompts
├── research.py            # search providers and research cache
├── research_tools.py      # bounded paper, GitHub, and documentation inspection
├── models/                # shared Pydantic models
├── pipeline/              # validation and reduction helpers
└── runtime/               # persistence, jobs, artifacts, operations, concurrency
```

Runtime state is stored under `.arch-council/`, reports under `reports/`, and credentials are
read from `.env`. These local files are ignored by Git.

## Validation

```bash
pytest -q
ruff check src tests justdowork.py
```

The tests cover structured discussion, claim ownership, malformed-output repair, transport and
attempt budgets, resume behavior, README summary caching, research degradation, provider routing,
and bounded prompts. Offline tests do not prove that a live provider account, model, network route,
or external research service is available.

## Scope

The current release focuses on evidence-grounded discussion and decision support. It does not
yet implement architecture drift validation, maturity scoring, a general rules engine, static
code analysis, CI integration, or pull-request comments.

The feature-research material that previously occupied this README remains a useful review
subject. Ask ArchCouncil to challenge feature novelty, leakage controls, deterministic validation,
out-of-sample experiments, and production risks. Those are review topics, not claims that
ArchCouncil itself implements a feature-generation platform.

## License

MIT
