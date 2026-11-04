# k8s-aiops-agent

An AI agent that investigates a local Kubernetes cluster, finds the root cause of a problem, backs
every claim with evidence from the tool call that produced it, and recommends the safest next
action. For one fault (a Service selector mismatch) it can also apply the fix after explicit human
approval, then verifies the result with fresh evidence.

Example run (abridged):

```
$ aiops investigate "Why is the checkout service unhealthy?" -n scenario-01

trace 7c1f…
└── investigate 9.8s
    ├── model.call 2.1s 6120+410 tok
    ├── tool.list_pods t1 {"namespace": "scenario-01"}
    ├── tool.list_deployments t2 {"namespace": "scenario-01"}
    ├── model.call …
    ├── tool.get_pod_logs t3 {"namespace": "scenario-01", "name": "checkout-…", "previous": true}
    ├── tool.get_configmap t5 {"namespace": "scenario-01", "name": "checkout-config"}
    └── validate_diagnosis
╭─ Diagnosis ─────────────────────────────────────────────────────────────────╮
│ bad_configuration (confidence 95%)                                          │
│ Root cause: ConfigMap checkout-config sets PAYMENT_TIMEOUT_MS to "30s", but │
│ checkout expects integer milliseconds, so it exits and crash loops.         │
│ Evidence                                                                    │
│   [t1] pod is crash looping          “CrashLoopBackOff”                     │
│   [t3] exits on config parse error   “FATAL config: PAYMENT_TIMEOUT_MS=…”   │
│   [t5] value comes from ConfigMap    “"PAYMENT_TIMEOUT_MS": "30s"”          │
│ Recommended action (risk: low): set it to "30000", then rollout restart.    │
╰─────────────────────────────────────────────────────────────────────────────╯
```

## Contents

- [Setup](#setup)
- [Run the agent](#run-the-agent)
- [Reproduce the failure scenarios](#reproduce-the-failure-scenarios)
- [Run the evaluations](#run-the-evaluations)
- [Observability](#observability)
- [Remediation with approval](#remediation-with-approval)
- [CI](#ci)
- [Project layout](#project-layout)
- [Design rationale](#design-rationale)
- [Trade-offs and next steps](#trade-offs-and-next-steps)

## Setup

Requirements: **Docker**, **kind** and **kubectl**. Everything else runs in containers. For local
development without Docker you also need [uv](https://docs.astral.sh/uv/).

```bash
make cluster-up          # kind cluster "aiops" + least-privilege agent RBAC + agent kubeconfigs
cp .env.example .env     # then set ANTHROPIC_API_KEY and AGENT_LLM_PROVIDER=anthropic
                         # (or GEMINI_API_KEY and AGENT_LLM_PROVIDER=gemini: free tier in Google AI Studio)
```

`make cluster-up` is idempotent. The agent never uses your admin kubeconfig: it gets two
ServiceAccount kubeconfigs in `.kube/` (mode 0600, gitignored). Their tokens last 24h; rerun
`make cluster-up` to refresh them. kind clusters do not survive a Docker/WSL restart; rerun it
then too.

All configuration comes from environment variables (`AGENT_*`, see `.env.example`). Without an API
key the agent runs with `AGENT_LLM_PROVIDER=scripted`, an offline test double (see
[evaluations](#run-the-evaluations)).

## Run the agent

In a container (only Docker needed):

```bash
docker compose build agent
docker compose run --rm agent investigate "Why is the checkout service unhealthy?" \
  -n scenario-01 --backend live
```

The compose service joins the `kind` Docker network, mounts `.kube/` read-only and runs as your
host uid so the private kubeconfigs stay private (override with `AGENT_UID`/`AGENT_GID` if yours
is not 1000).

Or locally with uv:

```bash
uv sync
uv run aiops investigate "The catalog service returns no data. What is wrong?" -n scenario-02 --backend live
uv run aiops investigate "..." -n scenario-02 --json   # machine-readable result
uv run aiops trace <trace_id>                          # replay a run's timeline later
```

`--backend fixture` replays recorded cluster snapshots instead of talking to the cluster, so the
agent also works without kind. The result is always the same structure (`InvestigationResult`):
`fault_found`, `category`, `root_cause`, `evidence[]` (`claim`, `tool_call_id`, verbatim
`excerpt`), `recommended_action`, `action_risk`, `confidence`, run stats (steps, tool calls,
tokens, latency, stop reason, trace id) and the full evidence ledger.

**Exit codes:** `0` for a completed run, including an inconclusive diagnosis; `2` for bad
configuration, such as an invalid setting or an unreadable kubeconfig. A tool call outside the
namespace allowlist does not stop the run: it is refused and recorded as a tool error. Model or API failures do not
crash the run: they end in an explicit inconclusive diagnosis with `stop_reason=model_error`.

## Reproduce the failure scenarios

Each scenario is a directory in `scenarios/` with `manifests.yaml` (what to apply), `scenario.yaml`
(the question, the expected result and a reference answer) and `fixture.json` (a recorded
snapshot of the namespace).

```bash
make scenario-up S=01     # apply one scenario (00-05); wait ~60s for restarts and events
make scenarios-down       # remove all scenario workloads
make fixtures             # re-record every fixture.json from the live cluster
```

| # | Scenario | Symptom | What the agent has to connect |
|---|---|---|---|
| 00 | healthy (control) | nothing wrong | must say *no fault* instead of inventing one |
| 01 | crashloop-bad-config | CrashLoopBackOff | pod state + previous logs (`FATAL … "30s"`) + the ConfigMap value (**multi-source**) |
| 02 | readiness-wrong-port | Running but 0/1 Ready | probe-failure events (`:8080 … refused`) + the deployment's probe vs. container port |
| 03 | service-selector-mismatch | healthy pods, Service has no endpoints | Service selector `app=frontend-v2` vs. pod label `app=frontend` (**remediation target**) |
| 04 | image-pull-error (held out) | ImagePullBackOff | events + image tag `1.27-alpne` |
| 05 | missing-configmap (held out) | CreateContainerConfigError | events + the referenced ConfigMap does not exist (the 404 is evidence) |

Scenarios 04 and 05 are **held out**: they were never used while writing or tuning the prompt and
are reported separately, as a check that the agent generalizes beyond the scenarios it was built
against. Nothing in the agent's code, tools or prompt mentions any scenario.

## Run the evaluations

```bash
make eval EVAL_ARGS="--llm scripted --backend fixture --min-pass-rate 1.0"   # offline, free (CI gate)
make eval EVAL_ARGS="--llm anthropic --backend fixture --trials 3 --judge"   # real model + Claude judge
make eval EVAL_ARGS="--llm gemini --backend fixture --trials 3 --judge"      # free-tier Gemini, judged by Gemini
make eval EVAL_ARGS="--llm anthropic --backend live --trials 1"              # full end to end
docker compose run --rm agent eval --llm anthropic --trials 3 --judge         # same, in Docker
```

Reports are written to `evals/results/<backend>-<llm>.md` (and `.json`), separating tuning and
held-out scenarios. The committed report is the [offline gate](evals/results/fixture-scripted.md).
The Claude report (3 trials plus the judge) comes from the second command above or from the manual
`real-llm-eval` CI job, which uploads it as the `eval-claude` artifact.

### Evaluation strategy

Every run is graded **programmatically**; an LLM judge adds a rubric score on top.

| Question | How it is checked |
|---|---|
| Is the diagnosis correct? | `fault_found` and `category` match the scenario's expected values; affected resources are named. The judge scores correctness against the reference answer (1-5), so a right category with a wrong cause still fails. |
| Is it supported by relevant evidence? | Every cited id exists in the ledger and every excerpt is verbatim from that tool output (also enforced at runtime). The cited outputs must contain the scenario's key signal (e.g. `30s`), and the required tools must have been called (trajectory check). The judge scores evidence support. |
| Is the action appropriate and safe? | A denylist rejects destructive or permission-widening actions (`kubectl delete namespace`, `--force`, disabling probes, `cluster-admin`, …) and `action_risk=high`. The judge scores action safety. |
| Did it stay within bounds? | The run finished with a submitted diagnosis, within the step and token budget. |

A trial passes only if every deterministic check passes and every judge score is at least 4.
Several trials per scenario report a pass rate, because model runs are not deterministic. Scores
are attached to the run's Langfuse trace, so any failure links to the exact execution.

**Why this mix:** deterministic checks are cheap, reproducible and gate CI. The judge catches what
string checks cannot ("right category, wrong reason"). Grounding is checked against the ledger
rather than the model's own claims, so a confident but unsupported answer cannot pass.

The scripted model (`evals/scripted/*.yaml`) replays a plausible good investigation per scenario.
It is **only a test double**: it proves the graph, tools, validation and eval harness work
offline in CI. Real quality numbers come from `--llm anthropic`.

## Observability

Every run is one OpenTelemetry trace: `investigate` (agent) → `model.call` (generation: model,
input/output, token usage, finish reason, latency) → `tool.<name>` (tool: arguments, output,
evidence id, latency, errors) → `validate_diagnosis` (evaluator) → `remediation.*`. Together these
cover what the agent did, which tools it used with their inputs and outputs, failures, latency,
and how it reached its conclusion: the final generation, its validation, and the evidence ids it
cites back to tool spans.

- **Always on:** spans are written to `traces/<trace_id>.jsonl` and printed as a timeline after
  each run (`aiops trace <id>` replays one).
- **Langfuse (optional, self-hosted):**

  ```bash
  docker compose --profile observability up -d     # Langfuse v4 on http://localhost:3000
  # in .env: LANGFUSE_PUBLIC_KEY=pk-lf-local  LANGFUSE_SECRET_KEY=sk-lf-local
  ```

  Log in with `admin@example.com` / `change-me-locally` (local defaults; override them in `.env`).
  The project and API keys are created headlessly on first start. Traces go over OTLP to
  `/api/public/otel`, with the observation types Langfuse renders (agent, generation, tool,
  evaluator), and eval scores attach to the same traces. Langfuse Cloud works by setting
  `LANGFUSE_HOST` and real keys.

The instrumentation is plain OpenTelemetry. Langfuse is only one exporter, so any OTLP backend
works.

## Remediation with approval

```bash
uv run aiops investigate "Requests to frontend time out. Why?" -n scenario-03 --backend live --allow-remediation
```

When the diagnosis is a Service selector mismatch and carries a proposed selector, the graph:

1. **Proposes:** re-reads the Service and pods, and refuses to propose unless the new selector
   matches at least one Ready pod and differs from the current one. It then shows the current
   selector, the proposed selector, the pods it will route to, and the expected effect.
2. **Pauses for approval:** LangGraph `interrupt()`. Nothing changes unless a human types `y`, and
   the default is no.
3. **Applies through one narrow tool:** a JSON patch that first `test`s that `spec.selector` is
   still the one the human approved, then `replace`s it. If anyone changed the Service in the
   meantime, the API server rejects the whole patch. It uses a separate remediator identity that
   RBAC only allows to `get`/`patch` Services in `scenario-03`.
4. **Verifies with fresh evidence:** polls the Service's endpoints with the read-only identity
   until the selected pods are ready endpoints, and reports `resolved` with that evidence (ids
   `v…`).

## MCP

`aiops mcp` serves the same read-only tools over MCP (stdio), marked `readOnlyHint`, behind the
same allowlist, redaction and RBAC identity. Other agents or IDEs can use the investigation tools
directly. No write tool is exposed.

## CI

`.github/workflows/ci.yml`:

| Job | What it proves |
|---|---|
| checks | ruff, ruff format, strict mypy, pytest (unit + offline agent tests) |
| eval gate | `aiops eval --llm scripted --backend fixture --min-pass-rate 1.0` on all scenarios |
| container build | the image builds, and the eval gate passes inside it |
| kind e2e | real cluster: RBAC, live tools (`pytest -m live`), and approved remediation resolving scenario 03 |
| Claude eval (manual) | `workflow_dispatch` with the `ANTHROPIC_API_KEY` secret: 3 trials per scenario + judge |

## Project layout

```
src/k8s_aiops_agent/   the package (src layout): one module per job
  agent.py             LangGraph loop: model ⇄ tools, budgets, validation, approval-gated fix
  tools.py             the tool surface: typed, read-only, guarded, recorded in the evidence ledger
  kube.py              read-only cluster access (Kubernetes Python client)
  fixture_kube.py      record a namespace, replay it offline behind the same interface
  remediation.py       the one write action: replace a Service selector
  guards.py            namespace allowlist, redaction, text normalization
  llm.py               Claude, Gemini and the scripted test double behind one interface
  prompts.py           system and control prompts
  schemas.py           the Diagnosis and InvestigationResult contracts
  evals.py             deterministic checks, LLM judge, reports
  telemetry.py         OpenTelemetry spans to local JSONL and Langfuse
  mcp_server.py        the read-only tools over MCP
  cli.py, config.py    the `aiops` CLI and environment-based settings
tests/                 unit and offline agent tests (+ `-m live` against kind)
scenarios/<id>/        manifests, expected result and recorded fixture per scenario
evals/                 scripted model turns and committed eval reports
deploy/, scripts/      kind config, RBAC, kubeconfig minting, fixture recording
```

The package is flat on purpose. At about 2.3k lines, each module does one job and is found by
name. Subpackages (for example `llm/`, `kube/`, `evals/`) are the next step if a module group
outgrows a single file.

## Design rationale

### Framework and orchestration: LangGraph + the Anthropic SDK

The loop is a small explicit `StateGraph` rather than a prebuilt ReAct agent, because the parts
that matter for an operations agent are control flow, not prompting. They are graph nodes you can
read and test:

- `agent ⇄ tools` until the model calls `submit_diagnosis`
- `validate`: schema, cited ids and verbatim excerpts, with one repair attempt
- budget routing to `finalize`, then a deterministic `fallback`
- `propose → approve (interrupt) → apply → verify` for the one fix

LangGraph's checkpointer and `interrupt()` give human-in-the-loop approval without custom plumbing.
Claude is called through the official Anthropic SDK rather than a LangChain wrapper, so it uses
the current API directly:

- `claude-opus-5-5` with explicit `effort`
- server-side refusal fallbacks
- prompt caching
- thinking blocks passed back unchanged (append-only history)

A `ChatModel` interface keeps a scripted test double behind the same seam, plus a Gemini adapter
(OpenAI-compatible endpoint, plain httpx) that translates the same message history at the edge.

**Stopping conditions** (`AGENT_*`): max steps (12), total token budget, wall-clock timeout, and
repeated identical tool calls. Hitting one triggers a single finalize turn ("submit with the
evidence you have"). If that fails too, the result is an explicit *inconclusive* diagnosis with
confidence 0 and the stop reason, never an unbounded loop or a crash. LangGraph's recursion limit
is a hard backstop below that.

### Tool boundaries

- **Nine typed, read-only tools.** Three discovery tools (`list_pods`, `list_deployments`,
  `list_services`) and six detail tools (`describe_pod`, `get_pod_logs`, `get_events`,
  `get_deployment`, `get_service_and_endpoints`, `get_configmap`). They use the Kubernetes Python
  client: no shell, no `kubectl`, no arbitrary API paths.
- **Compact output.** Tools return summaries rather than raw objects, so the model sees signal,
  not `managedFields`.
- **Guarded execution.** Arguments are validated with Pydantic, namespaces are checked against an
  allowlist (deny by default), output is redacted for secret-like values and truncated to keep
  the head and the tail, and API, connection and timeout errors come back as data, not
  exceptions.
- **Untrusted output.** Each result is wrapped as `evidence_id: tN` plus `<tool_output>`, and the
  prompt treats its contents as data, never as instructions.
- **Evidence ledger.** Every call is recorded with its id, arguments, output, error and latency.
  Diagnoses cite these ids, and runtime validation and the evals check them.
- **Errors are evidence.** For example, the 404 for a referenced ConfigMap in scenario 05.

### Evaluation approach

See [evaluation strategy](#evaluation-strategy): deterministic correctness, grounding, trajectory,
action-safety and budget checks on every run, plus a Claude judge against a reference answer,
multiple trials, and held-out scenarios reported separately.

### Safety boundary

The real boundary is **RBAC**, not the prompt. Each layer below assumes the one above can fail:

1. **Read-only identity:** `aiops-agent-readonly` can `get`/`list` pods, logs, events, services,
   endpoints, EndpointSlices, ConfigMaps and deployments, only in the scenario namespaces. It can
   never read Secrets. This is verified with `kubectl auth can-i` and a live test.
2. **No general execution:** the allowlisted, typed tools are the only way to reach the cluster.
   There is no shell or `kubectl` passthrough. The namespace allowlist is enforced in code too.
3. **Untrusted tool output:** output is delimited, secret-like values are redacted, and output is
   truncated.
4. **One write action, behind approval:** a separate identity that can only patch Services in
   `scenario-03`. Before asking, precondition checks run on fresh reads. The human sees the
   change and its expected effect, defaulting to no. Afterwards, verification uses fresh evidence.
5. **Safety in evals:** a denylist plus a judge score for every recommended action.

## Trade-offs and next steps

- **Fixtures are snapshots.** They make evals deterministic and free of cluster dependencies, but
  time-dependent behaviour such as restart counts and event ages is frozen. The live eval mode
  covers the rest.
- **Token budget counts cached input at full size.** Simple and conservative. A cost-weighted
  budget would let longer investigations through. Sizing: the first call is about 2.1k tokens
  (prompt plus tool schemas), and each tool round adds about 0.1–1.3k. The scripted runs total
  5–10k, so the 60k default only cuts in around turn 9–10, before `max_steps`, and then forces
  a final answer instead of failing. Calibrate it against real-Claude eval token counts.
- **Cluster discovery is intentionally narrow.** There are no nodes, PVCs, jobs, ingresses or
  metrics yet. Each would be one more typed tool plus RBAC rule, and a new scenario to prove it.
- **The judge uses the same model family as the agent.** Using a different judge model, or human
  calibration on a sample, would reduce shared blind spots.
- **More scenarios.** More scenarios with several plausible causes (for example OOMKilled vs. a
  bad config, or a failing dependency) would stress the agent's reasoning more than
  single-cause faults do.
