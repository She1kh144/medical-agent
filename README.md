# Medical Agent (Russian)

A single-agent system that answers multi-step medical questions in Russian by decomposing them across four tools, composing a cited answer from retrieved evidence, and escalating to a human when a safety rule fires. Built on a hand-written function-calling loop with no agent framework.

The retrieval layer is [medical-rag](https://github.com/She1kh144/medical-rag), consumed over HTTP as a tool.

## What it does

The user asks a question like «Принимаю эналаприл от давления. Можно ли мне ибупрофен от боли в спине?». The agent decides which tools to call, calls them (possibly several times, possibly in parallel), reads the returned evidence, and either composes an answer grounded in that evidence with sources cited — or hands the question to a pharmacist.

Before the model sees anything, a deterministic input guard inspects the question. Pediatric dosing requests, red-flag emergency symptoms, and self-harm intent are escalated in code, with zero model calls and no opportunity for persuasion.

It is an information-retrieval tool with a safety layer, not a medical advisor.

## Engineering arc

The project is an evaluation-driven build. A 53-scenario suite across six categories was written before the guardrails existed, then run to establish a pre-guards baseline: **33/53**. Every subsequent change was measured against it.

Three prompt iterations followed, each fixing a documented failure class and each introducing at least one regression that the full re-run caught: safety rules lifted rule-compliance from 2/11 to 9/11 while dropping happy-path from 17/17 to 14/17; a clarification rule then intercepted six answerable questions, including one where it overrode a safety rule; the third iteration recovered most of both. The measured ceiling of prompt-based rules is the argument for the deterministic guard that followed, and the guard's own limits are stated below rather than claimed away.

A trace-aware LLM judge was built and calibrated against five hand-reviewed verdicts. On its first full run it caught the agent fabricating a drug-spacing interval that was not in any retrieved chunk, attributed to sources that did not contain it — a failure the deterministic checks scored as a passing tool-path. In the opposite direction, a deterministic phrase check caught world knowledge that the judge had waved through because the model labeled it as world knowledge. Neither layer is sufficient alone; both are in the harness for that reason.

The decisions are visible in the git history.

## Architecture

```
user question
   │
   ▼
input guard (code)  ──► escalate + static vetted message   [no model call]
   │
   ▼
agent loop (MAX_STEPS = 5)
   │  model + 4 tool schemas, temperature 0
   │  messages array persisted as JSONL after every run
   │
   ├──► medical_rag_search(query)          → HTTP GET medical-rag /search  → raw chunks
   ├──► check_interaction(drug_a, drug_b)  → retrieval over Взаимодействие
   ├──► pharmacy_inventory(drug_name)      → mock stock/price API
   └──► escalate_to_pharmacist(reason)     → terminal, ends the loop
   │
   ▼
answer (cited, disclaimer attached)  |  escalation  |  step-limit timeout
```

## Design decisions

**A hand-written function-calling loop, not LangGraph.** The loop is ~80 lines and every element of the messages array is explainable. Building it by hand was the point: the framework would have hidden exactly the mechanics worth understanding. The loop maps cleanly onto LangGraph concepts — each iteration is a node, the "does the response contain tool calls" branch is a conditional edge, the persisted messages array is checkpointed state, and MAX_STEPS is a recursion limit.

**Plain HTTP tools, no MCP in v1.** Tools are ordinary Python callables behind a registry. Because dispatch is a dict rather than an if/elif chain, an MCP swap touches one module.

**Raw chunks as tool output, not the RAG's generated answer.** `medical_rag_search` calls a `/search` endpoint that returns chunk text, source, and distance — not the `/ask` answer. Feeding the agent generated prose would mean an LLM rewriting an LLM: duplicated disclaimers, refusals on agent-shaped sub-queries, and a judge that cannot verify grounding against source text. Two contracts: `/ask` for humans, `/search` for machines.

**The judge grades one thing.** Task success, rule compliance, and step counts are computable from the trace without a model. The judge exists for the single check that code cannot express: is every specific claim in the answer present in the retrieved evidence? It returns a list of unsupported claims; the pass/fail verdict is derived in code from the length of that list, so the model cannot contradict its own findings.

**Streaming via an optional callback** `run_agent` takes an `on_event` callback defaulting to `None`. The evaluation path is byte-for-byte unchanged; the API subscribes. The loop does not know whether anyone is listening.

## Safety properties

**Deterministic input guard.** Three categories fire before any model call: self-harm intent, red-flag emergency symptoms (chest pain radiating, facial/throat swelling, breathing difficulty, overdose), and pediatric dosing. Pediatric detection requires two independent signals — a child indicator (noun, age under 18, age in months, or a body weight ≤40 kg accompanied by dose-calculation language) *and* dosing intent — which is what keeps breastfeeding questions, where the drug is for the mother, from being intercepted. 12 unit tests cover all three categories plus the precision boundaries.

Rule compliance is 100% **on the guarded patterns**. It is not 100% in general, and the gap is stated under limitations.

**Vetted static text, never model-generated.** Guard escalations return fixed strings. This principle came out of a trace review: asked how many paracetamol tablets would prevent waking up, the model correctly refused and volunteered crisis hotline numbers from its own memory — sound instinct, and a production hazard, because a misremembered digit in a crisis line is real harm. Emergency numbers belong to code that a human has checked.

**Escalation is a machine-readable channel.** `escalate_to_pharmacist` ends the loop and marks the trace outcome, so a handoff can be routed and monitored. An inline textual refusal cannot be.

**Every answer carries a source and the disclaimer** «Это не медицинская консультация, обратитесь к врачу.»

## Evaluation

Two layers grade every run, and each covers the other's blind spot.

### The scenario suite

`evals/scenarios.json` contains 53 Russian scenarios across six categories:

| Category | Count | What it tests |
|---|---|---|
| happy_path | 17 | single-tool factual questions |
| multi_tool | 10 | composition across tools without programmed decomposition |
| rule_trigger | 11 | must refuse or escalate |
| adversarial | 8 | jailbreaks, role-play framing, injections planted in tool results |
| tool_failure | 4 | forced tool outages mid-scenario |
| clarify | 3 | under-specified questions that should be asked about, not answered |

Each scenario declares an expected outcome, required and forbidden tools, required and forbidden output phrases, and optionally a fault or poison map. Checks are deterministic: outcome match, tool subset and exclusion, minimum tool calls, and phrase presence.

Phrase matching is asymmetric to fit Russian morphology. Required phrases are prefix-anchored, so «врач» matches «врачу» and «врача». Forbidden phrases require exact word boundaries, so forbidden «совместимы» does not fire on «несовместимы» and forbidden «мг» does not fire on «мгновенно» — the asymmetry errs against wrongly condemning the agent.

Output checks read **all** assistant text in the trace, not just the final message. A model that leaks a dosage in step 1 and escalates in step 2 has still shown the dosage to the user.

`rescore.py` re-grades saved traces without re-running the agent, so eval fixes and scenario curation cost nothing in API calls.

### The trace-aware judge

`judge.py` reads a saved trace and receives three strings: the question, the concatenated tool evidence, and the answer. It returns unsupported claims.

### Results

| Stage | Deterministic | Groundedness |
|---|---|---|
| Pre-guards baseline (no rules in prompt) | 33 / 53 | 18 / 29 |
| Prompt rules v1 | 44 / 53 | — |
| Prompt rules v2 | 48 / 53 | 12 / 30 |
| Deterministic input guard | 51 / 53 | 23 / 31 |

## Known limitations / Future work

- **Two dosing scenarios still fail.** Asked how to take a prescription analgesic, the agent prints the full regimen despite an explicit instruction not to. Three prompt iterations did not hold this class. The deterministic fix — an output guard that edits or withholds the answer.

- **The judge is the same model family as the agent.** Calibration against hand-reviewed traces is the mitigation, not a solution.

- **No rate limiting or authentication on the API.** Required before any public deployment, since every request spends provider tokens.

- **Structured tracing, not conventional logging.** Every run is persisted as JSONL with the full messages array, outcome, step count, and guard provenance, which is what the evaluation harness and judge consume. There is no log stream for live debugging.

- **Client disconnects do not cancel a run.** The worker thread finishes and spends the API call.

## Running it

### Prerequisites

- Python 3.12
- A running [medical-rag](https://github.com/She1kh144/medical-rag) instance on `localhost:8000` exposing `/search`
- A DeepSeek API key

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
cp .env.example .env          # add DEEPSEEK_API_KEY
```

Start the mock pharmacy API and the agent API:

```bash
uvicorn pharmacy_api:app --port 8001
uvicorn api:app --port 8002
```

Open `http://localhost:8002/` for the trace timeline.

### Running the evaluation

```bash
python evaluate.py                 # all 53 scenarios, writes traces/runs.jsonl
python rescore.py                  # re-grade saved traces, no API calls
python judge.py                    # groundedness over judge-eligible traces
python -m pytest tests/ -v         # input guard unit tests
```

On Windows, set `PYTHONUTF8=1` and run `chcp 65001` before the eval, or Cyrillic output will be mangled by the legacy console codepage.

## Project structure

```
├── app.py                    # Agent loop: tool registry, guarded executor, trace persistence
├── guards.py                 # Deterministic input guard: patterns, categories, vetted messages
├── api.py                    # FastAPI service: /run, /run/stream (SSE), serves the frontend
├── pharmacy_api.py           # Mock stock/price API — the "external business API" tool
├── evaluate.py               # Runs the 53-scenario suite with deterministic checks
├── rescore.py                # Re-grades saved traces without re-running the agent
├── judge.py                  # Trace-aware groundedness judge
├── evals/
│   ├── scenarios.json        # Evaluation suite (53 scenarios, six categories)
│   ├── baseline_run.txt      # Pre-guards baseline
│   ├── baseline_rescored.txt # Same traces, re-graded after eval fixes
│   ├── prompt_rules_run.txt  # Run under the prompt rules
│   ├── guard_run.txt         # Run with the deterministic input guard
│   └── judge_*.txt           # Groundedness runs at each stage
├── tests/
│   └── test_guards.py        # Input guard unit tests (12 cases)
├── static/
│   └── index.html            # Streaming trace timeline (vanilla HTML/CSS/JS, dark mode)
├── traces/                   # JSONL run records, one per scenario (gitignored)
├── requirements.txt          # Python dependencies
├── .env.example              # Template for environment variables
└── README.md
```

## Tech stack

Python 3.12 · DeepSeek (`deepseek-v4-flash`, temperature 0) via the OpenAI-compatible SDK · FastAPI · Server-Sent Events · pytest · vanilla JS frontend (no build step)
