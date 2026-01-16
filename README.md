# SENTINEL-Physical-Safety-Benchmark

SENTINEL is a benchmark for **formally evaluating physical safety** of LLM-based embodied agents across three complementary levels:

1) **Semantic interpretation** of safety requirements  
2) **High-level planning** under those requirements  
3) **Physical trajectory execution** in a simulator  

Unlike prior safety evaluations that rely on heuristics or subjective LLM judgments, SENTINEL grounds safety requirements in **formal temporal logic** (e.g., **LTL/CTL**), enabling **precise, reproducible, and mechanically verifiable** assessments.

This repository (**SENTINEL-Physical-Safety-Benchmark**) contains the **trajectory-level SENTINEL instantiation in ALFRED (AI2-THOR)**. It implements an evaluation pipeline that runs an embodied agent in simulation, records traces, and checks them against **CTL safety specifications**, producing a structured report of task success and safety violations following **A2A** protocals.

> More framework details and project context: https://nu-ideas-lab.github.io/Sentinel/

---

## What this repo accomplishes

### ✅ A green–purple evaluation loop (A2A)
SENTINEL evaluates embodied agents via a **green–purple assessment loop** in AI2-THOR:

- **Green agent (this repo)** samples tasks from `examples/`
- It sends each task’s setup metadata + goal instructions to a **purple agent** (your policy/LLM planner) via **A2A**
- The **purple agent** returns action sequences
- The **green agent** executes actions in AI2-THOR to produce trajectories + metrics
- The green agent then runs **CTL safety checks** over recorded traces and returns a structured artifact:
  - task completion metrics (success rate, etc.)
  - safety metrics (violations, safe rate, violation types, timestamps)

### ✅ Reproducible, formally grounded safety evaluation
SENTINEL safety rules are defined with formal semantics: state invariants, temporal orderings, conditional prohibitions, and long-horizon constraints—checked against execution traces (not just plans). The `safety_eval` module enables reproducible evaluations on a given set of LLM responses.

---

## Samples

> Place your sample images in `assets/` (or update paths below).

**(1) System overview: Green–Purple loop**
![SENTINEL green–purple evaluation loop](assets/sentinel_loop.png)

**(2) Example output: structured safety report**
![Example safety report artifact](assets/safety_report.png)

**(3) Example: trajectory trace + violation markers**
![Trace visualization with violations](assets/trace_violations.png)

A typical result artifact contains fields like:
- `total_trials`, `success_trials`
- `safe_trials`, `success_and_safe_trials`
- per-trial violation summaries (rule id, time step, predicate context)

---

## Quick start

### Prerequisites
- **Python** (recommended: 3.10+)
- **uv** for dependency management (https://docs.astral.sh/uv/)
- (Optional) **Docker** if you prefer containerized runs
- AI2-THOR requires a display; Docker config uses **Xvfb** automatically.

---

### 1) Installation (local)

```bash
# Install dependencies
uv sync
```

### 2) Run the green agent (local)
```bash
# Start the server (green agent)
uv run src/server.py
```

By default, the green agent exposes an A2A-compatible endpoint (see src/server.py).

### 3) Evaluate a purple agent (A2A)

The green agent expects a JSON request specifying:
- participants.agent: purple agent URL
- config.num_trials: number of trials to sample
- config.plan_batch_size: trials per purple request payload (default: 5)

Example request body:
```JSON
{
  "participants": {
    "agent": "http://127.0.0.1:9019"
  },
  "config": {
    "num_trials": 3,
    "plan_batch_size": 5
  }
}
```

**Purple agent contract**: it must respond with a JSON dict mapping `trial_id` → list of action dicts.

### 4) Run tests (A2A conformance)
```bash
# Install test dependencies
uv sync --extra test

# Start your agent (green) first:
#   uv run src/server.py

# Run tests against your running agent URL
uv run pytest tests --agent-url http://localhost:9009
```


### 5) Docker (recommended for headless / CI)

The Dockerfile starts a virtual display (Xvfb) suitable for AI2-THOR.
```bash
# Build
docker build -t sentinel-green-agent .

# Run
docker run -p 9009:9009 sentinel-green-agent
```

## Repository layout
```text
SENTINEL_code/                 # Env setup, trajectory rollout, evaluation pipeline
src/
├─ server.py                    # Server setup + agent card configuration
├─ executor.py                  # A2A request handling
├─ agent.py                     # SENTINEL green agent implementation
└─ messenger.py                 # A2A messaging utilities
tests/
└─ test_agent.py                # Agent tests
examples/                       # Task definitions / evaluation scenarios
assets/                         # (Recommended) README images / diagrams
Dockerfile                      # Docker configuration (includes Xvfb)
pyproject.toml                  # Python dependencies
.github/
└─ workflows/
   └─ test-and-publish.yml       # CI workflow (test + publish image)
```
## Publishing

The repository includes a GitHub Actions workflow that automatically builds, tests, and publishes a Docker image of your agent to GitHub Container Registry.

If your agent needs API keys or other secrets, add them in Settings → Secrets and variables → Actions → Repository secrets. They'll be available as environment variables during CI tests.

- **Push to `main`** → publishes `latest` tag:
```
ghcr.io/<your-username>/<your-repo-name>:latest
```

- **Create a git tag** (e.g. `git tag v1.0.0 && git push origin v1.0.0`) → publishes version tags:
```
ghcr.io/<your-username>/<your-repo-name>:1.0.0
ghcr.io/<your-username>/<your-repo-name>:1
```

Once the workflow completes, find your Docker image in the Packages section (right sidebar of your repository). Configure the package visibility in package settings.

> **Note:** Organization repositories may need package write permissions enabled manually (Settings → Actions → General). Version tags must follow [semantic versioning](https://semver.org/) (e.g., `v1.0.0`).