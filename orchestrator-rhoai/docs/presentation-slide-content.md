# ToolOrchestra — Slide Content (What Goes ON Each Slide)

> Use alongside `presentation-talking-points.md` (speaker notes).
> Each section below = one slide. Keep slides clean — most detail is spoken, not shown.

---

## SLIDE 1 — Title

```
ToolOrchestra
Elevating Intelligence via Efficient
Model and Tool Orchestration

Implementing NVIDIA's #1-ranked agentic system
on Red Hat OpenShift AI

arxiv.org/abs/2511.21689
NVIDIA · University of Hong Kong
```

**Visual**: Paper title + NVIDIA / Red Hat logos side by side. Clean, dark background.

---

## SLIDE 2 — The Problem

**Headline**: *"One model does everything — at 2.5x the cost"*

Two-column layout:

| Option A: Single Large Model | Option B: Many Small Models |
|---|---|
| GPT-5 handles all tasks | Math model, Code model, Search model |
| Simple but expensive | Cheap but who decides which to use? |
| No coordination needed | Needs an intelligent coordinator |

**Bottom callout box**:
> Can a small 8B model learn to coordinate larger specialists and beat GPT-5 at a fraction of the cost?

---

## SLIDE 3 — The Answer

**Headline**: *"Yes — and the numbers prove it"*

Three result cards (large, bold numbers):

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│      HLE         │  │  FRAMES / τ²    │  │      GAIA       │
│                  │  │                  │  │                  │
│  37.1% vs 35.1% │  │  ~30% of cost   │  │    #1 Ranked    │
│  Beats GPT-5     │  │  Beats GPT-5    │  │   Dec 2, 2025   │
│  2.5x efficient  │  │  Wide margin    │  │                  │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

**Small text below**: *Trained via end-to-end RL with outcome, efficiency, and preference rewards on the ToolScale dataset*

---

## SLIDE 4 — What is ToolOrchestra?

**Headline**: *"Model-as-Tool: LLMs are callable tools"*

Three tool cards with icons:

```
🔍 search                ⚙️ enhance_reasoning       ✅ answer
Generates a query        Generates Python code      Calls a specialist
Executes web search      Executes in subprocess     to produce final answer
Returns documents        Returns results            Returns prediction

Each tool offers multiple model choices with different
pricing, latency, and capability — the Orchestrator picks.
```

**Key insight callout**:
> Tool definitions include pricing & latency metadata.
> The Orchestrator LEARNS cost-aware routing through RL.

---

## SLIDE 5 — How the Orchestrator Thinks

**Headline**: *"The Multi-Turn Loop"*

Animated step-through (or numbered diagram):

```
User: "Who won the Nobel Prize in Physics 2023?"

  Step 1  │ Orchestrator THINKS        │ <think> block
  Step 2  │ Calls search(search-2)     │ Cheap model → query generation
  Step 3  │ Llama-3.2-3B → DuckDuckGo │ 5 documents returned
  Step 4  │ Orchestrator THINKS again  │ Reviews documents
  Step 5  │ Calls answer(answer-1)     │ Strong model → synthesis
  Step 6  │ Gemini 2.5 Pro → answer    │ "Agostini, Krausz, L'Huillier"
```

**Callout arrow**: *search-2 (cheap) for query gen, answer-1 (powerful) for synthesis — this tradeoff was LEARNED, not programmed*

---

## SLIDE 6 — Architecture Diagram

**Headline**: *"Routing Flow"*

```
                    ┌──────────────────────┐
                    │   Orchestrator-8B    │
                    │  (fine-tuned Qwen3-8B │
                    │   8B parameters)     │
                    └──────┬───┬───┬───────┘
                           │   │   │
              ┌────────────┘   │   └────────────┐
              ▼                ▼                ▼
        ┌──────────┐    ┌──────────┐    ┌──────────┐
        │  search  │    │ reasoning│    │  answer  │
        └────┬─────┘    └────┬─────┘    └────┬─────┘
             │               │               │
    ┌────────┴────┐   ┌──────┴──────┐  ┌─────┴──────┐
    │Specialist   │   │Specialist   │  │Specialist  │
    │LLM → query  │   │LLM → code  │  │LLM → answer│
    │Web Search   │   │Python exec  │  │            │
    └─────────────┘   └─────────────┘  └────────────┘
```

Keep this clean. The live demo will show the interactive version.

---

## SLIDE 7 — Paper vs. Our Implementation

**Headline**: *"Same brain, different muscles"*

| Role | Paper | Ours | Where |
|---|---|---|---|
| **Orchestrator** | Nemotron-Orchestrator-8B | **Same** | L40S GPU |
| answer-1 | GPT-5 | Gemini 2.5 Pro | LlamaStack |
| answer-2 | gpt-5-mini | Gemini 2.5 Flash | LlamaStack |
| answer-3 | Llama-3.3-70B | Llama-3.2-3B | L4 GPU |
| answer-math-1 | Qwen2.5-Math-72B | Qwen2.5-Math-7B | L4 GPU |
| reasoner-1 | GPT-5 | Gemini 2.5 Flash | LlamaStack |
| search-2 | gpt-5-mini | Llama-3.2-3B | L4 GPU |

**Bottom callout**: *Orchestrator checkpoint is IDENTICAL. Specialists differ by available compute. We do NOT claim paper's benchmark scores with our specialists.*

---

## SLIDE 8 — Why OpenShift AI?

**Headline**: *"Enterprise platform for multi-model AI"*

Five icon + text rows:

```
🖥️  KServe Model Serving    Deploy & scale LLMs with GPU autoscaling
⚡  GPU MachineSet           Provision L4/L40S nodes on demand
📄  ConfigMap Config         Swap specialists by editing YAML — no code changes
🔒  Enterprise Ready         RBAC, TLS routes, network policies, monitoring
🏗️  Unified Platform         Models + UI + config in one place
```

---

## SLIDE 9 — How We Built It

**Headline**: *"3 on-prem models + cloud specialists + React UI"*

Deployment diagram:

```
┌─ OpenShift AI Cluster ──────────────────────────┐
│                                                  │
│  ┌──────────────┐  ┌──────────┐  ┌───────────┐  │
│  │Orchestrator-8B│  │Llama-3.2 │  │Qwen Math  │  │
│  │  L40S GPU     │  │  L4 GPU  │  │  L4 GPU   │  │
│  └──────┬───────┘  └────┬─────┘  └─────┬─────┘  │
│         │               │              │         │
│  ┌──────┴───────────────┴──────────────┴──────┐  │
│  │         FastAPI Backend (server.py)         │  │
│  │    SSE streaming · DuckDuckGo · subprocess  │  │
│  └──────────────────┬─────────────────────────┘  │
│                     │                            │
│  ┌──────────────────┴─────────────────────────┐  │
│  │          React Frontend (chat + diagram)    │  │
│  └─────────────────────────────────────────────┘  │
└──────────────────────┬───────────────────────────┘
                       │
              ┌────────┴─────────┐
              │  Gemini Models   │
              │  via LlamaStack  │
              │  (cloud/hybrid)  │
              └──────────────────┘

All config in one ConfigMap (config.yaml)
```

---

## (LIVE DEMO — no slide, just the UI)

Show the UI full-screen. Run 3 questions:
1. "Who is the prime minister of Italy?" — search + answer
2. "What is the 100th Fibonacci number?" — code execution
3. "Birth year of the ACE inhibitor chemist?" — multi-hop

---

## SLIDE 10 — Cost Efficiency & Differentiation

**Headline**: *"Intelligent routing = 2.5x savings"*

Two-column:

**Left — Cost examples from our demo**:
```
"What is 2+2?"
  → Qwen2.5-Math-7B
  → Cost: ~$0.00002

"Who synthesized the ACE inhibitor?"
  → 6 turns, Gemini Pro
  → Cost: ~$0.03
```

**Right — How we differ**:
```
vs. LangChain/CrewAI
  Hardcoded routing → RL-learned routing

vs. MCP
  Tool connectivity (transport) → Tool selection (intelligence)
  Complementary, not competing

Model-as-Tool
  LLMs exposed with pricing metadata
  Orchestrator makes cost-quality tradeoffs autonomously
```

---

## SLIDE 11 — Reproducibility

**Headline**: *"Open source, fully reproducible"*

```
github.com/rrbanda/ToolOrchestra

📁 evaluation/    Paper's original code (untouched)
📁 training/      RL training pipeline (untouched)
📁 data/          ToolScale dataset (untouched)

📁 orchestrator-rhoai/
   └── src/ui/           FastAPI + React frontend
   └── manifests/        KServe, ConfigMap, Deployment
   └── config.yaml       All settings in one file

To swap a specialist model:
  Edit config.yaml → restart pod → done
```

---

## SLIDE 12 — What's Next

**Headline**: *"Roadmap"*

Three cards:

```
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│  Larger Specialists │  │  Domain Fine-Tuning │  │  MCP Integration   │
│                     │  │                     │  │                     │
│  32B, 70B models    │  │  RL training on     │  │  Connect to        │
│  when GPU budget    │  │  domain-specific    │  │  enterprise tools  │
│  allows             │  │  tasks via RHOAI    │  │  via MCP protocol  │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
```

---

## SLIDE 13 — Key Takeaways

**Headline**: *"Remember these 4 things"*

```
1   A small 8B model can outperform GPT-5
    37.1% vs 35.1% on HLE — via RL-trained orchestration

2   OpenShift AI provides the platform
    KServe + vLLM + ConfigMaps = production multi-model serving

3   2.5x cost savings through intelligent routing
    The Orchestrator decides which model to call autonomously

4   Fully extensible
    Swap any specialist by editing a YAML file
```

**Bottom**: *Questions?*  |  *github.com/rrbanda/ToolOrchestra*  |  *arxiv.org/abs/2511.21689*

---

## Design Guidelines for All Slides

- **Font**: Clean sans-serif (Inter, Source Sans, or Red Hat Display)
- **Background**: Dark (#0f172a slate-900) or light (#ffffff) — pick one and stay consistent
- **Accent colors**: Indigo (#6366f1) for Orchestrator, Cyan (#06b6d4) for Llama, Purple (#a855f7) for Qwen, Amber (#f59e0b) for Gemini
- **Max bullet points per slide**: 4-5
- **Max words per bullet**: 10-12
- **Every number cited must match the paper exactly**
- **Diagrams over bullet lists wherever possible**
