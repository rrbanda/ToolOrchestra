# Third-Party Licenses

orchestrator-rhoai integrates with NVIDIA ToolOrchestra and various open-source models and libraries. This document summarizes the licenses of key models and dependencies. **Always verify the latest license terms** from the official sources before use.

## Summary Table

| Model / Library | License | Commercial Use | Status |
|-----------------|---------|----------------|--------|
| nvidia/Nemotron-Orchestrator-8B | NVIDIA Open Model License | Review needed | Review Needed |
| Qwen/Qwen3-8B | Apache 2.0 | Yes | Approved |
| Qwen/Qwen3-32B | Apache 2.0 | Yes | Approved |
| Qwen/Qwen2.5-Coder-32B-Instruct | Apache 2.0 | Yes | Approved |
| Qwen/Qwen2.5-Math-72B-Instruct | Apache 2.0 | Yes | Approved |
| Qwen/Qwen2.5-Math-7B-Instruct | Apache 2.0 | Yes | Approved |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | MIT | Yes | Approved |
| meta-llama/Llama-3.3-70B-Instruct | Llama 3.3 Community License | Conditional | Review Needed |
| NVIDIA ToolOrchestra (upstream) | Apache 2.0 | Yes | Approved |
| nvidia/ToolScale dataset | Custom | Review needed | Review Needed |

---

## Models

### NVIDIA

#### nvidia/Nemotron-Orchestrator-8B

- **License**: NVIDIA Open Model License
- **Commercial Use**: Review needed — terms may impose restrictions or attribution requirements
- **Status**: Review Needed
- **Notes**: Check [NVIDIA NGC](https://catalog.ngc.nvidia.com/) or Hugging Face for current license text before commercial deployment

#### NVIDIA ToolOrchestra (upstream code)

- **License**: Apache 2.0
- **Commercial Use**: Yes
- **Status**: Approved
- **Notes**: orchestrator-rhoai is built on top of this upstream project

#### nvidia/ToolScale dataset

- **License**: Custom (see NVIDIA repository)
- **Commercial Use**: Review needed
- **Status**: Review Needed
- **Notes**: Dataset terms may differ from code; verify before use in training or redistribution

---

### Qwen (Alibaba)

All Qwen models listed below use **Apache 2.0**, which permits commercial use with attribution.

| Model | License | Commercial Use | Status |
|-------|---------|----------------|--------|
| Qwen/Qwen3-8B | Apache 2.0 | Yes | Approved |
| Qwen/Qwen3-32B | Apache 2.0 | Yes | Approved |
| Qwen/Qwen2.5-Coder-32B-Instruct | Apache 2.0 | Yes | Approved |
| Qwen/Qwen2.5-Math-72B-Instruct | Apache 2.0 | Yes | Approved |
| Qwen/Qwen2.5-Math-7B-Instruct | Apache 2.0 | Yes | Approved |

---

### DeepSeek

#### deepseek-ai/DeepSeek-R1-Distill-Qwen-32B

- **License**: MIT
- **Commercial Use**: Yes
- **Status**: Approved
- **Notes**: MIT is permissive; attribution required per license terms

---

### Meta

#### meta-llama/Llama-3.3-70B-Instruct

- **License**: Llama 3.3 Community License
- **Commercial Use**: Conditional — check Meta’s current terms for revenue thresholds and usage limits
- **Status**: Review Needed
- **Notes**: Meta may update conditions; review [Meta’s license page](https://ai.meta.com/llama/license/) before commercial use

---

## Dependencies

Runtime and development dependencies (Python packages, container base images, etc.) are governed by their respective licenses. Key ones include:

- **Python ecosystem**: Check `requirements.txt`, `pyproject.toml`, or lock files
- **Container bases**: Red Hat UBI, NVIDIA NGC — verify on respective registries
- **Kubernetes/OpenShift tooling**: Apache 2.0, MIT, or as specified by upstream

Run `pip-licenses` or equivalent to generate a full dependency license report:

```bash
pip install pip-licenses
pip-licenses --format=markdown
```

---

## Recommendations

1. **Before production deployment**: Confirm all models and datasets in use are approved for your use case (commercial, internal, etc.).
2. **Attribution**: Apache 2.0 and MIT require attribution; maintain NOTICE files or equivalent.
3. **NVIDIA / Meta / custom licenses**: Review and comply with any restrictions, revenue caps, or reporting requirements.
4. **Updates**: Licenses may change; re-check when upgrading models or major dependencies.

---

## Contributing Updates

If you add a new model or dependency to orchestrator-rhoai, please:

1. Add it to the summary table and relevant section above
2. Specify license, commercial use, and status (Approved / Review Needed)
3. Link to the official license or source when possible
