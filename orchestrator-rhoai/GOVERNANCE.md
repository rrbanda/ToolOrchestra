# Governance

This document describes the governance model for the **orchestrator-rhoai** project—a production-grade open-source reference implementation that deploys NVIDIA's ToolOrchestra system on Red Hat OpenShift AI.

## Project Type

orchestrator-rhoai is a **Reference Implementation**. It demonstrates how to deploy and operate NVIDIA ToolOrchestra on Red Hat OpenShift AI, serving as a blueprint for production deployments. It is not a standalone product but a community-maintained template that can be adapted and extended.

## Decision Making

### Consensus Model

- **Maintainer consensus** is the primary decision-making mechanism
- Maintainers reach agreement through discussion in issues, PRs, and community channels

### Lazy Consensus

- **Minor changes** (bug fixes, documentation updates, dependency bumps, minor features) use **lazy consensus**
- If no maintainer objects within a reasonable period (typically 3–5 business days), the change is considered approved
- Silence is interpreted as approval

### Explicit Approval

- **Architectural changes** require **explicit approval** from at least two maintainers
- Examples: new components, major API changes, changes to deployment topology, security model updates
- These decisions are documented in an ADR (Architecture Decision Record) or design document when significant

### Tiers of Decisions

| Decision Type | Process | Examples |
|---------------|---------|----------|
| Minor | Lazy consensus | Docs fix, dependency update, small refactor |
| Significant | Explicit approval | New feature, API change, new integration |
| Architectural | Explicit approval + documentation | New service, security model, deployment structure |

## Roles

### Contributor

**Definition**: Anyone who submits a pull request, opens an issue, or participates in discussions.

**Responsibilities**:
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md)
- Follow [Contributing](CONTRIBUTING.md) guidelines
- Provide signed-off commits (DCO)
- Engage constructively in code review

**How to become one**: Submit your first contribution (issue or PR).

### Reviewer

**Definition**: A contributor who has been granted review privileges by maintainers.

**Responsibilities**:
- Review pull requests in their area of expertise
- Provide constructive feedback and approve when satisfied
- Help maintain code quality and consistency
- May merge PRs after approval when authorized

**How to advance**:
- Demonstrate consistent, high-quality contributions
- Be invited by maintainers based on sustained engagement and domain knowledge

### Maintainer

**Definition**: A core contributor who has merge rights and stewardship over the project.

**Responsibilities**:
- Merge approved pull requests
- Triage issues and guide roadmap
- Participate in architectural decisions
- Represent the project and enforce governance

**How to advance**:
- Demonstrate sustained, high-quality contributions over time
- Exhibit judgment, reliability, and alignment with project goals
- Be nominated by existing maintainers and confirmed by consensus
- See [MAINTAINERS.md](MAINTAINERS.md) for detailed criteria

## Conflict Resolution

### Escalation Path

1. **Direct discussion**: Parties attempt to resolve informally (issue comments, discussions)
2. **Maintainer mediation**: A maintainer not directly involved may mediate
3. **Maintainer vote**: If needed, maintainers vote; majority decides
4. **Project lead / tie-breaker**: If the project has a lead, they may break ties or make a final call when consensus cannot be reached

### Dispute Principles

- Disagreements should focus on technical merit and project goals, not personal preference
- Maintainers should recuse themselves from decisions where they have a conflict of interest
- All parties are expected to adhere to the [Code of Conduct](CODE_OF_CONDUCT.md)
- Harassment, ad hominem attacks, or unprofessional behavior are not tolerated

### When Consensus Fails

- Document the disagreement and options considered
- Escalate to maintainer vote
- If still unresolved, defer the decision until more information or alignment is available
- In rare cases, the project may fork or the disagreement may be documented for future resolution
