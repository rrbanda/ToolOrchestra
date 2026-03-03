# Security Policy

orchestrator-rhoai is a production-grade open-source reference implementation that deploys NVIDIA's ToolOrchestra system on Red Hat OpenShift AI. We take security seriously and encourage responsible disclosure of vulnerabilities.

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

If you discover a security vulnerability, please report it privately:

- **Email**: security@orchestrator-rhoai.dev
- Include a clear description of the vulnerability and steps to reproduce
- Provide any relevant logs, configurations, or proof-of-concept code if applicable

### Response Timeline

| Stage | Timeframe |
|-------|-----------|
| **Acknowledgment** | Within 48 hours of initial report |
| **Detailed response** | Within 7 days, including assessment and next steps |
| **Fix coordination** | Depends on severity; critical issues prioritized for immediate remediation |

Maintainers will work with you to understand the issue, confirm the vulnerability, and develop a fix. We appreciate your patience and collaboration.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.x    | ✅ Yes    |
| &lt; 0.x | ❌ No     |

Security updates are provided for the 0.x series. We recommend upgrading to the latest release within the supported series.

## Security Practices

The orchestrator-rhoai project follows these security practices:

### Container Image Scanning

- All container images are scanned with [Trivy](https://github.com/aquasecurity/trivy) as part of the CI/CD pipeline
- Critical and high-severity vulnerabilities must be addressed before images are published
- Scan results are reviewed on every build

### Dependency Monitoring

- **Dependabot** and/or **Renovate** are used to monitor dependencies for known vulnerabilities
- Security-related dependency updates are prioritized and reviewed promptly
- Maintainers are notified of new advisories affecting project dependencies

### Secrets Management

- **No secrets, credentials, or API keys** are stored in the repository
- Use Kubernetes Secrets, OpenShift secrets, or external secret managers (e.g., HashiCorp Vault) for sensitive data
- `.env` files with secrets are never committed; use `.env.example` templates only

### Model Weights

- **Model weights are never bundled in container images**
- Models are pulled at runtime from approved registries (e.g., Hugging Face, NGC) or mounted via PersistentVolumes
- Reduces image size and avoids license/compliance issues from embedding model artifacts

### Network Security

- **NetworkPolicies** restrict inter-service communication to only required traffic
- Services follow least-privilege network access
- Ingress/Egress rules are explicitly defined and reviewed

### Additional Practices

- Regular dependency audits and updates
- Immutable container tags for reproducibility
- Non-root container execution where possible
- Read-only filesystems where applicable

## Acknowledgments

We thank security researchers and contributors who responsibly disclose vulnerabilities. Acknowledgments will be made with your permission after the issue is resolved.
