# Security Policy

## Supported Versions

| Version | Supported | Notes |
|---------|-----------|-------|
| 0.x     | ✅ Yes    | Pre-release, actively developed |

Once the project reaches 1.0, a formal support policy for stable releases will be established.

## Reporting a Vulnerability

We're extremely grateful for security researchers and users that report vulnerabilities to the
Kubeflow Open Source Community. All reports are thoroughly investigated by project owners.

**Please do NOT report security vulnerabilities through public GitHub issues.**

You can use the following ways to report security vulnerabilities privately:

- Using the [GitHub Security Advisory](https://github.com/kubeflow/mcp-server/security/advisories/new) (preferred).
- Using the Kubeflow Steering Committee mailing list: [ksc@kubeflow.org](mailto:ksc@kubeflow.org).

Please provide detailed information to help us understand and address the issue promptly,
including: type of vulnerability, location of affected code, steps to reproduce, potential impact,
and suggested fix (if any).

## Disclosure Process

- **Acknowledgment**: We will acknowledge receipt of your report within 48 hours.
- **Assessment**: Project owners will investigate to determine validity and severity.
- **Resolution**: If confirmed, we will work on a fix and prepare a release.
- **Notification**: Once a fix is available, we will notify the reporter and coordinate disclosure.
- **Public Disclosure**: Details and the fix will be published in release notes.

Fix timelines depend on severity: critical — ASAP, high — 2 weeks, medium — 1 month.

## Prevention Mechanisms

- **Code Reviews**: All changes are reviewed by maintainers via Prow `/lgtm` + `/approve`.
- **Dependency Management**: Dependabot monitors dependencies; `pip-audit` runs in CI.
- **Continuous Integration**: Automated tests, linting, and security checks on every PR.
- **Image Scanning**: Container images are scanned for vulnerabilities via Trivy.
- **Input Validation**: AST-based script safety checks, K8s name validation, training parameter bounds (`core/security.py`).
- **Sensitive Data Masking**: Audit logs redact tokens, passwords, and credentials before storage.

## Security Model

For the full security model — trust boundaries, threat model, RBAC configuration,
known security considerations, hardening checklist, and deployment practices — see the
[Security Model](ARCHITECTURE.md#security-model) section in `ARCHITECTURE.md`.

## Communication Channels

- Kubeflow [Slack channels](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels)
- Kubeflow [mailing list](https://www.kubeflow.org/docs/about/community/#kubeflow-mailing-list)

Please **do not report** security vulnerabilities through public channels.
