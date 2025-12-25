# CI/CD Docker Image

Custom Docker image: **UV Python 3.12 + Docker CLI** for integration tests.

## Quick Info

- **Base**: `ghcr.io/astral-sh/uv:python3.12-bookworm-slim`
- **Tools**: UV, Docker CLI, Git, Make, curl
- **Purpose**: Run integration tests with Docker-in-Docker in CI/CD

## Documentation

**See complete setup guide:** [`docs/CI_IMAGE_SETUP.md`](../../docs/CI_IMAGE_SETUP.md)

## Quick Commands

```bash
# Build and push
make -f ../../Makefile.ci push-ci-image

# Test locally
make -f ../../Makefile.ci test-ci-image

# Quick reference
bash ../ci-image-quickref.sh
```
