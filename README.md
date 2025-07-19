# DRL Trading System
**AI-Driven Financial Trading with Deep Reinforcement Learning**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![uv](https://img.shields.io/badge/uv-package%20manager-green.svg)](https://github.com/astral-sh/uv)

## 🎯 Overview

A comprehensive AI trading system using Deep Reinforcement Learning for automated financial market trading. The system features a modern microservice architecture with ML pipeline automation, real-time inference, and production-grade observability.

### Key Features
- **Deep Reinforcement Learning**: Support for multiple algorithms (PPO, A2C, SAC, TD3, DQN) via Stable-Baselines3
- **Feature Engineering**: Decoupled feature implementations with custom technical indicator backends and Feast API integration
- **Microservice Architecture**: Event-driven pipeline with message bus communication
- **Risk Management**: Built-in risk controls and compliance framework
- **Production Ready**: MLflow integration, observability, and monitoring

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Ingest   │───▶│   Preprocessing  │───▶│    Training     │
│   (Market Data) │    │ (Feature Pipeline)│    │  (ML Pipeline)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Execution     │◀───│    Inference     │◀───│   Model Store   │
│ (Order Engine)  │    │  (Predictions)   │    │   (MLflow)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📦 System Components

### Core Libraries
| Component | Purpose | Status | Documentation |
|-----------|---------|--------|---------------|
| **drl-trading-core** | Framework backbone & feature engineering framework | ✅ Stable | [README](./drl-trading-core/README.md) |
| **drl-trading-common** | Shared messaging infrastructure & data models | ✅ Stable | [README](./drl-trading-common/README.md) |
| **drl-trading-strategy-***| Strategy modules (pluggable trading logic) | 🔄 Active Dev | [Example](./drl-trading-strategy-example/README.md) |

### Microservices (Deployable Units)
| Service | Purpose | Status | Documentation |
|---------|---------|--------|---------------|
| **drl-trading-preprocessing** | Feature computation & pipeline processing | 📝 Planned | [README](./drl-trading-preprocessing/README.md) |
| **drl-trading-training** | Model training service | 🔄 Active Dev | [README](./drl-trading-training/README.md) |
| **drl-trading-inference** | Real-time prediction service | 📝 Planned | [README](./drl-trading-inference/README.md) |
| **drl-trading-ingest** | Market data ingestion service | 📝 Planned | [README](./drl-trading-ingest/README.md) |
| **drl-trading-execution** | Trade execution service | 📝 Planned | [README](./drl-trading-execution/README.md) |

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- [uv](https://github.com/astral-sh/uv) package manager

### Development Setup
```bash
# Clone and setup workspace
git clone <repository>
cd ai_trading

# Install core framework
cd drl-trading-core
pip install uv
uv venv
source .venv/Scripts/activate  # Windows
# source .venv/bin/activate     # Linux/Mac
uv pip install -e .

# Install strategy module
cd ../drl-trading-strategy-example
uv pip install -e .

# Run preprocessing pipeline
python -m drl_trading_core.cli.preprocessing_cli
```

### Training a Model
```bash
# Setup training service
cd drl-trading-training
uv pip install -e .

# Run training with example strategy
python -m drl_trading_training --config config/default.json
```

### Docker Development
```bash
# Start training environment
docker-compose -f docker-compose.training.yml up

# Or full development stack (when available)
docker-compose -f docker-compose.production.yml up
```

## 🔧 Development

### Project Structure
```
ai_trading/
├── drl-trading-core/              # Framework backbone & feature engineering
├── drl-trading-common/            # Shared messaging & data models
├── drl-trading-strategy-example/  # Example strategy implementation
├── drl-trading-preprocessing/     # Feature computation microservice (planned)
├── drl-trading-training/          # Training microservice
├── drl-trading-inference/         # Inference microservice (planned)
├── drl-trading-ingest/           # Data ingestion microservice (planned)
├── drl-trading-execution/        # Execution microservice (planned)
├── .backlog/                     # Project management & documentation
├── scripts/                      # Utility scripts
└── docker-compose.*.yml         # Deployment configurations
```

### Development Workflow
1. **Strategy Development**: Create/modify strategy modules in `drl-trading-strategy-*/`
2. **Framework Development**: Extend core functionality in `drl-trading-core/`
3. **Service Development**: Build/modify microservices in `drl-trading-*/`
4. **Integration Testing**: Use Docker Compose for full system testing
5. **Deployment**: Production deployment via containerized microservices

### Code Quality
```bash
# Formatting and linting
ruff check <path> --fix
mypy <path>

# Testing
pytest tests/
```

## 📚 Documentation

- **[Project Management](./.backlog/README.md)** - Epics, tickets, and progress tracking
- **[Architecture Documentation](./.backlog/tickets/architecture-documentation/)** - Detailed system design
- **[API Documentation](./docs/api/)** - Service APIs and contracts
- **[Development Guide](./docs/development.md)** - Contribution guidelines

## 🎯 Current Status & Roadmap

### ✅ Completed
- Core feature engineering framework with Feast integration
- Strategy module system with dependency injection
- Training service with CLI interface
- Comprehensive testing infrastructure
- Feature engineering framework with pluggable indicators

### 🔄 In Progress
- Feast integration tests (95% complete)
- MLflow model management integration
- Feature normalization and encoding enhancements
- Data provider API expansion (Binance, TwelveData)

### 📝 Planned
- Real-time inference microservice
- Production deployment automation
- Advanced ML features (GNN pattern recognition)
- Observability stack (OpenTelemetry, Grafana)
- Multi-exchange execution service

## 🤝 Contributing

This project follows modern development practices with comprehensive testing and documentation standards.

### Getting Started
1. Fork the repository
2. Set up development environment (see Development Setup above)
3. Review [coding standards](./.github/instructions/) and [documentation guidelines](./docs/README-standards.md)
4. Create feature branch for your changes
5. Submit pull request with tests and documentation

### Development Standards
- **Code Quality**: All code must pass `ruff check`, `mypy`, and `pytest`
- **Testing**: Follow Given/When/Then structure for all tests
- **Documentation**: Update relevant READMEs and API documentation
- **Architecture**: Follow SOLID principles and dependency injection patterns

## 📄 License

[Add your license here]

## 🆘 Support

- **Issues**: [GitHub Issues](link)
- **Discussions**: [GitHub Discussions](link)
- **Documentation**: [Wiki](link)
