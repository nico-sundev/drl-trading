# DRL Trading Framework

> **A production-ready Deep Reinforcement Learning framework for algorithmic trading, showcasing modern ML engineering practices and enterprise software architecture.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/dependency%20manager-uv-blue)](https://github.com/astral-sh/uv)

## 🎯 Project Vision

This project demonstrates the intersection of **financial domain expertise**, **cutting-edge ML engineering**, and **enterprise software architecture**. It combines:

- **Deep Reinforcement Learning** applied to financial markets
- **Python mastery** through complex real-world implementation
- **ML Operations** with Feast feature store, MLflow model management
- **Microservices architecture** with hexagonal design patterns
- **Event-driven systems** with pluggable messaging infrastructure
- **AI-assisted development** workflows and best practices

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Ingest   │───▶│   Preprocessing │───▶│    Training     │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Execution     │◄───│   Inference     │◄───│   Strategy      │
│                 │    │                 │    │   (Pluggable)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

**Key Principles:**
- **Strategy Isolation**: Framework open-source, strategies proprietary
- **Hexagonal Architecture**: Clean business logic separation
- **Event-Driven**: Scalable messaging (in-memory → Kafka)
- **Cloud-Native**: Containerized AWS-ready services

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/drl-trading.git
cd drl-trading
uv sync --group dev-full

# Generate openapi clients by spec files
./scripts/openapi/generate-and-install-clients.sh

# Run example preprocess
cd drl-trading-preprocess
uv run main.py

# End-to-end test
docker-compose -f docker-compose.training.yml up
```

## 📚 Documentation

- **[Developer Guide](docs/DEVELOPER_GUIDE.md)** - Setup and development
- **[Strategy Development](docs/STRATEGY_DEVELOPMENT.md)** - Custom strategies
- **[Learning Journey](docs/LEARNING_JOURNEY.md)** - Skills and progression

## 🛠️ Technology Stack

| **ML** | **Architecture** | **Infrastructure** |
|--------|------------------|-------------------|
| Stable Baselines3 | Hexagonal patterns | Docker |
| Feast | Microservices | AWS |
| MLflow | Event-driven | uv |

## 🎯 Professional Skills Demonstrated

- **Software Architecture**: Microservices, hexagonal patterns, dependency injection
- **ML Engineering**: Feature stores, model management, real-time inference
- **DevOps & Cloud**: Containerization, CI/CD, AWS deployment
- **Financial Domain**: Technical analysis, risk management, backtesting

> **Strategy Separation**: Production strategies in private repo. Example strategy for learning/integration.

## 🤝 Contributing

See [DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) for setup and [STRATEGY_DEVELOPMENT.md](docs/STRATEGY_DEVELOPMENT.md) for custom strategies.

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🎤 About

Combining **financial markets expertise** with **modern software engineering** to explore algorithmic trading and machine learning integration.

**Connect**: [LinkedIn] | [GitHub]

---

*"Where financial domain knowledge meets cutting-edge ML engineering."*

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker & Docker Compose (optional)

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/drl-trading.git
cd drl-trading

# Install dependencies for all services
uv sync --group dev-full

# Run example strategy training
cd drl-trading-strategy-example
uv run python -m drl_trading_strategy_example.main
```

### End-to-End Test
```bash
# Start all services (docker-compose)
docker-compose -f docker-compose.training.yml up

# Verify pipeline
./scripts/e2e_test.sh
```

## 📚 Documentation

- **[Developer Guide](docs/DEVELOPER_GUIDE.md)** - Technical setup and development workflows
- **[Architecture Guide](docs/ARCHITECTURE.md)** - System design and patterns
- **[Strategy Development](docs/STRATEGY_DEVELOPMENT.md)** - How to create custom strategies
- **[Deployment Guide](docs/DEPLOYMENT.md)** - Production deployment patterns
- **[Learning Journey](docs/LEARNING_JOURNEY.md)** - Skills developed and lessons learned

## 🏛️ Framework Components

### Core Services
- **[drl-trading-core](drl-trading-core/)** - Framework foundation and preprocessing pipeline
- **[drl-trading-common](drl-trading-common/)** - Shared messaging and data models
- **[drl-trading-ingest](drl-trading-ingest/)** - Market data ingestion service
- **[drl-trading-training](drl-trading-training/)** - Model training orchestration
- **[drl-trading-inference](drl-trading-inference/)** - Real-time prediction service
- **[drl-trading-execution](drl-trading-execution/)** - Trade execution management
- **[drl-trading-preprocess](drl-trading-preprocess/)** - Feature computation service

### Strategy Module
- **[drl-trading-strategy-example](drl-trading-strategy-example/)** - Reference implementation

## 🛠️ Technology Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| **Language** | Python 3.11+ | Core development |
| **Package Management** | uv | Fast, reliable dependency management |
| **ML Framework** | Stable Baselines3 | Deep Reinforcement Learning |
| **Feature Store** | Feast | ML feature management |
| **Model Management** | MLflow | Experiment tracking and model registry |
| **Messaging** | Confluent Kafka | Event-driven communication |
| **Database** | PostgreSQL | Data persistence |
| **Containerization** | Docker | Service deployment |
| **Cloud Platform** | AWS | Production infrastructure |

## 🎯 Professional Skills Demonstrated

### Software Architecture
- **Microservices Design** with clear service boundaries
- **Hexagonal Architecture** for testability and maintainability
- **Event-Driven Patterns** for scalable system integration
- **Dependency Injection** for modular, testable code

### ML Engineering
- **Feature Engineering** pipelines with proper versioning
- **Model Training** orchestration and experiment management
- **Real-time Inference** with low-latency requirements
- **A/B Testing** framework for strategy comparison

### DevOps & Cloud
- **Infrastructure as Code** with Docker and docker-compose
- **CI/CD Pipelines** for automated testing and deployment
- **Monitoring & Observability** with structured logging
- **Cloud Deployment** patterns for AWS

## 🔧 Development

### Project Structure
```
drl-trading/
├── drl-trading-core/              # Framework foundation
├── drl-trading-common/            # Shared infrastructure
├── drl-trading-strategy-example/  # Example strategy (minimal)
├── drl-trading-training/          # Training service
├── drl-trading-inference/         # Inference service
├── drl-trading-ingest/           # Data ingestion service
├── drl-trading-execution/        # Trade execution service
├── drl-trading-preprocess/       # Feature computation service
├── docs/                         # Documentation
├── scripts/                      # Utility scripts
└── docker-compose.*.yml         # Deployment configurations
```

> **Strategy Separation**: Production trading strategies are maintained in a separate private repository (`drl-trading-strategy`). The example strategy provides minimal functionality for integration testing and learning.

### Development Workflow
1. **Framework Development**: Extend core functionality in `drl-trading-core/`
2. **Service Development**: Build/modify microservices following hexagonal architecture
3. **Strategy Development**: Create custom strategy modules (see [Strategy Development Guide](docs/STRATEGY_DEVELOPMENT.md))
4. **Integration Testing**: Use Docker Compose for full system testing
5. **Deployment**: Production deployment via containerized microservices

## 🤝 Contributing

This framework is designed to be extended with custom strategies. See [STRATEGY_DEVELOPMENT.md](docs/STRATEGY_DEVELOPMENT.md) for guidelines on creating strategy modules.

### Development Standards
- **Code Quality**: All code must pass `ruff check`, `mypy`, and `pytest`
- **Testing**: Follow Given/When/Then structure for all tests
- **Architecture**: Follow hexagonal architecture and SOLID principles
- **Documentation**: Update relevant docs with architectural decisions

## � License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎤 About the Author

Combining **financial markets expertise** with **modern software engineering** to explore the intersection of algorithmic trading and machine learning. This project represents a journey through Python mastery, ML operations, and enterprise architecture patterns.

**Skills Demonstrated:**
- Financial domain knowledge and quantitative trading
- Modern software architecture and microservices design
- Machine learning operations and MLOps best practices
- AI-assisted development workflows and productivity

**Connect:**
- LinkedIn: [Your Profile]
- GitHub: [Your Profile]

---

*"Where financial domain knowledge meets cutting-edge ML engineering."*
