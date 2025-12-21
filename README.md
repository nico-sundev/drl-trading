# DRL Trading Framework

> **A production-ready Deep Reinforcement Learning framework for algorithmic trading, showcasing modern ML engineering practices and enterprise software architecture.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/dependency%20manager-uv-blue)](https://github.com/astral-sh/uv)

## Current state

This project is a WIP. The framework is not ready-to-use yet.
Some information may be misleading, some files may be about to be cleaned up, GitLab Pipelines may be breaking and
hexagonal architecture violations may be still around somewhere.

Most mature service so far, which is also the backbone of the system: [drl-trading-preprocess](./drl-trading-preprocess) (~90% code cov and partially e2e tested)

> Side note: This repository is being mirrored from my GitLab Repository

## 🎯 Project Vision

This project demonstrates the intersection of **financial domain expertise**, **cutting-edge ML engineering**, and **enterprise software architecture**. It combines:

- **Deep Reinforcement Learning** applied to financial markets
- **Python mastery** through complex real-world implementation
- **ML Operations** with Feast feature store, MLflow model management
- **Microservices architecture** with hexagonal design patterns
- **Event-driven systems** with pluggable messaging infrastructure
- **AI-assisted development** workflows and best practices

## 🏗️ Architecture (TODO)

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

## 📚 Documentation (TODO)

- **[Developer Guide](docs/DEVELOPER_GUIDE.md)** - Technical setup and development workflows
- **[Architecture Guide](docs/ARCHITECTURE.md)** - System design and patterns (TODO)
- **[Strategy Development](docs/STRATEGY_DEVELOPMENT.md)** - How to create custom strategies (TODO)
- **[Deployment Guide](docs/DEPLOYMENT.md)** - Production deployment patterns (TODO)
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

> **Strategy Separation**: Production trading strategies are maintained in a separate private repository (`drl-trading-strategy`). The example strategy provides minimal functionality for integration testing and learning.

## 🛠️ Technology Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| **Language** | Python 3.11+ | Core development |
| **Package Management** | uv | Fast, reliable dependency management |
| **ML Framework** | Stable Baselines3 | Deep Reinforcement Learning |
| **Feature Store** | Feast | ML feature management |
| **Model Management** (TODO) | MLflow | Experiment tracking and model registry |
| **Messaging** | Confluent Kafka | Event-driven communication |
| **Database** | PostgreSQL | Data persistence |
| **Containerization** | Docker | Service deployment |
| **Cloud Platform** (TODO) | AWS | Production infrastructure |

## 🤝 Contributing

This framework is designed to be extended with custom strategies. See [strategy development](docs/STRATEGY_DEVELOPMENT.md) for guidelines on creating strategy modules and [the general developer guide](docs/DEVELOPER_GUIDE.md).


## � License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎤 About the Author

Combining **financial markets expertise** with **modern software engineering** to explore the intersection of algorithmic trading and machine learning. This project represents a journey through Python mastery, ML operations, and enterprise architecture patterns.

**Connect:**

- LinkedIn: [Nico Sonntag](https://www.linkedin.com/in/nico-sonntag-1671272bb/)
- GitHub: [nico-sundev](https://github.com/nico-sundev)

---

*"Where financial domain knowledge meets cutting-edge ML engineering."*
