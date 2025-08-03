# Architecture Decision Records (ADR)

## Overview
This directory contains Architecture Decision Records (ADRs) for the DRL Trading system. ADRs document significant architectural decisions, their context, alternatives considered, and consequences.

## Purpose
- **AI Agent Context**: Provides architectural context for AI assistants
- **Team Knowledge Sharing**: Preserves decision rationale for current and future team members
- **Consistent Decision Making**: Ensures new decisions align with established patterns
- **Onboarding**: Helps new team members understand architectural evolution

## ADR Process

### When to Create an ADR
Create an ADR for any decision that:
- Affects system architecture or design patterns
- Has significant impact on multiple services
- Involves trade-offs between different approaches
- Changes established patterns or conventions
- Has long-term consequences for the system

### How to Create an ADR
1. Copy `template.md` to a new file: `{number}-{short-title}.md`
2. Fill in the template sections
3. Submit as part of your pull request
4. Update the index after merging

### ADR Status Lifecycle
- **Proposed**: Decision is under consideration
- **Accepted**: Decision has been approved and should be implemented
- **Deprecated**: Decision is no longer recommended but not yet replaced
- **Superseded**: Decision has been replaced by a newer ADR

## ADR Index

### Active ADRs
| Number | Title | Status | Date | Tags |
|--------|-------|--------|------|------|
| [0001](0001-adr-system-adoption.md) | ADR System Adoption | Accepted | 2025-08-02 | documentation, process |
| [0002](0002-configuration-architecture-standardization.md) | Configuration Architecture Standardization | Accepted | 2025-08-02 | configuration, microservices, infrastructure |

### All ADRs by Topic

#### Documentation & Process
- ADR-0001: ADR System Adoption

#### Architecture Patterns
- (To be added)

#### Configuration Management
- ADR-0002: Configuration Architecture Standardization

#### Infrastructure
- (To be added)

## For AI Agents
See [ai-agent-context.md](ai-agent-context.md) for a summary of key architectural decisions.

## Quick Reference
- [ADR Template](template.md)
- [ADR Index](index.md) (auto-generated)
- [AI Context Summary](ai-context-summary.md) (auto-generated)
