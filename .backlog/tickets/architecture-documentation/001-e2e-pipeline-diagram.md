# E2E ML Pipeline Flow Diagram

**Epic:** Architecture Documentation
**Status:** 📝 Todo
**Assignee:** You
**Estimated:** 4 hours

## Description
Create comprehensive E2E ML pipeline flow diagram showing data flow from ingestion through execution, including Kafka messaging, service interactions, and decision points.

## Acceptance Criteria
- [ ] Visual diagram showing complete E2E flow
- [ ] Kafka topics and message flow documented
- [ ] Service interactions clearly illustrated
- [ ] Decision points and branching logic shown
- [ ] Batch vs incremental processing paths
- [ ] Database interactions included
- [ ] Error handling flows documented

## Technical Notes
- Use Mermaid.js for version-controlled diagrams
- Include all 5 services: ingest, preprocess, training, inference, execution
- Show warm-up process and feature computation
- Document sync vs async communication patterns
- Include TimescaleDB and Feast interactions

## Files to Create
- [ ] `/ARCHITECTURE.md` - Main architecture document
- [ ] `/docs/diagrams/e2e-pipeline-flow.md` - Mermaid diagram
- [ ] `/docs/diagrams/kafka-message-flow.md` - Message flow diagram

## Acceptance Criteria Detail
**Diagram must include:**
1. Data ingestion (real-time API + batch)
2. TimescaleDB storage and preprocessing trigger
3. Feature computation and warm-up process
4. Feast storage and materialization
5. Training vs inference branching
6. Model loading and prediction
7. Trade execution workflow

## Definition of Done
- [ ] Complete E2E diagram created
- [ ] Architecture document written
- [ ] Diagrams render correctly in GitHub
- [ ] Documentation reviewed for accuracy
- [ ] Links to relevant code sections included
