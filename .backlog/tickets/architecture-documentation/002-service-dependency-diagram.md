# Service Dependency Architecture Diagram

**Epic:** Architecture Documentation
**Status:** 📝 Todo
**Assignee:** You
**Estimated:** 3 hours

## Description
Create internal dependency architecture diagram showing how drl-trading-core and drl-trading-common support the microservice ecosystem, including dependency injection patterns and shared libraries.

## Acceptance Criteria
- [ ] Core and common library roles clearly illustrated
- [ ] Service dependency hierarchy shown
- [ ] Shared interfaces and contracts documented
- [ ] Dependency injection patterns visualized
- [ ] Configuration management flow shown
- [ ] Common messaging infrastructure depicted

## Technical Notes
- Focus on drl-trading-core as backbone
- Show drl-trading-common as shared library
- Include dependency injection containers
- Document configuration sharing patterns
- Show interface contracts between services

## Files to Create
- [ ] `/docs/diagrams/service-dependencies.md` - Mermaid diagram
- [ ] `/docs/architecture/dependency-patterns.md` - DI documentation
- [ ] `/docs/architecture/shared-libraries.md` - Core/common usage

## Diagram Components
**Must include:**
1. drl-trading-core (backbone/framework)
2. drl-trading-common (shared interfaces/config)
3. All 5 microservices
4. Dependency arrows and relationships
5. Configuration flow
6. Messaging infrastructure
7. Database dependencies

## Definition of Done
- [ ] Service dependency diagram complete
- [ ] DI patterns documented
- [ ] Shared library usage explained
- [ ] Configuration patterns documented
- [ ] Code examples included where helpful
