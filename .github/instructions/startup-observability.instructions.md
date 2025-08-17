---
applyTo: '**'
---
# Agent Instructions: Startup Observability

When modifying or creating service bootstrap code:
1. Always instantiate `StartupContext(service_name)` at the start of `ServiceBootstrap.start()`.
2. Wrap each discrete bootstrap step in `with ctx.phase('<phase_name>'):`
   - Standard phases: config, logging, dependency_injection, dependency_checks, health_checks, web_interface, business_start
   - Do NOT invent new phases unless the work exceeds ~50ms regularly or has external I/O.
3. Record environment & feature flags via `ctx.attribute(key, value)` (only scalar, non-sensitive values).
4. Add external readiness checks with `ctx.add_dependency_status(...)` instead of scattering log lines.
5. On any exception, allow it to propagate—`phase_exceptions` will capture the phase automatically.
6. Ensure exactly one `ctx.emit_summary(logger)` call on both success path and guarded failure path.
7. Avoid logging secrets or entire config blobs; whitelist keys explicitly.
8. If adding a dependency probe, prefer creating a reusable probe object (future `StartupDependencyProbe` pattern) so other services can adopt it.
9. Keep the `STARTUP SUMMARY` schema stable; if you must add a field, update:
   - `OBSERVABILITY_STARTUP.md`
   - ADR-0003 (amend with rationale)
   - This instruction file
10. Add or update unit tests for:
    - New phase timing (duration captured)
    - Degraded dependency path
    - Exception capture inside phase

Non-compliance increases drift and reduces automated reasoning power for downstream AI agents.
