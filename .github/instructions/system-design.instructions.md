---
applyTo: '**'
---

# hexagonal architecture
## components
- core
  - every module has a "drl_trading_{module_name}.core" package (right now, during development, sometimes the ".core" is omitted, but it should be there eventually)
  - there is a common core module drl_trading_core which contains core logic shared among multiple services
  - drl_trading_strategy_example is an entire module on its own, but acts like a decoupled extension of drl_trading_core
  - drl_trading_common contains a core package as well, but it is only used for shared domain objects, DTOs, exceptions, etc.
- adapters
  - every module has a "drl_trading_{module_name}.adapter" package (right now, during development, sometimes the ".adapter" is omitted, but it should be there eventually)
  - there is a common adapter module drl_trading_adapter which contains adapters shared among multiple services
  - drl_trading_common contains an adapter package as well (the reasonability is subject to discussion), but it is only used for shared interfaces, dtos, mappers etc. (no service implementations)
- application (currently still mistakenly referenced as infrastructure in some places, needs to be renamed)
  - every module has a "drl_trading_{module_name}.application" package (right now, during development, sometimes the ".application" is omitted, but it should be there eventually)
  - drl_trading_common contains an infrastructure package
  - drl_trading_common contains lots of common application code in other packages aswell, which has to be refactored at some point, to move application code into an application package

  ## violations and permissions
  - core packages should not depend on any other non-core packages
  - adapters can depend on core classes
  - if dtos, entitys etc need to be passes to a core package, adapters need to convert them first using mappers (the other direction, from core to adapter, is allowed to be done without mappings, but may be mapped as well to prevent hexarch violations)
  - application can depend on core classes

  ## known weaknesses
  - lack of archunit tests at the moment
  - core packages sometimes depend on application packages (needs to be refactored; replace by serviceconfig classes injected from application layer)
