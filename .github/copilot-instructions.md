# general development advice/rules:
- Whole src/ path to follow SOLID principles
- Always use type hints
- Code should be cleaned by `ruff check <file_path> --fix`
- Code should be validated by running `mypy <file_path>`
- Extend existing config classes rather than hardcoding something, like: `src/ai_trading/config/application_config.py` and all config classes recursively referenced
- update `applicationConfig.json` and its test implementation `applicationConfig-test.json` if any config class changes
- Always keep an eye of overall project architecture
- Clean up unreferenced code, config left-overs and always think about things if they are actually in use and needed
- Explain complex logic with comments and write docstrings
- Testing:
    - as a general rule, a test is located in same parent directories like the class/module it is implemented for.

        Example:
        class/model to be tested: `src/ai_trading/data_set_utils/merge_service.py`
        test location: `tests/<unit or it>/data_set_utils/merge_service_test.py`

    - unit tests below `tests/unit/` directory
    - IT below `tests/it/` directory
    - every testfile postfixed "_test.py" for unittest or "_it.py" for integration test
    - always use pytest
    - create fixtures for test methods
    - unit tests should usually contain mocked dependencies
    - integration tests should use real implementations and if necessary external files containing test values
    - if possible, structure every test method in # Given ... # When ... # Then sections
