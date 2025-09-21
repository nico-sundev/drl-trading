---
applyTo: '**'
---
# Instructions for software development tasks
## General development rules:
- Whole src/ path to follow SOLID principles
- Always use type hints for arguments and return types both
- Use dependency injection where it makes sense
- Always take care of proper error handling
- Code should be cleaned by `ruff check <file_path> --fix`
- Code should be validated by looking for type errors using the #problems tool
- Clean up unreferenced code, config left-overs and always think about things if they are actually in use and needed
- Try your best to find the easiest and most efficient solution to a problem
- If there is no easy understandable and simple solution, explain complex logic with comments
- Alway generate docstrings for all classes and methods
- For data structures used across services, consider creating
  - DTOs
  - Container Classes
  - Domain Objects
- To test **ANY** code ALWAYS use pytest and stick to the following specifications
- Testing:
    - as a general rule, a test is located in same parent directories like the class/module it is implemented for.
      Example:
      class/model to be tested: `src/drl_trading_core/data_set_utils/merge_service.py`
      test location: `tests/<unit or it>/data_set_utils/merge_service_test.py`
    - unit tests below `tests/unit/` directory
    - IT below `tests/integration/` directory
    - every testfile postfixed "_test.py" for unittest or "_it.py" for integration test
    - to run tests, use the tool #runTests, as fallback use `uv run python -m pytest`
    - only test public methods of a class
    - create fixtures for test methods, but use conftest.py for common fixtures among multiple test files
    - define test data builders for complex objects
    - define constants for common values used in multiple tests
    - if fixtures are about to be created, scan nearby conftest.py files and check for similar fixtures. combine them if possible and semantically correct
    - use pytest.mark.parametrize for parametrized tests
    - try to cluster similar kind of tests into multiple classes in one test file
    - unit tests should usually contain mocked dependencies and not depend on external files
    - integration tests should use real implementations and can make use of external files, like config files, data files, etc.
    - external files should be located in `tests/resources/` directory

    ** IMPORTANT: ALL test methods MUST follow the Given/When/Then structure with explicit comments **

    Example of proper test method structure:
    ```python
    def test_some_functionality(self, fixture1: X, fixture2: Y) -> None:
        """Test description."""
        # Given
        # Set up test preconditions
        input_data = [1, 2, 3]
        mock_dependency.return_value = "mocked_value"

        # When
        # Execute the function/method being tested
        result = system_under_test.do_something(input_data)

        # Then
        # Assert the expected outcomes
        assert result == expected_result
        mock_dependency.assert_called_once()
    ```

    **CRUCIAL:** Finding suitable test cases using your Superpower: QA-Glasses
      - Test cases should cover:
        - Bold cases
        - Boundary cases
        - Error cases
      - Leverage your superpower to rethink about real-world production scenarios
      - Bad assertions: Make the test green by assuming the current implementation is correct
      - Good assertions: Assert exactly what you expect, loosely coupled to the implementation

## Verification checklist for software development tasks:
Before generating any code, verify that:
- [ ] SOLID principles are followed
- [ ] Type hints are used for all arguments and return types
- [ ] All tests follow the Given/When/Then structure with explicit comments
- [ ] Docstrings generated and explanatory comments for complex logic
- [ ] The code can be validated with mypy and ruff
