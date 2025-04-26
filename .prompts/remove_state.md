I want to refactor the {SERVICE_NAME} class to be more stateless and create a proper interface for it using Abstract Base Classes. Please:

1. Remove state from {SERVICE_NAME} by:
   - Moving instance variables like datasets, symbols, or other stateful data from the constructor to method parameters
   - Keeping only configuration and stateless dependencies in the constructor

2. Create an {SERVICE_NAME}Interface abstract base class that:
   - Clearly defines the contract for all implementations
   - Uses @abstractmethod for all required methods
   - Includes proper type hints and comprehensive docstrings for all methods
   - Is placed in the same file as the implementation for now (unless you think a separate file would be better)

3. Update any classes that use {SERVICE_NAME} to work with the new stateless implementation.
Also consider about updating unit / integration tests, which depend on {SERVICE_NAME}

4. Use Python's underscore prefix convention (_method_name) for private/internal methods

Key questions I'd like you to address:
- Is it beneficial to create an interface for this particular service?
- Should the interface be in the same file or a separate file?
- Are there any patterns or practices I should consider for this type of refactoring?

Please follow my standard SOLID principles and ensure all code has proper type hints, docstrings, and would pass mypy/ruff validation.
