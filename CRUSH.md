# AI Therapist Voice Features - Development Commands & Guidelines

## Testing Commands

### Run Comprehensive Test Suite
```bash
# Run all tests with detailed reporting
python tests/test_runner.py

# Run specific test categories
python tests/test_runner.py tests/unit -v --tb=short
python tests/test_runner.py tests/integration -v --tb=short
python tests/test_runner.py tests/security -v --tb=short
python tests/test_runner.py tests/performance -v --tb=short

# Run single test file
python -m pytest tests/unit/test_voice_service.py -v --tb=short
python -m pytest tests/unit/test_audio_processor.py::TestAudioProcessor::test_initialization -v

# Run with coverage
python -m pytest tests/unit/ --cov=voice --cov-report=term-missing --cov-fail-under=90
```

### Application Commands
```bash
# Run main application
streamlit run app.py

# Build vector store
python build_vectorstore.py

# Test voice setup
python test_voice_setup.py
```

## Code Style Guidelines

### Import Structure
- Standard library imports first
- Third-party imports second
- Local imports last (relative imports preferred)
- Group imports with blank lines between groups

### Type Hints
- Use typing module for all function signatures
- Optional types marked with `Optional[Type]`
- Use `Union` for multiple return types
- Dataclasses for complex data structures

### Error Handling
- Use specific exception types
- Always include logging for debugging
- Graceful degradation for voice features
- Try-except blocks for optional dependencies

### Naming Conventions
- Classes: PascalCase (e.g., `VoiceService`)
- Functions/variables: snake_case (e.g., `process_audio`)
- Constants: UPPER_SNAKE_CASE (e.g., `MAX_AUDIO_LENGTH`)
- Private methods: underscore prefix (e.g., `_cleanup()`)

### Testing Patterns
- Use pytest fixtures for setup/teardown
- Mock external dependencies (APIs, hardware)
- Test both success and failure scenarios
- Use asyncio for async function testing
- Maintain 90%+ coverage requirement

### Security Requirements
- HIPAA compliance for all voice data
- PII detection and masking required
- Voice data encryption at rest/in-transit
- Audit logging for all voice operations