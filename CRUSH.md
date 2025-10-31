# AI Therapist - Development Commands & Guidelines

## Build/Lint/Test Commands
```bash
# Environment setup
python3 -m venv ai-therapist-env && source ai-therapist-env/bin/activate
pip install -r requirements.txt

# Run application
streamlit run app.py

# Testing commands
python tests/test_runner.py                           # All tests
python -m pytest tests/unit/test_voice_service.py    # Single test file
python -m pytest tests/unit/ -v --tb=short           # Unit tests
python -m pytest tests/integration/ -v --tb=short    # Integration tests
python -m pytest tests/security/ -v --tb=short       # Security tests
python -m pytest tests/unit/test_voice_service.py::test_initialization -v  # Single test method
python -m pytest tests/unit/ --cov=voice --cov-report=term-missing --cov-fail-under=90  # Coverage

# Build vector store
python build_vectorstore.py
```

## Code Style Guidelines
- **Imports**: Standard library → third-party → local (relative preferred)
- **Type hints**: Required for all functions, use `Optional[T]`, `Union`, dataclasses
- **Error handling**: Specific exceptions with logging, graceful degradation for voice features
- **Naming**: PascalCase classes, snake_case functions/vars, UPPER_SNAKE_CASE constants
- **Testing**: 90%+ coverage, mock external dependencies, pytest fixtures, use markers: unit/integration/security/performance
- **Security**: HIPAA compliance mandatory, PII masking, voice data encryption
- **Async**: Use `asyncio_mode = auto` from pytest.ini, test with proper await
- **Voice**: Multi-provider STT/TTS with fallbacks, mock services in testing, security controls in voice/security.py