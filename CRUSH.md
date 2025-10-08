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

## Testing Guidelines

### Test Structure
```
tests/
├── unit/                   # Isolated unit tests, no external dependencies
│   ├── auth_logic/        # Pure auth business logic tests
│   ├── test_voice_service.py
│   ├── test_audio_processor.py
│   └── ...
├── integration/           # Component integration tests
├── security/              # Security and compliance tests  
├── performance/           # Load and performance tests
├── auth/                  # Authentication-specific tests
├── database/              # Database layer tests
├── fixtures/              # Reusable test fixtures
│   ├── voice_fixtures.py
│   ├── security_fixtures.py
│   └── performance_fixtures.py
└── mocks/                 # Test utilities and mocks
```

### Core Fixtures (conftest.py)
- `isolated_test_env`: Clean test environment (function-scoped)
- `mock_database`: Isolated database instance (function-scoped)
- `auth_service`: Auth service with mocked dependencies
- `sample_audio_data`: Generated audio data for voice tests
- `mock_audio_data`: Mock AudioData object

### Category-Specific Fixtures
**Voice Testing (`tests/fixtures/voice_fixtures.py`):**
- `mock_voice_config`: Comprehensive voice configuration
- `mock_stt_service`: Mock speech-to-text service
- `mock_tts_service`: Mock text-to-speech service
- `mock_audio_processor`: Mock audio processing
- `voice_test_environment`: Complete voice test setup

**Security Testing (`tests/fixtures/security_fixtures.py`):**
- `mock_encryption_service`: Mock encryption/decryption
- `mock_pii_detector`: Mock PII detection and masking
- `mock_audit_logger`: Mock audit logging
- `security_test_environment`: Complete security test setup

**Performance Testing (`tests/fixtures/performance_fixtures.py`):**
- `performance_monitor`: Mock performance monitoring
- `memory_monitor`: Real memory leak detection
- `load_tester`: Concurrent load testing utility
- `performance_test_environment`: Complete performance test setup

### Test Categories

#### Unit Tests
- Test single functions/classes in isolation
- Use function-scoped fixtures for complete isolation
- Mock all external dependencies (APIs, databases, file systems)
- Focus on business logic, not infrastructure

#### Integration Tests
- Test component interactions
- Use realistic but controlled dependencies
- Verify data flow between modules
- Test error handling across boundaries

#### Security Tests
- HIPAA compliance validation
- PII protection and masking
- Authentication and authorization
- Encryption and audit logging

#### Performance Tests
- Load testing with concurrent users
- Memory leak detection
- Response time benchmarks
- Resource usage monitoring

### Best Practices

#### Test Structure
```python
class TestAuthService:
    def test_user_registration_success(self, auth_service):
        """Test successful user registration."""
        # Arrange - Setup test data and mocks
        user_data = {"email": "test@example.com", "password": "SecurePass123"}
        
        # Act - Execute the function being tested
        result = auth_service.register_user(**user_data)
        
        # Assert - Verify the results
        assert result.success is True
        assert result.user.email == "test@example.com"
```

#### Fixtures Usage
- Always use function-scoped fixtures unless absolutely necessary
- Prefer specific fixtures over generic ones
- Compose fixtures for complex test setups
- Clean up resources in fixture teardown

#### Mocking Strategy
- Mock external APIs, not internal logic
- Use realistic mock responses
- Test both success and failure scenarios
- Verify mock calls were made correctly

#### Test Data Management
- Use fixtures for consistent test data
- Generate unique data to avoid conflicts
- Clean up temporary files and databases
- Don't hardcode sensitive information

#### Naming Conventions
- Test classes: `TestClassName`
- Test methods: `test_function_name_scenario`
- Descriptive names that explain what's being tested
- Use snake_case for all test-related code

### Coverage Requirements
- Maintain 90%+ code coverage
- Focus on critical paths and edge cases
- Test error handling and exception scenarios
- Verify security controls are tested

### CI/CD Integration
- All tests must pass in CI environment
- Use mock services to avoid external dependencies
- Run performance tests with resource limits
- Generate coverage reports for PRs

## Code Style Guidelines
- **Imports**: Standard library → third-party → local (relative preferred)
- **Type hints**: Required for all functions, use `Optional[T]`, `Union`, dataclasses
- **Error handling**: Specific exceptions with logging, graceful degradation for voice features
- **Naming**: PascalCase classes, snake_case functions/vars, UPPER_SNAKE_CASE constants
- **Testing**: 90%+ coverage, mock external dependencies, pytest fixtures
- **Security**: HIPAA compliance mandatory, PII masking, voice data encryption
- **Async**: Use `asyncio_mode = auto` from pytest.ini, test with proper await