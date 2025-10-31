# AI Therapist Test Suite Documentation

## Overview

The AI Therapist project maintains a comprehensive test suite designed to ensure code quality, security, performance, and reliability. This document provides detailed information about all test suites, their purposes, coverage areas, and maintenance procedures.

## Test Suite Architecture

### Test Categories

| Category | Location | Purpose | Coverage Target |
|----------|----------|---------|-----------------|
| **Unit Tests** | `tests/unit/` | Component-level testing | 90% |
| **Integration Tests** | `tests/integration/` | Service integration testing | 85% |
| **Security Tests** | `tests/security/` | HIPAA compliance and security | 95% |
| **Performance Tests** | `tests/performance/` | Load and scalability testing | 80% |
| **UI Component Tests** | `tests/voice/test_voice_ui_components.py` | Streamlit UI component testing | 70% |
| **Edge Cases** | `tests/test_edge_cases_and_boundary_conditions.py` | Boundary conditions and cross-platform | 70% |

### Test Execution

#### Primary Test Runner
```bash
# Comprehensive test execution
python tests/test_runner.py

# Individual test categories
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v
python -m pytest tests/security/ -v
python -m pytest tests/performance/ -v
```

#### CI/CD Integration
Tests are automatically executed via GitHub Actions:
- **optimized-ci.yml**: Comprehensive testing on PRs and pushes
- **main-branch-ci.yml**: Quality gates for main branch

## Detailed Test Suite Documentation

### 1. Unit Tests (`tests/unit/`)

#### Purpose
Unit tests validate individual components in isolation, ensuring each function, class, and module works correctly under various conditions.

#### Coverage Areas
- **Core Application Logic** (`test_app_core.py`)
  - Security function validation
  - Input sanitization
  - Error handling
  - Authentication logic

- **Audio Processing** (`test_audio_processor.py`)
  - Audio format handling
  - Signal processing algorithms
  - Error conditions

- **Security Functions** (`test_security_functions.py`)
  - PII detection and masking
  - Encryption/decryption
  - Access control validation

- **Memory Management** (`test_memory_manager.py`)
  - Resource allocation
  - Memory leak prevention
  - Cleanup procedures

#### Key Test Files
```
tests/unit/
├── test_app_core.py              # Core application functionality
├── test_audio_processor.py       # Audio processing components
├── test_security_functions.py    # Security utilities
├── test_memory_manager.py        # Memory management
├── test_auth_middleware.py       # Authentication middleware
├── test_session_management.py    # Session handling
└── test_stt_service.py          # Speech-to-text services
```

#### Maintenance Procedures
- **Adding New Tests**: Create new test files following naming convention `test_*.py`
- **Test Isolation**: Each test should be independent with proper setup/teardown
- **Mock Usage**: Use `unittest.mock` for external dependencies
- **Coverage Requirements**: Maintain 90% coverage for unit tests

### 2. Integration Tests (`tests/integration/`)

#### Purpose
Integration tests verify that different components work together correctly, testing the interactions between modules and services.

#### Coverage Areas
- **Authentication Flow** (`test_auth_pii_integration.py`)
  - User registration and login
  - PII protection integration
  - Session management

- **Voice Service Integration** (`test_voice_service_integration.py`)
  - STT/TTS service coordination
  - Audio processing pipeline
  - Error propagation

- **Database Operations** (`test_database_integration.py`)
  - Connection pooling
  - Transaction integrity
  - Concurrent access patterns

#### Key Test Files
```
tests/integration/
├── test_auth_pii_integration.py         # Auth and PII integration
├── test_voice_service_integration.py    # Voice service coordination
├── test_database_integration.py         # Database operations
└── test_api_endpoints.py               # API endpoint testing
```

#### Maintenance Procedures
- **Test Data**: Use isolated test databases and mock external services
- **Setup/Teardown**: Ensure proper cleanup between tests
- **Async Testing**: Use `pytest-asyncio` for async integration tests
- **Performance**: Monitor test execution time for performance regressions

### 3. Security Tests (`tests/security/`)

#### Purpose
Security tests ensure HIPAA compliance, data protection, and resistance to common security vulnerabilities.

#### Coverage Areas
- **Access Control** (`test_access_control.py`)
  - Role-based permissions
  - Authentication validation
  - Authorization checks

- **Encryption** (`test_encryption_comprehensive.py`)
  - Data encryption/decryption
  - Key management
  - Secure storage

- **PII Protection** (`test_pii_protection.py`)
  - Sensitive data detection
  - Masking algorithms
  - Compliance validation

#### Key Test Files
```
tests/security/
├── test_access_control.py              # Access control validation
├── test_encryption_comprehensive.py    # Encryption functionality
├── test_pii_protection.py             # PII detection and masking
├── test_input_validation.py           # Input sanitization
└── test_audit_logging.py              # Security audit trails
```

#### Maintenance Procedures
- **Security Standards**: Regularly update tests for new security requirements
- **Compliance**: Ensure tests cover HIPAA and data protection regulations
- **Vulnerability Testing**: Include tests for common attack vectors
- **Audit Trails**: Maintain comprehensive logging for security events

### 4. Performance Tests (`tests/performance/`)

#### Purpose
Performance tests validate system scalability, memory efficiency, and response times under various load conditions.

#### Coverage Areas
- **Cache Management** (`test_cache_management.py`)
  - LRU eviction algorithms
  - TTL (Time-To-Live) handling
  - Concurrent access patterns
  - Compression and memory optimization

- **Memory Management** (`test_memory_management.py`)
  - Resource monitoring
  - Leak detection
  - Cleanup procedures
  - Memory alerts

- **Database Performance** (`test_database_integration.py`)
  - Connection pooling efficiency
  - Query optimization
  - Transaction performance
  - Concurrent access handling

#### Key Test Files
```
tests/performance/
├── test_cache_management.py           # Cache performance and algorithms
├── test_memory_management.py          # Memory usage and leaks
├── test_database_integration.py       # Database performance
├── test_simple_performance.py         # Basic performance benchmarks
└── test_cache_performance.py          # Cache-specific performance
```

#### Maintenance Procedures
- **Benchmarking**: Use `pytest-benchmark` for consistent performance measurements
- **Thresholds**: Set performance regression thresholds
- **Profiling**: Include memory profiling in performance tests
- **Load Testing**: Test under various load conditions

### 5. UI Component Tests (`tests/voice/test_voice_ui_components.py`)

#### Purpose
UI component tests validate Streamlit interface components, ensuring proper rendering, interaction handling, and accessibility.

#### Coverage Areas
- **Component Initialization** (38 tests)
  - Voice UI setup and configuration
  - Callback registration
  - CSS injection and styling

- **State Management** (15 tests)
  - UI state updates and persistence
  - Session state handling
  - State reset functionality

- **Component Rendering** (25 tests)
  - Voice interface rendering
  - Consent form display
  - Status indicators

- **User Interactions** (42 tests)
  - Button clicks and form submissions
  - Keyboard shortcuts
  - Touch gestures (mobile)

- **Accessibility Features** (12 tests)
  - Screen reader compatibility
  - Keyboard navigation
  - ARIA labels and descriptions

- **Emergency Protocols** (15 tests)
  - Crisis keyword detection
  - Emergency contact handling
  - Session logging for emergencies

- **Error Handling** (18 tests)
  - Microphone errors
  - Network connectivity issues
  - Audio device failures
  - Rate limiting feedback

- **Audio Processing Integration** (8 tests)
  - Waveform visualization
  - FFT computation
  - Volume level calculation

- **Async Operations** (15 tests)
  - Voice button press handling
  - Status announcements
  - Real-time display updates

#### Key Test Classes
```
Test Classes in test_voice_ui_components.py:
├── TestVoiceUIComponentsInitialization     # Component setup
├── TestVoiceUIStateManagement             # State handling
├── TestVoiceUIComponentRendering          # UI rendering
├── TestVoiceUIConsentForm                 # Consent management
├── TestVoiceUIHeaderAndStatus             # Header/status display
├── TestVoiceUIInputInterface              # Input controls
├── TestVoiceUITranscriptionDisplay        # Transcription UI
├── TestVoiceUIVisualization               # Audio visualization
├── TestVoiceUIOutputControls              # Output controls
├── TestVoiceUISettingsPanel               # Settings interface
├── TestVoiceUICommandsReference           # Command reference
├── TestVoiceUIKeyboardShortcuts           # Keyboard shortcuts
├── TestVoiceUIMobileResponsiveness        # Mobile optimization
├── TestVoiceUIAccessibility               # Accessibility features
├── TestVoiceUIEmergencyProtocol           # Emergency handling
├── TestVoiceUIErrorHandling               # Error states
├── TestVoiceUIAudioProcessingIntegration  # Audio processing
├── TestVoiceUIAsyncOperations             # Async operations
├── TestVoiceUIFactoryFunction             # Factory functions
├── TestVoiceUIUtilityFunctions            # Utility functions
└── TestVoiceUIStateQueries                # State queries
```

#### Maintenance Procedures
- **Mock Complexity**: Maintain comprehensive Streamlit mocks for context managers
- **Async Testing**: Use `@pytest.mark.asyncio` for async UI operations
- **Accessibility**: Regularly test with screen readers and accessibility tools
- **Cross-browser**: Test UI components across different browsers

### 6. Edge Cases and Boundary Conditions (`tests/test_edge_cases_and_boundary_conditions.py`)

#### Purpose
Edge case tests validate system behavior under extreme conditions, boundary values, and cross-platform scenarios.

#### Coverage Areas
- **Input Validation** (10 tests)
  - Empty/null inputs
  - Maximum length data
  - Malformed data structures

- **Resource Exhaustion** (12 tests)
  - Memory exhaustion scenarios
  - Thread pool limits
  - File handle exhaustion
  - Database connection limits

- **Data Corruption** (9 tests)
  - Corrupted audio data
  - Database corruption recovery
  - JSON parsing errors
  - Memory corruption simulation

- **Cross-platform Compatibility** (8 tests)
  - Path handling across platforms
  - File encoding compatibility
  - Platform-specific service availability

- **Integration Edge Cases** (8 tests)
  - Service initialization order dependencies
  - Component interaction boundary conditions
  - System shutdown simulation

#### Key Test Classes
```
Test Classes in test_edge_cases_and_boundary_conditions.py:
├── TestExtremeInputValidation            # Input boundary testing
├── TestSystemResourceExhaustion          # Resource limit testing
├── TestDataCorruptionAndIntegrity        # Data integrity testing
├── TestConcurrentStateCorruptionPrevention # Concurrency testing
├── TestCrossPlatformCompatibility        # Platform compatibility
└── TestIntegrationEdgeCases              # Integration edge cases
```

#### Maintenance Procedures
- **Platform Testing**: Test on Windows, macOS, and Linux when possible
- **Resource Limits**: Adjust tests based on system capabilities
- **Error Simulation**: Use realistic error conditions for testing
- **Boundary Expansion**: Update tests as system limits change

## Test Infrastructure

### Configuration Files

#### `pytest.ini`
```ini
[tool:pytest]
testpaths = .
python_files = test_*.py *_test.py
asyncio_mode = auto
markers =
    security: Security-related tests
    integration: Integration tests
    performance: Performance tests
timeout = 120
```

#### `conftest.py`
- Shared fixtures for test isolation
- Mock configurations
- Test environment management
- Async test support

### Test Execution Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=voice --cov=security --cov=auth --cov=performance --cov=database --cov-report=html

# Run specific test categories
python -m pytest tests/unit/ -m "not slow"
python -m pytest tests/security/ -v
python -m pytest tests/performance/ --benchmark-only

# Debug failing tests
python -m pytest tests/ -v --tb=long --maxfail=1
```

### CI/CD Integration

#### Coverage Requirements
- **Overall Target**: 60% minimum (enforced in CI)
- **Unit Tests**: 90% target
- **Security Tests**: 95% target
- **Integration Tests**: 85% target
- **Performance Tests**: 80% target

#### Automated Reporting
- **Codecov**: Trend analysis and PR comments
- **Test Reports**: JSON format with detailed results
- **Artifact Storage**: 7-30 day retention based on importance

## Maintenance Guidelines

### Adding New Tests

1. **Identify Test Category**: Determine appropriate test category (unit/integration/security/performance)
2. **Follow Naming Convention**: `test_*.py` for files, `test_*` for functions
3. **Use Proper Fixtures**: Leverage existing fixtures in `conftest.py`
4. **Include Documentation**: Add docstrings explaining test purpose
5. **Mock External Dependencies**: Use mocks for external services and I/O

### Test Maintenance Procedures

#### Regular Tasks
- **Weekly**: Review test failures and fix flaky tests
- **Monthly**: Update test data and mock configurations
- **Quarterly**: Review and update performance baselines

#### Coverage Monitoring
- Monitor coverage trends in Codecov
- Address coverage gaps in new code
- Maintain minimum coverage thresholds

#### Test Data Management
- Use realistic but anonymized test data
- Regularly update test datasets
- Ensure test data doesn't expose sensitive information

### Debugging Test Failures

#### Common Issues
- **Import Errors**: Check Python path and dependencies
- **Mock Failures**: Verify mock configurations and return values
- **Async Issues**: Ensure proper `@pytest.mark.asyncio` usage
- **Resource Leaks**: Check for proper cleanup in test teardown

#### Debugging Commands
```bash
# Verbose test execution
python -m pytest test_file.py -v -s

# Debug specific test
python -m pytest test_file.py::TestClass::test_method -v --tb=long

# Profile test performance
python -m pytest test_file.py --durations=10
```

### Performance Benchmarking

#### Benchmark Categories
- **Unit Test Performance**: < 0.1s per test
- **Integration Tests**: < 1.0s per test
- **Performance Tests**: < 5.0s per test
- **Full Suite**: < 10 minutes

#### Regression Detection
- Automated performance checks on main branch
- Benchmark result comparison with previous runs
- Alert on performance degradation > 10%

## Quality Assurance

### Test Quality Metrics
- **Test Pass Rate**: > 95% (excluding known issues)
- **Coverage Maintenance**: > 60% overall
- **Performance Stability**: < 5% variance between runs
- **Flaky Test Rate**: < 1%

### Code Review Requirements
- All new code must include corresponding tests
- Test coverage must not decrease
- Performance tests for performance-critical code
- Security tests for security-related changes

## Support and Troubleshooting

### Getting Help
1. Check this documentation first
2. Review pytest output for error details
3. Check CI/CD logs for environment-specific issues
4. Create issue with test failure details

### Common Problems
- **Mock Import Errors**: Update import paths in test files
- **Async Test Failures**: Add `@pytest.mark.asyncio` decorator
- **Coverage Issues**: Check for untested code paths
- **Performance Regressions**: Profile code for bottlenecks

---

*This documentation is maintained alongside the test suite. Please update when adding new test categories or changing test procedures.*
