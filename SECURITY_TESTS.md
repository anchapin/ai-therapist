# Voice Security Test Suite

This directory contains comprehensive security tests for the AI Therapist voice module, validating all security fixes implemented for PR #1.

## Overview

The security test suite validates:
1. **Input Validation** - Tests validation in `voice/security.py`
2. **Memory Leak Prevention** - Tests buffer management in `voice/audio_processor.py`
3. **Thread Safety** - Tests session management in `voice/voice_service.py`
4. **Integration Security** - Tests overall system security

## Files

### Core Test Files
- `test_voice_security_comprehensive.py` - Complete security test suite
- `test_security_quick.py` - Quick validation tests
- `test_security_fixes.py` - Original security fix tests

### Configuration Files
- `pytest.ini` - Pytest configuration
- `run_security_tests.py` - Test runner script
- `test_voice_config.py` - Test configuration and utilities

### Documentation
- `SECURITY_TESTS.md` - This documentation file

## Running Tests

### Quick Validation
Run quick validation tests to check if the test suite is ready:

```bash
python test_security_quick.py
```

### Full Security Test Suite
Run all security tests:

```bash
# Using the test runner (recommended)
python run_security_tests.py

# Using pytest directly
pytest test_voice_security_comprehensive.py -v

# With coverage report
python run_security_tests.py --coverage
pytest test_voice_security_comprehensive.py --cov=voice --cov-report=html
```

### Running Specific Test Categories

```bash
# Input validation tests only
python run_security_tests.py --validation
pytest test_voice_security_comprehensive.py -m input_validation

# Memory safety tests only
python run_security_tests.py --memory
pytest test_voice_security_comprehensive.py -m memory_safety

# Thread safety tests only
python run_security_tests.py --thread
pytest test_voice_security_comprehensive.py -m thread_safety

# Integration tests only
python run_security_tests.py --integration
pytest test_voice_security_comprehensive.py -m integration

# Unit tests only
python run_security_tests.py --unit
pytest test_voice_security_comprehensive.py -m unit
```

### Test Options

```bash
# Verbose output
python run_security_tests.py --verbose

# Quick tests only (skip slow/intensive tests)
python run_security_tests.py --quick

# Skip dependency checking
python run_security_tests.py --no-deps-check

# Custom pytest options
pytest test_voice_security_comprehensive.py -v --tb=short --strict-markers
```

## Test Categories

### 1. Input Validation Tests (`TestInputValidation`)

Validates that the security module properly validates and sanitizes input:

**Valid Input Tests:**
- User ID formats (alphanumeric, underscore, hyphen)
- IP address formats (IPv4)
- User agent strings
- Consent types
- Consent text length limits

**Invalid Input Tests:**
- SQL injection attempts
- XSS attacks
- Path traversal attacks
- Unicode/encoding attacks
- Overly long inputs
- Special characters
- Null bytes

**Security Features Tested:**
- Regular expression validation
- Length limits
- Character filtering
- Type checking
- Consent record persistence
- Audit log creation

### 2. Memory Leak Prevention Tests (`TestMemoryLeakPrevention`)

Validates that the audio processor prevents memory exhaustion:

**Buffer Management:**
- Bounded deque implementation
- Buffer size enforcement
- Memory usage tracking
- Automatic cleanup

**Memory Limit Enforcement:**
- Memory limit checking
- Large data rejection
- Buffer overflow protection
- Force cleanup functionality

**Recording Safety:**
- Memory monitoring during recording
- Chunk size validation
- Thread-safe cleanup
- Error handling

### 3. Thread Safety Tests (`TestThreadSafety`)

Validates that the voice service handles concurrent access safely:

**Session Management:**
- Concurrent session creation
- Thread-safe session access
- Concurrent session destruction
- High-load operations

**State Management:**
- Lock implementation
- Metrics updates
- Callback registration
- State consistency

**Event Loop Safety:**
- Event loop reference handling
- Audio callback safety
- Async/sync mixing prevention

### 4. Integration Security Tests (`TestIntegrationSecurity`)

Validates overall system security integration:

**System-wide Security:**
- Malicious input propagation
- DoS protection
- Error information disclosure prevention
- Consent flow security

**Compliance Features:**
- HIPAA compliance
- GDPR compliance
- Data retention
- Emergency protocols

**Audit and Logging:**
- Audit trail integrity
- File access security
- Concurrent operations

### 5. Edge Case Tests (`TestSecurityEdgeCases`)

Tests boundary conditions and edge cases:

**Boundary Values:**
- Maximum length inputs
- Empty/null inputs
- Unicode edge cases
- Rapid succession attacks

**Stress Testing:**
- Memory exhaustion resilience
- Filesystem boundary conditions
- Concurrent file access

## Security Fixes Validated

### 1. Input Validation in `voice/security.py`

**Fixes Validated:**
- ✅ User ID format validation with regex patterns
- ✅ IP address format validation
- ✅ User agent sanitization and length limits
- ✅ Consent type validation against allowed values
- ✅ Consent text length limits
- ✅ SQL injection prevention
- ✅ XSS attack prevention
- ✅ Unicode and encoding attack handling

**Code Coverage:**
- `_validate_user_id()` method
- `_validate_ip_address()` method
- `_validate_user_agent()` method
- `_validate_consent_type()` method
- `grant_consent()` method
- Consent record persistence
- Audit logging functionality

### 2. Memory Leak Prevention in `voice/audio_processor.py`

**Fixes Validated:**
- ✅ Bounded audio buffer using deque with maxlen
- ✅ Memory usage tracking and limits
- ✅ Buffer cleanup on memory limit reached
- ✅ Force cleanup functionality
- ✅ Thread-safe memory management
- ✅ Recording thread cleanup
- ✅ Error handling in cleanup

**Code Coverage:**
- Audio buffer initialization
- Memory monitoring in recording callback
- Buffer size enforcement
- Cleanup functionality
- Error handling

### 3. Thread Safety in `voice/voice_service.py`

**Fixes Validated:**
- ✅ Thread-safe session management with RLock
- ✅ Concurrent session creation/access/destruction
- ✅ High-load session operations
- ✅ State consistency under concurrency
- ✅ Event loop reference safety
- ✅ Metrics update thread safety
- ✅ Callback registration safety

**Code Coverage:**
- Session management methods
- Lock usage
- Concurrent operations
- State management

### 4. Integration Security

**Fixes Validated:**
- ✅ Malicious input blocking throughout system
- ✅ DoS protection through memory limits
- ✅ Error information disclosure prevention
- ✅ Consent flow security
- ✅ Emergency protocol security
- ✅ Audit trail integrity
- ✅ Compliance feature functionality
- ✅ Data retention security

## Dependencies

### Required Python Packages
- `pytest` - Test framework
- `numpy` - Numerical computing for audio data
- Standard library modules:
  - `threading`
  - `asyncio`
  - `tempfile`
  - `shutil`
  - `pathlib`
  - `json`
  - `re`
  - `logging`
  - `unittest.mock`
  - `concurrent.futures`

### Optional Dependencies
- `pytest-cov` - Coverage reporting
- `pytest-timeout` - Test timeout functionality

### Voice Module Dependencies
The tests require the voice module to be available:
- `voice.security`
- `voice.audio_processor`
- `voice.voice_service`
- `voice.config`

## Test Configuration

### Environment Setup
Tests use temporary directories and mock configurations to avoid affecting the production environment:

```python
# Temporary directories are created for:
- voice_data/
- voice_data/encrypted/
- voice_data/consents/
- voice_data/audit/
- voice_data/emergency/
```

### Mock Configurations
Tests use mock configurations with reduced limits for faster testing:
- Buffer size: 50 chunks (vs 300 in production)
- Memory limit: 10MB (vs 100MB in production)
- Small data sizes for validation tests

## Test Results Interpretation

### Success Indicators
- ✅ All tests pass without exceptions
- ✅ Memory usage stays within limits
- ✅ No race conditions detected
- ✅ Malicious inputs are properly rejected
- ✅ Valid inputs are properly accepted
- ✅ Audit logs are created correctly
- ✅ Thread safety is maintained

### Failure Indicators
- ❌ Security validation allows malicious input
- ❌ Memory limits are exceeded
- ❌ Race conditions or deadlocks
- ❌ Test files contain sensitive information
- ❌ Audit logs are missing or corrupted
- ❌ Thread safety violations

### Performance Metrics
Tests monitor:
- Execution time (should complete quickly)
- Memory usage (should stay within limits)
- Thread contention (should be minimal)
- File I/O operations (should be efficient)

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   ImportError: No module named 'voice.security'
   ```
   **Solution:** Ensure the voice module is in the Python path and properly installed.

2. **Permission Errors**
   ```
   PermissionError: [Errno 13] Permission denied
   ```
   **Solution:** Check file permissions for temporary directories and voice_data.

3. **Memory Errors**
   ```
   MemoryError: Unable to allocate array
   ```
   **Solution:** Reduce test memory limits or ensure sufficient system memory.

4. **Thread/Timeout Errors**
   ```
   TimeoutError: Test timed out
   ```
   **Solution:** Increase timeout limits or check for deadlocks.

### Debug Options

```bash
# Run with maximum verbosity
pytest test_voice_security_comprehensive.py -vv -s

# Run with specific test debugging
pytest test_voice_security_comprehensive.py::TestInputValidation::test_sql_injection_attempts -vv -s

# Run with logging
pytest test_voice_security_comprehensive.py --log-cli-level=DEBUG

# Run with traceback
pytest test_voice_security_comprehensive.py --tb=long
```

## Contributing

### Adding New Security Tests

1. **Add test class to appropriate category:**
   ```python
   class TestNewSecurityFeature:
       def test_new_feature_validation(self):
           # Test implementation
           pass
   ```

2. **Add appropriate markers:**
   ```python
   @pytest.mark.security
   @pytest.mark.input_validation
   def test_new_validation(self):
       pass
   ```

3. **Update test configuration:**
   - Add new markers to `pytest.ini`
   - Update test documentation
   - Add to test runner options

### Test Naming Conventions
- Test classes: `Test<FeatureName>`
- Test methods: `test_<specific_behavior>`
- Fixtures: `<feature>_config`, `mock_<component>`

### Test Data Management
- Use temporary directories for file operations
- Clean up resources in fixtures
- Mock external dependencies
- Use deterministic test data

## Continuous Integration

### GitHub Actions Integration
```yaml
- name: Run Security Tests
  run: |
    python test_security_quick.py
    python run_security_tests.py --coverage
```

### Pre-commit Hooks
```bash
# Quick validation before commits
python test_security_quick.py

# Full security validation for PRs
python run_security_tests.py --integration --thread --memory --validation
```

## Security Considerations

### Test Data Safety
- Tests use mock/temporary data only
- No production data is accessed
- Sensitive information is not logged
- Temporary files are cleaned up

### Test Environment Isolation
- Tests run in isolated temporary directories
- Mock configurations prevent system interaction
- Network access is mocked/not required
- Audio hardware access is mocked

### Vulnerability Disclosure
If security tests reveal new vulnerabilities:
1. Document the vulnerability clearly
2. Create appropriate test cases
3. Report to security team
4. Implement fixes
5. Validate fixes with tests

## Maintenance

### Regular Updates
- Update test cases for new security features
- Review and update mock configurations
- Monitor test performance and coverage
- Update dependencies as needed

### Test Coverage Monitoring
- Aim for >90% coverage of security-critical code
- Monitor coverage trends
- Add tests for uncovered security code
- Regular security audit reviews

---

For questions or issues with the security test suite, please refer to the test documentation or create an issue in the project repository.