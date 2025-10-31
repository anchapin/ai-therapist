# Test Troubleshooting Guide

## Overview

This guide provides systematic approaches to debugging test failures, identifying root causes, and implementing fixes for the AI Therapist test suite. Use this guide when encountering test failures in development or CI/CD pipelines.

## Quick Diagnosis Workflow

### Step 1: Isolate the Failure
```bash
# Run the specific failing test
python -m pytest tests/path/to/test_file.py::TestClass::test_method -v --tb=long

# Run with minimal output for speed
python -m pytest tests/path/to/test_file.py::TestClass::test_method -v --tb=line

# Run multiple times to check for flakiness
for i in {1..3}; do
  python -m pytest tests/path/to/test_file.py::TestClass::test_method -q
done
```

### Step 2: Gather Context
```bash
# Check Python environment
python --version
pip list | grep -E "(pytest|coverage|mock)"

# Check imports
python -c "import voice.voice_service; print('Import OK')"

# Check test environment
python -c "import sys; print('\\n'.join(sys.path))"
```

### Step 3: Analyze the Error

#### Common Error Patterns

## Specific Failure Types

### 1. Import Errors

#### Symptom
```
ImportError: No module named 'voice.voice_service'
ModuleNotFoundError: No module named 'package'
```

#### Root Causes
- **Missing Dependencies**: Package not installed
- **Path Issues**: Python path not configured correctly
- **Circular Imports**: Import cycles in code
- **Environment Issues**: Virtual environment not activated

#### Resolution Steps
```bash
# Check if package is installed
pip list | grep voice

# Install missing dependencies
pip install -r requirements.txt

# Check Python path
python -c "import sys; print('\\n'.join(sys.path))"

# Verify file exists
find . -name "voice_service.py"

# Test import in isolation
python -c "from voice.voice_service import VoiceService; print('Import successful')"
```

#### Prevention
- Always activate virtual environment before running tests
- Use `requirements.txt` for all dependencies
- Keep imports at the top of files
- Avoid circular import patterns

### 2. Mock Configuration Errors

#### Symptom
```
AttributeError: Mock object has no attribute 'method_name'
AssertionError: Expected 'method' to have been called
TypeError: 'Mock' object is not callable
```

#### Root Causes
- **Incorrect Mock Setup**: Mock not configured for expected calls
- **Context Manager Issues**: Mock not supporting `with` statements
- **Return Value Types**: Mock returning wrong data types
- **Patch Target**: Wrong module path being patched

#### Resolution Steps
```python
# Debug mock calls
def test_debug_mock():
    with patch('module.Class') as mock_class:
        # Your test code
        print("Mock calls:", mock_class.method.call_args_list)
        print("Mock return value:", mock_class.method.return_value)

# Check mock configuration
mock_service = Mock(spec=VoiceService)
mock_service.method.return_value = expected_value
mock_service.method.side_effect = None

# Fix context manager mocks
class MockContextManager:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

mock_st.columns.return_value = (MockContextManager(), MockContextManager())
```

#### Prevention
- Use `spec` parameter in Mock constructors
- Configure all expected return values
- Test mocks in isolation before integration
- Use `autospec=True` for better mock behavior

### 3. Async Test Failures

#### Symptom
```
RuntimeError: async def functions are not natively supported
AssertionError: coroutine 'function' was never awaited
```

#### Root Causes
- **Missing asyncio marker**: Async tests need `@pytest.mark.asyncio`
- **Incorrect async setup**: pytest-asyncio not configured
- **Await Issues**: Not awaiting coroutines properly
- **Event Loop Problems**: Multiple event loops or loop conflicts

#### Resolution Steps
```python
# Add asyncio marker
@pytest.mark.asyncio
async def test_async_function():
    result = await async_operation()
    assert result == expected

# Check asyncio configuration
# pytest.ini should contain:
# asyncio_mode = auto

# Fix event loop issues
@pytest.fixture
async def setup_async_test():
    # Setup async fixtures properly
    yield
    # Cleanup

# Debug async operations
async def debug_async():
    try:
        result = await operation()
        print(f"Operation result: {result}")
    except Exception as e:
        print(f"Async error: {e}")
```

#### Prevention
- Always use `@pytest.mark.asyncio` for async tests
- Configure pytest-asyncio properly
- Use async fixtures for async setup
- Test async functions in isolation

### 4. Database Test Failures

#### Symptom
```
sqlite3.OperationalError: disk I/O error
DatabaseError: Failed to initialize connection pool
AssertionError: Database state not as expected
```

#### Root Causes
- **File System Issues**: No write permissions or disk full
- **Connection Pool Problems**: Pool not properly initialized
- **Transaction Issues**: Uncommitted transactions
- **Concurrent Access**: Multiple tests accessing same database

#### Resolution Steps
```python
# Check file system permissions
ls -la /tmp/test_db.sqlite
df -h  # Check disk space

# Fix database isolation
@pytest.fixture
def test_db():
    db_path = tempfile.mktemp(suffix='.sqlite')
    db = DatabaseManager(db_path=db_path)
    yield db
    db.close()
    os.unlink(db_path)

# Debug database operations
def debug_db_operations():
    db = DatabaseManager()
    try:
        result = db.execute_query("SELECT * FROM users")
        print(f"Query result: {result}")
    except Exception as e:
        print(f"Database error: {e}")
        print(f"Database path: {db.db_path}")

# Reset database state
def reset_database():
    db = DatabaseManager()
    db.reset_tables()  # Custom reset method
    return db
```

#### Prevention
- Use temporary databases for each test
- Implement proper database cleanup
- Use database transactions for test isolation
- Mock database operations when possible

### 5. Memory and Resource Issues

#### Symptom
```
MemoryError: Out of memory
RuntimeError: can't start new thread
OSError: [Errno 24] Too many open files
```

#### Root Causes
- **Memory Leaks**: Resources not properly cleaned up
- **Thread Exhaustion**: Too many threads created
- **File Handle Leaks**: Files not closed properly
- **Large Test Data**: Tests using too much memory

#### Resolution Steps
```python
# Monitor memory usage
import tracemalloc

tracemalloc.start()
# Run test code
current, peak = tracemalloc.get_traced_memory()
print(f"Memory usage: {current / 1024 / 1024:.2f}MB, Peak: {peak / 1024 / 1024:.2f}MB")
tracemalloc.stop()

# Fix resource leaks
@pytest.fixture
def limited_resource():
    resource = acquire_resource()
    yield resource
    resource.release()  # Ensure cleanup

# Check thread limits
import threading
print(f"Active threads: {threading.active_count()}")
print(f"Thread limit: {threading.get_limit()}")

# Monitor file handles
import psutil
process = psutil.Process()
print(f"Open files: {len(process.open_files())}")
```

#### Prevention
- Use fixtures with proper cleanup
- Limit resource-intensive tests
- Monitor resource usage in CI/CD
- Implement resource pooling

### 6. Streamlit UI Test Failures

#### Symptom
```
TypeError: 'Mock' object does not support the context manager protocol
AttributeError: 'Mock' object has no attribute 'columns'
ValueError: not enough values to unpack (expected 2, got 0)
```

#### Root Causes
- **Mock Incompleteness**: Streamlit mocks not handling all methods
- **Context Manager Issues**: `st.columns()` returns context managers
- **Session State Problems**: `st.session_state` not properly mocked
- **UI Component Changes**: Streamlit API changes

#### Resolution Steps
```python
# Comprehensive Streamlit mock
@pytest.fixture
def mock_st():
    st_mock = MagicMock()

    # Mock columns with context managers
    def mock_columns(*args):
        num_cols = args[0] if len(args) == 1 else 2
        return tuple(MockContextManager() for _ in range(num_cols))

    st_mock.columns.side_effect = mock_columns

    # Mock session_state
    st_mock.session_state = {
        'voice_consent_given': True,
        'current_session': 'test_session',
        'ui_state': {}
    }

    return st_mock

# Fix context manager mocks
class MockContextManager:
    def __enter__(self):
        return Mock()
    def __exit__(self, *args):
        pass

# Test UI components in isolation
def test_ui_component_isolation():
    with patch('voice.voice_ui.st') as mock_st:
        # Configure mock for specific component
        mock_st.columns.return_value = (Mock(), Mock())
        mock_st.button.return_value = False

        component = VoiceUIComponents(service, config)
        result = component.render_component()
        assert result is not None
```

#### Prevention
- Create comprehensive Streamlit mocks
- Test UI components individually
- Keep mocks updated with Streamlit API changes
- Use integration tests for UI workflows

### 7. Performance Test Failures

#### Symptom
```
pytest_benchmark.exception.BenchmarkFailure: Benchmark did not reach minimum rounds
AssertionError: Performance threshold exceeded
```

#### Root Causes
- **Unstable Environment**: System load affecting benchmarks
- **Resource Contention**: Other processes interfering
- **Benchmark Configuration**: Incorrect benchmark settings
- **Code Regression**: Actual performance degradation

#### Resolution Steps
```python
# Stabilize benchmark environment
@pytest.mark.benchmark(
    group="performance-critical",
    min_rounds=5,
    max_time=1.0,
    warmup=True,
    calibration_precision=10
)
def test_performance_operation(benchmark):
    def operation():
        # Code to benchmark
        return expensive_operation()

    result = benchmark(operation)

    # Assert performance thresholds
    assert result.stats.mean < 0.1  # 100ms threshold
    assert result.stats.max < 0.5   # 500ms max

# Debug performance issues
def debug_performance():
    import cProfile
    profiler = cProfile.Profile()
    profiler.enable()

    # Run operation
    result = operation_to_profile()

    profiler.disable()
    profiler.print_stats(sort='cumulative')

# Compare benchmarks
pytest-benchmark compare benchmark-baseline.json benchmark-current.json
```

#### Prevention
- Run benchmarks in isolated environments
- Use statistical analysis for threshold setting
- Monitor system resources during benchmarks
- Establish performance regression baselines

## Systematic Debugging Process

### Level 1: Quick Diagnosis (5 minutes)

1. **Reproduce Locally**
   ```bash
   python -m pytest failing_test.py -v --tb=short
   ```

2. **Check Environment**
   ```bash
   python --version
   pip list | grep critical_packages
   ```

3. **Isolate Components**
   ```bash
   # Test imports
   python -c "import module; print('OK')"
   # Test basic functionality
   python -c "from service import Service; s = Service(); print('OK')"
   ```

### Level 2: Detailed Analysis (15 minutes)

1. **Examine Error Details**
   - Read full traceback carefully
   - Identify the exact failure point
   - Check variable values at failure point

2. **Check Dependencies**
   ```bash
   # Verify all dependencies
   pip check
   # Test related modules
   python -c "import voice, security, database; print('All imports OK')"
   ```

3. **Debug with Print Statements**
   ```python
   def test_debug():
       print("Starting test...")
       try:
           # Test code with debug prints
           step1_result = step1()
           print(f"Step 1 result: {step1_result}")
           step2_result = step2(step1_result)
           print(f"Step 2 result: {step2_result}")
           assert condition(step2_result)
       except Exception as e:
           print(f"Error at: {e}")
           raise
   ```

### Level 3: Deep Investigation (30+ minutes)

1. **Use Debugging Tools**
   ```python
   import pdb
   def test_with_debugger():
       # Set breakpoint
       pdb.set_trace()
       # Test code
   ```

2. **Profile Performance**
   ```python
   import cProfile
   cProfile.run('test_function()', 'profile_output.prof')
   ```

3. **Memory Analysis**
   ```python
   import tracemalloc
   tracemalloc.start()
   # Run test
   tracemalloc.stop()
   ```

## Common Test Anti-patterns

### 1. Over-mocking
```python
# Bad: Mocking everything
@patch('module.A')
@patch('module.B')
@patch('module.C')
def test_overmocked(mock_c, mock_b, mock_a):
    # Hard to maintain, tests implementation not behavior

# Good: Mock only external dependencies
@patch('external_api.Service')
def test_behavioral(mock_external):
    mock_external.return_value = expected_response
    result = function_under_test()
    assert result == expected
```

### 2. Brittle Tests
```python
# Bad: Testing implementation details
def test_internal_state():
    obj = Class()
    obj._internal_state = 'value'  # Testing private attribute
    assert obj._internal_state == 'value'

# Good: Testing behavior
def test_public_behavior():
    obj = Class()
    result = obj.public_method()
    assert result == expected_output
```

### 3. Slow Tests
```python
# Bad: Unnecessary delays
def test_with_sleep():
    operation()
    time.sleep(5)  # Arbitrary wait
    assert condition()

# Good: Event-driven waiting
def test_event_driven():
    operation()
    wait_for_condition(lambda: condition(), timeout=5)
```

## CI/CD Specific Issues

### GitHub Actions Failures

#### Timeout Issues
```yaml
# Increase timeout for slow tests
jobs:
  test:
    timeout-minutes: 30  # Increase from default 10
```

#### Resource Constraints
```yaml
# Use larger runner for resource-intensive tests
runs-on: ubuntu-latest-8-cores  # Instead of ubuntu-latest
```

#### Dependency Issues
```yaml
# Cache dependencies properly
- uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
```

### Parallel Execution Issues

#### Race Conditions
```python
# Use proper isolation
@pytest.fixture(scope="function")  # Not session
def isolated_resource():
    resource = create_resource()
    yield resource
    cleanup_resource(resource)
```

#### Shared State Problems
```python
# Avoid global state
class TestClass:
    def setup_method(self):
        self.resource = create_isolated_resource()

    def teardown_method(self):
        cleanup_isolated_resource(self.resource)
```

## Getting Help

### Internal Resources
1. **Team Documentation**: Check TEST_SUITE_DOCUMENTATION.md
2. **Maintenance Guide**: Review TEST_MAINTENANCE_GUIDELINES.md
3. **Performance Benchmarks**: Check PERFORMANCE_BENCHMARKS.md
4. **Code Comments**: Look for test-specific documentation

### External Resources
1. **pytest Documentation**: https://docs.pytest.org/
2. **pytest-asyncio**: https://pytest-asyncio.readthedocs.io/
3. **unittest.mock**: https://docs.python.org/3/library/unittest.mock.html
4. **GitHub Issues**: Search existing issues for similar problems

### Escalation Process
1. **Try Local Reproduction**: Can you reproduce the issue locally?
2. **Check CI Logs**: Are there environment-specific issues?
3. **Review Recent Changes**: Did recent code changes cause this?
4. **Create Minimal Test Case**: Isolate the issue
5. **Document and Report**: Create issue with full details

## Prevention Strategies

### Test Quality Gates
- **Code Review**: Require test review for all changes
- **Coverage Checks**: Enforce minimum coverage thresholds
- **Performance Monitoring**: Track performance regressions
- **Flakiness Detection**: Monitor for flaky tests

### Development Practices
- **Test-First Development**: Write tests before implementation
- **Continuous Testing**: Run tests frequently during development
- **Clean Test Data**: Use realistic but safe test data
- **Documentation**: Keep test documentation current

### Maintenance Rituals
- **Weekly Review**: Review test failures and fix issues
- **Monthly Audit**: Comprehensive test suite health check
- **Quarterly Updates**: Update test infrastructure and dependencies
- **Annual Review**: Complete test strategy assessment

---

*Remember: Most test failures have simple root causes. Start with isolation, check the environment, and work systematically through the debugging process.*
