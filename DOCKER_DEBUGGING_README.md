# AI Therapist Docker Debugging System

## Overview

This comprehensive Docker Compose setup provides systematic debugging and resolution of AI Therapist test failures. The system creates isolated, reproducible testing environments that can systematically identify and fix each category of test failure.

## Architecture

### Components

1. **Ollama Service** - Provides LLM functionality for testing
2. **Dependency Checker** - Validates all required dependencies
3. **Unit Test Debugger** - Analyzes and fixes unit test issues
4. **Integration Test Debugger** - Handles service integration and numpy recursion problems
5. **Security Test Debugger** - Addresses HIPAA compliance and access control issues
6. **Performance Test Debugger** - Resolves memory leaks and resource exhaustion
7. **Fix Applier** - Automatically applies identified fixes
8. **Test Validator** - Final comprehensive validation
9. **Debug Monitor** - Web-based monitoring dashboard

### Key Features

- **Systematic Debugging**: Each test category has dedicated debugging scripts
- **Automated Fix Application**: Common issues are automatically fixed
- **Comprehensive Monitoring**: Real-time dashboard shows progress
- **Evidence-Based**: All decisions based on actual test results
- **Reproducible Environment**: Docker ensures consistent testing environment

## Quick Start

### Prerequisites

- Docker and Docker Compose
- At least 8GB RAM
- 10GB available disk space

### Running the Complete Debugging Process

```bash
# Clone and navigate to the project
cd ai-therapist

# Run the complete debugging pipeline
docker-compose -f docker-compose.debug.yml up --build

# For background execution
docker-compose -f docker-compose.debug.yml up -d --build

# Monitor progress
open http://localhost:8080
```

### Running Individual Components

```bash
# Run dependency validation only
docker-compose -f docker-compose.debug.yml up dependency-checker

# Run unit test debugging only
docker-compose -f docker-compose.debug.yml up unit-test-debugger

# Run security test debugging only
docker-compose -f docker-compose.debug.yml up security-test-debugger
```

## Debugging Process

### Phase 1: Dependency Validation

The `dependency-checker` service systematically validates:

- **Python Package Imports**: Checks all required packages can be imported
- **System Dependencies**: Validates system-level dependencies (ffmpeg, portaudio, etc.)
- **Project Structure**: Ensures all required files exist
- **Version Compatibility**: Checks for version conflicts

**Output**: `reports/dependency-report.json`

### Phase 2: Unit Test Debugging

The `unit-test-debugger` service addresses:

- **Import Errors**: Missing or incorrect import statements
- **Mocking Issues**: `__spec__` attribute problems and mock configuration
- **Test Logic Flaws**: Logical errors in test assertions
- **Configuration Problems**: Missing test configuration

**Output**: `reports/unit-debug-report.json`

### Phase 3: Integration Test Debugging

The `integration-test-debugger` service handles:

- **Service Mocking**: External service integration issues
- **Numpy Recursion**: Recursion depth problems with numpy arrays
- **Dependency Issues**: Cross-service dependency problems
- **Configuration**: Integration test environment setup

**Output**: `reports/integration-debug-report.json`

### Phase 4: Security Test Debugging

The `security-test-debugger` service resolves:

- **HIPAA Compliance**: HIPAA requirement validation
- **Encryption Issues**: Cryptography module problems
- **Access Control**: Permission and role-based access issues
- **Audit Logging**: Security audit trail problems

**Output**: `reports/security-debug-report.json`

### Phase 5: Performance Test Debugging

The `performance-test-debugger` service addresses:

- **Memory Leaks**: Memory usage and garbage collection
- **Timeout Issues**: Test execution timeout problems
- **Resource Exhaustion**: File handles, processes, threads
- **Performance Metrics**: psutil and monitoring setup

**Output**: `reports/performance-debug-report.json`

### Phase 6: Automated Fix Application

The `fix-applier` service:

- **Analyzes Reports**: Reads all debug reports
- **Creates Backups**: Creates backup of original files
- **Applies Fixes**: Systematically applies identified fixes
- **Validates Fixes**: Ensures fixes don't break existing functionality

**Output**: `reports/fix-report.json`

### Phase 7: Final Validation

The `test-validator` service:

- **Runs All Tests**: Executes complete test suite
- **Analyzes Coverage**: Generates coverage reports
- **Validates Compliance**: Checks requirement compliance
- **Performance Analysis**: Analyzes test performance
- **Generates Report**: Creates final validation report

**Output**: `reports/final-validation-report.json`

## Monitoring Dashboard

Access the monitoring dashboard at `http://localhost:8080`:

### Features

- **Real-time Status**: Live status of all debugging phases
- **Progress Tracking**: Visual progress bars for each category
- **Report Access**: Direct links to all generated reports
- **Activity Log**: Recent debugging activities
- **Auto-refresh**: Automatically updates every 30 seconds

### Metrics Tracked

- **Dependency Validation**: Total/Successful imports
- **Unit Tests**: Passed/Failed tests, success rate
- **Integration Tests**: Service integration status
- **Security Tests**: HIPAA compliance status
- **Performance Tests**: Memory and resource usage
- **Overall Status**: Combined success rate and fixes applied

## Report Analysis

### Understanding Report Structure

Each report follows a consistent structure:

```json
{
  "metadata": {
    "timestamp": "2025-10-01T19:30:00",
    "script": "debug_unit_tests.py"
  },
  "summary": {
    "total_tests": 15,
    "successful_tests": 12,
    "failed_tests": 3,
    "success_rate": 0.8
  },
  "issues": [
    {
      "type": "IMPORT_ERROR",
      "file": "test_audio_processor.py",
      "message": "ModuleNotFoundError: No module named 'psutil'"
    }
  ],
  "fixes_applied": [
    {
      "type": "DEPENDENCY_INSTALL",
      "description": "Installed missing psutil dependency",
      "success": true
    }
  ]
}
```

### Key Report Locations

- **Dependency Report**: `/reports/dependency-report.json`
- **Unit Test Report**: `/reports/unit-debug-report.json`
- **Integration Report**: `/reports/integration-debug-report.json`
- **Security Report**: `/reports/security-debug-report.json`
- **Performance Report**: `/reports/performance-debug-report.json`
- **Fix Report**: `/reports/fix-report.json`
- **Final Validation**: `/reports/final-validation-report.json`

## Common Issues and Solutions

### Issue: Import Errors

**Symptoms**: `ModuleNotFoundError` or `ImportError` in tests

**Solutions Applied**:
- Install missing dependencies via pip
- Fix import statements in test files
- Add proper Python path configuration
- Mock unavailable modules

### Issue: Mock Configuration Problems

**Symptoms**: `AttributeError: __spec__` or mock-related failures

**Solutions Applied**:
- Add `__spec__=None` to MagicMock calls
- Fix patch decorators
- Add proper mock return values
- Configure mock attributes

### Issue: Numpy Recursion

**Symptoms**: `RecursionError` or infinite recursion with numpy arrays

**Solutions Applied**:
- Set recursion limit: `sys.setrecursionlimit(1000)`
- Fix nested numpy array creation
- Add numpy safety configurations
- Mock numpy operations where appropriate

### Issue: Security Test Logic

**Symptoms**: Access control assertion failures

**Solutions Applied**:
- Fix permission overlap logic
- Update role-based access tests
- Add proper HIPAA configuration
- Fix encryption mocking

### Issue: Performance Test Timeouts

**Symptoms**: Tests timing out or resource exhaustion

**Solutions Applied**:
- Add proper timeout handling
- Implement resource cleanup
- Add memory management
- Optimize test execution

## Manual Intervention

While the system is highly automated, some issues may require manual intervention:

### When to Intervene

1. **Fix Application Fails**: When automated fixes don't resolve issues
2. **Complex Mocking**: When service integration requires custom mocking
3. **Performance Issues**: When tests are too slow or consume too many resources
4. **Configuration Conflicts**: When environment settings conflict

### Manual Debugging Steps

1. **Check Logs**: Examine container logs for detailed error information
2. **Run Tests Locally**: Reproduce issues in local environment
3. **Examine Reports**: Review detailed debugging reports
4. **Modify Tests**: Make necessary changes to test files
5. **Re-run Pipeline**: Execute debugging pipeline again

### Accessing Container Logs

```bash
# View logs for specific service
docker-compose -f docker-compose.debug.yml logs unit-test-debugger

# View real-time logs
docker-compose -f docker-compose.debug.yml logs -f integration-test-debugger

# View all logs
docker-compose -f docker-compose.debug.yml logs
```

### Accessing Container Shell

```bash
# Access running container
docker-compose -f docker-compose.debug.yml exec unit-test-debugger bash

# Start specific service and access shell
docker-compose -f docker-compose.debug.yml run --rm unit-test-debugger bash
```

## Customization

### Adding New Debug Scripts

1. Create script in `/scripts/` directory
2. Add service to `docker-compose.debug.yml`
3. Update monitoring dashboard if needed
4. Modify validation script to include new category

### Modifying Debugging Logic

1. Edit scripts in `/scripts/` directory
2. Rebuild Docker image: `docker-compose -f docker-compose.debug.yml build`
3. Run updated debugging pipeline

### Custom Configuration

1. Modify environment variables in `docker-compose.debug.yml`
2. Update debugging thresholds in individual scripts
3. Adjust timeout values as needed

## Troubleshooting

### Common Docker Issues

**Issue: Out of Memory**
```bash
# Increase Docker memory allocation
# Or run services individually
docker-compose -f docker-compose.debug.yml up unit-test-debugger
```

**Issue: Port Conflicts**
```bash
# Check port usage
netstat -tulpn | grep :8080

# Change port in docker-compose.debug.yml
ports:
  - "8081:80"  # Use different host port
```

**Issue: Permission Errors**
```bash
# Fix volume permissions
sudo chown -R $USER:$USER reports/ logs/
```

### Service-Specific Issues

**Ollama Service Not Starting**
```bash
# Check Docker logs
docker-compose -f docker-compose.debug.yml logs ollama

# Manual Ollama setup
docker run -d -p 11434:11434 --name ollama ollama/ollama
```

**Test Services Failing**
```bash
# Check dependencies
docker-compose -f docker-compose.debug.yml exec dependency-checker python scripts/validate_dependencies.py

# Run individual test category
docker-compose -f docker-compose.debug.yml exec unit-test-debugger python -m pytest tests/unit/ -v
```

## Performance Optimization

### Resource Allocation

For optimal performance, ensure:

- **Memory**: At least 8GB RAM allocated to Docker
- **CPU**: 4+ CPU cores recommended
- **Disk**: SSD with at least 10GB free space
- **Network**: Stable internet connection for Ollama model downloads

### Execution Time

Typical execution times:

- **Dependency Validation**: 2-5 minutes
- **Unit Test Debugging**: 5-10 minutes
- **Integration Test Debugging**: 10-15 minutes
- **Security Test Debugging**: 8-12 minutes
- **Performance Test Debugging**: 15-20 minutes
- **Fix Application**: 2-5 minutes
- **Final Validation**: 10-15 minutes

**Total Time**: 45-80 minutes

### Parallel Execution

Some services can run in parallel:

```bash
# Run unit and integration debugging in parallel
docker-compose -f docker-compose.debug.yml up unit-test-debugger integration-test-debugger
```

## Best Practices

### Before Running

1. **Backup Code**: Ensure your code is committed to version control
2. **Free Resources**: Close unnecessary applications
3. **Check Disk Space**: Ensure sufficient disk space
4. **Network Connection**: Stable internet for Ollama models

### During Execution

1. **Monitor Dashboard**: Watch progress via web dashboard
2. **Check Logs**: Monitor container logs for errors
3. **Resource Usage**: Monitor system resource consumption
4. **Interrupt Gracefully**: Use Ctrl+C to stop pipeline cleanly

### After Completion

1. **Review Reports**: Examine all generated reports
2. **Validate Fixes**: Ensure fixes resolve intended issues
3. **Update Tests**: Commit fixed tests to version control
4. **Clean Up**: Remove unnecessary containers and images

## Support and Contributing

### Getting Help

1. **Check Logs**: Examine container logs first
2. **Review Reports**: Look at generated debugging reports
3. **Documentation**: Refer to this README and inline comments
4. **Issues**: Report problems via GitHub issues

### Contributing

1. **Fork Repository**: Create fork for changes
2. **Add Tests**: Include tests for new functionality
3. **Update Documentation**: Keep README updated
4. **Submit PR**: Create pull request with changes

## License

This debugging system is part of the AI Therapist project and follows the same license terms.

---

**Last Updated**: October 1, 2025
**Version**: 1.0.0
**Compatibility**: Docker Compose 3.8+, Docker 20.10+