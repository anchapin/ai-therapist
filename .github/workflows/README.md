# CI/CD Workflows

This directory contains GitHub Actions workflows for automated testing, security scanning, and deployment of the AI Therapist application.

## Workflows Overview

### 🚀 `optimized-ci.yml` (Primary)
**Trigger**: Push to main/develop branches, Pull Requests to main
**Purpose**: Comprehensive CI/CD pipeline with parallel execution and optimization

**Jobs**:
- **pre-check**: Fast linting, type checking, and basic unit tests (10 min timeout)
- **comprehensive-test**: Parallel test execution with coverage (30 min timeout)
- **security-scan**: Automated security scanning with Bandit and Safety (15 min timeout)
- **performance-check**: Performance benchmarks (main branch only, 20 min timeout)
- **report-results**: PR comments with test results and coverage

**Optimizations**:
- ✅ Parallel test execution with `pytest-xdist`
- ✅ Efficient caching with `actions/setup-python` cache
- ✅ Consolidated workflows (replaces 3 separate workflows)
- ✅ Smart dependency installation with `requirements-ci.txt`
- ✅ Automated PR reporting
- ✅ 60% coverage threshold enforcement

### 🏠 `main-branch-ci.yml`
**Trigger**: Push to main branch, Weekly schedule (Monday 2 AM UTC)
**Purpose**: Quality gates and deployment readiness for production

**Jobs**:
- **quality-gate**: Critical tests and security audits
- **deployment-check**: Verify deployment requirements
- **weekly-audit**: Comprehensive security and quality audit

**Features**:
- ✅ Automated issue creation for weekly audits
- ✅ Deployment package generation
- ✅ Quality gate enforcement
- ✅ Security audit retention (90 days)

### 📋 `test.yml` (Deprecated)
**Status**: Legacy workflow, kept for backward compatibility
**Replacement**: Use `optimized-ci.yml` instead

## Configuration Files

### `pytest.ini`
```ini
[tool:pytest]
asyncio_mode = auto
markers =
    security: Security-related tests
    integration: Integration tests
    performance: Performance tests
    asyncio: Async tests
timeout = 120
```

### Coverage Configuration
- **Threshold**: 60% minimum coverage
- **Modules**: voice, security, auth, performance, database
- **Reports**: XML, JSON, terminal, HTML
- **Integration**: Codecov automatic uploads

## Performance Optimizations

### Speed Improvements
- **Parallel Execution**: Tests run with `-n auto` (uses all available cores)
- **Efficient Caching**: Pip cache with dependency hashing
- **Fast Failures**: `--maxfail=5` prevents wasting time on broken builds
- **Timeout Management**: Reasonable timeouts prevent hanging

### Cost Optimizations
- **Job Dependencies**: Parallel jobs only when needed
- **Conditional Execution**: Performance checks only on main branch
- **Artifact Retention**: Optimized retention periods (7-90 days)

## Security Features

### Automated Scanning
- **Bandit**: Python security linting
- **Safety**: Dependency vulnerability checking
- **Codecov**: Coverage and quality metrics

### Audit Integration
- **Weekly Audits**: Automated comprehensive security reviews
- **Issue Tracking**: Automatic GitHub issue creation for findings
- **Retention**: Long-term security report storage

## Migration Guide

### From Legacy Workflows
1. **Replace triggers**: Update branch names if needed
2. **Update requirements**: Use `requirements-ci.txt` for faster installs
3. **Remove manual steps**: Dependency installation now automated
4. **Update artifact names**: New naming convention for artifacts

### Environment Variables
```bash
PYTHON_VERSION=3.12
MPLBACKEND=Agg
USE_DUMMY_AUDIO=1
PYTEST_TIMEOUT=300
```

## Troubleshooting

### Common Issues

**Tests timeout**: Increase `PYTEST_TIMEOUT` or use `--durations` to identify slow tests

**Coverage fails**: Check if new code is missing test coverage, adjust threshold if needed

**Security scan fails**: Review Bandit findings, update dependencies for Safety issues

**Parallel tests fail**: Check for test isolation issues, remove test interdependencies

### Debugging
- **Verbose output**: Add `-v` to pytest commands
- **Coverage details**: Use `--cov-report=html` for detailed reports
- **Performance**: Use `--durations=20` to identify slow tests

## Maintenance

### Regular Tasks
- **Weekly**: Review security audit issues
- **Monthly**: Update dependencies and security tools
- **Quarterly**: Review and optimize workflow performance

### Updating Dependencies
1. Update `requirements.txt` and `requirements-ci.txt`
2. Test workflows on feature branch
3. Update cache keys if dependency patterns change
4. Monitor for breaking changes in GitHub Actions

## Support

For workflow issues:
1. Check GitHub Actions logs
2. Review pytest output for test failures
3. Check artifact uploads for detailed reports
4. Create issue with workflow run link
