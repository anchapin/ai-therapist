# AI Therapist Performance Benchmarks

## Overview

This document establishes performance benchmarks, monitoring procedures, and regression detection thresholds for the AI Therapist application. Performance monitoring ensures system reliability, user experience quality, and resource efficiency.

## Performance Categories

### 1. Response Time Benchmarks

#### Voice Service Operations
| Operation | Target | Threshold | Critical |
|-----------|--------|-----------|----------|
| Voice Session Creation | < 500ms | < 1000ms | Yes |
| Speech-to-Text (1s audio) | < 2000ms | < 5000ms | Yes |
| Text-to-Speech (100 chars) | < 1500ms | < 3000ms | Yes |
| Audio Processing (real-time) | < 100ms latency | < 200ms latency | Yes |
| Voice Command Processing | < 300ms | < 700ms | Yes |

#### Database Operations
| Operation | Target | Threshold | Critical |
|-----------|--------|-----------|----------|
| User Authentication | < 200ms | < 500ms | Yes |
| Session Retrieval | < 100ms | < 300ms | Yes |
| Data Insertion | < 150ms | < 400ms | Yes |
| Complex Query | < 500ms | < 1000ms | No |

#### Cache Operations
| Operation | Target | Threshold | Critical |
|-----------|--------|-----------|----------|
| Cache Hit | < 5ms | < 20ms | No |
| Cache Miss | < 50ms | < 200ms | No |
| Cache Eviction | < 10ms | < 50ms | No |
| Cache Compression | < 100ms | < 300ms | No |

### 2. Throughput Benchmarks

#### Concurrent Users
| Load Level | Target TPS | Threshold TPS | Max Latency |
|------------|------------|---------------|-------------|
| Light (10 users) | 50 req/sec | 25 req/sec | 1000ms |
| Medium (50 users) | 100 req/sec | 50 req/sec | 2000ms |
| Heavy (100 users) | 150 req/sec | 75 req/sec | 3000ms |

#### Voice Processing
| Audio Format | Target | Threshold | Quality |
|--------------|--------|-----------|---------|
| WAV 16kHz | 10 concurrent | 5 concurrent | Lossless |
| MP3 128kbps | 15 concurrent | 8 concurrent | High |
| Real-time streaming | 20 concurrent | 10 concurrent | Medium |

### 3. Resource Usage Benchmarks

#### Memory Usage
| Component | Target | Threshold | Critical |
|-----------|--------|-----------|----------|
| Base Application | < 100MB | < 200MB | No |
| Voice Service (active) | < 150MB | < 300MB | No |
| Cache (1GB data) | < 200MB | < 400MB | No |
| Database Connections | < 50MB | < 100MB | No |
| Peak Load Memory | < 500MB | < 1GB | Yes |

#### CPU Usage
| Operation | Target | Threshold | Critical |
|-----------|--------|-----------|----------|
| Idle State | < 5% | < 15% | No |
| Voice Processing | < 30% | < 60% | No |
| Database Query | < 20% | < 40% | No |
| Concurrent Sessions | < 50% | < 80% | Yes |

#### Disk I/O
| Operation | Target | Threshold | Critical |
|-----------|--------|-----------|----------|
| Log Writing | < 10MB/min | < 50MB/min | No |
| Cache Persistence | < 100MB/min | < 300MB/min | No |
| Database WAL | < 50MB/min | < 150MB/min | No |

### 4. Scalability Benchmarks

#### Session Management
| Metric | Target | Threshold | Scaling Factor |
|--------|--------|-----------|----------------|
| Max Concurrent Sessions | 1000 | 500 | Linear |
| Session Creation Rate | 100/sec | 50/sec | Linear |
| Session Cleanup | < 100ms | < 500ms | Constant |

#### Cache Performance
| Cache Size | Hit Rate Target | Threshold | Eviction Time |
|------------|-----------------|-----------|---------------|
| 100MB | > 85% | > 70% | < 50ms |
| 500MB | > 90% | > 80% | < 100ms |
| 1GB | > 95% | > 85% | < 200ms |

## Benchmark Measurement Tools

### pytest-benchmark Integration

#### Running Benchmarks
```bash
# Run all performance tests with benchmarking
python -m pytest tests/performance/ --benchmark-only --benchmark-json=benchmark-results.json

# Run specific benchmark
python -m pytest tests/performance/test_cache_performance.py::test_cache_operations --benchmark-only

# Compare benchmarks
pytest-benchmark compare benchmark-baseline.json benchmark-current.json
```

#### Benchmark Configuration
```python
import pytest_benchmark

@pytest.mark.benchmark(
    group="voice-processing",
    min_rounds=10,
    max_time=2.0,
    warmup=True
)
def test_voice_session_creation(benchmark):
    def create_session():
        service = VoiceService(config)
        return service.create_session()

    result = benchmark(create_session)
    assert result is not None
```

### Custom Performance Monitoring

#### Memory Profiling
```python
import tracemalloc
import psutil

def monitor_memory_usage(operation_func, *args, **kwargs):
    tracemalloc.start()
    process = psutil.Process()

    start_mem = process.memory_info().rss
    start_time = time.time()

    result = operation_func(*args, **kwargs)

    end_time = time.time()
    end_mem = process.memory_info().rss

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        'execution_time': end_time - start_time,
        'memory_delta': end_mem - start_mem,
        'peak_memory': peak,
        'result': result
    }
```

#### CPU Profiling
```python
import cProfile
import pstats

def profile_cpu_usage(func, *args, **kwargs):
    profiler = cProfile.Profile()
    profiler.enable()

    result = func(*args, **kwargs)

    profiler.disable()
    stats = pstats.Stats(profiler)

    # Get top 10 time-consuming functions
    stats.sort_stats('cumulative').print_stats(10)

    return result
```

## Monitoring Procedures

### Daily Monitoring

#### Automated CI/CD Checks
- Performance regression detection in GitHub Actions
- Benchmark comparison with baseline
- Resource usage alerts
- Response time monitoring

#### Manual Health Checks
```bash
# Quick performance check
python -c "
from performance.cache_manager import CacheManager
import time

cache = CacheManager()
start = time.time()
for i in range(1000):
    cache.set(f'key_{i}', f'value_{i}')
    cache.get(f'key_{i}')
end = time.time()
print(f'Cache operations: {(end-start)*1000:.2f}ms')
"
```

### Weekly Monitoring

#### Trend Analysis
- Review performance metrics over past week
- Identify performance degradation patterns
- Update baseline benchmarks quarterly
- Analyze resource usage patterns

#### Capacity Planning
- Monitor concurrent user patterns
- Assess database connection usage
- Review cache hit rates and sizes
- Plan for scaling requirements

### Monthly Performance Audit

#### Comprehensive Assessment
- Full benchmark suite execution
- Memory leak detection
- CPU profiling analysis
- Database performance optimization
- Cache efficiency review

## Regression Detection

### Automated Alerts

#### Performance Regression Thresholds
```yaml
# .github/workflows/performance-alerts.yml
performance_thresholds:
  response_time_degradation: 10%  # Alert if >10% slower
  memory_usage_increase: 20%     # Alert if >20% more memory
  cpu_usage_spike: 50%          # Alert if >50% CPU increase
  throughput_drop: 15%          # Alert if >15% fewer operations/sec
```

#### Alert Conditions
- **Critical**: System performance impacts user experience
- **Warning**: Performance degradation detected
- **Info**: Minor performance changes for monitoring

### Manual Regression Analysis

#### Identifying Regressions
1. **Compare Benchmarks**: Use pytest-benchmark comparison tools
2. **Profile Code**: Use cProfile to identify bottlenecks
3. **Memory Analysis**: Check for memory leaks with tracemalloc
4. **Database Queries**: Analyze slow query performance

#### Regression Response Protocol
1. **Immediate**: Stop deployment if critical regression
2. **Investigation**: Profile and identify root cause
3. **Fix**: Implement performance optimization
4. **Validation**: Confirm fix with benchmarks
5. **Monitoring**: Add performance test for regression prevention

## Performance Optimization Guidelines

### Code-Level Optimizations

#### Cache Optimization
```python
# Efficient cache usage
class OptimizedCacheManager:
    def __init__(self):
        self._cache = {}
        self._access_times = {}

    def get(self, key):
        if key in self._cache:
            self._access_times[key] = time.time()
            return self._cache[key]
        return None

    def set(self, key, value, ttl=None):
        self._cache[key] = value
        self._access_times[key] = time.time()
        if ttl:
            # Schedule cleanup
            pass
```

#### Database Optimization
```python
# Connection pooling
class DatabaseManager:
    def __init__(self):
        self._pool = DatabaseConnectionPool(min_size=5, max_size=50)

    def execute_query(self, query, params=None):
        with self._pool.get_connection() as conn:
            return conn.execute(query, params)
```

### System-Level Optimizations

#### Memory Management
- Implement object pooling for frequently used objects
- Use weak references for cache entries
- Regular garbage collection scheduling
- Memory-mapped files for large datasets

#### CPU Optimization
- Asynchronous processing for I/O operations
- Thread pooling for concurrent operations
- CPU affinity for performance-critical threads
- Algorithm optimization for computational tasks

### Infrastructure Optimizations

#### Caching Strategy
- Multi-level caching (memory → disk → database)
- Cache warming for frequently accessed data
- Intelligent cache eviction policies
- Distributed caching for scalability

#### Database Tuning
- Query optimization and indexing
- Connection pooling configuration
- Read/write splitting
- Database maintenance scheduling

## Benchmark Maintenance

### Updating Baselines

#### Quarterly Baseline Updates
```bash
# Generate new baseline
python -m pytest tests/performance/ --benchmark-only --benchmark-save=baseline

# Update baseline file
mv .benchmarks/baseline.json benchmarks/baseline-$(date +%Y-%m-%d).json
ln -sf benchmarks/baseline-$(date +%Y-%m-%d).json benchmarks/current-baseline.json
```

#### Baseline Validation
- Ensure baseline represents typical production load
- Validate benchmarks across different environments
- Document any environmental factors affecting results
- Maintain historical baseline data for trend analysis

### Benchmark Documentation

#### Performance Test Documentation
Each performance test must include:
- **Purpose**: What performance aspect is being tested
- **Setup**: Required environment and data
- **Metrics**: What is being measured
- **Thresholds**: Acceptable performance ranges
- **Interpretation**: How to interpret results

#### Example Documentation
```python
def test_voice_session_creation_performance(benchmark):
    """
    Performance test for voice session creation.

    Purpose: Ensure voice session creation is fast enough for good UX
    Setup: Clean database, mock external services
    Metrics: Time to create session, memory usage
    Thresholds: < 500ms, < 50MB memory delta
    Interpretation: Times > 500ms indicate UX degradation
    """
    def create_session():
        service = VoiceService(config)
        return service.create_session()

    result = benchmark(create_session)
    assert result.execution_time < 0.5  # 500ms threshold
```

## Reporting and Analytics

### Performance Dashboards

#### CI/CD Integration
- Automated performance reporting in GitHub Actions
- Performance trend charts in Codecov
- Alert notifications for regressions
- Historical performance data retention

#### Custom Reporting
```python
# Generate performance report
def generate_performance_report(results):
    report = {
        'timestamp': datetime.now().isoformat(),
        'benchmarks': results,
        'thresholds': PERFORMANCE_THRESHOLDS,
        'regressions': detect_regressions(results),
        'recommendations': generate_recommendations(results)
    }

    with open('performance-report.json', 'w') as f:
        json.dump(report, f, indent=2)

    return report
```

### Stakeholder Communication

#### Performance Status Reports
- Weekly performance summary emails
- Monthly detailed performance reports
- Alert notifications for critical issues
- Performance improvement tracking

#### Documentation Updates
- Update performance benchmarks after optimizations
- Document performance improvements
- Maintain change log for performance-related changes

## Emergency Procedures

### Performance Incident Response

#### Critical Performance Issues
1. **Detection**: Automated monitoring alerts
2. **Assessment**: Quick performance check
3. **Isolation**: Identify affected components
4. **Mitigation**: Implement temporary fixes
5. **Resolution**: Deploy permanent fix
6. **Review**: Post-mortem analysis

#### Rollback Procedures
- Maintain performance baselines for rollback validation
- Quick rollback capability for performance regressions
- Gradual rollout procedures for high-risk changes

## Support Resources

### Tools and Utilities
- **pytest-benchmark**: Automated benchmarking
- **tracemalloc**: Memory profiling
- **cProfile**: CPU profiling
- **psutil**: System resource monitoring
- **Codecov**: Coverage and performance tracking

### Getting Help
1. Check this performance benchmarks document
2. Review CI/CD performance logs
3. Consult performance test results
4. Contact development team for optimization help

---

*Performance benchmarks should be reviewed quarterly and updated as the system evolves. All performance-critical changes must include corresponding benchmark updates.*
