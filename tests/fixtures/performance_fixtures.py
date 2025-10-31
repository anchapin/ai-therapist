"""
Performance testing fixtures for consistent performance testing.

Provides fixtures for load testing, memory monitoring, cache management,
and performance analysis to ensure reliable and isolated performance testing.
"""

import pytest
import threading
import time
import psutil
import gc
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from typing import Dict, List, Any
import tempfile
import os


@pytest.fixture
def performance_monitor():
    """Mock performance monitor for testing."""
    with patch('performance.monitor.PerformanceMonitor') as mock_monitor_class:
        mock_monitor = MagicMock()
        mock_monitor_class.return_value = mock_monitor
        
        # Mock memory monitoring
        mock_monitor.get_memory_usage.return_value = {
            'rss': 50 * 1024 * 1024,  # 50MB
            'vms': 100 * 1024 * 1024,  # 100MB
            'percent': 25.5,
            'available': 150 * 1024 * 1024  # 150MB available
        }
        
        # Mock CPU monitoring
        mock_monitor.get_cpu_usage.return_value = {
            'cpu_percent': 15.5,
            'cpu_count': 4,
            'load_average': [0.5, 0.8, 1.2]
        }
        
        # Mock disk I/O monitoring
        mock_monitor.get_disk_usage.return_value = {
            'read_bytes': 1024 * 1024,  # 1MB
            'write_bytes': 2 * 1024 * 1024,  # 2MB
            'read_count': 100,
            'write_count': 50
        }
        
        # Mock network I/O monitoring
        mock_monitor.get_network_usage.return_value = {
            'bytes_sent': 1024 * 1024,  # 1MB
            'bytes_recv': 3 * 1024 * 1024,  # 3MB
            'packets_sent': 500,
            'packets_recv': 1500
        }
        
        # Mock process monitoring
        mock_monitor.get_process_info.return_value = {
            'pid': 12345,
            'name': 'python',
            'status': 'running',
            'create_time': datetime.now().timestamp(),
            'cpu_times': {'user': 1.5, 'system': 0.8},
            'memory_info': {'rss': 50 * 1024 * 1024, 'vms': 100 * 1024 * 1024}
        }
        
        # Mock performance history
        mock_monitor.get_performance_history.return_value = [
            {
                'timestamp': datetime.now().isoformat(),
                'memory_usage': 45 * 1024 * 1024,
                'cpu_usage': 12.5,
                'response_time': 0.25
            }
        ]
        
        # Mock alerts
        mock_monitor.check_performance_thresholds.return_value = {
            'memory_alert': False,
            'cpu_alert': False,
            'disk_alert': False,
            'network_alert': False
        }
        
        # Mock statistics
        mock_monitor.get_performance_statistics.return_value = {
            'avg_memory_usage': 45 * 1024 * 1024,
            'max_memory_usage': 60 * 1024 * 1024,
            'avg_cpu_usage': 15.5,
            'max_cpu_usage': 25.0,
            'total_requests': 1000,
            'avg_response_time': 0.35,
            'max_response_time': 1.2
        }
        
        yield mock_monitor


@pytest.fixture
def memory_monitor():
    """Real memory monitor for testing."""
    class MemoryMonitor:
        def __init__(self):
            self.process = psutil.Process()
            self.baseline_memory = None
            self.measurements = []
        
        def start_monitoring(self):
            """Start memory monitoring and set baseline."""
            gc.collect()  # Force garbage collection
            self.baseline_memory = self.get_current_memory()
            self.measurements = []
        
        def get_current_memory(self):
            """Get current memory usage."""
            memory_info = self.process.memory_info()
            return {
                'rss': memory_info.rss,
                'vms': memory_info.vms,
                'percent': self.process.memory_percent()
            }
        
        def take_measurement(self, label: str = None):
            """Take a memory measurement."""
            current = self.get_current_memory()
            measurement = {
                'timestamp': datetime.now(),
                'label': label,
                'memory': current
            }
            self.measurements.append(measurement)
            return measurement
        
        def get_memory_growth(self):
            """Calculate memory growth since baseline."""
            if not self.baseline_memory:
                return None
            
            current = self.get_current_memory()
            return {
                'rss_growth': current['rss'] - self.baseline_memory['rss'],
                'vms_growth': current['vms'] - self.baseline_memory['vms'],
                'percent_growth': current['percent'] - self.baseline_memory['percent']
            }
        
        def detect_memory_leak(self, threshold_mb: float = 10.0):
            """Detect potential memory leaks."""
            if len(self.measurements) < 2:
                return False
            
            growth = self.get_memory_growth()
            if growth and growth['rss_growth'] > threshold_mb * 1024 * 1024:
                return True
            
            # Check for steady growth trend
            if len(self.measurements) >= 5:
                recent_measurements = self.measurements[-5:]
                rss_values = [m['memory']['rss'] for m in recent_measurements]
                
                # Simple linear regression to detect trend
                n = len(rss_values)
                x = list(range(n))
                sum_x = sum(x)
                sum_y = sum(rss_values)
                sum_xy = sum(x[i] * y for i, y in enumerate(rss_values))
                sum_x2 = sum(xi * xi for xi in x)
                
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                
                # If slope is positive and significant, suspect memory leak
                return slope > 1024 * 1024  # Growing more than 1MB per measurement
            
            return False
        
        def get_summary(self):
            """Get memory monitoring summary."""
            if not self.measurements:
                return {}
            
            rss_values = [m['memory']['rss'] for m in self.measurements]
            vms_values = [m['memory']['vms'] for m in self.measurements]
            
            return {
                'baseline': self.baseline_memory,
                'current': self.get_current_memory(),
                'growth': self.get_memory_growth(),
                'max_rss': max(rss_values),
                'min_rss': min(rss_values),
                'max_vms': max(vms_values),
                'min_vms': min(vms_values),
                'measurement_count': len(self.measurements),
                'memory_leak_detected': self.detect_memory_leak()
            }
    
    return MemoryMonitor()


@pytest.fixture
def cache_manager():
    """Mock cache manager for testing."""
    with patch('performance.cache_manager.CacheManager') as mock_cache_class:
        mock_cache = MagicMock()
        mock_cache_class.return_value = mock_cache
        
        # Mock cache operations
        mock_cache.get.return_value = None  # Cache miss
        mock_cache.set.return_value = True
        mock_cache.delete.return_value = True
        mock_cache.clear.return_value = True
        
        # Mock cache statistics
        mock_cache.get_stats.return_value = {
            'total_items': 100,
            'memory_usage': 25 * 1024 * 1024,  # 25MB
            'hit_rate': 0.85,
            'miss_rate': 0.15,
            'evictions': 5
        }
        
        # Mock cache configuration
        mock_cache.get_config.return_value = {
            'max_size': 1000,
            'ttl': 3600,  # 1 hour
            'cleanup_interval': 300,  # 5 minutes
            'max_memory': 100 * 1024 * 1024  # 100MB
        }
        
        # Mock advanced operations
        mock_cache.get_multiple.return_value = {'key1': 'value1', 'key2': None}
        mock_cache.set_multiple.return_value = True
        mock_cache.cleanup.return_value = 10  # Cleaned up 10 items
        
        # Mock performance testing
        mock_cache.benchmark_cache.return_value = {
            'set_ops_per_second': 10000,
            'get_ops_per_second': 15000,
            'avg_set_latency': 0.1,  # ms
            'avg_get_latency': 0.067  # ms
        }
        
        yield mock_cache


@pytest.fixture
def load_tester():
    """Load testing utility for performance testing."""
    class LoadTester:
        def __init__(self):
            self.results = []
            self.errors = []
            self.concurrency = 1
            self.duration = 0
            self.total_requests = 0
        
        def run_concurrent_requests(self, func, *args, concurrency: int = 10, 
                                 duration: float = 10.0, **kwargs):
            """Run concurrent requests for load testing."""
            self.concurrency = concurrency
            self.duration = duration
            self.results = []
            self.errors = []
            
            def worker():
                start_time = time.time()
                while time.time() - start_time < duration:
                    try:
                        request_start = time.time()
                        result = func(*args, **kwargs)
                        request_time = time.time() - request_start
                        
                        self.results.append({
                            'timestamp': datetime.now(),
                            'response_time': request_time,
                            'success': True,
                            'result': result
                        })
                    except Exception as e:
                        self.errors.append({
                            'timestamp': datetime.now(),
                            'error': str(e),
                            'exception': e
                        })
            
            # Start worker threads
            threads = []
            for _ in range(concurrency):
                thread = threading.Thread(target=worker)
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            self.total_requests = len(self.results) + len(self.errors)
            return self.get_results()
        
        def run_ramp_up_test(self, func, *args, max_concurrency: int = 50,
                           ramp_up_time: float = 30.0, test_duration: float = 60.0, **kwargs):
            """Run ramp-up load test."""
            self.results = []
            self.errors = []
            
            def worker():
                start_time = time.time()
                while time.time() - start_time < test_duration:
                    try:
                        request_start = time.time()
                        result = func(*args, **kwargs)
                        request_time = time.time() - request_start
                        
                        self.results.append({
                            'timestamp': datetime.now(),
                            'response_time': request_time,
                            'success': True,
                            'result': result
                        })
                        time.sleep(0.1)  # Small delay between requests
                    except Exception as e:
                        self.errors.append({
                            'timestamp': datetime.now(),
                            'error': str(e),
                            'exception': e
                        })
            
            # Gradually increase concurrency
            threads = []
            start_time = time.time()
            
            while time.time() - start_time < ramp_up_time:
                current_concurrency = min(
                    int((time.time() - start_time) / ramp_up_time * max_concurrency),
                    max_concurrency
                )
                
                while len(threads) < current_concurrency:
                    thread = threading.Thread(target=worker)
                    threads.append(thread)
                    thread.start()
                
                time.sleep(1)
            
            # Wait for remaining threads to complete
            for thread in threads:
                thread.join()
            
            self.total_requests = len(self.results) + len(self.errors)
            return self.get_results()
        
        def get_results(self):
            """Get load test results summary."""
            if not self.results and not self.errors:
                return {'status': 'No test results available'}
            
            response_times = [r['response_time'] for r in self.results if r['success']]
            
            summary = {
                'total_requests': self.total_requests,
                'successful_requests': len(self.results),
                'failed_requests': len(self.errors),
                'success_rate': len(self.results) / self.total_requests if self.total_requests > 0 else 0,
                'concurrency': self.concurrency,
                'duration': self.duration,
                'requests_per_second': self.total_requests / self.duration if self.duration > 0 else 0,
                'response_time_stats': {
                    'min': min(response_times) if response_times else 0,
                    'max': max(response_times) if response_times else 0,
                    'avg': sum(response_times) / len(response_times) if response_times else 0,
                    'median': sorted(response_times)[len(response_times) // 2] if response_times else 0,
                    'p95': sorted(response_times)[int(len(response_times) * 0.95)] if response_times else 0,
                    'p99': sorted(response_times)[int(len(response_times) * 0.99)] if response_times else 0
                },
                'errors': self.errors[-10:] if self.errors else []  # Last 10 errors
            }
            
            return summary
    
    return LoadTester()


@pytest.fixture
def performance_test_environment(performance_monitor, memory_monitor, 
                               cache_manager, load_tester):
    """Complete performance test environment."""
    return {
        'monitor': performance_monitor,
        'memory_monitor': memory_monitor,
        'cache_manager': cache_manager,
        'load_tester': load_tester
    }


@pytest.fixture
def performance_benchmark_data():
    """Sample performance benchmark data for testing."""
    return {
        'api_endpoints': {
            '/auth/login': {
                'avg_response_time': 0.25,
                'p95_response_time': 0.5,
                'p99_response_time': 1.0,
                'requests_per_second': 1000,
                'error_rate': 0.01
            },
            '/voice/transcribe': {
                'avg_response_time': 2.5,
                'p95_response_time': 5.0,
                'p99_response_time': 10.0,
                'requests_per_second': 100,
                'error_rate': 0.02
            },
            '/voice/synthesize': {
                'avg_response_time': 1.5,
                'p95_response_time': 3.0,
                'p99_response_time': 6.0,
                'requests_per_second': 200,
                'error_rate': 0.015
            }
        },
        'memory_usage': {
            'baseline_mb': 50,
            'peak_mb': 150,
            'average_mb': 75,
            'memory_leak_detected': False
        },
        'cpu_usage': {
            'average_percent': 25.5,
            'peak_percent': 85.0,
            'cpu_efficient': True
        },
        'cache_performance': {
            'hit_rate': 0.85,
            'miss_rate': 0.15,
            'evictions_per_hour': 5,
            'memory_usage_mb': 25
        }
    }


@pytest.fixture
def temporary_performance_logs():
    """Create temporary directory for performance log testing."""
    temp_dir = tempfile.mkdtemp(prefix="perf_logs_test_")
    
    log_files = {
        'performance.log': os.path.join(temp_dir, 'performance.log'),
        'memory.log': os.path.join(temp_dir, 'memory.log'),
        'cache.log': os.path.join(temp_dir, 'cache.log'),
        'load_test.log': os.path.join(temp_dir, 'load_test.log')
    }
    
    # Write sample log entries
    sample_logs = {
        'performance.log': '2024-01-01 10:00:00 INFO Response time: 0.25s\n2024-01-01 10:00:01 INFO CPU: 25%\n',
        'memory.log': '2024-01-01 10:00:00 INFO Memory: 50MB RSS, 100MB VMS\n',
        'cache.log': '2024-01-01 10:00:00 INFO Cache hit: 0.85, miss: 0.15\n',
        'load_test.log': '2024-01-01 10:00:00 INFO 1000 req/s, avg response: 0.25s\n'
    }
    
    for log_file, content in sample_logs.items():
        with open(log_files[log_file], 'w') as f:
            f.write(content)
    
    yield log_files
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_profiler():
    """Mock profiler for performance testing."""
    with patch('cProfile.Profile') as mock_profile_class:
        mock_profile = MagicMock()
        mock_profile_class.return_value = mock_profile
        
        # Mock profiling methods
        mock_profile.enable.return_value = None
        mock_profile.disable.return_value = None
        mock_profile.dump_stats.return_value = None
        mock_profile.print_stats.return_value = None
        
        # Mock profiling results
        mock_profile.getstats.return_value = [
            {
                'filename': 'test.py',
                'lineno': 10,
                'function': 'test_function',
                'ncalls': 100,
                'tottime': 1.5,
                'cumtime': 2.0
            }
        ]
        
        # Mock statistics
        mock_profile.stats = {
            'test_function': {
                'ncalls': 100,
                'tottime': 1.5,
                'cumtime': 2.0,
                'filename': 'test.py',
                'lineno': 10
            }
        }
        
        yield mock_profile


@pytest.fixture
def stress_test_config():
    """Stress test configuration for testing."""
    return {
        'max_concurrent_users': 1000,
        'ramp_up_time': 300,  # 5 minutes
        'test_duration': 1800,  # 30 minutes
        'think_time': 1.0,  # seconds between requests
        'timeout': 30.0,  # request timeout
        'expected_response_time': 2.0,  # 95th percentile
        'max_error_rate': 0.01,  # 1%
        'memory_limit_mb': 500,
        'cpu_limit_percent': 80,
        'endpoints': [
            {'path': '/auth/login', 'weight': 30},
            {'path': '/voice/transcribe', 'weight': 50},
            {'path': '/voice/synthesize', 'weight': 20}
        ]
    }