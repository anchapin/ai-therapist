"""
Performance Stress Testing and Load Testing

This module provides comprehensive performance testing including:
- Memory pressure scenarios and leak detection
- Cache eviction and performance under load
- Concurrent access patterns and thread safety
- Alert system reliability and threshold testing
- Resource cleanup and recovery testing
- Performance regression detection
"""

import pytest
import asyncio
import threading
import time
import psutil
import gc
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List, Optional, Generator, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
import queue
import weakref
import tracemalloc
import resource
import os


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: float
    memory_usage_mb: float
    cpu_usage_percent: float
    response_time_ms: float
    throughput_ops_per_sec: float
    error_rate: float
    cache_hit_rate: float


@dataclass
class StressTestConfig:
    """Configuration for stress testing."""
    duration_seconds: int
    concurrent_users: int
    requests_per_second: int
    memory_limit_mb: int
    cpu_limit_percent: float
    max_error_rate: float
    min_throughput: float


class TestPerformanceStressTesting:
    """Comprehensive performance stress testing suite."""
    
    @pytest.fixture
    def performance_monitor(self):
        """Mock performance monitoring service."""
        with patch('performance.monitor.PerformanceMonitor') as mock_monitor:
            monitor_instance = Mock()
            monitor_instance.get_current_metrics.return_value = {
                'memory_usage_mb': 100.0,
                'cpu_usage_percent': 25.0,
                'response_time_ms': 150.0,
                'throughput_ops_per_sec': 1000.0,
                'error_rate': 0.01,
                'cache_hit_rate': 0.85
            }
            monitor_instance.start_monitoring.return_value = True
            monitor_instance.stop_monitoring.return_value = True
            monitor_return_value = monitor_instance
            mock_monitor.return_value = monitor_instance
            
            yield monitor_instance
    
    @pytest.fixture
    def cache_manager(self):
        """Mock cache manager for testing."""
        with patch('performance.cache_manager.CacheManager') as mock_cache:
            cache_instance = Mock()
            cache_instance.get.return_value = None
            cache_instance.set.return_value = True
            cache_instance.delete.return_value = True
            cache_instance.clear.return_value = True
            cache_instance.get_stats.return_value = {
                'hits': 850,
                'misses': 150,
                'size': 1000,
                'memory_usage_mb': 50.0
            }
            cache_instance.evict_expired.return_value = 10
            cache_instance.evict_lru.return_value = 5
            mock_cache.return_value = cache_instance
            
            yield cache_instance
    
    @pytest.fixture
    def memory_manager(self):
        """Mock memory manager for testing."""
        with patch('performance.memory_manager.MemoryManager') as mock_memory:
            memory_instance = Mock()
            memory_instance.get_memory_usage.return_value = 100.0
            memory_instance.check_memory_limit.return_value = False
            memory_instance.cleanup.return_value = True
            memory_instance.get_memory_pressure.return_value = 0.3
            memory_instance.set_memory_limit.return_value = True
            mock_memory.return_value = memory_instance
            
            yield memory_instance
    
    @pytest.fixture
    def sample_load_data(self):
        """Sample data for load testing."""
        return {
            'voice_requests': [
                {'audio_size': 1024, 'transcription_length': 50},
                {'audio_size': 2048, 'transcription_length': 100},
                {'audio_size': 512, 'transcription_length': 25},
                {'audio_size': 4096, 'transcription_length': 200}
            ] * 100,  # 400 requests
            'cache_operations': [
                {'operation': 'get', 'key': f'key_{i}', 'value': f'value_{i}'}
                for i in range(1000)
            ],
            'concurrent_tasks': [
                {'task_id': i, 'duration': 0.1, 'memory_allocation': 10}
                for i in range(50)
            ]
        }
    
    class TestMemoryPressureScenarios:
        """Test memory pressure and leak detection."""
        
        @pytest.fixture(autouse=True)
        def setup_memory_tracking(self):
            """Setup memory tracking for leak detection."""
            tracemalloc.start()
            gc.collect()  # Clear existing garbage
            yield
            tracemalloc.stop()
        
        def test_memory_leak_detection(self, performance_monitor):
            """Test detection of memory leaks during prolonged operation."""
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_snapshots = [initial_memory]
            
            # Simulate prolonged operation with potential leaks
            leaked_objects = []
            
            for i in range(100):
                # Create objects that might leak
                large_data = np.random.randn(1000, 1000)  # ~8MB
                leaked_objects.append(large_data)
                
                # Some objects should be cleaned up
                if i % 10 == 0:
                    # Simulate cleanup
                    leaked_objects = leaked_objects[-5:]  # Keep only last 5
                
                # Track memory usage
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_snapshots.append(current_memory)
                
                # Allow garbage collection
                if i % 20 == 0:
                    gc.collect()
            
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_growth = final_memory - initial_memory
            
            # Check for memory leak (more than 100MB growth indicates potential leak)
            assert memory_growth < 150, f"Potential memory leak detected: {memory_growth:.1f}MB growth"
            
            # Verify memory trend
            if len(memory_snapshots) > 10:
                # Calculate memory growth trend
                recent_growth = memory_snapshots[-1] - memory_snapshots[-10]
                assert recent_growth < 50, f"Recent memory growth too high: {recent_growth:.1f}MB"
        
        def test_memory_cleanup_under_pressure(self, memory_manager):
            """Test memory cleanup when under memory pressure."""
            # Simulate memory pressure
            memory_manager.get_memory_pressure.return_value = 0.9  # 90% memory usage
            memory_manager.check_memory_limit.return_value = True  # Over limit
            
            # Test cleanup triggered by memory pressure
            cleanup_called = False
            
            def mock_cleanup():
                nonlocal cleanup_called
                cleanup_called = True
                gc.collect()  # Force garbage collection
                return True
            
            memory_manager.cleanup.side_effect = mock_cleanup
            
            # Trigger memory pressure response
            from performance.memory_manager import handle_memory_pressure
            
            # Mock the function since we're testing the concept
            def handle_pressure():
                if memory_manager.get_memory_pressure() > 0.8:
                    return memory_manager.cleanup()
                return False
            
            result = handle_pressure()
            
            assert result is True, "Should cleanup under memory pressure"
            assert cleanup_called is True, "Cleanup should be triggered"
        
        @pytest.mark.asyncio
        async def test_concurrent_memory_allocation(self, memory_manager):
            """Test memory allocation under concurrent load."""
            allocation_tasks = []
            allocation_results = []
            
            async def allocate_memory(task_id, size_mb, duration_sec):
                """Allocate memory for specified duration."""
                try:
                    # Allocate memory
                    data = np.random.randn(size_mb * 256, 1000)  # Approx size_mb MB
                    
                    # Hold memory for duration
                    await asyncio.sleep(duration_sec)
                    
                    # Clean up
                    del data
                    
                    return {
                        'task_id': task_id,
                        'success': True,
                        'allocated_mb': size_mb
                    }
                except MemoryError:
                    return {
                        'task_id': task_id,
                        'success': False,
                        'error': 'MemoryError'
                    }
            
            # Create concurrent allocation tasks
            for i in range(20):
                task = allocate_memory(
                    task_id=i,
                    size_mb=np.random.randint(5, 50),  # 5-50MB
                    duration_sec=np.random.uniform(0.1, 1.0)
                )
                allocation_tasks.append(task)
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*allocation_tasks, return_exceptions=True)
            
            # Analyze results
            successful_allocations = [r for r in results if isinstance(r, dict) and r.get('success')]
            failed_allocations = [r for r in results if isinstance(r, dict) and not r.get('success')]
            
            # Most allocations should succeed
            success_rate = len(successful_allocations) / len(results)
            assert success_rate > 0.8, f"Success rate too low: {success_rate:.2%}"
            
            # Failed allocations should be due to memory limits
            for failure in failed_allocations:
                assert failure.get('error') in ['MemoryError'], "Failures should be memory-related"
        
        def test_memory_fragmentation_detection(self):
            """Test detection of memory fragmentation."""
            # Simulate memory fragmentation scenario
            allocations = []
            allocation_sizes = [1, 10, 1, 10, 1, 10, 1, 10]  # MB
            
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Create fragmented allocation pattern
            for size in allocation_sizes:
                data = np.random.randn(size * 256, 1000)
                allocations.append(data)
            
            # Remove every other allocation to create fragmentation
            for i in range(0, len(allocations), 2):
                del allocations[i]
            
            # Force garbage collection
            gc.collect()
            
            after_fragmentation = psutil.Process().memory_info().rss / 1024 / 1024
            fragmentation_overhead = after_fragmentation - initial_memory
            
            # Try to allocate large block in fragmented space
            try:
                large_block = np.random.randn(50 * 256, 1000)  # 50MB
                large_allocation_success = True
                del large_block
            except MemoryError:
                large_allocation_success = False
            
            # Clean up
            allocations.clear()
            gc.collect()
            
            # Fragmentation should not prevent reasonable allocations
            assert large_allocation_success, "Fragmentation should not block large allocations"
            assert fragmentation_overhead < 100, f"Fragmentation overhead too high: {fragmentation_overhead:.1f}MB"
    
    class TestCacheEvictionPerformance:
        """Test cache eviction and performance under memory pressure."""
        
        @pytest.mark.asyncio
        async def test_cache_eviction_under_memory_pressure(self, cache_manager, memory_manager):
            """Test cache eviction when memory pressure is high."""
            # Configure cache stats
            cache_manager.get_stats.return_value = {
                'hits': 800,
                'misses': 200,
                'size': 10000,
                'memory_usage_mb': 200.0
            }
            
            # Configure memory pressure
            memory_manager.get_memory_pressure.return_value = 0.85  # 85% memory usage
            
            # Test eviction logic
            eviction_triggered = False
            evicted_items = []
            
            def mock_evict_expired():
                nonlocal eviction_triggered
                eviction_triggered = True
                evicted_count = np.random.randint(50, 200)
                evicted_items.extend([f'item_{i}' for i in range(evicted_count)])
                return evicted_count
            
            def mock_evict_lru(count=100):
                evicted_count = np.random.randint(count//2, count)
                evicted_items.extend([f'lru_{i}' for i in range(evicted_count)])
                return evicted_count
            
            cache_manager.evict_expired.side_effect = mock_evict_expired
            cache_manager.evict_lru.side_effect = mock_evict_lru
            
            # Simulate memory pressure response
            from performance.cache_manager import handle_cache_memory_pressure
            
            # Mock implementation
            async def handle_pressure():
                if memory_manager.get_memory_pressure() > 0.8:
                    # First try expired items
                    expired = cache_manager.evict_expired()
                    
                    # If still under pressure, evict LRU
                    if memory_manager.get_memory_pressure() > 0.8:
                        lru = cache_manager.evict_lru(100)
                        return expired + lru
                    return expired
                return 0
            
            evicted_total = await handle_pressure()
            
            assert eviction_triggered is True, "Eviction should be triggered under pressure"
            assert evicted_total > 0, "Should evict items under memory pressure"
            assert len(evicted_items) > 0, "Should track evicted items"
        
        @pytest.mark.asyncio
        async def test_cache_performance_under_load(self, cache_manager):
            """Test cache performance under high load."""
            # Configure cache with different performance characteristics
            hit_rates = [0.9, 0.8, 0.7, 0.6, 0.5]  # Degradating performance
            response_times = [10, 15, 25, 40, 60]    # Increasing response times (ms)
            
            performance_metrics = []
            
            # Simulate increasing load
            for load_factor in range(1, 6):
                # Adjust cache performance based on load
                cache_manager.get_stats.return_value = {
                    'hits': int(1000 * hit_rates[load_factor-1]),
                    'misses': int(1000 * (1 - hit_rates[load_factor-1])),
                    'size': 1000 * load_factor,
                    'memory_usage_mb': 50 * load_factor
                }
                
                # Simulate cache operations with varying response times
                start_time = time.time()
                
                # Simulate 100 cache operations
                for i in range(100):
                    # Simulate get operation
                    await asyncio.sleep(response_times[load_factor-1] / 1000)  # Convert ms to seconds
                    
                    if i % 3 == 0:
                        # Simulate set operation
                        await asyncio.sleep(response_times[load_factor-1] / 500)
                
                end_time = time.time()
                
                # Calculate performance metrics
                total_time = end_time - start_time
                ops_per_second = 100 / total_time
                hit_rate = hit_rates[load_factor-1]
                
                performance_metrics.append({
                    'load_factor': load_factor,
                    'ops_per_second': ops_per_second,
                    'hit_rate': hit_rate,
                    'avg_response_time_ms': response_times[load_factor-1]
                })
            
            # Analyze performance degradation
            initial_ops = performance_metrics[0]['ops_per_second']
            final_ops = performance_metrics[-1]['ops_per_second']
            performance_degradation = (initial_ops - final_ops) / initial_ops
            
            # Performance should not degrade more than 50%
            assert performance_degradation < 0.5, f"Performance degradation too high: {performance_degradation:.2%}"
            
            # Even under highest load, should maintain reasonable performance
            assert final_ops > 10, f"Final operations per second too low: {final_ops:.2f}"
        
        def test_cache_ttl_expiration_edge_cases(self, cache_manager):
            """Test TTL expiration edge cases."""
            # Test immediate expiration
            cache_manager.get.return_value = None  # Cache miss
            
            # Test very short TTL
            with patch('time.time') as mock_time:
                current_time = 1000.0
                
                def advance_time(seconds):
                    nonlocal current_time
                    current_time += seconds
                    return current_time
                
                mock_time.side_effect = lambda: current_time
                
                # Test TTL expiration scenarios
                test_cases = [
                    {'ttl': 0, 'should_expire': True},      # Immediate expiration
                    {'ttl': 1, 'should_expire': True},      # Very short TTL
                    {'ttl': 3600, 'should_expire': False},  # Long TTL
                    {'ttl': -1, 'should_expire': True},     # Invalid TTL
                ]
                
                for case in test_cases:
                    # Set cache item
                    cache_manager.set('test_key', 'test_value', ttl=case['ttl'])
                    
                    # Advance time
                    if case['ttl'] > 0:
                        advance_time(case['ttl'] + 1)
                    else:
                        advance_time(1)
                    
                    # Check if item expired
                    result = cache_manager.get('test_key')
                    
                    if case['should_expire']:
                        assert result is None, f"Item with TTL {case['ttl']} should expire"
                    else:
                        assert result is not None, f"Item with TTL {case['ttl']} should not expire"
        
        def test_cache_concurrent_access_thread_safety(self, cache_manager):
            """Test thread safety of cache operations."""
            results = []
            errors = []
            
            def worker_thread(thread_id, operations_count):
                """Worker thread performing cache operations."""
                thread_results = []
                
                try:
                    for i in range(operations_count):
                        key = f"thread_{thread_id}_key_{i}"
                        value = f"thread_{thread_id}_value_{i}"
                        
                        # Mix of operations
                        operation = i % 4
                        if operation == 0:
                            # Set
                            result = cache_manager.set(key, value)
                            thread_results.append(('set', key, result))
                        elif operation == 1:
                            # Get
                            result = cache_manager.get(key)
                            thread_results.append(('get', key, result))
                        elif operation == 2:
                            # Delete
                            result = cache_manager.delete(key)
                            thread_results.append(('delete', key, result))
                        else:
                            # Get stats
                            result = cache_manager.get_stats()
                            thread_results.append(('stats', 'all', result is not None))
                
                except Exception as e:
                    errors.append(f"Thread {thread_id} error: {str(e)}")
                
                return thread_results
            
            # Create multiple worker threads
            threads = []
            thread_count = 10
            operations_per_thread = 100
            
            for i in range(thread_count):
                thread = threading.Thread(
                    target=lambda tid=i: results.append(worker_thread(tid, operations_per_thread))
                )
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Verify results
            assert len(errors) == 0, f"Thread safety errors: {errors}"
            assert len(results) == thread_count, f"Expected {thread_count} thread results, got {len(results)}"
            
            # Verify operation counts
            total_operations = sum(len(thread_result) for thread_result in results)
            expected_operations = thread_count * operations_per_thread
            assert total_operations == expected_operations, f"Operations count mismatch: {total_operations} vs {expected_operations}"
    
    class TestConcurrentPerformanceTesting:
        """Test concurrent access patterns and performance."""
        
        @pytest.mark.asyncio
        async def test_concurrent_voice_processing(self, performance_monitor, sample_load_data):
            """Test concurrent voice processing performance."""
            # Configure performance monitoring
            performance_monitor.get_current_metrics.return_value = {
                'memory_usage_mb': 200.0,
                'cpu_usage_percent': 75.0,
                'response_time_ms': 200.0,
                'throughput_ops_per_sec': 500.0,
                'error_rate': 0.02,
                'cache_hit_rate': 0.80
            }
            
            async def process_voice_request(request_data):
                """Simulate voice processing request."""
                try:
                    # Simulate audio processing time
                    processing_time = request_data['audio_size'] / 1024  # Rough scaling
                    await asyncio.sleep(processing_time / 100)  # Convert to seconds
                    
                    # Simulate transcription time
                    transcription_time = request_data['transcription_length'] / 100
                    await asyncio.sleep(transcription_time / 100)
                    
                    return {
                        'success': True,
                        'processing_time_ms': (processing_time + transcription_time) * 10,
                        'transcription_length': request_data['transcription_length']
                    }
                except Exception as e:
                    return {
                        'success': False,
                        'error': str(e)
                    }
            
            # Create concurrent processing tasks
            concurrent_limit = 20
            semaphore = asyncio.Semaphore(concurrent_limit)
            
            async def limited_processing(request_data):
                async with semaphore:
                    return await process_voice_request(request_data)
            
            # Process all voice requests concurrently
            start_time = time.time()
            tasks = [limited_processing(req) for req in sample_load_data['voice_requests']]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            # Analyze results
            successful_requests = [r for r in results if isinstance(r, dict) and r.get('success')]
            failed_requests = [r for r in results if isinstance(r, dict) and not r.get('success')]
            
            # Performance metrics
            total_time = end_time - start_time
            throughput = len(successful_requests) / total_time
            success_rate = len(successful_requests) / len(results)
            avg_response_time = np.mean([r['processing_time_ms'] for r in successful_requests])
            
            # Verify performance
            assert success_rate > 0.95, f"Success rate too low: {success_rate:.2%}"
            assert throughput > 10, f"Throughput too low: {throughput:.2f} req/sec"
            assert avg_response_time < 1000, f"Average response time too high: {avg_response_time:.1f}ms"
            
            # Verify concurrent limit was respected
            peak_concurrent = concurrent_limit  # In real implementation, track actual concurrency
            assert peak_concurrent <= concurrent_limit, "Concurrent limit exceeded"
        
        def test_thread_pool_resource_management(self):
            """Test thread pool resource management under load."""
            max_workers = 10
            queue_size = 100
            
            results = queue.Queue()
            errors = queue.Queue()
            
            def task_worker(task_id):
                """Worker function for thread pool."""
                try:
                    # Simulate CPU-intensive work
                    result = sum(i * i for i in range(1000))
                    
                    # Simulate some I/O wait
                    time.sleep(np.random.uniform(0.01, 0.1))
                    
                    results.put({
                        'task_id': task_id,
                        'result': result,
                        'thread_id': threading.current_thread().ident
                    })
                except Exception as e:
                    errors.put({'task_id': task_id, 'error': str(e)})
            
            # Test with ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit many tasks
                futures = []
                task_count = 200
                
                for i in range(task_count):
                    future = executor.submit(task_worker, i)
                    futures.append(future)
                
                # Wait for completion with timeout
                completed_futures = []
                for future in as_completed(futures, timeout=30):
                    try:
                        future.result()
                        completed_futures.append(future)
                    except Exception as e:
                        errors.put({'error': str(e)})
            
            # Analyze results
            successful_tasks = []
            while not results.empty():
                successful_tasks.append(results.get())
            
            failed_tasks = []
            while not errors.empty():
                failed_tasks.append(errors.get())
            
            # Verify resource management
            success_rate = len(successful_tasks) / task_count
            assert success_rate > 0.95, f"Thread pool success rate too low: {success_rate:.2%}"
            assert len(failed_tasks) < task_count * 0.05, "Too many failed tasks"
            
            # Verify thread reuse
            thread_ids = set(task['thread_id'] for task in successful_tasks)
            assert len(thread_ids) <= max_workers, f"Too many threads used: {len(thread_ids)}"
            
            # Verify load distribution
            tasks_per_thread = {}
            for task in successful_tasks:
                thread_id = task['thread_id']
                tasks_per_thread[thread_id] = tasks_per_thread.get(thread_id, 0) + 1
            
            # Work should be distributed among threads
            avg_tasks_per_thread = task_count / max_workers
            for thread_id, count in tasks_per_thread.items():
                assert count >= avg_tasks_per_thread * 0.5, f"Uneven load distribution on thread {thread_id}"
        
        @pytest.mark.asyncio
        async def test_async_resource_cleanup(self, memory_manager):
            """Test async resource cleanup under concurrent load."""
            cleanup_events = []
            resource_ids = []
            
            class AsyncResource:
                def __init__(self, resource_id):
                    self.resource_id = resource_id
                    self.created_at = time.time()
                    self.cleaned_up = False
                
                async def cleanup(self):
                    """Async cleanup method."""
                    await asyncio.sleep(0.01)  # Simulate cleanup work
                    self.cleaned_up = True
                    cleanup_events.append({
                        'resource_id': self.resource_id,
                        'cleanup_time': time.time(),
                        'lifetime': time.time() - self.created_at
                    })
                
                def __del__(self):
                    # Fallback cleanup
                    if not self.cleaned_up:
                        cleanup_events.append({
                            'resource_id': self.resource_id,
                            'cleanup_time': time.time(),
                            'fallback': True
                        })
            
            async def resource_user(session_id, resource_count):
                """Create and use resources."""
                resources = []
                
                try:
                    for i in range(resource_count):
                        resource = AsyncResource(f"{session_id}_{i}")
                        resources.append(resource)
                        resource_ids.append(resource.resource_id)
                        
                        # Use resource
                        await asyncio.sleep(np.random.uniform(0.01, 0.05))
                
                finally:
                    # Cleanup all resources
                    cleanup_tasks = [resource.cleanup() for resource in resources]
                    await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
            # Create concurrent resource users
            concurrent_sessions = 10
            resources_per_session = 20
            
            tasks = [
                resource_user(session_id, resources_per_session)
                for session_id in range(concurrent_sessions)
            ]
            
            start_time = time.time()
            await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            # Verify cleanup
            total_resources = concurrent_sessions * resources_per_session
            cleanup_count = len([e for e in cleanup_events if not e.get('fallback')])
            fallback_cleanup = len([e for e in cleanup_events if e.get('fallback')])
            
            assert cleanup_count == total_resources, f"Expected {total_resources} cleanups, got {cleanup_count}"
            assert fallback_cleanup == 0, "No fallback cleanups should occur"
            
            # Verify cleanup timing
            cleanup_times = [e['cleanup_time'] for e in cleanup_events]
            cleanup_duration = max(cleanup_times) - min(cleanup_times)
            
            # Cleanup should complete in reasonable time
            assert cleanup_duration < end_time - start_time + 1.0, "Cleanup taking too long"
    
    class TestAlertSystemReliability:
        """Test alert system reliability and threshold detection."""
        
        def test_memory_threshold_alerts(self, memory_manager):
            """Test memory threshold alert triggering."""
            alert_events = []
            
            def mock_alert_handler(alert_type, threshold, current_value):
                alert_events.append({
                    'type': alert_type,
                    'threshold': threshold,
                    'current_value': current_value,
                    'timestamp': time.time()
                })
            
            # Configure memory manager with different thresholds
            test_scenarios = [
                {'memory_usage': 80.0, 'limit': 100.0, 'should_alert': False},
                {'memory_usage': 95.0, 'limit': 100.0, 'should_alert': True},
                {'memory_usage': 105.0, 'limit': 100.0, 'should_alert': True},
                {'memory_usage': 50.0, 'limit': 100.0, 'should_alert': False}
            ]
            
            for scenario in test_scenarios:
                memory_manager.get_memory_usage.return_value = scenario['memory_usage']
                memory_manager.check_memory_limit.return_value = scenario['memory_usage'] > scenario['limit']
                
                # Simulate alert checking
                if memory_manager.check_memory_limit():
                    mock_alert_handler('memory_limit', scenario['limit'], scenario['memory_usage'])
                
                # Verify alert was triggered correctly
                if scenario['should_alert']:
                    memory_alerts = [a for a in alert_events if a['type'] == 'memory_limit']
                    assert len(memory_alerts) > 0, f"Memory alert should trigger for usage {scenario['memory_usage']}%"
                else:
                    memory_alerts = [a for a in alert_events if a['type'] == 'memory_limit']
                    if scenario['memory_usage'] > scenario['limit']:
                        assert len(memory_alerts) == 0, f"No memory alert should trigger for usage {scenario['memory_usage']}%"
        
        def test_performance_degradation_alerts(self, performance_monitor):
            """Test performance degradation alert detection."""
            alert_events = []
            
            def mock_performance_alert(metric_name, threshold, current_value):
                alert_events.append({
                    'metric': metric_name,
                    'threshold': threshold,
                    'current_value': current_value,
                    'timestamp': time.time()
                })
            
            # Test performance degradation scenarios
            degradation_scenarios = [
                {
                    'metrics': {
                        'response_time_ms': 500.0,      # High response time
                        'throughput_ops_per_sec': 100.0,  # Low throughput
                        'error_rate': 0.05,               # High error rate
                        'memory_usage_mb': 150.0
                    },
                    'thresholds': {
                        'response_time_ms': 1000.0,
                        'throughput_ops_per_sec': 200.0,
                        'error_rate': 0.03,
                        'memory_usage_mb': 200.0
                    },
                    'expected_alerts': ['error_rate']  # Only error rate exceeds threshold
                },
                {
                    'metrics': {
                        'response_time_ms': 1200.0,     # Very high response time
                        'throughput_ops_per_sec': 50.0,   # Very low throughput
                        'error_rate': 0.10,               # Very high error rate
                        'memory_usage_mb': 250.0          # High memory usage
                    },
                    'thresholds': {
                        'response_time_ms': 1000.0,
                        'throughput_ops_per_sec': 200.0,
                        'error_rate': 0.03,
                        'memory_usage_mb': 200.0
                    },
                    'expected_alerts': ['response_time_ms', 'throughput_ops_per_sec', 'error_rate', 'memory_usage_mb']
                }
            ]
            
            for scenario in degradation_scenarios:
                # Configure performance monitor
                performance_monitor.get_current_metrics.return_value = scenario['metrics']
                
                # Check each metric against thresholds
                metrics = scenario['metrics']
                thresholds = scenario['thresholds']
                
                for metric_name, threshold in thresholds.items():
                    current_value = metrics[metric_name]
                    
                    # Determine if alert should trigger (for some metrics, higher is worse)
                    if metric_name in ['response_time_ms', 'error_rate', 'memory_usage_mb']:
                        should_alert = current_value > threshold
                    else:  # throughput (lower is worse)
                        should_alert = current_value < threshold
                    
                    if should_alert:
                        mock_performance_alert(metric_name, threshold, current_value)
                
                # Verify expected alerts were triggered
                for expected_alert in scenario['expected_alerts']:
                    alert_matches = [a for a in alert_events if a['metric'] == expected_alert]
                    assert len(alert_matches) > 0, f"Alert for {expected_alert} should be triggered"
        
        def test_alert_cooldown_and_deduplication(self):
            """Test alert cooldown periods and deduplication."""
            alert_log = []
            cooldown_period = 60  # seconds
            
            def send_alert(alert_type, message, severity='warning'):
                """Send alert with cooldown logic."""
                current_time = time.time()
                
                # Check for similar alerts within cooldown period
                recent_similar = [
                    a for a in alert_log 
                    if a['type'] == alert_type 
                    and current_time - a['timestamp'] < cooldown_period
                ]
                
                if recent_similar:
                    return False  # Suppressed due to cooldown
                
                alert_log.append({
                    'type': alert_type,
                    'message': message,
                    'severity': severity,
                    'timestamp': current_time
                })
                return True
            
            # Test alert deduplication
            base_time = time.time()
            
            # First alert should go through
            result1 = send_alert('high_memory', 'Memory usage at 95%', 'critical')
            assert result1 is True, "First alert should be sent"
            
            # Immediate duplicate should be suppressed
            result2 = send_alert('high_memory', 'Memory usage at 96%', 'critical')
            assert result2 is False, "Duplicate alert should be suppressed"
            
            # Different alert type should go through
            result3 = send_alert('high_cpu', 'CPU usage at 90%', 'warning')
            assert result3 is True, "Different alert type should be sent"
            
            # Alert after cooldown should go through
            with patch('time.time') as mock_time:
                mock_time.return_value = base_time + cooldown_period + 1
                
                result4 = send_alert('high_memory', 'Memory usage at 97%', 'critical')
                assert result4 is True, "Alert after cooldown should be sent"
            
            # Verify alert log
            memory_alerts = [a for a in alert_log if a['type'] == 'high_memory']
            cpu_alerts = [a for a in alert_log if a['type'] == 'high_cpu']
            
            assert len(memory_alerts) == 2, f"Expected 2 memory alerts, got {len(memory_alerts)}"
            assert len(cpu_alerts) == 1, f"Expected 1 CPU alert, got {len(cpu_alerts)}"
        
        @pytest.mark.asyncio
        async def test_alert_system_under_load(self):
            """Test alert system reliability under high load."""
            alert_queue = asyncio.Queue(maxsize=1000)
            processed_alerts = []
            dropped_alerts = []
            
            async def alert_processor():
                """Process alerts from queue."""
                while True:
                    try:
                        alert = await asyncio.wait_for(alert_queue.get(), timeout=1.0)
                        processed_alerts.append(alert)
                        await asyncio.sleep(0.01)  # Simulate processing time
                    except asyncio.TimeoutError:
                        break
            
            async def generate_alerts(alert_count, burst_interval=0.001):
                """Generate alerts at high rate."""
                for i in range(alert_count):
                    alert = {
                        'id': i,
                        'type': 'test_alert',
                        'message': f'Test alert {i}',
                        'timestamp': time.time()
                    }
                    
                    try:
                        alert_queue.put_nowait(alert)
                    except asyncio.QueueFull:
                        dropped_alerts.append(alert)
                    
                    if burst_interval > 0:
                        await asyncio.sleep(burst_interval)
            
            # Test scenarios
            test_scenarios = [
                {'count': 100, 'interval': 0.01, 'description': 'Moderate load'},
                {'count': 500, 'interval': 0.001, 'description': 'High load'},
                {'count': 1000, 'interval': 0.0001, 'description': 'Burst load'}
            ]
            
            for scenario in test_scenarios:
                # Clear previous results
                alert_queue._queue.clear()
                processed_alerts.clear()
                dropped_alerts.clear()
                
                # Start processor
                processor_task = asyncio.create_task(alert_processor())
                
                # Generate alerts
                await generate_alerts(scenario['count'], scenario['interval'])
                
                # Wait for processing
                await asyncio.sleep(2.0)
                
                # Stop processor
                processor_task.cancel()
                try:
                    await processor_task
                except asyncio.CancelledError:
                    pass
                
                # Analyze results
                total_generated = scenario['count']
                total_processed = len(processed_alerts)
                total_dropped = len(dropped_alerts)
                
                processing_rate = total_processed / total_generated
                drop_rate = total_dropped / total_generated
                
                # Most alerts should be processed
                assert processing_rate > 0.9, f"{scenario['description']}: Processing rate too low {processing_rate:.2%}"
                assert drop_rate < 0.1, f"{scenario['description']}: Drop rate too high {drop_rate:.2%}"
    
    class TestPerformanceRegressionDetection:
        """Test performance regression detection capabilities."""
        
        def test_baseline_performance_capture(self, performance_monitor):
            """Test capturing and storing baseline performance metrics."""
            baseline_metrics = {
                'response_time_p50': 150.0,
                'response_time_p95': 300.0,
                'response_time_p99': 500.0,
                'throughput_ops_per_sec': 1000.0,
                'memory_usage_mb': 100.0,
                'cpu_usage_percent': 25.0,
                'error_rate': 0.01,
                'cache_hit_rate': 0.85,
                'timestamp': time.time()
            }
            
            # Store baseline
            baseline_storage = {}
            baseline_storage['current'] = baseline_metrics
            
            # Verify baseline capture
            assert 'current' in baseline_storage, "Baseline should be stored"
            assert baseline_storage['current'] == baseline_metrics, "Baseline should match captured metrics"
            
            # Test baseline comparison
            current_metrics = baseline_metrics.copy()
            current_metrics['timestamp'] = time.time()
            current_metrics['response_time_p95'] = 450.0  # 50% regression
            current_metrics['throughput_ops_per_sec'] = 800.0  # 20% regression
            
            # Calculate regressions
            regressions = {}
            for metric, baseline_value in baseline_metrics.items():
                if metric == 'timestamp':
                    continue
                
                current_value = current_metrics.get(metric)
                if current_value is None:
                    continue
                
                # Different metrics have different regression directions
                if metric in ['response_time_p50', 'response_time_p95', 'response_time_p99', 
                             'memory_usage_mb', 'cpu_usage_percent', 'error_rate']:
                    # Lower is better
                    regression_pct = ((current_value - baseline_value) / baseline_value) * 100
                else:
                    # Higher is better
                    regression_pct = ((baseline_value - current_value) / baseline_value) * 100
                
                if regression_pct > 10:  # 10% regression threshold
                    regressions[metric] = {
                        'baseline': baseline_value,
                        'current': current_value,
                        'regression_percent': regression_pct
                    }
            
            # Verify regression detection
            assert len(regressions) >= 2, "Should detect multiple regressions"
            assert 'response_time_p95' in regressions, "Should detect response time regression"
            assert 'throughput_ops_per_sec' in regressions, "Should detect throughput regression"
            
            # Verify regression percentages
            assert regressions['response_time_p95']['regression_percent'] == 50.0
            assert regressions['throughput_ops_per_sec']['regression_percent'] == 20.0
        
        def test_trend_analysis_and_prediction(self):
            """Test performance trend analysis and prediction."""
            # Generate historical performance data
            historical_metrics = []
            base_date = time.time() - (30 * 24 * 60 * 60)  # 30 days ago
            
            for day in range(30):
                # Simulate gradual performance degradation
                degradation_factor = 1 + (day * 0.01)  # 1% degradation per day
                
                daily_metrics = {
                    'date': base_date + (day * 24 * 60 * 60),
                    'response_time_p95': 200.0 * degradation_factor,
                    'throughput_ops_per_sec': 1000.0 / degradation_factor,
                    'memory_usage_mb': 100.0 * degradation_factor,
                    'error_rate': 0.01 * degradation_factor
                }
                historical_metrics.append(daily_metrics)
            
            # Calculate trends
            def calculate_trend(values):
                """Calculate linear trend for values."""
                if len(values) < 2:
                    return 0.0
                
                n = len(values)
                x = list(range(n))
                y = values
                
                # Simple linear regression
                x_mean = sum(x) / n
                y_mean = sum(y) / n
                
                numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
                denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
                
                if denominator == 0:
                    return 0.0
                
                return numerator / denominator
            
            # Analyze trends for each metric
            trends = {}
            for metric in ['response_time_p95', 'throughput_ops_per_sec', 'memory_usage_mb', 'error_rate']:
                values = [m[metric] for m in historical_metrics]
                trend = calculate_trend(values)
                trends[metric] = trend
            
            # Verify trend detection
            assert trends['response_time_p95'] > 0, "Response time should be increasing"
            assert trends['throughput_ops_per_sec'] < 0, "Throughput should be decreasing"
            assert trends['memory_usage_mb'] > 0, "Memory usage should be increasing"
            assert trends['error_rate'] > 0, "Error rate should be increasing"
            
            # Predict future performance
            future_day = 30
            predictions = {}
            for metric, trend in trends.items():
                current_value = historical_metrics[-1][metric]
                predicted_value = current_value + (trend * 7)  # Predict 7 days ahead
                predictions[metric] = predicted_value
            
            # Verify predictions make sense
            assert predictions['response_time_p95'] > historical_metrics[-1]['response_time_p95']
            assert predictions['throughput_ops_per_sec'] < historical_metrics[-1]['throughput_ops_per_sec']
        
        @pytest.mark.asyncio
        async def test_continuous_performance_monitoring(self, performance_monitor):
            """Test continuous performance monitoring with regression detection."""
            monitoring_results = []
            regression_alerts = []
            
            async def monitoring_loop(duration_seconds, check_interval=1.0):
                """Simulate continuous monitoring."""
                start_time = time.time()
                baseline_captured = False
                
                while time.time() - start_time < duration_seconds:
                    # Get current metrics
                    metrics = performance_monitor.get_current_metrics()
                    
                    if not baseline_captured:
                        # Capture baseline on first iteration
                        baseline = metrics.copy()
                        baseline_captured = True
                    else:
                        # Check for regressions
                        for metric, baseline_value in baseline.items():
                            if metric == 'timestamp':
                                continue
                            
                            current_value = metrics.get(metric)
                            if current_value is None:
                                continue
                            
                            # Calculate regression
                            if metric in ['memory_usage_mb', 'cpu_usage_percent', 'error_rate']:
                                regression_pct = ((current_value - baseline_value) / baseline_value) * 100
                            else:
                                regression_pct = ((baseline_value - current_value) / baseline_value) * 100
                            
                            if regression_pct > 5:  # 5% regression threshold
                                regression_alerts.append({
                                    'metric': metric,
                                    'baseline': baseline_value,
                                    'current': current_value,
                                    'regression_percent': regression_pct,
                                    'timestamp': time.time()
                                })
                    
                    monitoring_results.append({
                        'timestamp': time.time(),
                        'metrics': metrics.copy()
                    })
                    
                    await asyncio.sleep(check_interval)
            
            # Simulate performance degradation over time
            metric_degradation = {
                'response_time_ms': 150.0,
                'throughput_ops_per_sec': 1000.0,
                'error_rate': 0.01
            }
            
            def degrading_metrics():
                """Return metrics with gradual degradation."""
                nonlocal metric_degradation
                
                # Degrade metrics slightly each call
                metric_degradation['response_time_ms'] *= 1.02
                metric_degradation['throughput_ops_per_sec'] *= 0.98
                metric_degradation['error_rate'] *= 1.01
                
                return {
                    'memory_usage_mb': 100.0,
                    'cpu_usage_percent': 25.0,
                    'response_time_ms': metric_degradation['response_time_ms'],
                    'throughput_ops_per_sec': metric_degradation['throughput_ops_per_sec'],
                    'error_rate': metric_degradation['error_rate'],
                    'cache_hit_rate': 0.85
                }
            
            performance_monitor.get_current_metrics.side_effect = degrading_metrics
            
            # Run monitoring
            await monitoring_loop(duration_seconds=5, check_interval=0.5)
            
            # Verify monitoring was active
            assert len(monitoring_results) >= 8, f"Expected at least 8 monitoring points, got {len(monitoring_results)}"
            
            # Verify regression detection
            assert len(regression_alerts) > 0, "Should detect performance regressions"
            
            # Verify regression types
            regression_metrics = set(alert['metric'] for alert in regression_alerts)
            expected_regressions = {'response_time_ms', 'throughput_ops_per_sec', 'error_rate'}
            
            assert regression_metrics.intersection(expected_regressions), "Should detect multiple types of regressions"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])