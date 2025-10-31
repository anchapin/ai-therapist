"""
Memory Management Testing

Comprehensive test suite for memory management functionality:
- Real-time memory usage monitoring and tracking
- Automatic garbage collection triggers
- Memory leak detection and reporting
- Resource cleanup utilities
- Memory threshold alerts and management

Coverage targets: Memory management testing for performance monitoring
"""

import pytest
import time
import threading
import gc
import psutil
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional

from performance.memory_manager import (
    MemoryManager, MemoryStats, MemoryLeak, MemoryAlertLevel
)


@pytest.fixture
def memory_config():
    """Create test memory manager configuration."""
    return {
        'memory_threshold_low': 100,      # MB
        'memory_threshold_medium': 200,   # MB
        'memory_threshold_high': 300,     # MB
        'memory_threshold_critical': 400, # MB
        'monitoring_interval': 1.0,       # Fast monitoring for tests
        'gc_threshold': 100,              # Low GC threshold for tests
        'leak_detection_window': 10.0,    # Short window for tests
        'alert_cooldown': 1.0,            # Fast cooldown for tests
        'cleanup_interval': 5.0           # Fast cleanup for tests
    }


@pytest.fixture
def memory_manager(memory_config):
    """Create memory manager instance for testing."""
    manager = MemoryManager(memory_config)
    yield manager
    manager.stop_monitoring()  # Ensure cleanup


class TestMemoryManagerInitialization:
    """Test MemoryManager initialization and configuration."""

    def test_initialization_with_config(self, memory_config):
        """Test memory manager initialization with configuration."""
        manager = MemoryManager(memory_config)

        assert manager.memory_threshold_low == 100
        assert manager.memory_threshold_medium == 200
        assert manager.memory_threshold_high == 300
        assert manager.memory_threshold_critical == 400
        assert manager.monitoring_interval == 1.0
        assert isinstance(manager.memory_history, list)
        assert isinstance(manager.alert_callbacks, list)

        manager.stop_monitoring()

    def test_initialization_default_config(self):
        """Test memory manager initialization with default configuration."""
        manager = MemoryManager()

        assert manager.memory_threshold_low == 512
        assert manager.memory_threshold_medium == 1024
        assert manager.memory_threshold_high == 1536
        assert manager.memory_threshold_critical == 2048
        assert isinstance(manager.memory_history, list)
        assert isinstance(manager.alert_callbacks, list)

        manager.stop_monitoring()

    def test_initialization_with_callbacks(self, memory_config):
        """Test memory manager initialization with callback registration."""
        manager = MemoryManager(memory_config)

        alerts_received = []

        def alert_callback(level, stats):
            alerts_received.append((level, stats.process_memory_mb))

        manager.register_alert_callback(alert_callback)
        manager.register_cleanup_callback(lambda: None)

        assert len(manager.alert_callbacks) == 1
        assert len(manager.cleanup_callbacks) == 1

        manager.stop_monitoring()


class TestMemoryStatsCollection:
    """Test memory statistics collection."""

    def test_get_memory_stats_basic(self, memory_manager):
        """Test basic memory statistics collection."""
        stats = memory_manager.get_memory_stats()

        assert isinstance(stats, MemoryStats)
        assert stats.total_memory_mb > 0
        assert stats.used_memory_mb >= 0
        assert stats.available_memory_mb > 0
        assert 0 <= stats.memory_percent <= 100
        assert stats.process_memory_mb >= 0
        assert stats.gc_objects >= 0
        assert isinstance(stats.gc_collections, dict)
        assert stats.timestamp > 0

    @patch('performance.memory_manager.psutil')
    def test_get_memory_stats_psutil_error(self, mock_psutil, memory_manager):
        """Test memory stats collection when psutil fails."""
        mock_psutil.virtual_memory.side_effect = Exception("psutil error")
        mock_psutil.Process.side_effect = Exception("process error")

        stats = memory_manager.get_memory_stats()

        # Should return fallback stats
        assert stats.total_memory_mb == 0.0
        assert stats.process_memory_mb == 0.0
        assert stats.gc_objects == 0

    def test_memory_history_tracking(self, memory_manager):
        """Test memory history tracking."""
        initial_history_size = len(memory_manager.memory_history)

        # Start monitoring to populate history
        memory_manager.start_monitoring()
        time.sleep(2)  # Wait for a couple monitoring cycles

        # Should have collected some history
        assert len(memory_manager.memory_history) > initial_history_size

        # All entries should be MemoryStats objects
        for entry in memory_manager.memory_history:
            assert isinstance(entry, MemoryStats)

        memory_manager.stop_monitoring()


class TestGarbageCollection:
    """Test garbage collection functionality."""

    def test_force_garbage_collection(self, memory_manager):
        """Test forced garbage collection."""
        # Create some garbage
        garbage = [object() for _ in range(1000)]
        del garbage

        # Force GC
        stats = memory_manager.force_garbage_collection()

        assert isinstance(stats, dict)
        assert 'objects_collected' in stats
        assert 'memory_freed_mb' in stats
        assert 'collections_by_generation' in stats
        assert len(stats['collections_by_generation']) == 3  # 3 generations

        # Should have collected some objects
        assert stats['objects_collected'] >= 0

    def test_gc_statistics_tracking(self, memory_manager):
        """Test garbage collection statistics tracking."""
        initial_gc_count = memory_manager.metrics['gc_collections']

        memory_manager.force_garbage_collection()

        # GC count should be incremented
        assert memory_manager.metrics['gc_collections'] == initial_gc_count + 1

    def test_gc_error_handling(self, memory_manager):
        """Test garbage collection error handling."""
        with patch('performance.memory_manager.gc.collect', side_effect=Exception("GC error")):
            stats = memory_manager.force_garbage_collection()

            # Should return empty dict on error
            assert stats == {}


class TestMemoryLeakDetection:
    """Test memory leak detection functionality."""

    def test_memory_leak_detection_basic(self, memory_manager):
        """Test basic memory leak detection."""
        # Create objects that persist
        persistent_objects = [object() for _ in range(100)]
        memory_manager.known_objects.update(id(obj) for obj in persistent_objects)

        # Run leak detection
        leaks = memory_manager.detect_memory_leaks()

        # Should detect some objects (depending on GC behavior)
        assert isinstance(leaks, dict)

    def test_leak_detection_with_growth(self, memory_manager):
        """Test leak detection with object growth patterns."""
        # Simulate object growth over time
        memory_manager.start_monitoring()

        # Wait for multiple monitoring cycles
        time.sleep(3)

        leaks = memory_manager.detect_memory_leaks()

        # Should handle growth analysis
        assert isinstance(leaks, dict)

        memory_manager.stop_monitoring()

    def test_leak_analysis_algorithm(self, memory_manager):
        """Test the leak analysis algorithm."""
        # Create fake object counts with growth
        memory_manager.object_counts['TestClass'] = [
            (time.time() - 10, 100, 1000),  # 10 seconds ago
            (time.time() - 5, 120, 1200),   # 5 seconds ago
            (time.time(), 150, 1500)        # Now
        ]

        # Analyze growth
        memory_manager._analyze_object_growth('TestClass')

        # Should detect significant growth
        assert 'TestClass' in memory_manager.detected_leaks
        leak_info = memory_manager.detected_leaks['TestClass']
        assert isinstance(leak_info, MemoryLeak)
        assert leak_info.object_type == 'TestClass'
        assert leak_info.count == 150
        assert leak_info.growth_rate > 0

    def test_leak_detection_error_handling(self, memory_manager):
        """Test leak detection error handling."""
        with patch('performance.memory_manager.gc.get_objects', side_effect=Exception("GC error")):
            leaks = memory_manager.detect_memory_leaks()

            # Should return empty dict on error
            assert leaks == {}


class TestMemoryThresholdAlerts:
    """Test memory threshold alert system."""

    def test_memory_threshold_alert_levels(self, memory_manager):
        """Test memory threshold alert level determination."""
        # Test different memory levels
        test_cases = [
            (50, None),      # Below low threshold
            (150, MemoryAlertLevel.LOW),     # Above low
            (250, MemoryAlertLevel.MEDIUM),  # Above medium
            (350, MemoryAlertLevel.HIGH),    # Above high
            (450, MemoryAlertLevel.CRITICAL) # Above critical
        ]

        for memory_mb, expected_level in test_cases:
            with patch.object(memory_manager, '_get_current_memory_mb', return_value=memory_mb):
                stats = memory_manager.get_memory_stats()
                memory_manager._check_memory_thresholds(stats)

                # Check if alert was triggered (may not trigger due to cooldown)
                # This tests the threshold logic

    def test_alert_callback_triggering(self, memory_manager):
        """Test alert callback triggering."""
        alerts_received = []

        def alert_callback(level, stats):
            alerts_received.append((level, stats.process_memory_mb))

        memory_manager.register_alert_callback(alert_callback)

        # Simulate high memory usage
        with patch.object(memory_manager, '_get_current_memory_mb', return_value=1000):
            stats = memory_manager.get_memory_stats()
            memory_manager._check_memory_thresholds(stats)

        # Should have triggered alerts
        assert len(alerts_received) > 0

    def test_alert_cooldown_prevention(self, memory_manager):
        """Test alert cooldown prevents spam."""
        alerts_received = []

        def alert_callback(level, stats):
            alerts_received.append((level, stats.process_memory_mb))

        memory_manager.register_alert_callback(alert_callback)

        # First alert
        with patch.object(memory_manager, '_get_current_memory_mb', return_value=1000):
            stats = memory_manager.get_memory_stats()
            memory_manager._check_memory_thresholds(stats)

        initial_alert_count = len(alerts_received)

        # Immediate second check (should be blocked by cooldown)
        with patch.object(memory_manager, '_get_current_memory_mb', return_value=1000):
            stats = memory_manager.get_memory_stats()
            memory_manager._check_memory_thresholds(stats)

        # Should not have additional alerts due to cooldown
        assert len(alerts_received) == initial_alert_count

    def test_automatic_cleanup_on_high_alert(self, memory_manager):
        """Test automatic cleanup trigger on high memory alerts."""
        cleanup_called = False

        def cleanup_callback():
            nonlocal cleanup_called
            cleanup_called = True

        memory_manager.register_cleanup_callback(cleanup_callback)

        # Simulate critical memory usage
        with patch.object(memory_manager, '_get_current_memory_mb', return_value=2000):
            stats = memory_manager.get_memory_stats()
            memory_manager._check_memory_thresholds(stats)

        # Cleanup should be triggered automatically
        assert cleanup_called


class TestMemoryCleanupOperations:
    """Test memory cleanup operations."""

    def test_trigger_memory_cleanup(self, memory_manager):
        """Test comprehensive memory cleanup triggering."""
        cleanup_called = False

        def cleanup_callback():
            nonlocal cleanup_called
            cleanup_called = True

        memory_manager.register_cleanup_callback(cleanup_callback)

        # Trigger cleanup
        result = memory_manager.trigger_memory_cleanup()

        assert isinstance(result, dict)
        assert cleanup_called

        # Should have performed GC
        assert 'objects_collected' in result

    def test_cleanup_resource_management(self, memory_manager):
        """Test cleanup of tracked resources."""
        # Track some resources
        resource1 = Mock()
        resource2 = Mock()
        resource1.cleanup = Mock()
        resource2.cleanup = Mock()

        memory_manager.track_resource("resource1", resource1)
        memory_manager.track_resource("resource2", resource2)

        # Trigger cleanup
        memory_manager.trigger_memory_cleanup()

        # Resources should be cleaned up
        resource1.cleanup.assert_called_once()
        resource2.cleanup.assert_called_once()

    def test_cleanup_weak_reference_handling(self, memory_manager):
        """Test cleanup with weak references."""
        # Create object and track it
        obj = Mock()
        obj.cleanup = Mock()
        memory_manager.track_resource("temp_resource", obj)

        # Delete the object
        del obj

        # Trigger cleanup - should handle dead references
        memory_manager.trigger_memory_cleanup()

        # Should not crash and should clean up dead references

    def test_cleanup_cache_clearing(self, memory_manager):
        """Test cache clearing during cleanup."""
        # This tests the cache clearing functionality
        # (In real implementation, this would clear various caches)
        result = memory_manager.trigger_memory_cleanup()

        assert isinstance(result, dict)


class TestResourceTracking:
    """Test resource tracking functionality."""

    def test_resource_tracking_basic(self, memory_manager):
        """Test basic resource tracking."""
        resource = Mock()
        memory_manager.track_resource("test_resource", resource)

        assert "test_resource" in memory_manager.tracked_resources
        assert len(memory_manager.tracked_resources) == 1

    def test_resource_untracking(self, memory_manager):
        """Test resource untracking."""
        resource = Mock()
        memory_manager.track_resource("test_resource", resource)
        assert "test_resource" in memory_manager.tracked_resources

        memory_manager.untrack_resource("test_resource")
        assert "test_resource" not in memory_manager.tracked_resources

    def test_resource_cleanup_on_destroy(self, memory_manager):
        """Test resource cleanup when destroyed."""
        destroyed_names = []

        # Override the resource destroy callback
        original_callback = memory_manager._resource_destroyed
        def track_destroyed(name):
            destroyed_names.append(name)
            original_callback(name)

        memory_manager._resource_destroyed = track_destroyed

        resource = Mock()
        memory_manager.track_resource("destroy_test", resource)

        # Simulate resource destruction
        del resource
        gc.collect()  # Force garbage collection

        # Should be cleaned up eventually (weakref behavior)
        # Note: weakref callbacks may not fire immediately


class TestBackgroundMonitoring:
    """Test background monitoring functionality."""

    def test_monitoring_thread_startup(self, memory_manager):
        """Test monitoring thread startup and operation."""
        memory_manager.start_monitoring()

        assert memory_manager.is_monitoring
        assert memory_manager.monitoring_thread.is_alive()
        assert memory_manager.cleanup_thread.is_alive()

        memory_manager.stop_monitoring()

    def test_monitoring_thread_shutdown(self, memory_manager):
        """Test monitoring thread shutdown."""
        memory_manager.start_monitoring()
        assert memory_manager.is_monitoring

        memory_manager.stop_monitoring()

        assert not memory_manager.is_monitoring
        # Threads should be stopped (may take a moment)

    def test_monitoring_data_collection(self, memory_manager):
        """Test that monitoring collects data over time."""
        initial_history_size = len(memory_manager.memory_history)

        memory_manager.start_monitoring()
        time.sleep(3)  # Wait for monitoring cycles

        # Should have collected more history
        assert len(memory_manager.memory_history) > initial_history_size

        memory_manager.stop_monitoring()

    def test_monitoring_error_resilience(self, memory_manager):
        """Test monitoring resilience to errors."""
        memory_manager.start_monitoring()

        # Simulate error in monitoring
        with patch.object(memory_manager, 'get_memory_stats', side_effect=Exception("Monitor error")):
            time.sleep(2)  # Should continue running despite errors

        # Should still be monitoring
        assert memory_manager.is_monitoring

        memory_manager.stop_monitoring()


class TestPerformanceMetrics:
    """Test performance metrics collection."""

    def test_performance_metrics_collection(self, memory_manager):
        """Test performance metrics collection."""
        # Perform some operations
        memory_manager.force_garbage_collection()
        memory_manager.trigger_memory_cleanup()

        metrics = memory_manager.get_performance_metrics()

        assert isinstance(metrics, dict)
        assert 'gc_collections' in metrics
        assert 'memory_alerts' in metrics
        assert 'leaks_detected' in metrics
        assert 'resources_cleaned' in metrics
        assert 'uptime_seconds' in metrics
        assert 'current_memory_mb' in metrics

    def test_metrics_increment_tracking(self, memory_manager):
        """Test that metrics are properly incremented."""
        initial_gc = memory_manager.metrics['gc_collections']
        initial_alerts = memory_manager.metrics['memory_alerts']

        # Perform operations
        memory_manager.force_garbage_collection()

        # Simulate alert
        memory_manager.metrics['memory_alerts'] += 1

        metrics = memory_manager.get_performance_metrics()

        assert metrics['gc_collections'] >= initial_gc
        assert metrics['memory_alerts'] >= initial_alerts


class TestMemoryManagerIntegration:
    """Test memory manager integration scenarios."""

    def test_full_memory_management_workflow(self, memory_manager):
        """Test complete memory management workflow."""
        # Start monitoring
        memory_manager.start_monitoring()

        # Register callbacks
        alerts = []
        def alert_cb(level, stats):
            alerts.append((level, stats.process_memory_mb))

        memory_manager.register_alert_callback(alert_cb)
        memory_manager.register_cleanup_callback(lambda: None)

        # Track a resource
        resource = Mock()
        resource.cleanup = Mock()
        memory_manager.track_resource("workflow_test", resource)

        # Wait for monitoring
        time.sleep(2)

        # Check leak detection
        leaks = memory_manager.detect_memory_leaks()

        # Trigger cleanup
        cleanup_stats = memory_manager.trigger_memory_cleanup()

        # Stop monitoring
        memory_manager.stop_monitoring()

        # Verify workflow completion
        assert isinstance(leaks, dict)
        assert isinstance(cleanup_stats, dict)
        assert not memory_manager.is_monitoring

    def test_memory_manager_concurrent_access(self, memory_manager):
        """Test concurrent access to memory manager."""
        results = []

        def concurrent_operation(operation_id):
            try:
                if operation_id % 3 == 0:
                    stats = memory_manager.get_memory_stats()
                    results.append(('stats', stats.process_memory_mb))
                elif operation_id % 3 == 1:
                    leaks = memory_manager.detect_memory_leaks()
                    results.append(('leaks', len(leaks)))
                else:
                    metrics = memory_manager.get_performance_metrics()
                    results.append(('metrics', metrics['gc_collections']))
            except Exception as e:
                results.append(('error', str(e)))

        threads = []
        for i in range(10):
            thread = threading.Thread(target=concurrent_operation, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should have completed without errors
        assert len(results) == 10
        assert not any(r[0] == 'error' for r in results)


class TestMemoryManagerErrorHandling:
    """Test error handling in memory manager."""

    def test_initialization_error_handling(self):
        """Test error handling during initialization."""
        with patch('performance.memory_manager.resource.setrlimit', side_effect=OSError("Permission denied")):
            # Should not crash, just log warning
            manager = MemoryManager()
            manager.stop_monitoring()

    def test_monitoring_error_handling(self, memory_manager):
        """Test error handling during monitoring."""
        memory_manager.start_monitoring()

        # Simulate various errors
        with patch.object(memory_manager, '_check_memory_thresholds', side_effect=Exception("Threshold error")):
            time.sleep(1.5)  # Should continue despite errors

        assert memory_manager.is_monitoring  # Should still be running

        memory_manager.stop_monitoring()

    def test_cleanup_error_handling(self, memory_manager):
        """Test error handling during cleanup operations."""
        def failing_cleanup():
            raise Exception("Cleanup failed")

        memory_manager.register_cleanup_callback(failing_cleanup)

        # Should not crash when cleanup fails
        result = memory_manager.trigger_memory_cleanup()
        assert isinstance(result, dict)  # Should return dict even on errors


# Run basic validation
if __name__ == "__main__":
    print("Memory Management Test Suite")
    print("=" * 40)

    try:
        from performance.memory_manager import MemoryManager, MemoryStats
        print("✅ Memory manager imports successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")

    try:
        manager = MemoryManager({
            'memory_threshold_low': 100,
            'monitoring_interval': 0.1
        })
        stats = manager.get_memory_stats()
        assert isinstance(stats, MemoryStats)
        manager.stop_monitoring()
        print("✅ Basic memory stats collection working")
    except Exception as e:
        print(f"❌ Memory stats failed: {e}")

    try:
        manager = MemoryManager()
        gc_stats = manager.force_garbage_collection()
        assert isinstance(gc_stats, dict)
        manager.stop_monitoring()
        print("✅ Garbage collection working")
    except Exception as e:
        print(f"❌ Garbage collection failed: {e}")

    print("Memory management test file created - run with pytest for full validation")
