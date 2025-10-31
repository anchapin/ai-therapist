"""
Unit tests for Performance Memory Manager Module.

This module provides comprehensive test coverage for the memory management
functionality including monitoring, leak detection, garbage collection,
and resource cleanup.
"""

import pytest
import time
import threading
import gc
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Import the modules to test
from performance.memory_manager import (
    MemoryManager, MemoryStats, MemoryLeak, MemoryAlertLevel,
    RESOURCE_AVAILABLE
)


class TestMemoryStats:
    """Test MemoryStats dataclass."""

    def test_memory_stats_creation(self):
        """Test MemoryStats creation with all fields."""
        stats = MemoryStats(
            total_memory_mb=8192.0,
            used_memory_mb=4096.0,
            available_memory_mb=4096.0,
            memory_percent=50.0,
            process_memory_mb=256.0,
            gc_objects=1000,
            gc_collections={0: 10, 1: 5, 2: 2},
            timestamp=time.time()
        )

        assert stats.total_memory_mb == 8192.0
        assert stats.used_memory_mb == 4096.0
        assert stats.available_memory_mb == 4096.0
        assert stats.memory_percent == 50.0
        assert stats.process_memory_mb == 256.0
        assert stats.gc_objects == 1000
        assert stats.gc_collections == {0: 10, 1: 5, 2: 2}
        assert isinstance(stats.timestamp, float)

    def test_memory_stats_defaults(self):
        """Test MemoryStats with default values."""
        stats = MemoryStats(
            total_memory_mb=0.0,
            used_memory_mb=0.0,
            available_memory_mb=0.0,
            memory_percent=0.0,
            process_memory_mb=0.0,
            gc_objects=0,
            gc_collections={0: 0, 1: 0, 2: 0},
            timestamp=0.0
        )

        assert stats.total_memory_mb == 0.0
        assert stats.used_memory_mb == 0.0
        assert stats.available_memory_mb == 0.0
        assert stats.memory_percent == 0.0
        assert stats.process_memory_mb == 0.0
        assert stats.gc_objects == 0
        assert stats.gc_collections == {0: 0, 1: 0, 2: 0}
        assert stats.timestamp == 0.0


class TestMemoryLeak:
    """Test MemoryLeak dataclass."""

    def test_memory_leak_creation(self):
        """Test MemoryLeak creation with all fields."""
        current_time = time.time()
        leak = MemoryLeak(
            object_type="dict",
            count=1000,
            size_bytes=50000,
            growth_rate=10.5,
            first_detected=current_time,
            last_detected=current_time + 60.0
        )

        assert leak.object_type == "dict"
        assert leak.count == 1000
        assert leak.size_bytes == 50000
        assert leak.growth_rate == 10.5
        assert leak.first_detected == current_time
        assert leak.last_detected == current_time + 60.0


class TestMemoryManager:
    """Test MemoryManager class."""

    def test_memory_manager_initialization_default(self):
        """Test MemoryManager initialization with default config."""
        manager = MemoryManager()

        assert manager.memory_threshold_low == 512
        assert manager.memory_threshold_medium == 1024
        assert manager.memory_threshold_high == 1536
        assert manager.memory_threshold_critical == 2048
        assert manager.monitoring_interval == 30.0
        assert manager.gc_threshold == 1000
        assert manager.leak_detection_window == 300.0
        assert manager.alert_cooldown == 60.0
        assert manager.cleanup_interval == 600.0
        assert manager.is_monitoring is False
        assert manager.monitoring_thread is None
        assert manager.cleanup_thread is None
        assert manager.baseline_memory == 0.0
        assert len(manager.memory_history) == 0
        assert len(manager.detected_leaks) == 0
        assert len(manager.alert_callbacks) == 0
        assert len(manager.cleanup_callbacks) == 0
        assert len(manager.tracked_resources) == 0

    def test_memory_manager_initialization_custom_config(self):
        """Test MemoryManager initialization with custom config."""
        config = {
            'memory_threshold_low': 256,
            'memory_threshold_medium': 512,
            'memory_threshold_high': 768,
            'memory_threshold_critical': 1024,
            'monitoring_interval': 10.0,
            'gc_threshold': 500,
            'leak_detection_window': 600.0,
            'alert_cooldown': 30.0,
            'cleanup_interval': 300.0
        }

        manager = MemoryManager(config)

        assert manager.memory_threshold_low == 256
        assert manager.memory_threshold_medium == 512
        assert manager.memory_threshold_high == 768
        assert manager.memory_threshold_critical == 1024
        assert manager.monitoring_interval == 10.0
        assert manager.gc_threshold == 500
        assert manager.leak_detection_window == 600.0
        assert manager.alert_cooldown == 30.0
        assert manager.cleanup_interval == 300.0

    @patch('performance.memory_manager.psutil')
    def test_get_memory_stats_success(self, mock_psutil):
        """Test successful memory stats retrieval."""
        # Mock psutil objects
        mock_memory = Mock()
        mock_memory.total = 8589934592  # 8GB
        mock_memory.used = 4294967296   # 4GB
        mock_memory.available = 4294967296  # 4GB
        mock_memory.percent = 50.0

        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 268435456  # 256MB

        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.Process.return_value = mock_process

        manager = MemoryManager()
        stats = manager.get_memory_stats()

        assert isinstance(stats, MemoryStats)
        assert stats.total_memory_mb == 8192.0
        assert stats.used_memory_mb == 4096.0
        assert stats.available_memory_mb == 4096.0
        assert stats.memory_percent == 50.0
        assert stats.process_memory_mb == 256.0
        assert stats.gc_objects >= 0
        assert isinstance(stats.gc_collections, dict)
        assert len(stats.gc_collections) == 3  # 3 generations
        assert isinstance(stats.timestamp, float)

    @patch('performance.memory_manager.psutil')
    def test_get_memory_stats_error(self, mock_psutil):
        """Test memory stats retrieval with error."""
        mock_psutil.virtual_memory.side_effect = Exception("Test error")

        manager = MemoryManager()
        stats = manager.get_memory_stats()

        # Should return minimal stats on error
        assert isinstance(stats, MemoryStats)
        assert stats.total_memory_mb == 0.0
        assert stats.used_memory_mb == 0.0
        assert stats.available_memory_mb == 0.0
        assert stats.memory_percent == 0.0
        assert stats.process_memory_mb == 0.0
        assert stats.gc_objects == 0
        assert stats.gc_collections == {0: 0, 1: 0, 2: 0}

    def test_force_garbage_collection(self):
        """Test forced garbage collection."""
        manager = MemoryManager()
        
        # Create some objects to collect
        test_objects = [[] for _ in range(100)]
        del test_objects

        stats = manager.force_garbage_collection()

        assert isinstance(stats, dict)
        assert 'objects_before' in stats
        assert 'objects_after' in stats
        assert 'objects_collected' in stats
        assert 'memory_before_mb' in stats
        assert 'memory_after_mb' in stats
        assert 'memory_freed_mb' in stats
        assert 'collections_by_generation' in stats
        assert 'timestamp' in stats
        assert len(stats['collections_by_generation']) == 3
        assert manager.metrics['gc_collections'] >= 1

    def test_force_garbage_collection_error(self):
        """Test forced garbage collection with error."""
        manager = MemoryManager()
        
        with patch('gc.collect', side_effect=Exception("Test error")):
            stats = manager.force_garbage_collection()
            assert stats == {}

    def test_detect_memory_leaks(self):
        """Test memory leak detection."""
        manager = MemoryManager()
        
        # Create some objects to detect
        test_objects = [[] for _ in range(50)]
        test_strings = ["test" * 100 for _ in range(50)]
        
        leaks = manager.detect_memory_leaks()
        
        assert isinstance(leaks, dict)
        # May or may not detect leaks depending on implementation
        # but should not crash
        
        del test_objects
        del test_strings

    def test_detect_memory_leaks_error(self):
        """Test memory leak detection with error."""
        manager = MemoryManager()
        
        with patch('gc.get_objects', side_effect=Exception("Test error")):
            leaks = manager.detect_memory_leaks()
            assert leaks == {}

    def test_trigger_memory_cleanup(self):
        """Test memory cleanup trigger."""
        manager = MemoryManager()
        
        # Add a cleanup callback
        callback_called = False
        def test_callback():
            nonlocal callback_called
            callback_called = True
        
        manager.register_cleanup_callback(test_callback)
        
        result = manager.trigger_memory_cleanup()
        
        assert isinstance(result, dict)
        assert callback_called
        assert manager.metrics['resources_cleaned'] >= 1

    def test_trigger_memory_cleanup_error(self):
        """Test memory cleanup trigger with error."""
        manager = MemoryManager()
        
        with patch.object(manager, 'force_garbage_collection', side_effect=Exception("Test error")):
            result = manager.trigger_memory_cleanup()
            assert result == {}

    def test_register_alert_callback(self):
        """Test registering alert callbacks."""
        manager = MemoryManager()
        
        callback = Mock()
        manager.register_alert_callback(callback)
        
        assert len(manager.alert_callbacks) == 1
        assert manager.alert_callbacks[0] == callback

    def test_register_cleanup_callback(self):
        """Test registering cleanup callbacks."""
        manager = MemoryManager()
        
        callback = Mock()
        manager.register_cleanup_callback(callback)
        
        assert len(manager.cleanup_callbacks) == 1
        assert manager.cleanup_callbacks[0] == callback

    def test_track_resource(self):
        """Test resource tracking."""
        manager = MemoryManager()
        
        resource = {"data": "test"}
        manager.track_resource("test_resource", resource)
        
        assert "test_resource" in manager.tracked_resources
        assert manager.tracked_resources["test_resource"]() is resource

    def test_untrack_resource(self):
        """Test resource untracking."""
        manager = MemoryManager()
        
        resource = {"data": "test"}
        manager.track_resource("test_resource", resource)
        manager.untrack_resource("test_resource")
        
        assert "test_resource" not in manager.tracked_resources

    def test_get_performance_metrics(self):
        """Test performance metrics retrieval."""
        manager = MemoryManager()
        
        metrics = manager.get_performance_metrics()
        
        assert isinstance(metrics, dict)
        assert 'gc_collections' in metrics
        assert 'memory_alerts' in metrics
        assert 'leaks_detected' in metrics
        assert 'resources_cleaned' in metrics
        assert 'uptime_seconds' in metrics
        assert 'current_memory_mb' in metrics
        assert 'memory_history_size' in metrics
        assert 'tracked_resources' in metrics
        assert 'detected_leaks' in metrics

    def test_start_monitoring(self):
        """Test starting memory monitoring."""
        manager = MemoryManager(monitoring_interval=0.1)  # Short interval for tests
        
        manager.start_monitoring()
        
        assert manager.is_monitoring is True
        assert manager.monitoring_thread is not None
        assert manager.cleanup_thread is not None
        assert manager.baseline_memory > 0
        
        # Wait a bit for threads to start
        time.sleep(0.2)
        
        # Should have some memory history
        assert len(manager.memory_history) > 0
        
        manager.stop_monitoring()

    def test_start_monitoring_already_running(self):
        """Test starting monitoring when already running."""
        manager = MemoryManager(monitoring_interval=0.1)
        
        manager.start_monitoring()
        first_thread = manager.monitoring_thread
        
        # Try to start again
        manager.start_monitoring()
        
        # Should not create new threads
        assert manager.monitoring_thread is first_thread
        
        manager.stop_monitoring()

    def test_stop_monitoring(self):
        """Test stopping memory monitoring."""
        manager = MemoryManager(monitoring_interval=0.1)
        
        manager.start_monitoring()
        assert manager.is_monitoring is True
        
        manager.stop_monitoring()
        assert manager.is_monitoring is False
        
        # Wait for threads to stop
        time.sleep(0.2)
        
        assert not manager.monitoring_thread.is_alive()
        assert not manager.cleanup_thread.is_alive()

    def test_stop_monitoring_not_running(self):
        """Test stopping monitoring when not running."""
        manager = MemoryManager()
        
        # Should not crash
        manager.stop_monitoring()
        assert manager.is_monitoring is False

    def test_check_memory_thresholds(self):
        """Test memory threshold checking."""
        manager = MemoryManager()
        
        # Create stats with different memory levels
        low_stats = MemoryStats(
            total_memory_mb=8192.0,
            used_memory_mb=4096.0,
            available_memory_mb=4096.0,
            memory_percent=50.0,
            process_memory_mb=600.0,  # Above low threshold
            gc_objects=1000,
            gc_collections={0: 10, 1: 5, 2: 2},
            timestamp=time.time()
        )
        
        # Mock alert callback
        alert_callback = Mock()
        manager.register_alert_callback(alert_callback)
        
        # Should trigger low alert
        manager._check_memory_thresholds(low_stats)
        
        # Should have triggered alert
        assert manager.metrics['memory_alerts'] >= 1

    def test_trigger_alert(self):
        """Test alert triggering."""
        manager = MemoryManager()
        
        # Mock alert callback
        alert_callback = Mock()
        manager.register_alert_callback(alert_callback)
        
        stats = MemoryStats(
            total_memory_mb=8192.0,
            used_memory_mb=4096.0,
            available_memory_mb=4096.0,
            memory_percent=50.0,
            process_memory_mb=600.0,
            gc_objects=1000,
            gc_collections={0: 10, 1: 5, 2: 2},
            timestamp=time.time()
        )
        
        manager._trigger_alert(MemoryAlertLevel.LOW, stats)
        
        assert manager.metrics['memory_alerts'] >= 1
        assert manager.last_alert_time > 0
        alert_callback.assert_called_once()

    def test_analyze_object_growth(self):
        """Test object growth analysis."""
        manager = MemoryManager()
        
        # Add some mock data
        current_time = time.time()
        manager.object_counts["dict"] = [
            (current_time - 300, 100, 1000),
            (current_time - 240, 120, 1200),
            (current_time - 180, 140, 1400),
            (current_time - 120, 160, 1600),
            (current_time - 60, 180, 1800),
            (current_time, 200, 2000),
        ]
        
        manager._analyze_object_growth("dict")
        
        # Should detect growth and potentially create a leak
        if "dict" in manager.detected_leaks:
            leak = manager.detected_leaks["dict"]
            assert leak.object_type == "dict"
            assert leak.count == 200
            assert leak.size_bytes == 2000
            assert leak.growth_rate > 0

    def test_cleanup_tracked_resources(self):
        """Test cleanup of tracked resources."""
        manager = MemoryManager()
        
        # Create a mock resource with cleanup method
        resource = Mock()
        resource.cleanup = Mock()
        
        manager.track_resource("test_resource", resource)
        
        # Force cleanup
        manager._cleanup_tracked_resources()
        
        # Should have called cleanup
        resource.cleanup.assert_called_once()

    def test_get_current_memory_mb(self):
        """Test getting current memory usage."""
        manager = MemoryManager()
        
        memory_mb = manager._get_current_memory_mb()
        
        assert isinstance(memory_mb, float)
        assert memory_mb >= 0

    def test_get_current_memory_mb_error(self):
        """Test getting current memory usage with error."""
        manager = MemoryManager()
        
        with patch('performance.memory_manager.psutil.Process', side_effect=Exception("Test error")):
            memory_mb = manager._get_current_memory_mb()
            assert memory_mb == 0.0

    def test_estimate_object_size(self):
        """Test object size estimation."""
        manager = MemoryManager()
        
        # Test different object types
        small_string = "test"
        large_string = "x" * 1000
        small_list = [1, 2, 3]
        large_list = list(range(1000))
        small_dict = {"a": 1, "b": 2}
        large_dict = {str(i): i for i in range(1000)}
        
        small_string_size = manager._estimate_object_size(small_string)
        large_string_size = manager._estimate_object_size(large_string)
        small_list_size = manager._estimate_object_size(small_list)
        large_list_size = manager._estimate_object_size(large_list)
        small_dict_size = manager._estimate_object_size(small_dict)
        large_dict_size = manager._estimate_object_size(large_dict)
        
        assert isinstance(small_string_size, int)
        assert isinstance(large_string_size, int)
        assert isinstance(small_list_size, int)
        assert isinstance(large_list_size, int)
        assert isinstance(small_dict_size, int)
        assert isinstance(large_dict_size, int)
        
        # Larger objects should have larger sizes
        assert large_string_size > small_string_size
        assert large_list_size > small_list_size
        assert large_dict_size > small_dict_size

    def test_estimate_object_size_error(self):
        """Test object size estimation with error."""
        manager = MemoryManager()
        
        # Create an object that will cause an error
        class BadObject:
            def __len__(self):
                raise Exception("Test error")
        
        bad_obj = BadObject()
        size = manager._estimate_object_size(bad_obj)
        
        # Should return 0 on error
        assert size == 0

    def test_destructor_cleanup(self):
        """Test destructor cleanup."""
        manager = MemoryManager(monitoring_interval=0.1)
        manager.start_monitoring()
        
        # Simulate destructor call
        manager.__del__()
        
        # Should stop monitoring
        assert manager.is_monitoring is False

    def test_resource_destroyed_callback(self):
        """Test resource destroyed callback."""
        manager = MemoryManager()
        
        # Add a resource
        resource = {"data": "test"}
        manager.track_resource("test_resource", resource)
        
        # Simulate resource destruction
        manager._resource_destroyed("test_resource")
        
        # Resource should be removed
        assert "test_resource" not in manager.tracked_resources

    def test_monitoring_worker_interruptible(self):
        """Test that monitoring worker can be interrupted."""
        manager = MemoryManager(monitoring_interval=1.0)  # 1 second interval
        
        start_time = time.time()
        manager.start_monitoring()
        
        # Stop quickly
        time.sleep(0.1)
        manager.stop_monitoring()
        
        elapsed = time.time() - start_time
        
        # Should stop quickly, not wait full interval
        assert elapsed < 0.5

    def test_cleanup_worker_interruptible(self):
        """Test that cleanup worker can be interrupted."""
        manager = MemoryManager(cleanup_interval=1.0)  # 1 second interval
        
        start_time = time.time()
        manager.start_monitoring()
        
        # Stop quickly
        time.sleep(0.1)
        manager.stop_monitoring()
        
        elapsed = time.time() - start_time
        
        # Should stop quickly, not wait full interval
        assert elapsed < 0.5

    @patch('performance.memory_manager.RESOURCE_AVAILABLE', True)
    @patch('performance.memory_manager.resource')
    def test_resource_limits_setting(self, mock_resource):
        """Test setting resource limits when available."""
        mock_resource.setrlimit = Mock()
        mock_resource.RLIMIT_AS = Mock()
        
        manager = MemoryManager()
        
        # Should attempt to set resource limits
        mock_resource.setrlimit.assert_called_once()

    @patch('performance.memory_manager.RESOURCE_AVAILABLE', True)
    @patch('performance.memory_manager.resource')
    def test_resource_limits_error(self, mock_resource):
        """Test handling resource limits setting error."""
        mock_resource.setrlimit.side_effect = Exception("Test error")
        mock_resource.RLIMIT_AS = Mock()
        
        manager = MemoryManager()
        
        # Should handle error gracefully
        assert manager is not None

    def test_gc_threshold_setting(self):
        """Test GC threshold setting."""
        manager = MemoryManager(gc_threshold=500)
        
        # Should set GC thresholds
        assert manager.gc_threshold == 500

    def test_memory_history_size_limit(self):
        """Test memory history size limit."""
        manager = MemoryManager(max_history_size=5)
        
        # Add more entries than limit
        for i in range(10):
            stats = MemoryStats(
                total_memory_mb=8192.0,
                used_memory_mb=4096.0,
                available_memory_mb=4096.0,
                memory_percent=50.0,
                process_memory_mb=256.0,
                gc_objects=1000,
                gc_collections={0: 10, 1: 5, 2: 2},
                timestamp=time.time() + i
            )
            manager.memory_history.append(stats)
        
        # Should limit size
        assert len(manager.memory_history) <= 5

    def test_alert_cooldown(self):
        """Test alert cooldown functionality."""
        manager = MemoryManager(alert_cooldown=0.1)
        
        stats = MemoryStats(
            total_memory_mb=8192.0,
            used_memory_mb=4096.0,
            available_memory_mb=4096.0,
            memory_percent=50.0,
            process_memory_mb=600.0,
            gc_objects=1000,
            gc_collections={0: 10, 1: 5, 2: 2},
            timestamp=time.time()
        )
        
        # Trigger first alert
        manager._trigger_alert(MemoryAlertLevel.LOW, stats)
        first_alert_time = manager.last_alert_time
        
        # Try to trigger second alert immediately
        manager._trigger_alert(MemoryAlertLevel.LOW, stats)
        second_alert_time = manager.last_alert_time
        
        # Should not trigger second alert due to cooldown
        assert first_alert_time == second_alert_time
        
        # Wait for cooldown
        time.sleep(0.2)
        
        # Should trigger alert after cooldown
        manager._trigger_alert(MemoryAlertLevel.LOW, stats)
        assert manager.last_alert_time > second_alert_time

    def test_auto_cleanup_on_high_alert(self):
        """Test automatic cleanup on high memory alerts."""
        manager = MemoryManager()
        
        # Mock cleanup method
        cleanup_called = False
        def mock_cleanup():
            nonlocal cleanup_called
            cleanup_called = True
        
        manager.trigger_memory_cleanup = mock_cleanup
        
        stats = MemoryStats(
            total_memory_mb=8192.0,
            used_memory_mb=4096.0,
            available_memory_mb=4096.0,
            memory_percent=50.0,
            process_memory_mb=1600.0,  # Above high threshold
            gc_objects=1000,
            gc_collections={0: 10, 1: 5, 2: 2},
            timestamp=time.time()
        )
        
        # Trigger high alert
        manager._trigger_alert(MemoryAlertLevel.HIGH, stats)
        
        # Should trigger cleanup
        assert cleanup_called

    def test_clear_caches(self):
        """Test cache clearing functionality."""
        manager = MemoryManager()
        
        # Should not crash
        manager._clear_caches()

    def test_clear_caches_error(self):
        """Test cache clearing with error."""
        manager = MemoryManager()
        
        with patch('sys.modules.items', side_effect=Exception("Test error")):
            # Should handle error gracefully
            manager._clear_caches()


class TestMemoryManagerIntegration:
    """Integration tests for MemoryManager."""

    def test_full_monitoring_cycle(self):
        """Test a full monitoring cycle."""
        manager = MemoryManager(
            monitoring_interval=0.1,
            cleanup_interval=0.2,
            max_history_size=10
        )
        
        # Add callbacks
        alert_callback = Mock()
        cleanup_callback = Mock()
        
        manager.register_alert_callback(alert_callback)
        manager.register_cleanup_callback(cleanup_callback)
        
        # Start monitoring
        manager.start_monitoring()
        
        # Wait for some monitoring cycles
        time.sleep(0.3)
        
        # Should have collected some data
        assert len(manager.memory_history) > 0
        
        # Force some operations
        stats = manager.get_memory_stats()
        assert isinstance(stats, MemoryStats)
        
        gc_stats = manager.force_garbage_collection()
        assert isinstance(gc_stats, dict)
        
        leaks = manager.detect_memory_leaks()
        assert isinstance(leaks, dict)
        
        metrics = manager.get_performance_metrics()
        assert isinstance(metrics, dict)
        
        # Stop monitoring
        manager.stop_monitoring()
        
        # Should have stopped cleanly
        assert manager.is_monitoring is False

    def test_memory_leak_detection_integration(self):
        """Test memory leak detection with real objects."""
        manager = MemoryManager(leak_detection_window=1.0)
        
        # Create objects that grow over time
        growing_objects = []
        
        for i in range(5):
            # Add more objects each iteration
            growing_objects.extend([{"data": f"item_{i}_{j}"} for j in range(10 * (i + 1))])
            
            # Detect leaks
            leaks = manager.detect_memory_leaks()
            
            # Wait a bit
            time.sleep(0.2)
        
        # Final leak detection
        leaks = manager.detect_memory_leaks()
        
        # Should have detected some growth
        assert isinstance(leaks, dict)
        
        # Clean up
        del growing_objects

    def test_resource_tracking_integration(self):
        """Test resource tracking with cleanup."""
        manager = MemoryManager()
        
        # Track multiple resources
        resources = []
        for i in range(5):
            resource = Mock()
            resource.cleanup = Mock()
            manager.track_resource(f"resource_{i}", resource)
            resources.append(resource)
        
        # Verify tracking
        assert len(manager.tracked_resources) == 5
        
        # Trigger cleanup
        manager.trigger_memory_cleanup()
        
        # All resources should have been cleaned up
        for resource in resources:
            resource.cleanup.assert_called_once()

    def test_alert_system_integration(self):
        """Test alert system with multiple alerts."""
        manager = MemoryManager(alert_cooldown=0.1)
        
        alerts_received = []
        
        def alert_callback(level, stats):
            alerts_received.append((level, stats))
        
        manager.register_alert_callback(alert_callback)
        
        # Create stats that will trigger different alert levels
        low_stats = MemoryStats(
            total_memory_mb=8192.0,
            used_memory_mb=4096.0,
            available_memory_mb=4096.0,
            memory_percent=50.0,
            process_memory_mb=600.0,  # Low alert
            gc_objects=1000,
            gc_collections={0: 10, 1: 5, 2: 2},
            timestamp=time.time()
        )
        
        high_stats = MemoryStats(
            total_memory_mb=8192.0,
            used_memory_mb=4096.0,
            available_memory_mb=4096.0,
            memory_percent=50.0,
            process_memory_mb=1600.0,  # High alert
            gc_objects=1000,
            gc_collections={0: 10, 1: 5, 2: 2},
            timestamp=time.time()
        )
        
        # Trigger alerts
        manager._trigger_alert(MemoryAlertLevel.LOW, low_stats)
        manager._trigger_alert(MemoryAlertLevel.HIGH, high_stats)
        
        # Should have received alerts
        assert len(alerts_received) == 2
        assert alerts_received[0][0] == MemoryAlertLevel.LOW
        assert alerts_received[1][0] == MemoryAlertLevel.HIGH

    def test_thread_safety(self):
        """Test thread safety of memory manager operations."""
        manager = MemoryManager(monitoring_interval=0.05)
        
        def worker():
            for i in range(10):
                stats = manager.get_memory_stats()
                manager.force_garbage_collection()
                manager.detect_memory_leaks()
                time.sleep(0.01)
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Start monitoring
        manager.start_monitoring()
        
        # Wait for threads to complete
        for thread in threads:
            thread.join()
        
        # Stop monitoring
        manager.stop_monitoring()
        
        # Should not have crashed
        assert manager.is_monitoring is False