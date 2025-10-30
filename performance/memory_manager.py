"""
Performance Memory Management Module

This module provides comprehensive memory management and monitoring capabilities
for the AI Therapist voice services. It handles memory usage tracking, leak detection,
automatic garbage collection, and resource cleanup.

Features:
- Real-time memory usage monitoring and tracking
- Automatic garbage collection triggers
- Memory leak detection and reporting
- Resource cleanup utilities
- Performance metrics collection
- Memory threshold alerts and management
"""

import gc
import time
import psutil
import threading
import logging
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass
from collections import defaultdict
import weakref
import os
from enum import Enum

try:
    import resource
    RESOURCE_AVAILABLE = True
except ImportError:
    RESOURCE_AVAILABLE = False

class MemoryAlertLevel(Enum):
    """Memory alert levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_memory_mb: float
    used_memory_mb: float
    available_memory_mb: float
    memory_percent: float
    process_memory_mb: float
    gc_objects: int
    gc_collections: Dict[int, int]
    timestamp: float

@dataclass
class MemoryLeak:
    """Memory leak detection information."""
    object_type: str
    count: int
    size_bytes: int
    growth_rate: float
    first_detected: float
    last_detected: float

class MemoryManager:
    """
    Comprehensive memory management system for voice services.

    Features:
    - Real-time memory monitoring
    - Automatic garbage collection
    - Memory leak detection
    - Resource cleanup coordination
    - Performance metrics collection
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize memory manager."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Memory thresholds (MB)
        self.memory_threshold_low = self.config.get('memory_threshold_low', 512)
        self.memory_threshold_medium = self.config.get('memory_threshold_medium', 1024)
        self.memory_threshold_high = self.config.get('memory_threshold_high', 1536)
        self.memory_threshold_critical = self.config.get('memory_threshold_critical', 2048)

        # Configuration
        self.monitoring_interval = self.config.get('monitoring_interval', 30.0)  # seconds
        self.gc_threshold = self.config.get('gc_threshold', 1000)  # GC trigger threshold
        self.leak_detection_window = self.config.get('leak_detection_window', 300.0)  # 5 minutes
        self.alert_cooldown = self.config.get('alert_cooldown', 60.0)  # seconds
        self.cleanup_interval = self.config.get('cleanup_interval', 600.0)  # 10 minutes

        # State tracking
        self.is_monitoring = False
        self.monitoring_thread = None
        self.cleanup_thread = None
        self.last_gc_time = 0.0
        self.last_cleanup_time = 0.0
        self.last_alert_time = 0.0

        # Memory tracking
        self.baseline_memory = 0.0
        self.memory_history: List[MemoryStats] = []
        self.max_history_size = 1000

        # Leak detection
        self.object_counts: Dict[str, List[tuple]] = defaultdict(list)  # (timestamp, count, size)
        self.detected_leaks: Dict[str, MemoryLeak] = {}
        self.known_objects: Set[int] = set()

        # Callbacks
        self.alert_callbacks: List[Callable[[MemoryAlertLevel, MemoryStats], None]] = []
        self.cleanup_callbacks: List[Callable[[], None]] = []

        # Resource tracking
        self.tracked_resources: Dict[str, weakref.ReferenceType] = {}
        self.resource_lock = threading.RLock()

        # Performance metrics
        self.metrics = {
            'gc_collections': 0,
            'memory_alerts': 0,
            'leaks_detected': 0,
            'resources_cleaned': 0,
            'uptime_seconds': 0.0
        }

        # Set GC thresholds
        gc.set_threshold(self.gc_threshold, 10, 10)

        # Set resource limits if available
        if RESOURCE_AVAILABLE:
            try:
                # Set memory limit (256MB soft, 512MB hard)
                resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 512 * 1024 * 1024))
                self.logger.info("Resource limits set successfully")
            except Exception as e:
                self.logger.warning(f"Failed to set resource limits: {e}")

        self.logger.info("Memory manager initialized")

    def start_monitoring(self):
        """Start memory monitoring."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.baseline_memory = self._get_current_memory_mb()

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_worker,
            daemon=True,
            name="memory-monitor"
        )
        self.monitoring_thread.start()

        # Start cleanup thread
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_worker,
            daemon=True,
            name="memory-cleanup"
        )
        self.cleanup_thread.start()

        self.logger.info("Memory monitoring started")

    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.is_monitoring = False

        # Wait for threads to stop with longer timeout for tests
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=10.0)
            if self.monitoring_thread.is_alive():
                self.logger.warning("Monitoring thread did not stop gracefully")

        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=10.0)
            if self.cleanup_thread.is_alive():
                self.logger.warning("Cleanup thread did not stop gracefully")

        self.logger.info("Memory monitoring stopped")

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        try:
            # System memory
            memory = psutil.virtual_memory()
            process = psutil.Process()

            # GC information
            gc_objects = len(gc.get_objects())
            gc_collections = {}
            for gen in range(3):
                gc_collections[gen] = gc.get_count()[gen]

            return MemoryStats(
                total_memory_mb=memory.total / (1024 * 1024),
                used_memory_mb=memory.used / (1024 * 1024),
                available_memory_mb=memory.available / (1024 * 1024),
                memory_percent=memory.percent,
                process_memory_mb=process.memory_info().rss / (1024 * 1024),
                gc_objects=gc_objects,
                gc_collections=gc_collections,
                timestamp=time.time()
            )
        except Exception as e:
            self.logger.error(f"Error getting memory stats: {e}")
            # Return minimal stats on error
            return MemoryStats(
                total_memory_mb=0.0,
                used_memory_mb=0.0,
                available_memory_mb=0.0,
                memory_percent=0.0,
                process_memory_mb=0.0,
                gc_objects=0,
                gc_collections={0: 0, 1: 0, 2: 0},
                timestamp=time.time()
            )

    def force_garbage_collection(self) -> Dict[str, Any]:
        """Force garbage collection and return statistics."""
        try:
            start_objects = len(gc.get_objects())
            start_memory = self._get_current_memory_mb()

            # Force collection of all generations
            collected_counts = []
            for generation in range(3):
                collected = gc.collect(generation)
                collected_counts.append(collected)

            end_objects = len(gc.get_objects())
            end_memory = self._get_current_memory_mb()

            stats = {
                'objects_before': start_objects,
                'objects_after': end_objects,
                'objects_collected': start_objects - end_objects,
                'memory_before_mb': start_memory,
                'memory_after_mb': end_memory,
                'memory_freed_mb': start_memory - end_memory,
                'collections_by_generation': collected_counts,
                'timestamp': time.time()
            }

            self.metrics['gc_collections'] += 1
            self.last_gc_time = time.time()

            self.logger.info(f"Garbage collection completed: {stats['objects_collected']} objects, {stats['memory_freed_mb']:.2f} MB freed")
            return stats

        except Exception as e:
            self.logger.error(f"Error during garbage collection: {e}")
            return {}

    def detect_memory_leaks(self) -> Dict[str, MemoryLeak]:
        """Detect potential memory leaks by analyzing object growth."""
        try:
            # Get current object counts by type
            current_objects = defaultdict(lambda: {'count': 0, 'size': 0})

            # Sample objects (limit to avoid performance impact)
            objects = gc.get_objects()
            sample_size = min(len(objects), 10000)

            for obj in objects[:sample_size]:
                obj_type = type(obj).__name__
                size = self._estimate_object_size(obj)
                current_objects[obj_type]['count'] += 1
                current_objects[obj_type]['size'] += size

            # Record current state
            current_time = time.time()
            for obj_type, info in current_objects.items():
                self.object_counts[obj_type].append((current_time, info['count'], info['size']))

                # Keep only recent data
                cutoff_time = current_time - self.leak_detection_window
                self.object_counts[obj_type] = [
                    entry for entry in self.object_counts[obj_type]
                    if entry[0] >= cutoff_time
                ]

                # Check for leaks if we have enough data points
                if len(self.object_counts[obj_type]) >= 5:
                    self._analyze_object_growth(obj_type)

            return self.detected_leaks.copy()

        except Exception as e:
            self.logger.error(f"Error detecting memory leaks: {e}")
            return {}

    def trigger_memory_cleanup(self):
        """Trigger comprehensive memory cleanup."""
        try:
            self.logger.info("Starting comprehensive memory cleanup")

            # Force garbage collection
            gc_stats = self.force_garbage_collection()

            # Call cleanup callbacks
            for callback in self.cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    self.logger.error(f"Error in cleanup callback: {e}")

            # Clean up tracked resources
            self._cleanup_tracked_resources()

            # Clear caches if available
            self._clear_caches()

            self.metrics['resources_cleaned'] += 1
            self.last_cleanup_time = time.time()

            self.logger.info("Memory cleanup completed")
            return gc_stats

        except Exception as e:
            self.logger.error(f"Error during memory cleanup: {e}")
            return {}

    def register_alert_callback(self, callback: Callable[[MemoryAlertLevel, MemoryStats], None]):
        """Register callback for memory alerts."""
        self.alert_callbacks.append(callback)

    def register_cleanup_callback(self, callback: Callable[[], None]):
        """Register callback for cleanup operations."""
        self.cleanup_callbacks.append(callback)

    def track_resource(self, name: str, resource):
        """Track a resource for cleanup."""
        with self.resource_lock:
            self.tracked_resources[name] = weakref.ref(resource, lambda ref: self._resource_destroyed(name))

    def untrack_resource(self, name: str):
        """Stop tracking a resource."""
        with self.resource_lock:
            self.tracked_resources.pop(name, None)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            'gc_collections': self.metrics['gc_collections'],
            'memory_alerts': self.metrics['memory_alerts'],
            'leaks_detected': self.metrics['leaks_detected'],
            'resources_cleaned': self.metrics['resources_cleaned'],
            'uptime_seconds': time.time() - (time.time() - self.metrics['uptime_seconds']),
            'current_memory_mb': self._get_current_memory_mb(),
            'memory_history_size': len(self.memory_history),
            'tracked_resources': len(self.tracked_resources),
            'detected_leaks': len(self.detected_leaks)
        }

    def _monitoring_worker(self):
        """Background monitoring worker."""
        while self.is_monitoring:
            try:
                # Get current stats
                stats = self.get_memory_stats()
                self.memory_history.append(stats)

                # Maintain history size
                if len(self.memory_history) > self.max_history_size:
                    self.memory_history = self.memory_history[-self.max_history_size:]

                # Check memory thresholds
                self._check_memory_thresholds(stats)

                # Update metrics
                self.metrics['uptime_seconds'] = time.time() - (time.time() - self.metrics['uptime_seconds'])

                # Periodic leak detection
                if len(self.memory_history) % 10 == 0:  # Every 10 monitoring cycles
                    self.detect_memory_leaks()

                # Use a shorter sleep with interruptible wait for tests
                for _ in range(int(self.monitoring_interval * 10)):  # Break into 0.1s chunks
                    if not self.is_monitoring:
                        break
                    time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Error in monitoring worker: {e}")
                # Use interruptible sleep for error recovery
                for _ in range(50):  # 5 seconds in 0.1s chunks
                    if not self.is_monitoring:
                        break
                    time.sleep(0.1)

    def _cleanup_worker(self):
        """Background cleanup worker."""
        while self.is_monitoring:
            try:
                current_time = time.time()
                if current_time - self.last_cleanup_time >= self.cleanup_interval:
                    self.trigger_memory_cleanup()

                # Use interruptible sleep for tests
                for _ in range(600):  # 60 seconds in 0.1s chunks
                    if not self.is_monitoring:
                        break
                    time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Error in cleanup worker: {e}")
                # Use interruptible sleep for error recovery
                for _ in range(300):  # 30 seconds in 0.1s chunks
                    if not self.is_monitoring:
                        break
                    time.sleep(0.1)

    def _check_memory_thresholds(self, stats: MemoryStats):
        """Check memory usage against thresholds and trigger alerts."""
        current_memory = stats.process_memory_mb
        alert_level = None

        if current_memory >= self.memory_threshold_critical:
            alert_level = MemoryAlertLevel.CRITICAL
        elif current_memory >= self.memory_threshold_high:
            alert_level = MemoryAlertLevel.HIGH
        elif current_memory >= self.memory_threshold_medium:
            alert_level = MemoryAlertLevel.MEDIUM
        elif current_memory >= self.memory_threshold_low:
            alert_level = MemoryAlertLevel.LOW

        if alert_level and (time.time() - self.last_alert_time) >= self.alert_cooldown:
            self._trigger_alert(alert_level, stats)

    def _trigger_alert(self, level: MemoryAlertLevel, stats: MemoryStats):
        """Trigger memory alert."""
        self.metrics['memory_alerts'] += 1
        self.last_alert_time = time.time()

        self.logger.warning(f"Memory alert triggered: {level.value.upper()} - {stats.process_memory_mb:.1f} MB used")

        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(level, stats)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")

        # Auto-cleanup for high alerts
        if level in [MemoryAlertLevel.HIGH, MemoryAlertLevel.CRITICAL]:
            self.logger.info("Triggering automatic cleanup due to high memory usage")
            threading.Thread(target=self.trigger_memory_cleanup, daemon=True).start()

    def _analyze_object_growth(self, obj_type: str):
        """Analyze object count growth for potential leaks."""
        entries = self.object_counts[obj_type]
        if len(entries) < 3:
            return

        # Calculate growth rate
        recent_entries = entries[-10:]  # Last 10 entries
        if len(recent_entries) < 2:
            return

        # Simple linear regression for growth rate
        times = [entry[0] for entry in recent_entries]
        counts = [entry[1] for entry in recent_entries]

        # Calculate slope (growth rate)
        n = len(times)
        sum_x = sum(times)
        sum_y = sum(counts)
        sum_xy = sum(x * y for x, y in zip(times, counts))
        sum_xx = sum(x * x for x in times)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x) if (n * sum_xx - sum_x * sum_x) != 0 else 0

        # Significant growth indicates potential leak
        if slope > 10:  # Growing by more than 10 objects per time unit
            current_count = recent_entries[-1][1]
            current_size = recent_entries[-1][2]

            if obj_type not in self.detected_leaks:
                self.detected_leaks[obj_type] = MemoryLeak(
                    object_type=obj_type,
                    count=current_count,
                    size_bytes=current_size,
                    growth_rate=slope,
                    first_detected=time.time(),
                    last_detected=time.time()
                )
                self.metrics['leaks_detected'] += 1
                self.logger.warning(f"Memory leak detected: {obj_type} growing at rate {slope:.2f} objects/time")
            else:
                leak_info = self.detected_leaks[obj_type]
                leak_info.count = current_count
                leak_info.size_bytes = current_size
                leak_info.growth_rate = slope
                leak_info.last_detected = time.time()

    def _cleanup_tracked_resources(self):
        """Clean up tracked resources."""
        with self.resource_lock:
            resources_to_remove = []
            for name, ref in self.tracked_resources.items():
                resource = ref()
                if resource is None:
                    resources_to_remove.append(name)
                elif hasattr(resource, 'cleanup'):
                    try:
                        resource.cleanup()
                    except Exception as e:
                        self.logger.error(f"Error cleaning up resource {name}: {e}")

            # Remove dead references
            for name in resources_to_remove:
                del self.tracked_resources[name]

    def _clear_caches(self):
        """Clear various caches to free memory."""
        try:
            # Clear Python module cache
            import sys
            modules_to_clear = []
            for module_name, module in sys.modules.items():
                if module_name.startswith(('cache', 'temp', 'tmp')):
                    modules_to_clear.append(module_name)

            # Actually clearing modules can be dangerous, so just log
            if modules_to_clear:
                self.logger.debug(f"Found {len(modules_to_clear)} cache-related modules")

        except Exception as e:
            self.logger.error(f"Error clearing caches: {e}")

    def _resource_destroyed(self, name: str):
        """Callback when a tracked resource is destroyed."""
        with self.resource_lock:
            self.tracked_resources.pop(name, None)
        self.logger.debug(f"Tracked resource destroyed: {name}")

    def _get_current_memory_mb(self) -> float:
        """Get current process memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0

    def _estimate_object_size(self, obj) -> int:
        """Estimate the size of a Python object in bytes."""
        try:
            # Use sys.getsizeof for basic size
            size = sys.getsizeof(obj)

            # Add size of contained objects for containers
            if hasattr(obj, '__len__') and hasattr(obj, '__iter__'):
                try:
                    # Sample a few items to estimate
                    items = list(obj)[:10] if len(obj) > 10 else list(obj)
                    if items:
                        avg_item_size = sum(sys.getsizeof(item) for item in items) / len(items)
                        size += int(avg_item_size * len(obj))
                except Exception as e:
                    # Log error during size estimation but continue
                    import logging
                    logging.getLogger(__name__).debug(f"Error estimating collection size: {e}")

            return size
        except Exception as e:
            # Log error during size calculation but return 0
            import logging
            logging.getLogger(__name__).debug(f"Error calculating object size: {e}")
            return 0

    def __del__(self):
        """Destructor - ensure cleanup."""
        try:
            self.stop_monitoring()
        except Exception as e:
            # Log error during cleanup but don't raise in destructor
            import logging
            logging.getLogger(__name__).debug(f"Error during memory manager cleanup: {e}")