"""
Mock Memory Manager for testing.

This module provides a mock memory manager that handles MemoryStats dataclass
properly, fixing dictionary access issues in tests.
"""

import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from unittest.mock import MagicMock

# Import the real MemoryStats class
try:
    from performance.memory_manager import MemoryStats, MemoryAlertLevel
except ImportError:
    # Fallback for testing
    @dataclass
    class MemoryStats:
        total_memory_mb: float
        used_memory_mb: float
        available_memory_mb: float
        memory_percent: float
        process_memory_mb: float
        gc_objects: int
        gc_collections: Dict[int, int]
        timestamp: float
    
    class MemoryAlertLevel:
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"


class MockMemoryManager:
    """Mock memory manager that handles MemoryStats properly."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize mock memory manager."""
        self.config = config or {}
        self.is_monitoring = False
        self.monitoring_thread = None
        self.cleanup_thread = None
        
        # Memory thresholds
        self.memory_threshold_low = self.config.get('memory_threshold_low', 512)
        self.memory_threshold_medium = self.config.get('memory_threshold_medium', 1024)
        self.memory_threshold_high = self.config.get('memory_threshold_high', 1536)
        self.memory_threshold_critical = self.config.get('memory_threshold_critical', 2048)
        
        # Configuration
        self.monitoring_interval = self.config.get('monitoring_interval', 30.0)
        self.gc_threshold = self.config.get('gc_threshold', 1000)
        self.leak_detection_window = self.config.get('leak_detection_window', 300.0)
        self.alert_cooldown = self.config.get('alert_cooldown', 60.0)
        self.cleanup_interval = self.config.get('cleanup_interval', 600.0)
        
        # State tracking
        self.last_gc_time = 0.0
        self.last_cleanup_time = 0.0
        self.last_alert_time = 0.0
        
        # Memory tracking
        self.baseline_memory = 0.0
        self.memory_history: List[MemoryStats] = []
        self.max_history_size = 1000
        
        # Callbacks
        self.alert_callbacks: List[Callable[[MemoryAlertLevel, MemoryStats], None]] = []
        self.cleanup_callbacks: List[Callable[[], None]] = []
        
        # Performance metrics
        self.metrics = {
            'gc_collections': 0,
            'memory_alerts': 0,
            'leaks_detected': 0,
            'resources_cleaned': 0,
            'uptime_seconds': 0.0
        }
        
        # Mock methods
        self.start_monitoring = MagicMock(side_effect=self._mock_start_monitoring)
        self.stop_monitoring = MagicMock(side_effect=self._mock_stop_monitoring)
        self.force_garbage_collection = MagicMock(side_effect=self._mock_force_garbage_collection)
        self.detect_memory_leaks = MagicMock(return_value={})
        self.trigger_memory_cleanup = MagicMock(side_effect=self._mock_trigger_memory_cleanup)
        self.register_alert_callback = MagicMock(side_effect=self._mock_register_alert_callback)
        self.register_cleanup_callback = MagicMock(side_effect=self._mock_register_cleanup_callback)
        self.track_resource = MagicMock()
        self.untrack_resource = MagicMock()
        self.get_performance_metrics = MagicMock(return_value=self.metrics)
    
    def _mock_start_monitoring(self):
        """Mock start monitoring."""
        self.is_monitoring = True
        self.baseline_memory = self._get_current_memory_mb()
    
    def _mock_stop_monitoring(self):
        """Mock stop monitoring."""
        self.is_monitoring = False
    
    def _mock_force_garbage_collection(self) -> Dict[str, Any]:
        """Mock force garbage collection."""
        self.metrics['gc_collections'] += 1
        self.last_gc_time = time.time()
        
        return {
            'objects_before': 1000,
            'objects_after': 800,
            'objects_collected': 200,
            'memory_before_mb': 50.0,
            'memory_after_mb': 45.0,
            'memory_freed_mb': 5.0,
            'collections_by_generation': [100, 70, 30],
            'timestamp': time.time()
        }
    
    def _mock_trigger_memory_cleanup(self) -> Dict[str, Any]:
        """Mock trigger memory cleanup."""
        gc_stats = self._mock_force_garbage_collection()
        
        # Call cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception:
                pass  # Ignore errors in callbacks
        
        self.metrics['resources_cleaned'] += 1
        self.last_cleanup_time = time.time()
        
        return gc_stats
    
    def _mock_register_alert_callback(self, callback: Callable[[MemoryAlertLevel, MemoryStats], None]):
        """Mock register alert callback."""
        self.alert_callbacks.append(callback)
    
    def _mock_register_cleanup_callback(self, callback: Callable[[], None]):
        """Mock register cleanup callback."""
        self.cleanup_callbacks.append(callback)
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        # Create mock memory stats
        stats = MemoryStats(
            total_memory_mb=8000.0,
            used_memory_mb=4000.0,
            available_memory_mb=4000.0,
            memory_percent=50.0,
            process_memory_mb=100.0,
            gc_objects=10000,
            gc_collections={0: 10, 1: 5, 2: 2},
            timestamp=time.time()
        )
        
        # Add to history
        self.memory_history.append(stats)
        
        # Maintain history size
        if len(self.memory_history) > self.max_history_size:
            self.memory_history = self.memory_history[-self.max_history_size:]
        
        return stats
    
    def _get_current_memory_mb(self) -> float:
        """Get current process memory usage in MB."""
        return 100.0  # Mock implementation


class MockMemoryStatsWithDictAccess:
    """Wrapper for MemoryStats that provides dictionary access for backward compatibility."""
    
    def __init__(self, memory_stats: MemoryStats):
        """Initialize with a MemoryStats object."""
        self._memory_stats = memory_stats
    
    def __getattr__(self, name):
        """Get attribute from underlying MemoryStats."""
        return getattr(self._memory_stats, name)
    
    def __getitem__(self, key):
        """Provide dictionary access to MemoryStats fields."""
        return asdict(self._memory_stats)[key]
    
    def get(self, key, default=None):
        """Get value with default."""
        return asdict(self._memory_stats).get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self._memory_stats)


def create_mock_memory_manager(config: Optional[Dict[str, Any]] = None) -> MockMemoryManager:
    """Create a mock memory manager for testing."""
    return MockMemoryManager(config)


def wrap_memory_stats_for_dict_access(memory_stats: MemoryStats) -> MockMemoryStatsWithDictAccess:
    """Wrap MemoryStats to provide dictionary access for tests."""
    return MockMemoryStatsWithDictAccess(memory_stats)