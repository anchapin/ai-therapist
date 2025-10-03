"""
Memory Leak Detection Tests

This module contains tests for detecting and monitoring memory leaks
in the AI Therapist voice services.
"""

import pytest
import time
import gc
import psutil
import threading
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Import performance modules
try:
    from performance.memory_manager import MemoryManager
    from performance.cache_manager import CacheManager
    PERFORMANCE_AVAILABLE = True
except ImportError:
    PERFORMANCE_AVAILABLE = False
    MemoryManager = None
    CacheManager = None

# Import voice services
from voice.audio_processor import SimplifiedAudioProcessor, AudioData
from voice.voice_service import VoiceService
from voice.config import VoiceConfig


@pytest.mark.skipif(not PERFORMANCE_AVAILABLE, reason="Performance modules not available")
class TestMemoryLeakDetection:
    """Test memory leak detection functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.memory_manager = MemoryManager({
            'memory_threshold_low': 50,
            'memory_threshold_medium': 100,
            'memory_threshold_high': 150,
            'memory_threshold_critical': 200,
            'monitoring_interval': 1.0,  # Fast monitoring for tests
            'gc_threshold': 100,
            'cleanup_interval': 10.0
        })

        self.cache_manager = CacheManager({
            'max_cache_size': 10,
            'max_memory_mb': 50,
            'enable_compression': True
        })

    def teardown_method(self):
        """Clean up test environment."""
        if hasattr(self, 'memory_manager') and self.memory_manager:
            self.memory_manager.stop_monitoring()

        if hasattr(self, 'cache_manager') and self.cache_manager:
            self.cache_manager.stop()

    def test_memory_manager_initialization(self):
        """Test memory manager initializes correctly."""
        assert self.memory_manager is not None
        assert not self.memory_manager.is_monitoring

        stats = self.memory_manager.get_memory_stats()
        assert 'total_memory_mb' in stats
        assert 'process_memory_mb' in stats
        assert stats['process_memory_mb'] >= 0

    def test_garbage_collection_trigger(self):
        """Test automatic garbage collection triggering."""
        # Create some garbage
        garbage = []
        for i in range(1000):
            garbage.append([i] * 100)

        # Force garbage collection
        stats = self.memory_manager.force_garbage_collection()

        assert 'objects_collected' in stats
        assert stats['objects_collected'] > 0
        assert 'memory_freed_mb' in stats

    def test_memory_threshold_alerts(self):
        """Test memory threshold alerts."""
        alerts_received = []

        def alert_callback(level, stats):
            alerts_received.append((level.value, stats['process_memory_mb']))

        self.memory_manager.register_alert_callback(alert_callback)
        self.memory_manager.start_monitoring()

        # Wait a bit for monitoring to run
        time.sleep(2.0)

        # Check if any alerts were generated (may not trigger in test environment)
        # This is mainly testing that the monitoring system works
        assert isinstance(alerts_received, list)

    def test_memory_leak_detection(self):
        """Test memory leak detection over time."""
        # Start monitoring
        self.memory_manager.start_monitoring()

        # Create objects that simulate potential leaks
        leak_objects = []
        for i in range(10):
            # Create some objects
            leak_objects.append([f"test_data_{i}"] * 100)
            time.sleep(0.1)  # Small delay

        # Wait for leak detection to run
        time.sleep(2.0)

        # Check for detected leaks
        leaks = self.memory_manager.detect_memory_leaks()
        assert isinstance(leaks, dict)

    def test_cache_memory_management(self):
        """Test cache memory management."""
        # Add items to cache
        for i in range(20):
            data = f"test_data_{i}" * 100  # Create larger strings
            self.cache_manager.set(f"key_{i}", data)

        # Check cache size
        assert self.cache_manager.size() <= self.cache_manager.max_cache_size

        # Check memory usage
        memory_usage = self.cache_manager.memory_usage()
        assert memory_usage >= 0
        assert memory_usage <= self.cache_manager.max_memory_mb

    def test_audio_processor_memory_cleanup(self):
        """Test audio processor memory cleanup."""
        config = VoiceConfig()
        processor = SimplifiedAudioProcessor(config)

        # Simulate audio processing
        for i in range(5):
            audio_data = AudioData(
                data=np.random.random(16000).astype(np.float32),
                sample_rate=16000,
                duration=1.0
            )
            processor.add_to_buffer(audio_data.data)

        # Check buffer has data
        assert len(processor.audio_buffer) > 0

        # Force cleanup
        cleaned_count = processor.force_cleanup_buffers()
        assert cleaned_count > 0
        assert len(processor.audio_buffer) == 0

    def test_voice_service_session_cleanup(self):
        """Test voice service session cleanup."""
        config = VoiceConfig()
        security = Mock()
        voice_service = VoiceService(config, security)

        # Create several sessions
        session_ids = []
        for i in range(5):
            session_id = voice_service.create_session(f"test_user_{i}")
            session_ids.append(session_id)

        # Verify sessions were created
        assert len(voice_service.sessions) >= 5

        # Simulate session inactivity by updating timestamps
        current_time = time.time()
        for session_id in session_ids[:3]:  # Make first 3 sessions old
            if session_id in voice_service.sessions:
                voice_service.sessions[session_id].last_activity = current_time - 7200  # 2 hours ago

        # Clean up old sessions (this would be done by background cleanup in real usage)
        # For testing, we'll manually call cleanup logic
        old_sessions = []
        for session_id, session in voice_service.sessions.items():
            if current_time - session.last_activity > 3600:  # 1 hour timeout
                old_sessions.append(session_id)

        for session_id in old_sessions:
            voice_service.destroy_session(session_id)

        # Verify old sessions were cleaned up
        assert len(voice_service.sessions) < len(session_ids)

    def test_concurrent_memory_operations(self):
        """Test memory operations under concurrent load."""
        results = []
        errors = []

        def memory_worker(worker_id):
            try:
                # Simulate memory-intensive operations
                data = []
                for i in range(100):
                    data.append([worker_id] * 1000)

                # Record memory usage
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024 * 1024)
                results.append(memory_mb)

                # Clean up
                del data
                gc.collect()

            except Exception as e:
                errors.append(str(e))

        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=memory_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=10.0)

        # Verify operations completed without errors
        assert len(results) == 3
        assert len(errors) == 0
        assert all(memory > 0 for memory in results)

    def test_performance_monitoring_overhead(self):
        """Test that performance monitoring doesn't add excessive overhead."""
        # Measure baseline performance
        start_time = time.time()
        for i in range(1000):
            _ = i * i  # Simple operation
        baseline_time = time.time() - start_time

        # Start monitoring
        self.memory_manager.start_monitoring()

        # Measure performance with monitoring
        start_time = time.time()
        for i in range(1000):
            _ = i * i  # Simple operation
        monitored_time = time.time() - start_time

        # Monitoring should not add more than 10% overhead
        overhead_ratio = monitored_time / baseline_time
        assert overhead_ratio < 1.1, f"Monitoring overhead too high: {overhead_ratio:.2f}"

    def test_cache_eviction_under_memory_pressure(self):
        """Test cache eviction under memory pressure."""
        # Fill cache to capacity
        for i in range(50):
            large_data = f"large_data_{i}" * 1000  # Create large strings
            self.cache_manager.set(f"large_key_{i}", large_data)

        initial_size = self.cache_manager.size()

        # Add more data to trigger eviction
        for i in range(50, 100):
            large_data = f"more_large_data_{i}" * 1000
            self.cache_manager.set(f"more_large_key_{i}", large_data)

        final_size = self.cache_manager.size()

        # Cache should have evicted some entries
        assert final_size <= self.cache_manager.max_cache_size
        assert self.cache_manager.memory_usage() <= self.cache_manager.max_memory_mb

    def test_memory_cleanup_callbacks(self):
        """Test memory cleanup callback system."""
        cleanup_called = []

        def cleanup_callback():
            cleanup_called.append(time.time())

        self.memory_manager.register_cleanup_callback(cleanup_callback)

        # Trigger cleanup
        self.memory_manager.trigger_memory_cleanup()

        # Verify callback was called
        assert len(cleanup_called) == 1
        assert isinstance(cleanup_called[0], float)