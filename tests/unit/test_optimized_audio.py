"""
Comprehensive unit tests for voice/optimized_audio_processor.py

Covers critical gaps in coverage analysis for optimized audio processing:
- Memory-efficient audio data handling
- Real-time performance metrics
- High-performance audio processing
- Memory pooling and optimization
- Voice activity detection
- Performance optimization modes
- Thread-safe operations
- Error handling and edge cases
"""

import os
import sys
import tempfile
import shutil
import unittest
from unittest.mock import Mock, patch, MagicMock, call
import pytest
import time
import threading
import numpy as np
from pathlib import Path
from queue import Queue, Empty

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from voice.optimized_audio_processor import (
    OptimizedAudioData,
    OptimizedAudioProcessorState,
    AudioProcessingMetrics,
    OptimizedAudioProcessor,
    create_optimized_audio_processor
)


class TestOptimizedAudioData(unittest.TestCase):
    """Test OptimizedAudioData class."""

    def test_optimized_audio_data_creation(self):
        """Test basic audio data creation."""
        data = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        sample_rate = 16000
        duration = 0.25

        audio_data = OptimizedAudioData(data, sample_rate, duration)

        self.assertTrue(np.array_equal(audio_data.data, data))
        self.assertEqual(audio_data.sample_rate, sample_rate)
        self.assertEqual(audio_data.duration, duration)
        self.assertEqual(audio_data.channels, 1)
        self.assertEqual(audio_data.format, "wav")
        self.assertGreater(audio_data.timestamp, 0)

    def test_optimized_audio_data_dtype_conversion(self):
        """Test automatic dtype conversion."""
        data = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.int16)  # Wrong dtype
        audio_data = OptimizedAudioData(data, 16000, 0.25)

        # Should be converted to float32
        self.assertEqual(audio_data.data.dtype, np.float32)

    def test_optimized_audio_data_timestamp_default(self):
        """Test timestamp default value."""
        data = np.array([0.1, 0.2, 0.3])
        audio_data = OptimizedAudioData(data, 16000, 0.001)

        # Timestamp should be set to current time
        self.assertGreater(audio_data.timestamp, time.time() - 1)
        self.assertLess(audio_data.timestamp, time.time() + 1)

    def test_optimized_audio_data_to_bytes(self):
        """Test conversion to bytes."""
        data = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

        with patch('voice.optimized_audio_processor.SOUNDDEVICE_AVAILABLE', True), \
             patch('voice.optimized_audio_processor.sf') as mock_sf:

            # Mock soundfile operations
            mock_buffer = Mock()
            mock_buffer.getvalue.return_value = b'fake_wav_data'
            mock_sf.io.BytesIO.return_value = mock_buffer

            audio_data = OptimizedAudioData(data, 16000, 0.25)
            result = audio_data.to_bytes()

            # Verify soundfile was used for WAV encoding
            mock_sf.io.BytesIO.assert_called_once()
            mock_sf.write.assert_called_once()
            self.assertEqual(result, b'fake_wav_data')

    def test_optimized_audio_data_to_bytes_fallback(self):
        """Test fallback bytes conversion when soundfile unavailable."""
        data = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

        with patch('voice.optimized_audio_processor.SOUNDDEVICE_AVAILABLE', False):
            audio_data = OptimizedAudioData(data, 16000, 0.25)
            result = audio_data.to_bytes()

            # Should use numpy tobytes fallback
            self.assertEqual(result, data.tobytes())

    def test_optimized_audio_data_from_bytes(self):
        """Test creation from bytes."""
        original_data = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        sample_rate = 16000

        with patch('voice.optimized_audio_processor.SOUNDDEVICE_AVAILABLE', True), \
             patch('voice.optimized_audio_processor.sf') as mock_sf:

            # Mock soundfile read
            mock_sf.read.return_value = (original_data, sample_rate)

            fake_wav_data = b'fake_wav_data'
            audio_data = OptimizedAudioData.from_bytes(fake_wav_data, sample_rate)

            # Verify soundfile was used for decoding
            mock_sf.io.BytesIO.assert_called_once()
            mock_sf.read.assert_called_once()

            self.assertTrue(np.array_equal(audio_data.data, original_data))
            self.assertEqual(audio_data.sample_rate, sample_rate)

    def test_optimized_audio_data_from_bytes_fallback(self):
        """Test fallback creation when soundfile unavailable."""
        original_data = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        sample_rate = 16000

        with patch('voice.optimized_audio_processor.SOUNDDEVICE_AVAILABLE', False):
            fake_data = original_data.tobytes()
            audio_data = OptimizedAudioData.from_bytes(fake_data, sample_rate)

            # Should use numpy frombuffer fallback
            self.assertTrue(np.array_equal(audio_data.data, original_data))
            self.assertEqual(audio_data.sample_rate, sample_rate)


class TestAudioProcessingMetrics(unittest.TestCase):
    """Test AudioProcessingMetrics class."""

    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = AudioProcessingMetrics()

        self.assertEqual(metrics.capture_latency_ms, 0.0)
        self.assertEqual(metrics.processing_latency_ms, 0.0)
        self.assertEqual(metrics.buffer_utilization_percent, 0.0)
        self.assertEqual(metrics.memory_usage_mb, 0.0)
        self.assertEqual(metrics.drop_rate_percent, 0.0)
        self.assertEqual(metrics.quality_score, 0.0)

    def test_metrics_to_dict(self):
        """Test metrics conversion to dictionary."""
        metrics = AudioProcessingMetrics()
        metrics.capture_latency_ms = 15.5
        metrics.processing_latency_ms = 8.2
        metrics.buffer_utilization_percent = 75.0
        metrics.memory_usage_mb = 25.5
        metrics.drop_rate_percent = 2.1
        metrics.quality_score = 0.85

        result = metrics.to_dict()

        expected = {
            'capture_latency_ms': 15.5,
            'processing_latency_ms': 8.2,
            'buffer_utilization_percent': 75.0,
            'memory_usage_mb': 25.5,
            'drop_rate_percent': 2.1,
            'quality_score': 0.85
        }

        self.assertEqual(result, expected)


class TestOptimizedAudioProcessor(unittest.TestCase):
    """Test OptimizedAudioProcessor class."""

    def setUp(self):
        """Set up audio processor tests."""
        # Mock config to avoid dependencies
        self.mock_config = Mock()
        self.mock_config.audio = Mock()
        self.mock_config.audio.max_buffer_size = 50
        self.mock_config.audio.sample_rate = 16000
        self.mock_config.audio.channels = 1

        self.processor = OptimizedAudioProcessor(self.mock_config)

    def test_processor_initialization(self):
        """Test processor initialization."""
        self.assertEqual(self.processor.state, OptimizedAudioProcessorState.READY)
        self.assertFalse(self.processor.is_recording)
        self.assertFalse(self.processor.is_streaming)
        self.assertEqual(self.processor.sample_rate, 16000)
        self.assertEqual(self.processor.channels, 1)
        self.assertEqual(self.processor.chunk_size, 512)

    def test_processor_initialization_defaults(self):
        """Test processor initialization with default config."""
        processor = OptimizedAudioProcessor()

        self.assertEqual(processor.sample_rate, 16000)
        self.assertEqual(processor.channels, 1)
        self.assertEqual(processor.chunk_size, 512)
        self.assertEqual(processor._buffer_size, 50)

    def test_processor_features_detection(self):
        """Test feature availability detection."""
        # Test with mocked availability
        with patch('voice.optimized_audio_processor.SOUNDDEVICE_AVAILABLE', True), \
             patch('voice.optimized_audio_processor.NOISEREDUCE_AVAILABLE', True), \
             patch('voice.optimized_audio_processor.VAD_AVAILABLE', True), \
             patch('voice.optimized_audio_processor.LIBROSA_AVAILABLE', True):

            processor = OptimizedAudioProcessor()
            features = processor.features

            self.assertTrue(features['audio_capture'])
            self.assertTrue(features['noise_reduction'])
            self.assertTrue(features['vad'])
            self.assertTrue(features['quality_analysis'])

    def test_start_recording_success(self):
        """Test successful recording start."""
        with patch('voice.optimized_audio_processor.SOUNDDEVICE_AVAILABLE', True), \
             patch('voice.optimized_audio_processor.threading.Thread') as mock_thread:

            mock_thread_instance = Mock()
            mock_thread.return_value = mock_thread_instance

            result = self.processor.start_recording()

            # Should return True on success
            self.assertTrue(result)
            self.assertEqual(self.processor.state, OptimizedAudioProcessorState.RECORDING)
            self.assertTrue(self.processor.is_recording)

    def test_start_recording_unavailable(self):
        """Test recording start when audio capture unavailable."""
        with patch('voice.optimized_audio_processor.SOUNDDEVICE_AVAILABLE', False):
            result = self.processor.start_recording()

            # Should return False when unavailable
            self.assertFalse(result)

    def test_start_recording_already_recording(self):
        """Test recording start when already recording."""
        self.processor.state = OptimizedAudioProcessorState.RECORDING
        self.processor.is_recording = True

        result = self.processor.start_recording()

        # Should return False when already recording
        self.assertFalse(result)

    def test_stop_recording_success(self):
        """Test successful recording stop."""
        # Setup recording state
        self.processor.is_recording = True
        self.processor.state = OptimizedAudioProcessorState.RECORDING

        # Mock audio buffer with data
        test_audio_data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        self.processor._audio_buffer.append(test_audio_data)

        with patch('voice.optimized_audio_processor.threading.Thread'), \
             patch('voice.optimized_audio_processor.SOUNDDEVICE_AVAILABLE', True):

            # Start recording first
            self.processor.start_recording()

            result = self.processor.stop_recording()

            # Should return OptimizedAudioData
            self.assertIsNotNone(result)
            self.assertIsInstance(result, OptimizedAudioData)

    def test_stop_recording_not_recording(self):
        """Test recording stop when not recording."""
        self.processor.is_recording = False

        result = self.processor.stop_recording()

        # Should return None when not recording
        self.assertIsNone(result)

    def test_memory_pool_operations(self):
        """Test memory pool operations."""
        # Test getting chunk from empty pool
        test_data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        chunk = self.processor._get_pooled_chunk(test_data.copy())

        # Should return new chunk
        self.assertTrue(np.array_equal(chunk, test_data))

        # Test returning chunk to pool
        self.processor._return_chunk_to_pool(chunk)

        # Should be in pool now
        self.assertEqual(len(self.processor._memory_pool), 1)

        # Test getting chunk from pool
        pooled_chunk = self.processor._get_pooled_chunk(test_data.copy())

        # Should get pooled chunk
        self.assertEqual(len(self.processor._memory_pool), 0)
        self.assertTrue(np.array_equal(pooled_chunk, test_data))

    def test_memory_limit_enforcement(self):
        """Test memory limit enforcement."""
        # Set low memory limit for testing
        self.processor._memory_limit_bytes = 1000  # 1KB limit

        # Create large chunk that exceeds limit
        large_data = np.zeros(1000, dtype=np.float32)  # ~4KB
        large_chunk = self.processor._get_pooled_chunk(large_data)

        # Should exceed memory limit
        self.processor._current_memory_bytes = 2000  # Set current usage high

        # Try to add chunk (should be dropped due to memory limit)
        self.processor._audio_buffer.append(large_chunk)
        self.processor._current_memory_bytes += large_chunk.nbytes

        # Verify chunk was dropped
        self.assertEqual(self.processor._dropped_chunks, 1)

    def test_voice_activity_detection(self):
        """Test voice activity detection."""
        # Test with high energy (speech)
        speech_data = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)
        result = self.processor._fast_voice_activity_detection(speech_data)
        self.assertTrue(result)

        # Test with low energy (silence)
        silence_data = np.array([0.001, 0.002, 0.001], dtype=np.float32)
        result = self.processor._fast_voice_activity_detection(silence_data)
        self.assertFalse(result)

    def test_noise_reduction(self):
        """Test noise reduction processing."""
        test_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)

        # Enable noise reduction for testing
        self.processor._enable_noise_reduction = True

        with patch('voice.optimized_audio_processor.NOISEREDUCE_AVAILABLE', True):
            result = self.processor._fast_noise_reduction(test_data.copy())

            # Should process data (even if simplified)
            self.assertIsInstance(result, np.ndarray)

    def test_quality_analysis(self):
        """Test audio quality analysis."""
        test_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)

        # Enable quality analysis for testing
        self.processor._enable_quality_analysis = True

        self.processor._fast_quality_analysis(test_data)

        # Should update quality score
        self.assertGreaterEqual(self.processor._metrics.quality_score, 0.0)
        self.assertLessEqual(self.processor._metrics.quality_score, 1.0)

    def test_performance_optimization_modes(self):
        """Test performance optimization modes."""
        # Test latency optimization
        self.processor.optimize_for_latency()

        self.assertEqual(self.processor.chunk_size, 256)
        self.assertEqual(self.processor._buffer_size, 20)
        self.assertFalse(self.processor._enable_noise_reduction)
        self.assertFalse(self.processor._enable_quality_analysis)

        # Test quality optimization
        self.processor.optimize_for_quality()

        self.assertEqual(self.processor.chunk_size, 1024)
        self.assertEqual(self.processor._buffer_size, 100)
        self.assertTrue(self.processor._enable_noise_reduction)
        self.assertTrue(self.processor._enable_quality_analysis)

    def test_callback_systems(self):
        """Test callback systems."""
        # Test audio callback
        callback_results = []

        def test_audio_callback(audio_data):
            callback_results.append(audio_data)

        self.processor.add_audio_callback(test_audio_callback)

        # Trigger callback
        test_data = np.array([0.1, 0.2, 0.3])
        for callback in self.processor._audio_callbacks:
            callback(test_data)

        # Verify callback was called
        self.assertEqual(len(callback_results), 1)
        self.assertTrue(np.array_equal(callback_results[0], test_data))

    def test_metrics_callback(self):
        """Test metrics callback."""
        callback_results = []

        def test_metrics_callback(metrics):
            callback_results.append(metrics)

        self.processor.add_metrics_callback(test_metrics_callback)

        # Trigger metrics update
        test_metrics = AudioProcessingMetrics()
        test_metrics.capture_latency_ms = 15.0

        for callback in self.processor._metrics_callbacks:
            callback(test_metrics)

        # Verify callback was called
        self.assertEqual(len(callback_results), 1)
        self.assertEqual(callback_results[0].capture_latency_ms, 15.0)

    def test_get_status(self):
        """Test status reporting."""
        status = self.processor.get_status()

        # Verify status contains all expected fields
        expected_fields = [
            'state', 'is_recording', 'features', 'metrics', 'buffer_size',
            'max_buffer_size', 'memory_pool_size', 'processed_chunks',
            'dropped_chunks', 'sample_rate', 'chunk_size'
        ]

        for field in expected_fields:
            self.assertIn(field, status)

        # Verify specific values
        self.assertEqual(status['state'], 'ready')
        self.assertFalse(status['is_recording'])
        self.assertIsInstance(status['features'], dict)
        self.assertIsInstance(status['metrics'], dict)

    def test_metrics_update(self):
        """Test metrics updating."""
        # Set some initial values
        self.processor._processed_chunks = 10
        self.processor._dropped_chunks = 2
        self.processor._current_memory_bytes = 1024 * 1024  # 1MB

        # Update metrics
        self.processor._update_metrics()

        # Verify calculations
        self.assertEqual(self.processor._metrics.buffer_utilization_percent,
                        len(self.processor._audio_buffer) / self.processor._buffer_size * 100)
        self.assertEqual(self.processor._metrics.drop_rate_percent, 20.0)  # 2/10 * 100
        self.assertEqual(self.processor._metrics.memory_usage_mb, 1.0)  # 1024*1024 / 1024 / 1024

    def test_cleanup(self):
        """Test processor cleanup."""
        # Setup some state
        self.processor.is_recording = True
        self.processor._audio_buffer.append(np.array([0.1, 0.2]))
        self.processor._memory_pool.append(np.array([0.3, 0.4]))

        # Add callbacks
        self.processor._audio_callbacks.append(lambda x: None)
        self.processor._metrics_callbacks.append(lambda x: None)

        # Perform cleanup
        self.processor.cleanup()

        # Verify cleanup
        self.assertFalse(self.processor.is_recording)
        self.assertEqual(len(self.processor._audio_buffer), 0)
        self.assertEqual(len(self.processor._memory_pool), 0)
        self.assertEqual(len(self.processor._audio_callbacks), 0)
        self.assertEqual(len(self.processor._metrics_callbacks), 0)
        self.assertEqual(self.processor.state, OptimizedAudioProcessorState.IDLE)


class TestOptimizedAudioProcessorErrorHandling(unittest.TestCase):
    """Test error handling in OptimizedAudioProcessor."""

    def setUp(self):
        """Set up error handling tests."""
        self.mock_config = Mock()
        self.mock_config.audio = Mock()
        self.mock_config.audio.max_buffer_size = 50
        self.mock_config.audio.sample_rate = 16000
        self.mock_config.audio.channels = 1

        self.processor = OptimizedAudioProcessor(self.mock_config)

    def test_recording_exception_handling(self):
        """Test exception handling during recording."""
        with patch('voice.optimized_audio_processor.SOUNDDEVICE_AVAILABLE', False):
            # Should handle gracefully when sounddevice unavailable
            result = self.processor.start_recording()
            self.assertFalse(result)

    def test_audio_callback_exception_handling(self):
        """Test exception handling in audio callbacks."""
        def failing_callback(audio_data):
            raise Exception("Callback error")

        self.processor.add_audio_callback(failing_callback)

        # Should not raise exception when callback fails
        test_data = np.array([0.1, 0.2, 0.3])

        # This should not raise an exception
        try:
            for callback in self.processor._audio_callbacks:
                callback(test_data)
        except Exception:
            self.fail("Exception should be caught in callback system")

    def test_metrics_callback_exception_handling(self):
        """Test exception handling in metrics callbacks."""
        def failing_metrics_callback(metrics):
            raise Exception("Metrics callback error")

        self.processor.add_metrics_callback(failing_metrics_callback)

        # Should not raise exception when metrics callback fails
        test_metrics = AudioProcessingMetrics()

        try:
            for callback in self.processor._metrics_callbacks:
                callback(test_metrics)
        except Exception:
            self.fail("Exception should be caught in metrics callback system")

    def test_memory_pool_exception_handling(self):
        """Test exception handling in memory pool operations."""
        # Test with corrupted chunk
        corrupted_chunk = np.array([float('inf'), float('nan')])

        # Should handle gracefully
        try:
            self.processor._return_chunk_to_pool(corrupted_chunk)
        except Exception:
            self.fail("Memory pool should handle corrupted data gracefully")


class TestCreateOptimizedAudioProcessor(unittest.TestCase):
    """Test factory function for creating optimized audio processor."""

    def test_create_processor_latency_mode(self):
        """Test creating processor in latency optimization mode."""
        mock_config = Mock()
        mock_config.audio = Mock()
        mock_config.audio.max_buffer_size = 50

        processor = create_optimized_audio_processor(mock_config, "latency")

        # Should be optimized for latency
        self.assertEqual(processor.chunk_size, 256)
        self.assertEqual(processor._buffer_size, 20)
        self.assertFalse(processor._enable_noise_reduction)
        self.assertFalse(processor._enable_quality_analysis)

    def test_create_processor_quality_mode(self):
        """Test creating processor in quality optimization mode."""
        mock_config = Mock()
        mock_config.audio = Mock()
        mock_config.audio.max_buffer_size = 50

        processor = create_optimized_audio_processor(mock_config, "quality")

        # Should be optimized for quality
        self.assertEqual(processor.chunk_size, 1024)
        self.assertEqual(processor._buffer_size, 100)
        self.assertTrue(processor._enable_noise_reduction)
        self.assertTrue(processor._enable_quality_analysis)

    def test_create_processor_default_mode(self):
        """Test creating processor with default mode."""
        mock_config = Mock()
        mock_config.audio = Mock()
        mock_config.audio.max_buffer_size = 50

        processor = create_optimized_audio_processor(mock_config)

        # Should default to latency mode
        self.assertEqual(processor.chunk_size, 256)
        self.assertEqual(processor._buffer_size, 20)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)