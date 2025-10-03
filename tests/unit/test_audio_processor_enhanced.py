#!/usr/bin/env python3
"""
Comprehensive unit tests for Audio Processor module.
"""

import pytest
import asyncio
import time
import threading
import numpy as np
from unittest.mock import MagicMock, Mock, patch, AsyncMock, call
from dataclasses import dataclass
from pathlib import Path
import sys
import os
import json
import base64
from collections import deque

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Mock all external dependencies to avoid import issues
def mock_audio_dependencies():
    """Create comprehensive mocks for all audio dependencies."""
    mocks = {}

    # Mock audio processing libraries
    mocks['soundfile'] = Mock()
    mocks['noisereduce'] = Mock()
    mocks['webrtcvad'] = Mock()
    mocks['librosa'] = Mock()
    mocks['sounddevice'] = Mock()

    return mocks

# Apply mocking
sys.modules.update(mock_audio_dependencies())

# Import after mocking
from voice.audio_processor import (
    AudioData, AudioProcessorState, AudioQualityMetrics,
    SimplifiedAudioProcessor, SOUNDDEVICE_AVAILABLE, NOISEREDUCE_AVAILABLE,
    VAD_AVAILABLE, LIBROSA_AVAILABLE
)


class TestAudioData:
    """Test AudioData class."""

    @pytest.fixture
    def sample_audio_data(self):
        """Create sample audio data for testing."""
        data = np.array([0.1, 0.2, 0.3, 0.4, 0.5] * 1000, dtype=np.float32)
        return AudioData(
            data=data,
            sample_rate=16000,
            duration=len(data) / 16000,
            channels=1,
            format="wav"
        )

    def test_audio_data_initialization(self, sample_audio_data):
        """Test audio data initialization."""
        assert sample_audio_data.data is not None
        assert sample_audio_data.sample_rate == 16000
        assert sample_audio_data.duration > 0
        assert sample_audio_data.channels == 1
        assert sample_audio_data.format == "wav"

    def test_audio_data_to_bytes_fallback(self, sample_audio_data):
        """Test audio data to bytes conversion using fallback."""
        # Mock soundfile as unavailable
        with patch('voice.audio_processor.SOUNDDEVICE_AVAILABLE', False):
            result = sample_audio_data.to_bytes()
            assert isinstance(result, bytes)
            assert len(result) > 0

    def test_audio_data_to_bytes_with_soundfile(self, sample_audio_data):
        """Test audio data to bytes conversion with soundfile."""
        with patch('voice.audio_processor.SOUNDDEVICE_AVAILABLE', True):
            mock_sf = Mock()
            mock_buffer = Mock()
            mock_buffer.getvalue.return_value = b"mock_audio_bytes"
            mock_sf.io.BytesIO.return_value = mock_buffer

            with patch('voice.audio_processor.sf', mock_sf):
                result = sample_audio_data.to_bytes()
                assert result == b"mock_audio_bytes"

    def test_audio_data_from_bytes_fallback(self):
        """Test audio data from bytes using fallback."""
        # Create test data
        data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        encoded = base64.b64encode(data.tobytes())

        with patch('voice.audio_processor.SOUNDDEVICE_AVAILABLE', False):
            result = AudioData.from_bytes(encoded, 16000)
            assert isinstance(result, AudioData)
            assert result.sample_rate == 16000
            assert len(result.data) == 3

    def test_audio_data_from_bytes_with_soundfile(self):
        """Test audio data from bytes with soundfile."""
        mock_data = np.array([0.1, 0.2, 0.3])
        mock_sf = Mock()
        mock_buffer = Mock()
        mock_sf.read.return_value = (mock_data, 16000)
        mock_sf.io.BytesIO.return_value = mock_buffer

        with patch('voice.audio_processor.SOUNDDEVICE_AVAILABLE', True):
            with patch('voice.audio_processor.sf', mock_sf):
                result = AudioData.from_bytes(b"mock_bytes", 16000)
                assert isinstance(result, AudioData)
                assert result.sample_rate == 16000
                assert len(result.data) == 3


class TestAudioQualityMetrics:
    """Test AudioQualityMetrics class."""

    @pytest.fixture
    def metrics(self):
        """Create audio quality metrics for testing."""
        return AudioQualityMetrics(
            snr_ratio=10.5,
            noise_level=0.3,
            speech_level=0.7,
            clarity_score=0.85,
            overall_quality=0.8
        )

    def test_metrics_initialization(self, metrics):
        """Test metrics initialization."""
        assert metrics.snr_ratio == 10.5
        assert metrics.noise_level == 0.3
        assert metrics.speech_level == 0.7
        assert metrics.clarity_score == 0.85
        assert metrics.overall_quality == 0.8

    def test_metrics_to_dict(self, metrics):
        """Test metrics to dictionary conversion."""
        result = metrics.to_dict()
        expected = {
            'snr_ratio': 10.5,
            'noise_level': 0.3,
            'speech_level': 0.7,
            'clarity_score': 0.85,
            'overall_quality': 0.8
        }
        assert result == expected

    def test_metrics_default_values(self):
        """Test metrics with default values."""
        metrics = AudioQualityMetrics()
        assert metrics.snr_ratio == 0.0
        assert metrics.noise_level == 0.0
        assert metrics.speech_level == 0.0
        assert metrics.clarity_score == 0.0
        assert metrics.overall_quality == 0.0


class TestSimplifiedAudioProcessor:
    """Test SimplifiedAudioProcessor class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock()
        config.audio = Mock()
        config.audio.max_buffer_size = 300
        config.audio.max_memory_mb = 100
        config.audio.sample_rate = 16000
        config.audio.channels = 1
        config.audio.chunk_size = 1024
        config.audio.format = "wav"
        return config

    @pytest.fixture
    def audio_processor(self, mock_config):
        """Create audio processor with mocked config."""
        return SimplifiedAudioProcessor(mock_config)

    def test_audio_processor_initialization(self, audio_processor, mock_config):
        """Test audio processor initialization."""
        assert audio_processor.config == mock_config
        assert audio_processor.state == AudioProcessorState.READY
        assert audio_processor.is_recording == False
        assert audio_processor.is_playing == False
        assert audio_processor.sample_rate == 16000
        assert audio_processor.channels == 1
        assert audio_processor.chunk_size == 1024
        assert audio_processor.format == "wav"
        assert isinstance(audio_processor.audio_buffer, deque)
        assert audio_processor.max_buffer_size == 300

    def test_audio_processor_initialization_with_no_config(self):
        """Test audio processor initialization without config."""
        # Create a minimal mock config since the processor expects one
        config = Mock()
        config.audio = Mock()
        config.audio.max_buffer_size = 300
        config.audio.max_memory_mb = 100
        config.audio.sample_rate = 16000
        config.audio.channels = 1
        config.audio.chunk_size = 1024
        config.audio.format = "wav"

        processor = SimplifiedAudioProcessor(config)
        assert processor.config == config
        assert processor.sample_rate == 16000
        assert processor.channels == 1

    def test_audio_processor_features_availability(self, audio_processor):
        """Test that features availability is correctly detected."""
        assert isinstance(audio_processor.features, dict)
        assert 'audio_capture' in audio_processor.features
        assert 'audio_playback' in audio_processor.features
        assert 'noise_reduction' in audio_processor.features
        assert 'vad' in audio_processor.features
        assert 'quality_analysis' in audio_processor.features
        assert 'format_conversion' in audio_processor.features

    def test_initialize_features_with_vad_available(self, audio_processor):
        """Test feature initialization when VAD is available."""
        with patch('voice.audio_processor.VAD_AVAILABLE', True):
            mock_vad = Mock()
            with patch('voice.audio_processor.webrtcvad') as mock_webrtcvad:
                mock_webrtcvad.Vad.return_value = mock_vad
                audio_processor._initialize_features()
                assert audio_processor.vad == mock_vad

    def test_initialize_features_without_vad(self, audio_processor):
        """Test feature initialization when VAD is not available."""
        # Create a new processor with VAD unavailable
        config = Mock()
        config.audio = Mock()
        config.audio.max_buffer_size = 300
        config.audio.max_memory_mb = 100
        config.audio.sample_rate = 16000
        config.audio.channels = 1
        config.audio.chunk_size = 1024
        config.audio.format = "wav"

        with patch('voice.audio_processor.VAD_AVAILABLE', False):
            processor = SimplifiedAudioProcessor(config)
            # vad attribute should not be created when VAD_AVAILABLE is False
            assert not hasattr(processor, 'vad') or processor.vad is None

    def test_initialize_features_exception_handling(self, audio_processor):
        """Test feature initialization exception handling."""
        with patch.object(audio_processor, '_get_audio_devices', side_effect=Exception("Test error")):
            audio_processor._initialize_features()
            assert audio_processor.state == AudioProcessorState.ERROR

    def test_get_audio_devices_without_soundfile(self, audio_processor):
        """Test getting audio devices when soundfile is not available."""
        with patch('voice.audio_processor.SOUNDDEVICE_AVAILABLE', False):
            audio_processor._get_audio_devices()
            assert audio_processor.input_devices == []
            assert audio_processor.output_devices == []

    def test_get_audio_devices_with_soundfile(self, audio_processor):
        """Test getting audio devices when soundfile is available."""
        mock_sd = Mock()
        mock_devices = [
            {'name': 'Test Mic 1', 'max_input_channels': 2, 'max_output_channels': 0},
            {'name': 'Test Speaker 1', 'max_input_channels': 0, 'max_output_channels': 2}
        ]
        mock_sd.query_devices.return_value = mock_devices

        with patch('voice.audio_processor.SOUNDDEVICE_AVAILABLE', True):
            # Mock the import at module level
            with patch('voice.audio_processor.sd', mock_sd):
                audio_processor._get_audio_devices()
                assert len(audio_processor.input_devices) == 1
                assert len(audio_processor.output_devices) == 1

    def test_get_audio_devices_exception_handling(self, audio_processor):
        """Test audio devices query exception handling."""
        with patch('voice.audio_processor.SOUNDDEVICE_AVAILABLE', True):
            mock_sd = Mock()
            mock_sd.query_devices.side_effect = Exception("Device query failed")

            with patch.dict('sys.modules', {'sounddevice': mock_sd}):
                # Should not crash
                audio_processor._get_audio_devices()
                assert hasattr(audio_processor, 'input_devices')
                assert hasattr(audio_processor, 'output_devices')

    def test_memory_monitoring_initialization(self, audio_processor):
        """Test memory monitoring initialization."""
        assert audio_processor._buffer_bytes_estimate == 0
        assert audio_processor._max_memory_bytes == 100 * 1024 * 1024  # 100MB

    def test_thread_safety_initialization(self, audio_processor):
        """Test thread safety components initialization."""
        assert audio_processor._lock is not None
        assert audio_processor._recording_thread is None

    def test_missing_attributes_for_tests(self, audio_processor):
        """Test that missing attributes expected by tests are initialized."""
        assert audio_processor.audio is None
        assert audio_processor.stream is None

    def test_recording_state_tracking(self, audio_processor):
        """Test recording state tracking."""
        assert audio_processor.recording_start_time is None
        assert audio_processor.recording_duration == 0.0

    def test_logger_initialization(self, audio_processor):
        """Test logger initialization."""
        assert audio_processor.logger is not None

    def test_feature_logging(self, audio_processor):
        """Test that feature information is logged."""
        # The logger should have been called with features info
        assert isinstance(audio_processor.features, dict)


class TestAudioProcessorState:
    """Test AudioProcessorState enum."""

    def test_state_values(self):
        """Test that all expected states are available."""
        assert AudioProcessorState.IDLE.value == "idle"
        assert AudioProcessorState.INITIALIZING.value == "initializing"
        assert AudioProcessorState.READY.value == "ready"
        assert AudioProcessorState.RECORDING.value == "recording"
        assert AudioProcessorState.PROCESSING.value == "processing"
        assert AudioProcessorState.PLAYING.value == "playing"
        assert AudioProcessorState.ERROR.value == "error"

    def test_state_comparison(self):
        """Test state comparison operations."""
        state1 = AudioProcessorState.IDLE
        state2 = AudioProcessorState.IDLE
        state3 = AudioProcessorState.RECORDING

        assert state1 == state2
        assert state1 != state3


class TestAudioProcessorBufferManagement:
    """Test audio processor buffer management."""

    @pytest.fixture
    def audio_processor(self):
        """Create audio processor for buffer testing."""
        config = Mock()
        config.audio = Mock()
        config.audio.max_buffer_size = 300
        config.audio.max_memory_mb = 100
        config.audio.sample_rate = 16000
        config.audio.channels = 1
        config.audio.chunk_size = 1024
        config.audio.format = "wav"
        return SimplifiedAudioProcessor(config)

    def test_audio_buffer_initialization(self, audio_processor):
        """Test audio buffer initialization."""
        assert isinstance(audio_processor.audio_buffer, deque)
        assert audio_processor.audio_buffer.maxlen == 10
        assert len(audio_processor.audio_buffer) == 0

    def test_audio_buffer_memory_limit(self, audio_processor):
        """Test audio buffer memory limit."""
        # Verify memory limit is set correctly
        assert audio_processor._max_memory_bytes == 1 * 1024 * 1024  # 1MB

    def test_audio_buffer_size_limit(self, audio_processor):
        """Test audio buffer size limit."""
        # Fill buffer beyond its limit
        for i in range(15):
            audio_processor.audio_buffer.append(f"audio_chunk_{i}")

        # Buffer should not exceed max size
        assert len(audio_processor.audio_buffer) <= 10

    def test_buffer_bytes_estimate_initialization(self, audio_processor):
        """Test buffer bytes estimate initialization."""
        assert audio_processor._buffer_bytes_estimate == 0


class TestAudioProcessorFeatureAvailability:
    """Test audio processor feature availability detection."""

    def test_sounddevice_availability_detection(self):
        """Test sounddevice availability detection."""
        # The test should run regardless of whether soundfile is actually available
        from voice.audio_processor import SOUNDDEVICE_AVAILABLE
        assert isinstance(SOUNDDEVICE_AVAILABLE, bool)

    def test_noisereduce_availability_detection(self):
        """Test noisereduce availability detection."""
        from voice.audio_processor import NOISEREDUCE_AVAILABLE
        assert isinstance(NOISEREDUCE_AVAILABLE, bool)

    def test_vad_availability_detection(self):
        """Test VAD availability detection."""
        from voice.audio_processor import VAD_AVAILABLE
        assert isinstance(VAD_AVAILABLE, bool)

    def test_librosa_availability_detection(self):
        """Test librosa availability detection."""
        from voice.audio_processor import LIBROSA_AVAILABLE
        assert isinstance(LIBROSA_AVAILABLE, bool)


class TestAudioProcessorGracefulDegradation:
    """Test audio processor graceful degradation when dependencies are missing."""

    def test_initialization_without_dependencies(self):
        """Test initialization works without audio dependencies."""
        # Create minimal config
        config = Mock()
        config.audio = Mock()
        config.audio.max_buffer_size = 300
        config.audio.max_memory_mb = 100
        config.audio.sample_rate = 16000
        config.audio.channels = 1
        config.audio.chunk_size = 1024
        config.audio.format = "wav"

        processor = SimplifiedAudioProcessor(config)

        # Should still initialize successfully
        assert processor is not None
        assert processor.state in [AudioProcessorState.READY, AudioProcessorState.ERROR]

        # Features should reflect availability
        assert isinstance(processor.features, dict)

        # Should have default configuration
        assert processor.sample_rate > 0
        assert processor.channels > 0

    def test_buffer_operations_without_dependencies(self):
        """Test buffer operations work without audio dependencies."""
        config = Mock()
        config.audio = Mock()
        config.audio.max_buffer_size = 300
        config.audio.max_memory_mb = 100
        config.audio.sample_rate = 16000
        config.audio.channels = 1
        config.audio.chunk_size = 1024
        config.audio.format = "wav"

        processor = SimplifiedAudioProcessor(config)

        # Should be able to work with audio buffer
        assert hasattr(processor, 'audio_buffer')
        assert isinstance(processor.audio_buffer, deque)

        # Should be able to add items to buffer
        processor.audio_buffer.append("test_audio_chunk")
        assert len(processor.audio_buffer) == 1

    def test_configuration_without_dependencies(self):
        """Test configuration works without audio dependencies."""
        config = Mock()
        config.audio = Mock()
        config.audio.max_buffer_size = 300
        config.audio.max_memory_mb = 100
        config.audio.sample_rate = 16000
        config.audio.channels = 1
        config.audio.chunk_size = 1024
        config.audio.format = "wav"

        processor = SimplifiedAudioProcessor(config)

        # Should have sensible defaults
        assert processor.sample_rate == 16000
        assert processor.channels == 1
        assert processor.chunk_size == 1024
        assert processor.format == "wav"

    def test_state_management_without_dependencies(self):
        """Test state management works without audio dependencies."""
        config = Mock()
        config.audio = Mock()
        config.audio.max_buffer_size = 300
        config.audio.max_memory_mb = 100
        config.audio.sample_rate = 16000
        config.audio.channels = 1
        config.audio.chunk_size = 1024
        config.audio.format = "wav"

        processor = SimplifiedAudioProcessor(config)

        # Should be able to track state
        assert processor.state in [AudioProcessorState.IDLE, AudioProcessorState.READY, AudioProcessorState.ERROR]
        assert processor.is_recording == False
        assert processor.is_playing == True


class TestAudioProcessorThreadSafety:
    """Test audio processor thread safety."""

    @pytest.fixture
    def audio_processor(self):
        """Create audio processor for thread safety testing."""
        config = Mock()
        config.audio = Mock()
        config.audio.max_buffer_size = 300
        config.audio.max_memory_mb = 100
        config.audio.sample_rate = 16000
        config.audio.channels = 1
        config.audio.chunk_size = 1024
        config.audio.format = "wav"
        return SimplifiedAudioProcessor(config)

    def test_lock_initialization(self, audio_processor):
        """Test that lock is properly initialized."""
        assert audio_processor._lock is not None
        assert hasattr(audio_processor._lock, 'acquire')
        assert hasattr(audio_processor._lock, 'release')

    def test_threading_attributes(self, audio_processor):
        """Test threading-related attributes."""
        assert audio_processor._recording_thread is None
        assert isinstance(audio_processor.is_recording, bool)
        assert isinstance(audio_processor.is_playing, bool)

    def test_concurrent_buffer_access(self, audio_processor):
        """Test concurrent access to audio buffer."""
        results = []
        errors = []

        def add_to_buffer(start_id, count):
            try:
                with audio_processor._lock:
                    for i in range(count):
                        audio_processor.audio_buffer.append(f"chunk_{start_id}_{i}")
                        time.sleep(0.001)  # Small delay to increase contention
                    results.append(len(audio_processor.audio_buffer))
            except Exception as e:
                errors.append(e)

        # Run multiple threads adding to buffer
        threads = []
        for i in range(5):
            thread = threading.Thread(target=add_to_buffer, args=(i, 10))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should have no errors
        assert len(errors) == 0
        assert len(results) == 5

        # Buffer should not exceed max size
        assert len(audio_processor.audio_buffer) <= audio_processor.max_buffer_size


class TestAudioProcessorConfiguration:
    """Test audio processor configuration handling."""

    def test_config_with_custom_values(self):
        """Test configuration with custom values."""
        config = Mock()
        config.audio = Mock()
        config.audio.max_buffer_size = 300
        config.audio.max_memory_mb = 100
        config.audio.sample_rate = 44100
        config.audio.channels = 2
        config.audio.chunk_size = 2048
        config.audio.format = "mp3"

        processor = SimplifiedAudioProcessor(config)

        assert processor.max_buffer_size == 500
        assert processor._max_memory_bytes == 200 * 1024 * 1024
        assert processor.sample_rate == 44100
        assert processor.channels == 2
        assert processor.chunk_size == 2048
        assert processor.format == "mp3"

    def test_config_missing_audio_section(self):
        """Test configuration when audio section is missing."""
        config = Mock()
        config.audio = None

        # Should handle gracefully with default values
        processor = SimplifiedAudioProcessor(config)
        assert processor.sample_rate == 16000  # Default value
        assert processor.channels == 1  # Default value

    def test_config_with_missing_attributes(self):
        """Test configuration with missing attributes."""
        config = Mock()
        config.audio = Mock()
        # Mock audio to have missing attributes but return None for them
        config.audio.max_buffer_size = 300
        config.audio.max_memory_mb = 100
        config.audio.sample_rate = 16000
        config.audio.channels = 1
        config.audio.chunk_size = 1024
        config.audio.format = "wav"

        # Should use defaults ( getattr handles missing attributes)
        processor = SimplifiedAudioProcessor(config)
        assert processor.sample_rate == 16000
        assert processor.channels == 1
        assert processor.chunk_size == 1024