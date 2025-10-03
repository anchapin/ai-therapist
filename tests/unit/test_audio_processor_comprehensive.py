#!/usr/bin/env python3
"""
Comprehensive unit tests for AudioProcessor module.
"""

import pytest
import tempfile
import os
import threading
import time
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Optional, Dict, List, Any

# Mock the audio libraries that might not be available
with patch.dict('sys.modules', {
    'soundfile': Mock(),
    'noisereduce': Mock(),
    'webrtcvad': Mock(),
    'librosa': Mock(),
    'pyaudio': Mock(),
    'sounddevice': Mock()
}):
    from voice.audio_processor import (
        SimplifiedAudioProcessor, AudioData, AudioProcessorState,
        AudioQualityMetrics, create_audio_processor
    )


class TestAudioData:
    """Test AudioData class."""

    @pytest.fixture
    def audio_data(self):
        """Create test audio data."""
        return AudioData(
            data=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            sample_rate=16000,
            duration=0.25,
            channels=1,
            format="wav"
        )

    def test_audio_data_initialization(self, audio_data):
        """Test audio data initialization."""
        assert audio_data.sample_rate == 16000
        assert audio_data.duration == 0.25
        assert audio_data.channels == 1
        assert audio_data.format == "wav"
        assert len(audio_data.data) == 5

    def test_to_bytes_with_soundfile(self, audio_data):
        """Test to_bytes method when soundfile is available."""
        with patch('voice.audio_processor.SOUNDDEVICE_AVAILABLE', True):
            with patch('voice.audio_processor.sf') as mock_sf:
                mock_buffer = Mock()
                mock_sf.io.BytesIO.return_value = mock_buffer

                result = audio_data.to_bytes()

                mock_sf.io.BytesIO.assert_called_once()
                mock_sf.write.assert_called_once_with(mock_buffer, audio_data.data, audio_data.sample_rate, format='WAV')
                mock_buffer.getvalue.assert_called_once()

    def test_to_bytes_without_soundfile(self, audio_data):
        """Test to_bytes method when soundfile is not available."""
        with patch('voice.audio_processor.SOUNDDEVICE_AVAILABLE', False):
            with patch('voice.audio_processor.base64') as mock_base64:
                mock_base64.b64encode.return_value = b'encoded_data'

                result = audio_data.to_bytes()

                mock_base64.b64encode.assert_called_once_with(audio_data.data.tobytes())
                assert result == b'encoded_data'

    def test_from_bytes_with_soundfile(self):
        """Test from_bytes method when soundfile is available."""
        test_bytes = b'test_audio_data'

        with patch('voice.audio_processor.SOUNDDEVICE_AVAILABLE', True):
            with patch('voice.audio_processor.sf') as mock_sf:
                mock_buffer = Mock()
                mock_sf.io.BytesIO.return_value = mock_buffer
                mock_sf.read.return_value = (np.array([0.1, 0.2]), 16000)

                result = AudioData.from_bytes(test_bytes, 16000)

                mock_sf.io.BytesIO.assert_called_once_with(test_bytes)
                mock_sf.read.assert_called_once_with(mock_buffer)
                assert result.sample_rate == 16000
                assert result.duration == 2/16000

    def test_from_bytes_without_soundfile(self):
        """Test from_bytes method when soundfile is not available."""
        test_bytes = b'test_audio_data'

        with patch('voice.audio_processor.SOUNDDEVICE_AVAILABLE', False):
            with patch('voice.audio_processor.base64') as mock_base64:
                with patch('voice.audio_processor.np') as mock_np:
                    mock_base64.b64decode.return_value = b'decoded'
                    mock_np.frombuffer.return_value = np.array([0.1, 0.2])

                    result = AudioData.from_bytes(test_bytes, 16000)

                    mock_base64.b64decode.assert_called_once_with(test_bytes)
                    mock_np.frombuffer.assert_called_once_with(b'decoded', dtype=np.float32)
                    assert result.sample_rate == 16000
                    assert result.duration == 2/16000


class TestAudioProcessorState:
    """Test AudioProcessorState enum."""

    def test_state_values(self):
        """Test state enum values."""
        assert AudioProcessorState.IDLE.value == "idle"
        assert AudioProcessorState.INITIALIZING.value == "initializing"
        assert AudioProcessorState.READY.value == "ready"
        assert AudioProcessorState.RECORDING.value == "recording"
        assert AudioProcessorState.PROCESSING.value == "processing"
        assert AudioProcessorState.PLAYING.value == "playing"
        assert AudioProcessorState.ERROR.value == "error"


class TestAudioQualityMetrics:
    """Test AudioQualityMetrics class."""

    @pytest.fixture
    def metrics(self):
        """Create test metrics."""
        return AudioQualityMetrics(
            snr_ratio=25.0,
            noise_level=0.1,
            speech_level=0.7,
            clarity_score=0.8,
            overall_quality=0.85
        )

    def test_metrics_initialization(self, metrics):
        """Test metrics initialization."""
        assert metrics.snr_ratio == 25.0
        assert metrics.noise_level == 0.1
        assert metrics.speech_level == 0.7
        assert metrics.clarity_score == 0.8
        assert metrics.overall_quality == 0.85

    def test_to_dict(self, metrics):
        """Test to_dict method."""
        result = metrics.to_dict()
        expected = {
            'snr_ratio': 25.0,
            'noise_level': 0.1,
            'speech_level': 0.7,
            'clarity_score': 0.8,
            'overall_quality': 0.85
        }
        assert result == expected


class TestSimplifiedAudioProcessor:
    """Test SimplifiedAudioProcessor class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock()
        # Create nested audio config
        audio_config = Mock()
        audio_config.max_buffer_size = 50
        audio_config.max_memory_mb = 100
        audio_config.sample_rate = 16000
        audio_config.channels = 1
        audio_config.chunk_size = 1024
        audio_config.max_recording_duration = 30.0
        audio_config.vad_enabled = True
        audio_config.noise_reduction_enabled = True
        audio_config.noise_reduction_strength = 0.3
        audio_config.auto_gain_control = True
        audio_config.voice_activity_threshold = 0.5
        audio_config.buffer_size = 1024

        config.audio = audio_config
        config.audio_sample_rate = 16000
        config.audio_channels = 1
        config.max_recording_duration = 30.0
        config.vad_enabled = True
        config.noise_reduction_enabled = True
        config.noise_reduction_strength = 0.3
        config.auto_gain_control = True
        config.voice_activity_threshold = 0.5
        config.audio_buffer_size = 1024

        return config

    @pytest.fixture
    def processor(self, mock_config):
        """Create audio processor for testing."""
        with patch('voice.audio_processor.SOUNDDEVICE_AVAILABLE', False):
            return SimplifiedAudioProcessor(mock_config)

    def test_initialization(self, processor, mock_config):
        """Test processor initialization."""
        assert processor.config == mock_config
        assert processor.state in [AudioProcessorState.IDLE, AudioProcessorState.READY]
        assert processor.sample_rate == 16000
        assert processor.channels == 1
        assert processor.is_recording == False
        assert processor.is_playing == False
        assert processor.recording_duration == 0.0

    def test_initialize_features(self, processor):
        """Test feature initialization."""
        processor._initialize_features()
        assert hasattr(processor, 'vad')
        assert hasattr(processor, 'noise_reducer')
        assert hasattr(processor, 'audio_analyzer')

    def test_get_audio_devices_mocked(self, processor):
        """Test getting audio devices when mocked."""
        input_devices, output_devices = processor._get_audio_devices()
        assert isinstance(input_devices, list)
        assert isinstance(output_devices, list)

    def test_detect_audio_devices(self, processor):
        """Test audio device detection."""
        with patch.object(processor, '_get_audio_devices', return_value=(['mic1', 'mic2'], ['speaker1'])):
            input_devices, output_devices = processor.detect_audio_devices()
            assert input_devices == ['mic1', 'mic2']
            assert output_devices == ['speaker1']

    def test_audio_callback(self, processor):
        """Test audio callback function."""
        test_data = np.array([0.1, 0.2, 0.3])
        frames = 1024
        time_info = {'input_buffer_adc_time': 0.0}
        status = Mock()

        with patch.object(processor, 'add_to_buffer') as mock_add:
            processor.audio_callback(test_data, frames, time_info, status)
            mock_add.assert_called_once_with(test_data)

    def test_get_available_features(self, processor):
        """Test getting available features."""
        features = processor.get_available_features()
        assert isinstance(features, dict)
        assert 'vad' in features
        assert 'noise_reduction' in features
        assert 'audio_analysis' in features

    def test_start_recording_success(self, processor):
        """Test successful recording start."""
        with patch.object(processor, '_record_audio'):
            result = processor.start_recording()
            assert result == True
            assert processor.state == AudioProcessorState.RECORDING
            assert processor.is_recording == True

    def test_start_recording_already_recording(self, processor):
        """Test starting recording when already recording."""
        processor.is_recording = True
        result = processor.start_recording()
        assert result == False

    def test_stop_recording_success(self, processor):
        """Test successful recording stop."""
        processor.is_recording = True
        processor.state = AudioProcessorState.RECORDING
        processor.recording_start_time = time.time() - 1.0
        processor.audio_buffer = [np.array([0.1, 0.2])]

        with patch.object(processor, '_process_audio') as mock_process:
            mock_process.return_value = AudioData(
                data=np.array([0.1, 0.2]),
                sample_rate=16000,
                duration=0.125,
                channels=1
            )

            result = processor.stop_recording()

            assert isinstance(result, AudioData)
            assert processor.is_recording == False
            assert processor.state == AudioProcessorState.IDLE
            mock_process.assert_called_once()

    def test_stop_recording_not_recording(self, processor):
        """Test stopping recording when not recording."""
        result = processor.stop_recording()
        assert result is None

    def test_process_audio_with_noise_reduction(self, processor):
        """Test audio processing with noise reduction."""
        input_audio = AudioData(
            data=np.array([0.1, 0.2, 0.3]),
            sample_rate=16000,
            duration=0.1875,
            channels=1
        )

        with patch.object(processor, 'reduce_background_noise') as mock_reduce:
            with patch.object(processor, 'normalize_audio_level') as mock_normalize:
                mock_reduce.return_value = np.array([0.15, 0.25, 0.35])
                mock_normalize.return_value = np.array([0.2, 0.3, 0.4])

                result = processor._process_audio(input_audio)

                mock_reduce.assert_called_once()
                mock_normalize.assert_called_once()
                assert isinstance(result, AudioData)

    def test_process_audio_without_noise_reduction(self, processor):
        """Test audio processing without noise reduction."""
        processor.config.noise_reduction_enabled = False

        input_audio = AudioData(
            data=np.array([0.1, 0.2, 0.3]),
            sample_rate=16000,
            duration=0.1875,
            channels=1
        )

        with patch.object(processor, 'normalize_audio_level') as mock_normalize:
            mock_normalize.return_value = np.array([0.2, 0.3, 0.4])

            result = processor._process_audio(input_audio)

            mock_normalize.assert_called_once()
            assert isinstance(result, AudioData)

    def test_analyze_audio_quality(self, processor):
        """Test audio quality analysis."""
        audio_data = AudioData(
            data=np.array([0.1, 0.2, 0.3]),
            sample_rate=16000,
            duration=0.1875,
            channels=1
        )

        with patch.object(processor, 'calculate_audio_quality_metrics') as mock_calculate:
            mock_metrics = AudioQualityMetrics(
                snr_ratio=20.0,
                noise_level=0.1,
                speech_level=0.6,
                clarity_score=0.7,
                overall_quality=0.75
            )
            mock_calculate.return_value = mock_metrics

            result = processor._analyze_audio_quality(audio_data)

            assert result == mock_metrics
            mock_calculate.assert_called_once_with(audio_data.data)

    def test_detect_voice_activity(self, processor):
        """Test voice activity detection."""
        audio_data = AudioData(
            data=np.array([0.1, 0.2, 0.3]),
            sample_rate=16000,
            duration=0.1875,
            channels=1
        )

        with patch.object(processor, 'detect_voice_activity_simple') as mock_detect:
            mock_detect.return_value = True

            result = processor.detect_voice_activity(audio_data)

            assert isinstance(result, list)
            assert len(result) > 0
            mock_detect.assert_called_once_with(audio_data.data)

    def test_play_audio_success(self, processor):
        """Test successful audio playback."""
        audio_data = AudioData(
            data=np.array([0.1, 0.2, 0.3]),
            sample_rate=16000,
            duration=0.1875,
            channels=1
        )

        with patch('threading.Thread') as mock_thread:
            result = processor.play_audio(audio_data)
            assert result == True
            assert processor.is_playing == True
            mock_thread.assert_called_once()

    def test_play_audio_already_playing(self, processor):
        """Test playing audio when already playing."""
        processor.is_playing = True
        audio_data = AudioData(
            data=np.array([0.1, 0.2, 0.3]),
            sample_rate=16000,
            duration=0.1875,
            channels=1
        )

        result = processor.play_audio(audio_data)
        assert result == False

    def test_stop_playback_success(self, processor):
        """Test successful playback stop."""
        processor.is_playing = True
        processor.state = AudioProcessorState.PLAYING

        result = processor.stop_playback()
        assert result == True
        assert processor.is_playing == False
        assert processor.state == AudioProcessorState.IDLE

    def test_stop_playback_not_playing(self, processor):
        """Test stopping playback when not playing."""
        result = processor.stop_playback()
        assert result == False

    def test_get_recording_duration(self, processor):
        """Test getting recording duration."""
        processor.recording_start_time = time.time() - 2.5
        result = processor.get_recording_duration()
        assert result >= 2.5  # Allow for small timing differences

    def test_get_audio_level(self, processor):
        """Test getting audio level."""
        processor.audio_buffer = [np.array([0.1, 0.2, 0.3])]
        with patch.object(processor, 'calculate_rms_level') as mock_calculate:
            mock_calculate.return_value = 0.25
            result = processor.get_audio_level()
            assert result == 0.25
            mock_calculate.assert_called_once()

    def test_create_silent_audio(self, processor):
        """Test creating silent audio."""
        duration = 1.0
        result = processor.create_silent_audio(duration)

        assert isinstance(result, AudioData)
        assert result.duration == duration
        assert result.sample_rate == processor.sample_rate
        assert len(result.data) == int(duration * processor.sample_rate)
        assert np.all(result.data == 0.0)

    def test_save_audio_success(self, processor):
        """Test successful audio saving."""
        audio_data = AudioData(
            data=np.array([0.1, 0.2, 0.3]),
            sample_rate=16000,
            duration=0.1875,
            channels=1
        )

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            with patch('voice.audio_processor.np') as mock_np:
                mock_np.save.return_value = True

                result = processor.save_audio(audio_data, temp_path)
                assert result == True

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_save_audio_failure(self, processor):
        """Test audio saving failure."""
        audio_data = AudioData(
            data=np.array([0.1, 0.2, 0.3]),
            sample_rate=16000,
            duration=0.1875,
            channels=1
        )

        result = processor.save_audio(audio_data, "/invalid/path/audio.wav")
        assert result == False

    def test_load_audio_success(self, processor):
        """Test successful audio loading."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Create a dummy file
            with open(temp_path, 'w') as f:
                f.write("dummy content")

            with patch('voice.audio_processor.np') as mock_np:
                mock_data = np.array([0.1, 0.2, 0.3])
                mock_np.load.return_value = mock_data

                result = processor.load_audio(temp_path)

                assert isinstance(result, AudioData)
                assert result.sample_rate == 16000

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_load_audio_failure(self, processor):
        """Test audio loading failure."""
        result = processor.load_audio("/nonexistent/file.wav")
        assert result is None

    def test_get_state(self, processor):
        """Test getting processor state."""
        processor.state = AudioProcessorState.READY
        result = processor.get_state()
        assert result == AudioProcessorState.READY

    def test_is_available(self, processor):
        """Test availability check."""
        result = processor.is_available()
        assert isinstance(result, bool)

    def test_get_status(self, processor):
        """Test getting status."""
        processor.sample_rate = 16000
        processor.channels = 1
        processor.is_recording = True
        processor.recording_duration = 5.0

        status = processor.get_status()

        assert isinstance(status, dict)
        assert 'state' in status
        assert 'sample_rate' in status
        assert 'channels' in status
        assert 'is_recording' in status
        assert 'recording_duration' in status
        assert status['sample_rate'] == 16000
        assert status['channels'] == 1
        assert status['is_recording'] == True
        assert status['recording_duration'] == 5.0

    def test_get_memory_usage(self, processor):
        """Test getting memory usage."""
        processor.audio_buffer = [np.array([0.1, 0.2]) for _ in range(5)]

        with patch('voice.audio_processor.np') as mock_np:
            mock_np.nbytes = 8

            usage = processor.get_memory_usage()

            assert isinstance(usage, dict)
            assert 'buffer_size' in usage
            assert 'buffer_count' in usage
            assert 'estimated_memory_mb' in usage
            assert usage['buffer_count'] == 5

    def test_force_cleanup_buffers(self, processor):
        """Test buffer cleanup."""
        processor.audio_buffer = [np.array([0.1, 0.2]) for _ in range(10)]
        processor.force_cleanup_buffers()
        assert len(processor.audio_buffer) == 0

    def test_get_audio_chunk(self, processor):
        """Test getting audio chunk."""
        processor.audio_buffer = [np.array([0.1, 0.2])]

        with patch.object(processor, '_encode_audio_chunk') as mock_encode:
            mock_encode.return_value = b'encoded_chunk'

            result = processor.get_audio_chunk()

            assert result == b'encoded_chunk'
            mock_encode.assert_called_once()

    def test_reduce_background_noise_with_library(self, processor):
        """Test noise reduction with library available."""
        audio_data = np.array([0.1, 0.2, 0.3])

        with patch('voice.audio_processor.NOISEREDUCE_AVAILABLE', True):
            with patch('voice.audio_processor.nr') as mock_nr:
                mock_nr.reduce_noise.return_value = np.array([0.15, 0.25, 0.35])

                result = processor.reduce_background_noise(audio_data)

                mock_nr.reduce_noise.assert_called_once()
                assert np.array_equal(result, np.array([0.15, 0.25, 0.35]))

    def test_reduce_background_noise_without_library(self, processor):
        """Test noise reduction without library available."""
        audio_data = np.array([0.1, 0.2, 0.3])

        with patch('voice.audio_processor.NOISEREDUCE_AVAILABLE', False):
            result = processor.reduce_background_noise(audio_data)
            # Should return original data when library not available
            assert np.array_equal(result, audio_data)

    def test_convert_audio_format(self, processor):
        """Test audio format conversion."""
        audio_data = AudioData(
            data=np.array([0.1, 0.2, 0.3]),
            sample_rate=16000,
            duration=0.1875,
            channels=1,
            format="wav"
        )

        result = processor.convert_audio_format(audio_data, "mp3")

        assert isinstance(result, AudioData)
        assert result.format == "mp3"
        assert np.array_equal(result.data, audio_data.data)

    def test_calculate_audio_quality_metrics(self, processor):
        """Test calculating audio quality metrics."""
        audio_data = np.array([0.1, 0.2, 0.3, -0.1, -0.2])

        result = processor.calculate_audio_quality_metrics(audio_data)

        assert isinstance(result, AudioQualityMetrics)
        assert hasattr(result, 'snr_ratio')
        assert hasattr(result, 'noise_level')
        assert hasattr(result, 'speech_level')
        assert hasattr(result, 'clarity_score')
        assert hasattr(result, 'overall_quality')

    def test_normalize_audio_level(self, processor):
        """Test audio level normalization."""
        audio_data = np.array([0.1, 0.2, 0.3])
        target_level = 0.5

        result = processor.normalize_audio_level(audio_data, target_level)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(audio_data)
        # Should be normalized to target level
        current_max = np.max(np.abs(result))
        assert current_max <= target_level

    def test_add_to_buffer(self, processor):
        """Test adding data to buffer."""
        audio_data = np.array([0.1, 0.2, 0.3])
        processor.audio_buffer = []

        processor.add_to_buffer(audio_data)

        assert len(processor.audio_buffer) == 1
        assert np.array_equal(processor.audio_buffer[0], audio_data)

    def test_get_buffer_contents(self, processor):
        """Test getting buffer contents."""
        buffer_data = [np.array([0.1]), np.array([0.2])]
        processor.audio_buffer = buffer_data

        result = processor.get_buffer_contents()

        assert result == buffer_data

    def test_clear_buffer(self, processor):
        """Test clearing buffer."""
        processor.audio_buffer = [np.array([0.1]), np.array([0.2])]

        processor.clear_buffer()

        assert len(processor.audio_buffer) == 0

    def test_detect_voice_activity_simple_with_speech(self, processor):
        """Test simple voice activity detection with speech."""
        audio_data = np.array([0.5, 0.6, 0.7, -0.5, -0.6])  # Higher amplitude

        with patch.object(processor, 'calculate_rms_level') as mock_rms:
            mock_rms.return_value = 0.4  # Above threshold
            processor.config.voice_activity_threshold = 0.3

            result = processor.detect_voice_activity_simple(audio_data)

            assert result == True
            mock_rms.assert_called_once_with(audio_data)

    def test_detect_voice_activity_simple_without_speech(self, processor):
        """Test simple voice activity detection without speech."""
        audio_data = np.array([0.01, 0.02, 0.01])  # Low amplitude

        with patch.object(processor, 'calculate_rms_level') as mock_rms:
            mock_rms.return_value = 0.1  # Below threshold
            processor.config.voice_activity_threshold = 0.3

            result = processor.detect_voice_activity_simple(audio_data)

            assert result == False
            mock_rms.assert_called_once_with(audio_data)

    def test_select_input_device(self, processor):
        """Test input device selection."""
        processor.input_devices = ['mic1', 'mic2']
        processor.default_input_device = 'mic1'

        result = processor.select_input_device(1)

        assert result == True
        assert processor.selected_input_device == 'mic2'

    def test_select_input_device_invalid(self, processor):
        """Test selecting invalid input device."""
        processor.input_devices = ['mic1', 'mic2']

        result = processor.select_input_device(5)

        assert result == False

    def test_save_audio_to_file(self, processor):
        """Test saving audio to file."""
        audio_data = AudioData(
            data=np.array([0.1, 0.2, 0.3]),
            sample_rate=16000,
            duration=0.1875,
            channels=1
        )

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            with patch.object(processor, 'save_audio') as mock_save:
                mock_save.return_value = True

                result = processor.save_audio_to_file(audio_data, temp_path)

                assert result == True
                mock_save.assert_called_once_with(audio_data, temp_path)

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_load_audio_from_file(self, processor):
        """Test loading audio from file."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            expected_audio = AudioData(
                data=np.array([0.1, 0.2, 0.3]),
                sample_rate=16000,
                duration=0.1875,
                channels=1
            )

            with patch.object(processor, 'load_audio') as mock_load:
                mock_load.return_value = expected_audio

                result = processor.load_audio_from_file(temp_path)

                assert result == expected_audio
                mock_load.assert_called_once_with(temp_path)

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_cleanup(self, processor):
        """Test cleanup."""
        processor.is_recording = True
        processor.is_playing = True
        processor.audio_buffer = [np.array([0.1])]

        processor.cleanup()

        assert processor.is_recording == False
        assert processor.is_playing == False
        assert processor.state == AudioProcessorState.IDLE
        assert len(processor.audio_buffer) == 0


class TestCreateAudioProcessor:
    """Test create_audio_processor function."""

    def test_create_audio_processor_with_config(self):
        """Test creating audio processor with config."""
        config = Mock()
        config.audio_sample_rate = 16000

        with patch('voice.audio_processor.SimplifiedAudioProcessor') as mock_processor:
            create_audio_processor(config)
            mock_processor.assert_called_once_with(config)

    def test_create_audio_processor_without_config(self):
        """Test creating audio processor without config."""
        with patch('voice.audio_processor.SimplifiedAudioProcessor') as mock_processor:
            create_audio_processor()
            mock_processor.assert_called_once_with(None)


# Integration tests
class TestAudioProcessorIntegration:
    """Integration tests for AudioProcessor."""

    @pytest.fixture
    def processor(self):
        """Create real audio processor for integration tests."""
        config = Mock()
        config.audio_sample_rate = 16000
        config.audio_channels = 1
        config.max_recording_duration = 30.0
        config.vad_enabled = False
        config.noise_reduction_enabled = False
        config.auto_gain_control = False
        config.voice_activity_threshold = 0.5
        config.audio_buffer_size = 1024

        with patch('voice.audio_processor.SOUNDDEVICE_AVAILABLE', False):
            return SimplifiedAudioProcessor(config)

    def test_full_recording_cycle(self, processor):
        """Test full recording cycle."""
        # Start recording
        result = processor.start_recording()
        assert result == True
        assert processor.is_recording == True

        # Add some audio data
        audio_data = np.array([0.1, 0.2, 0.3])
        processor.add_to_buffer(audio_data)

        # Stop recording
        recorded_audio = processor.stop_recording()
        assert recorded_audio is not None
        assert processor.is_recording == False

        # Test playback
        playback_result = processor.play_audio(recorded_audio)
        assert playback_result == True
        assert processor.is_playing == True

        # Stop playback
        stop_result = processor.stop_playback()
        assert stop_result == True
        assert processor.is_playing == False

    def test_audio_data_pipeline(self, processor):
        """Test audio data processing pipeline."""
        # Create test audio
        audio_data = AudioData(
            data=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            sample_rate=16000,
            duration=0.3125,
            channels=1
        )

        # Process audio
        processed_audio = processor._process_audio(audio_data)
        assert processed_audio is not None

        # Analyze quality
        quality = processor._analyze_audio_quality(processed_audio)
        assert quality is not None

        # Test voice activity detection
        vad_results = processor.detect_voice_activity(processed_audio)
        assert isinstance(vad_results, list)

        # Test save and load
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            save_result = processor.save_audio(processed_audio, temp_path)
            assert save_result == True

            loaded_audio = processor.load_audio(temp_path)
            assert loaded_audio is not None

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)