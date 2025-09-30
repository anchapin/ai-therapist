"""
Unit tests for audio processing functionality.

Tests SPEECH_PRD.md requirements:
- Audio quality handling
- Background noise scenarios
- Voice activity detection
- Audio format conversion
- Device management
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import MagicMock, patch
import tempfile
import os

from voice.audio_processor import AudioProcessor
from voice.config import VoiceConfig


class TestAudioProcessor:
    """Test audio processing functionality."""

    @pytest.fixture
    def processor(self, mock_voice_config):
        """Create AudioProcessor instance for testing."""
        with patch('sounddevice.default'):
            return AudioProcessor(mock_voice_config)

    def test_initialization(self, processor, mock_voice_config):
        """Test AudioProcessor initialization."""
        assert processor.config == mock_voice_config
        assert processor.sample_rate == mock_voice_config.audio_sample_rate
        assert processor.channels == mock_voice_config.audio_channels
        assert processor.chunk_size == mock_voice_config.audio_chunk_size
        assert processor.audio is None
        assert processor.stream is None

    def test_audio_device_detection(self, processor):
        """Test audio device detection."""
        # Mock sounddevice to return test devices
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [
            {'name': 'Test Microphone', 'max_input_channels': 2, 'max_output_channels': 0},
            {'name': 'Test Speakers', 'max_input_channels': 0, 'max_output_channels': 2},
            {'name': 'Test Headset', 'max_input_channels': 1, 'max_output_channels': 1}
        ]

        with patch('sounddevice.query_devices', mock_sd.query_devices):
            input_devices, output_devices = processor.detect_audio_devices()

            assert len(input_devices) == 2  # Test Microphone and Test Headset
            assert len(output_devices) == 2  # Test Speakers and Test Headset
            assert input_devices[0]['name'] == 'Test Microphone'
            assert output_devices[0]['name'] == 'Test Speakers'

    def test_audio_recording_start(self, processor):
        """Test starting audio recording."""
        # Mock audio stream
        mock_stream = MagicMock()
        processor.audio = MagicMock()
        processor.audio.open.return_value = mock_stream

        success = processor.start_recording()

        assert success
        processor.audio.open.assert_called_once()
        mock_stream.start_stream.assert_called_once()
        assert processor.stream == mock_stream
        assert processor.recording

    def test_audio_recording_stop(self, processor):
        """Test stopping audio recording."""
        # Setup mock stream
        mock_stream = MagicMock()
        processor.stream = mock_stream
        processor.recording = True

        success = processor.stop_recording()

        assert success
        mock_stream.stop_stream.assert_called_once()
        mock_stream.close.assert_called_once()
        assert processor.stream is None
        assert not processor.recording

    def test_audio_chunk_processing(self, processor, mock_audio_data):
        """Test audio chunk processing."""
        # Mock audio stream
        mock_stream = MagicMock()
        mock_stream.read.return_value = mock_audio_data['data'].tobytes()
        processor.stream = mock_stream
        processor.recording = True

        # Process one chunk
        chunk = processor.get_audio_chunk()

        assert chunk is not None
        assert len(chunk) > 0
        mock_stream.read.assert_called_once_with(processor.chunk_size)

    @pytest.mark.asyncio
    async def test_background_noise_reduction(self, processor, mock_audio_data):
        """Test background noise reduction."""
        # Add noise to audio data
        noise = np.random.normal(0, 0.1, len(mock_audio_data['data']))
        noisy_audio = mock_audio_data['data'] + noise

        # Process noise reduction
        processed_audio = await processor.reduce_background_noise(noisy_audio)

        assert processed_audio is not None
        assert len(processed_audio) == len(noisy_audio)
        # In a real test, we'd verify noise reduction effectiveness

    def test_voice_activity_detection(self, processor, mock_audio_data):
        """Test voice activity detection."""
        # Test with speech audio
        is_speech = processor.detect_voice_activity(mock_audio_data['data'])
        assert isinstance(is_speech, bool)

        # Test with silence
        silence = np.zeros(1000, dtype=np.int16)
        is_speech_silence = processor.detect_voice_activity(silence)
        assert isinstance(is_speech_silence, bool)

    def test_audio_format_conversion(self, processor, mock_audio_data):
        """Test audio format conversion."""
        # Convert to different format
        converted = processor.convert_audio_format(
            mock_audio_data['data'],
            from_format='int16',
            to_format='float32'
        )

        assert converted is not None
        assert converted.dtype == np.float32
        assert len(converted) == len(mock_audio_data['data'])

    def test_audio_quality_metrics(self, processor, mock_audio_data):
        """Test audio quality metrics calculation."""
        metrics = processor.calculate_audio_quality_metrics(mock_audio_data['data'])

        assert 'rms_energy' in metrics
        assert 'zero_crossing_rate' in metrics
        assert 'spectral_centroid' in metrics
        assert 'signal_to_noise_ratio' in metrics
        assert isinstance(metrics['rms_energy'], float)

    def test_audio_level_normalization(self, processor, mock_audio_data):
        """Test audio level normalization."""
        # Scale down audio
        quiet_audio = mock_audio_data['data'] // 4

        normalized = processor.normalize_audio_level(quiet_audio)

        assert normalized is not None
        assert len(normalized) == len(quiet_audio)
        # Verify normalization increased volume
        assert np.max(np.abs(normalized)) > np.max(np.abs(quiet_audio))

    def test_audio_buffer_management(self, processor):
        """Test audio buffer management."""
        # Add test data to buffer
        test_data = b'test_audio_data'
        processor.add_to_buffer(test_data)

        # Retrieve buffer contents
        buffer_contents = processor.get_buffer_contents()
        assert buffer_contents == test_data

        # Clear buffer
        processor.clear_buffer()
        assert len(processor.get_buffer_contents()) == 0

    def test_error_handling(self, processor):
        """Test error handling scenarios."""
        # Test recording without audio device
        processor.audio = None
        success = processor.start_recording()
        assert not success

        # Test stopping when not recording
        processor.recording = False
        processor.stream = None
        success = processor.stop_recording()
        assert not success

    def test_audio_device_selection(self, processor):
        """Test audio device selection."""
        # Mock available devices
        processor.input_devices = [
            {'name': 'Device 1', 'index': 0},
            {'name': 'Device 2', 'index': 1}
        ]

        # Select valid device
        success = processor.select_input_device(1)
        assert success
        assert processor.selected_input_device == 1

        # Select invalid device
        success = processor.select_input_device(99)
        assert not success

    @pytest.mark.parametrize("audio_format", ['wav', 'mp3', 'flac'])
    def test_audio_file_io(self, processor, mock_audio_data, audio_format):
        """Test audio file input/output operations."""
        with tempfile.NamedTemporaryFile(suffix=f'.{audio_format}', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Save audio to file
            success = processor.save_audio_to_file(
                mock_audio_data['data'],
                temp_path,
                format=audio_format
            )
            assert success
            assert os.path.exists(temp_path)

            # Load audio from file
            loaded_audio = processor.load_audio_from_file(temp_path)
            assert loaded_audio is not None
            assert len(loaded_audio) > 0

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_audio_streaming_capabilities(self, processor):
        """Test audio streaming capabilities."""
        # Test stream creation
        mock_stream = MagicMock()
        processor.audio = MagicMock()
        processor.audio.open.return_value = mock_stream

        stream = processor.create_audio_stream()
        assert stream == mock_stream
        processor.audio.open.assert_called_once()

    def test_audio_callback_function(self, processor):
        """Test audio callback function."""
        # Mock callback data
        in_data = b'test_audio_data'
        frame_count = 1024
        time_info = {'input_buffer_adc_time': 0.0}
        status_flags = 0

        # Test callback
        result = processor.audio_callback(
            in_data, frame_count, time_info, status_flags
        )

        assert isinstance(result, tuple)
        assert len(result) == 2  # (data, status_flags)

    def test_cleanup(self, processor):
        """Test cleanup functionality."""
        # Setup mock objects
        processor.audio = MagicMock()
        processor.stream = MagicMock()

        # Test cleanup
        processor.cleanup()

        processor.stream.stop_stream.assert_called_once()
        processor.stream.close.assert_called_once()
        processor.audio.terminate.assert_called_once()
        assert processor.audio is None
        assert processor.stream is None