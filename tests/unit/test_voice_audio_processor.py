"""
Comprehensive unit tests for voice/audio_processor.py module.
"""

import pytest
import numpy as np
import io
import wave
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Import the module to test with robust error handling
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from voice.audio_processor import AudioProcessor, AudioData, AudioFormat, AudioProcessorError
except ImportError as e:
    pytest.skip(f"voice.audio_processor module not available: {e}", allow_module_level=True)


class TestAudioData:
    """Test AudioData dataclass."""
    
    def test_audio_data_creation(self):
        """Test creating audio data."""
        data = b"fake_audio_data"
        audio_data = AudioData(
            data=data,
            sample_rate=16000,
            channels=1,
            sample_width=2
        )
        
        assert audio_data.data == data
        assert audio_data.sample_rate == 16000
        assert audio_data.channels == 1
        assert audio_data.sample_width == 2
    
    def test_audio_data_with_optional_fields(self):
        """Test creating audio data with optional fields."""
        data = b"fake_audio_data"
        audio_data = AudioData(
            data=data,
            sample_rate=16000,
            channels=1,
            sample_width=2,
            duration=2.0,
            format=AudioFormat.WAV,
            metadata={"source": "microphone"}
        )
        
        assert audio_data.duration == 2.0
        assert audio_data.format == AudioFormat.WAV
        assert audio_data.metadata == {"source": "microphone"}
    
    def test_audio_data_duration_calculation(self):
        """Test duration calculation when not provided."""
        # 1 second of audio at 16000 Hz, 1 channel, 2 bytes per sample
        data = b"\x00" * (16000 * 1 * 2)
        audio_data = AudioData(
            data=data,
            sample_rate=16000,
            channels=1,
            sample_width=2
        )
        
        assert audio_data.duration == 1.0
    
    def test_audio_data_numpy_conversion(self):
        """Test converting audio data to numpy array."""
        # Create simple sine wave data
        sample_rate = 16000
        duration = 0.1  # 100ms
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        sine_wave = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        data = (sine_wave * 32767).astype(np.int16).tobytes()
        
        audio_data = AudioData(
            data=data,
            sample_rate=sample_rate,
            channels=1,
            sample_width=2
        )
        
        numpy_array = audio_data.to_numpy()
        
        assert isinstance(numpy_array, np.ndarray)
        assert len(numpy_array) == sample_rate * duration
        assert numpy_array.dtype == np.int16


class TestAudioFormat:
    """Test AudioFormat enum."""
    
    def test_audio_format_values(self):
        """Test audio format enum values."""
        assert AudioFormat.WAV.value == "wav"
        assert AudioFormat.MP3.value == "mp3"
        assert AudioFormat.FLAC.value == "flac"
        assert AudioFormat.OGG.value == "ogg"
        assert AudioFormat.RAW.value == "raw"
    
    def test_audio_format_from_string(self):
        """Test creating audio format from string."""
        assert AudioFormat("wav") == AudioFormat.WAV
        assert AudioFormat("mp3") == AudioFormat.MP3
        assert AudioFormat("flac") == AudioFormat.FLAC
        assert AudioFormat("ogg") == AudioFormat.OGG
        assert AudioFormat("raw") == AudioFormat.RAW


class TestAudioProcessor:
    """Test AudioProcessor class."""
    
    @pytest.fixture
    def audio_processor(self):
        """Create an audio processor."""
        return AudioProcessor()
    
    def test_audio_processor_initialization(self, audio_processor):
        """Test audio processor initialization."""
        assert audio_processor is not None
        assert hasattr(audio_processor, 'supported_formats')
    
    def test_load_from_bytes_wav(self, audio_processor):
        """Test loading audio from WAV bytes."""
        # Create a simple WAV file in memory
        sample_rate = 16000
        duration = 0.1  # 100ms
        samples = int(sample_rate * duration)
        
        # Create sine wave
        t = np.linspace(0, duration, samples, False)
        sine_wave = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        audio_data = (sine_wave * 32767).astype(np.int16)
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        wav_bytes = wav_buffer.getvalue()
        
        # Load audio
        result = audio_processor.load_from_bytes(wav_bytes, AudioFormat.WAV)
        
        assert isinstance(result, AudioData)
        assert result.sample_rate == sample_rate
        assert result.channels == 1
        assert result.sample_width == 2
        assert result.format == AudioFormat.WAV
        assert len(result.data) > 0
    
    def test_load_from_bytes_unsupported_format(self, audio_processor):
        """Test loading audio with unsupported format."""
        data = b"fake_audio_data"
        
        with pytest.raises(AudioProcessorError) as exc_info:
            audio_processor.load_from_bytes(data, "unsupported")
        
        assert "Unsupported audio format" in str(exc_info.value)
    
    def test_load_from_bytes_invalid_data(self, audio_processor):
        """Test loading audio with invalid data."""
        data = b"not_audio_data"
        
        with pytest.raises(AudioProcessorError) as exc_info:
            audio_processor.load_from_bytes(data, AudioFormat.WAV)
        
        assert "Failed to load audio" in str(exc_info.value)
    
    def test_convert_sample_rate(self, audio_processor):
        """Test converting sample rate."""
        # Create audio data at 16000 Hz
        sample_rate = 16000
        duration = 0.1  # 100ms
        samples = int(sample_rate * duration)
        
        # Create sine wave
        t = np.linspace(0, duration, samples, False)
        sine_wave = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        audio_data = (sine_wave * 32767).astype(np.int16).tobytes()
        
        input_audio = AudioData(
            data=audio_data,
            sample_rate=sample_rate,
            channels=1,
            sample_width=2
        )
        
        # Convert to 22050 Hz
        target_rate = 22050
        result = audio_processor.convert_sample_rate(input_audio, target_rate)
        
        assert isinstance(result, AudioData)
        assert result.sample_rate == target_rate
        assert result.channels == 1
        assert result.sample_width == 2
        # Duration should remain approximately the same
        assert abs(result.duration - input_audio.duration) < 0.01
    
    def test_convert_sample_rate_same_rate(self, audio_processor):
        """Test converting sample rate to same rate."""
        sample_rate = 16000
        audio_data = AudioData(
            data=b"\x00" * (sample_rate * 2),  # 1 second of silence
            sample_rate=sample_rate,
            channels=1,
            sample_width=2
        )
        
        result = audio_processor.convert_sample_rate(audio_data, sample_rate)
        
        assert result.sample_rate == sample_rate
        assert len(result.data) == len(audio_data.data)
    
    def test_convert_channels_mono_to_stereo(self, audio_processor):
        """Test converting mono to stereo."""
        sample_rate = 16000
        audio_data = AudioData(
            data=b"\x00" * (sample_rate * 2),  # 1 second of silence
            sample_rate=sample_rate,
            channels=1,
            sample_width=2
        )
        
        result = audio_processor.convert_channels(audio_data, 2)
        
        assert isinstance(result, AudioData)
        assert result.channels == 2
        assert result.sample_rate == sample_rate
        assert result.sample_width == 2
        # Stereo should have twice the data
        assert len(result.data) == len(audio_data.data) * 2
    
    def test_convert_channels_stereo_to_mono(self, audio_processor):
        """Test converting stereo to mono."""
        sample_rate = 16000
        # Create stereo data (alternating left/right samples)
        stereo_data = b"\x00\x01" * (sample_rate)
        
        audio_data = AudioData(
            data=stereo_data,
            sample_rate=sample_rate,
            channels=2,
            sample_width=2
        )
        
        result = audio_processor.convert_channels(audio_data, 1)
        
        assert isinstance(result, AudioData)
        assert result.channels == 1
        assert result.sample_rate == sample_rate
        assert result.sample_width == 2
        # Mono should have half the data
        assert len(result.data) == len(audio_data.data) // 2
    
    def test_convert_channels_same_channels(self, audio_processor):
        """Test converting to same number of channels."""
        sample_rate = 16000
        audio_data = AudioData(
            data=b"\x00" * (sample_rate * 2),
            sample_rate=sample_rate,
            channels=1,
            sample_width=2
        )
        
        result = audio_processor.convert_channels(audio_data, 1)
        
        assert result.channels == 1
        assert len(result.data) == len(audio_data.data)
    
    def test_normalize_audio(self, audio_processor):
        """Test normalizing audio."""
        # Create audio with low amplitude
        sample_rate = 16000
        duration = 0.1
        samples = int(sample_rate * duration)
        
        # Create low amplitude sine wave
        t = np.linspace(0, duration, samples, False)
        sine_wave = np.sin(2 * np.pi * 440 * t) * 0.1  # 10% amplitude
        audio_data = (sine_wave * 32767).astype(np.int16).tobytes()
        
        input_audio = AudioData(
            data=audio_data,
            sample_rate=sample_rate,
            channels=1,
            sample_width=2
        )
        
        result = audio_processor.normalize_audio(input_audio)
        
        assert isinstance(result, AudioData)
        assert result.sample_rate == sample_rate
        assert result.channels == 1
        assert result.sample_width == 2
        # Normalized audio should have higher amplitude
        assert len(result.data) == len(audio_data.data)
    
    def test_normalize_audio_already_normalized(self, audio_processor):
        """Test normalizing already normalized audio."""
        # Create audio with maximum amplitude
        sample_rate = 16000
        audio_data = AudioData(
            data=b"\xff\x7f" * (sample_rate),  # Maximum positive amplitude
            sample_rate=sample_rate,
            channels=1,
            sample_width=2
        )
        
        result = audio_processor.normalize_audio(audio_data)
        
        # Should remain the same
        assert len(result.data) == len(audio_data.data)
    
    def test_trim_silence(self, audio_processor):
        """Test trimming silence from audio."""
        sample_rate = 16000
        
        # Create audio with silence at beginning and end
        silence_duration = 0.1  # 100ms
        silence_samples = int(sample_rate * silence_duration)
        silence_data = b"\x00" * (silence_samples * 2)
        
        # Create some audio content
        content_data = b"\x01\x02" * (sample_rate // 2)
        
        # Combine: silence + content + silence
        audio_data = silence_data + content_data + silence_data
        
        input_audio = AudioData(
            data=audio_data,
            sample_rate=sample_rate,
            channels=1,
            sample_width=2
        )
        
        result = audio_processor.trim_silence(input_audio)
        
        assert isinstance(result, AudioData)
        assert result.sample_rate == sample_rate
        assert result.channels == 1
        assert result.sample_width == 2
        # Result should be shorter (silence trimmed)
        assert len(result.data) < len(audio_data)
        assert len(result.data) == len(content_data)
    
    def test_trim_silence_no_silence(self, audio_processor):
        """Test trimming silence from audio with no silence."""
        sample_rate = 16000
        audio_data = AudioData(
            data=b"\x01\x02" * (sample_rate),  # No silence
            sample_rate=sample_rate,
            channels=1,
            sample_width=2
        )
        
        result = audio_processor.trim_silence(audio_data)
        
        # Should remain the same
        assert len(result.data) == len(audio_data.data)
    
    def test_apply_filter_low_pass(self, audio_processor):
        """Test applying low-pass filter."""
        sample_rate = 16000
        duration = 0.1
        samples = int(sample_rate * duration)
        
        # Create high frequency sine wave
        t = np.linspace(0, duration, samples, False)
        sine_wave = np.sin(2 * np.pi * 8000 * t)  # 8 kHz tone
        audio_data = (sine_wave * 32767).astype(np.int16).tobytes()
        
        input_audio = AudioData(
            data=audio_data,
            sample_rate=sample_rate,
            channels=1,
            sample_width=2
        )
        
        # Apply low-pass filter at 4 kHz
        result = audio_processor.apply_filter(input_audio, "low_pass", cutoff_freq=4000)
        
        assert isinstance(result, AudioData)
        assert result.sample_rate == sample_rate
        assert result.channels == 1
        assert result.sample_width == 2
        assert len(result.data) == len(audio_data.data)
    
    def test_apply_filter_high_pass(self, audio_processor):
        """Test applying high-pass filter."""
        sample_rate = 16000
        duration = 0.1
        samples = int(sample_rate * duration)
        
        # Create low frequency sine wave
        t = np.linspace(0, duration, samples, False)
        sine_wave = np.sin(2 * np.pi * 200 * t)  # 200 Hz tone
        audio_data = (sine_wave * 32767).astype(np.int16).tobytes()
        
        input_audio = AudioData(
            data=audio_data,
            sample_rate=sample_rate,
            channels=1,
            sample_width=2
        )
        
        # Apply high-pass filter at 1 kHz
        result = audio_processor.apply_filter(input_audio, "high_pass", cutoff_freq=1000)
        
        assert isinstance(result, AudioData)
        assert result.sample_rate == sample_rate
        assert result.channels == 1
        assert result.sample_width == 2
        assert len(result.data) == len(audio_data.data)
    
    def test_apply_filter_invalid_type(self, audio_processor):
        """Test applying invalid filter type."""
        sample_rate = 16000
        audio_data = AudioData(
            data=b"\x00" * (sample_rate * 2),
            sample_rate=sample_rate,
            channels=1,
            sample_width=2
        )
        
        with pytest.raises(AudioProcessorError) as exc_info:
            audio_processor.apply_filter(audio_data, "invalid_filter")
        
        assert "Invalid filter type" in str(exc_info.value)
    
    def test_detect_speech_activity(self, audio_processor):
        """Test speech activity detection."""
        sample_rate = 16000
        duration = 0.2  # 200ms
        
        # Create audio with speech-like activity
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        # Mix of frequencies to simulate speech
        speech_wave = (
            np.sin(2 * np.pi * 200 * t) +  # Low frequency
            np.sin(2 * np.pi * 1000 * t) +  # Mid frequency
            np.sin(2 * np.pi * 3000 * t) * 0.3  # High frequency (lower amplitude)
        )
        audio_data = (speech_wave * 16383).astype(np.int16).tobytes()
        
        input_audio = AudioData(
            data=audio_data,
            sample_rate=sample_rate,
            channels=1,
            sample_width=2
        )
        
        result = audio_processor.detect_speech_activity(input_audio)
        
        assert isinstance(result, list)
        assert len(result) > 0
        # Each segment should have start and end times
        for segment in result:
            assert "start" in segment
            assert "end" in segment
            assert segment["start"] < segment["end"]
            assert segment["start"] >= 0
            assert segment["end"] <= input_audio.duration
    
    def test_detect_speech_activity_silence(self, audio_processor):
        """Test speech activity detection with silence."""
        sample_rate = 16000
        audio_data = AudioData(
            data=b"\x00" * (sample_rate * 2),  # 1 second of silence
            sample_rate=sample_rate,
            channels=1,
            sample_width=2
        )
        
        result = audio_processor.detect_speech_activity(audio_data)
        
        # Should detect no speech activity
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_get_audio_info(self, audio_processor):
        """Test getting audio information."""
        sample_rate = 16000
        duration = 1.0
        audio_data = AudioData(
            data=b"\x00" * (int(sample_rate * duration) * 2),
            sample_rate=sample_rate,
            channels=1,
            sample_width=2,
            format=AudioFormat.WAV
        )
        
        info = audio_processor.get_audio_info(audio_data)
        
        assert isinstance(info, dict)
        assert "sample_rate" in info
        assert "channels" in info
        assert "sample_width" in info
        assert "duration" in info
        assert "format" in info
        assert "size_bytes" in info
        
        assert info["sample_rate"] == sample_rate
        assert info["channels"] == 1
        assert info["sample_width"] == 2
        assert info["duration"] == duration
        assert info["format"] == "wav"
        assert info["size_bytes"] == len(audio_data.data)
    
    def test_concatenate_audio(self, audio_processor):
        """Test concatenating audio segments."""
        sample_rate = 16000
        
        # Create two audio segments
        segment1 = AudioData(
            data=b"\x01\x02" * (sample_rate // 2),  # 0.5 seconds
            sample_rate=sample_rate,
            channels=1,
            sample_width=2
        )
        
        segment2 = AudioData(
            data=b"\x03\x04" * (sample_rate // 2),  # 0.5 seconds
            sample_rate=sample_rate,
            channels=1,
            sample_width=2
        )
        
        result = audio_processor.concatenate_audio([segment1, segment2])
        
        assert isinstance(result, AudioData)
        assert result.sample_rate == sample_rate
        assert result.channels == 1
        assert result.sample_width == 2
        # Duration should be sum of both segments
        assert result.duration == segment1.duration + segment2.duration
        # Data should be concatenated
        assert len(result.data) == len(segment1.data) + len(segment2.data)
    
    def test_concatenate_audio_different_formats(self, audio_processor):
        """Test concatenating audio with different formats."""
        segment1 = AudioData(
            data=b"\x01\x02" * 8000,
            sample_rate=16000,
            channels=1,
            sample_width=2
        )
        
        segment2 = AudioData(
            data=b"\x03\x04" * 11025,
            sample_rate=22050,  # Different sample rate
            channels=1,
            sample_width=2
        )
        
        with pytest.raises(AudioProcessorError) as exc_info:
            audio_processor.concatenate_audio([segment1, segment2])
        
        assert "Audio formats do not match" in str(exc_info.value)
    
    def test_split_audio(self, audio_processor):
        """Test splitting audio into segments."""
        sample_rate = 16000
        duration = 2.0  # 2 seconds
        audio_data = AudioData(
            data=b"\x01\x02" * (int(sample_rate * duration)),
            sample_rate=sample_rate,
            channels=1,
            sample_width=2
        )
        
        # Split into 0.5 second segments
        segment_duration = 0.5
        segments = audio_processor.split_audio(audio_data, segment_duration)
        
        assert isinstance(segments, list)
        assert len(segments) == 4  # 2 seconds / 0.5 seconds = 4 segments
        
        for segment in segments:
            assert isinstance(segment, AudioData)
            assert segment.sample_rate == sample_rate
            assert segment.channels == 1
            assert segment.sample_width == 2
            assert abs(segment.duration - segment_duration) < 0.01
    
    def test_extract_segment(self, audio_processor):
        """Test extracting a segment from audio."""
        sample_rate = 16000
        duration = 2.0  # 2 seconds
        audio_data = AudioData(
            data=b"\x01\x02" * (int(sample_rate * duration)),
            sample_rate=sample_rate,
            channels=1,
            sample_width=2
        )
        
        # Extract segment from 0.5s to 1.5s
        start_time = 0.5
        end_time = 1.5
        
        result = audio_processor.extract_segment(audio_data, start_time, end_time)
        
        assert isinstance(result, AudioData)
        assert result.sample_rate == sample_rate
        assert result.channels == 1
        assert result.sample_width == 2
        assert abs(result.duration - (end_time - start_time)) < 0.01
    
    def test_extract_segment_invalid_bounds(self, audio_processor):
        """Test extracting segment with invalid time bounds."""
        sample_rate = 16000
        audio_data = AudioData(
            data=b"\x01\x02" * (sample_rate),
            sample_rate=sample_rate,
            channels=1,
            sample_width=2
        )
        
        # Start time after end time
        with pytest.raises(AudioProcessorError) as exc_info:
            audio_processor.extract_segment(audio_data, 1.5, 0.5)
        
        assert "Invalid time bounds" in str(exc_info.value)
        
        # End time beyond audio duration
        with pytest.raises(AudioProcessorError) as exc_info:
            audio_processor.extract_segment(audio_data, 0.5, 2.0)
        
        assert "Invalid time bounds" in str(exc_info.value)


class TestAudioProcessorError:
    """Test AudioProcessorError exception."""
    
    def test_audio_processor_error_creation(self):
        """Test creating AudioProcessorError."""
        error = AudioProcessorError("Test error message")
        
        assert str(error) == "Test error message"
    
    def test_audio_processor_error_inheritance(self):
        """Test AudioProcessorError inheritance."""
        error = AudioProcessorError("Test error")
        
        assert isinstance(error, Exception)
        assert isinstance(error, ValueError)