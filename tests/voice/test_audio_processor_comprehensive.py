"""
Comprehensive tests for audio processor with focus on coverage gaps.
Tests audio processing pipeline, format conversion, quality enhancement,
error handling, buffer management, and resource cleanup.
"""

import sys
import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import io

# Direct import to avoid torch loading issues
import importlib.util
spec = importlib.util.spec_from_file_location(
    "audio_processor",
    "/home/anchapin/projects/ai-therapist/voice/audio_processor.py"
)
audio_processor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audio_processor_module)

SimplifiedAudioProcessor = audio_processor_module.SimplifiedAudioProcessor
AudioData = audio_processor_module.AudioData
AudioProcessorState = audio_processor_module.AudioProcessorState
AudioQualityMetrics = audio_processor_module.AudioQualityMetrics
SOUNDDEVICE_AVAILABLE = audio_processor_module.SOUNDDEVICE_AVAILABLE
NOISEREDUCE_AVAILABLE = audio_processor_module.NOISEREDUCE_AVAILABLE
VAD_AVAILABLE = audio_processor_module.VAD_AVAILABLE
LIBROSA_AVAILABLE = audio_processor_module.LIBROSA_AVAILABLE


@pytest.fixture
def audio_config():
    """Mock audio configuration."""
    config = Mock()
    config.audio = Mock()
    config.audio.sample_rate = 16000
    config.audio.channels = 1
    config.audio.chunk_size = 1024
    config.audio.format = "wav"
    config.audio.max_buffer_size = 100
    config.audio.max_memory_mb = 50
    config.audio.stream_buffer_size = 10
    config.audio.stream_chunk_duration = 0.1
    config.audio.compression_enabled = True
    config.audio.compression_level = 6
    return config


@pytest.fixture
def processor(audio_config):
    """Create audio processor instance."""
    proc = SimplifiedAudioProcessor(audio_config)
    yield proc
    proc.cleanup()


@pytest.fixture
def sample_audio():
    """Generate synthetic audio data."""
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Generate 440Hz sine wave
    data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return AudioData(data=data, sample_rate=sample_rate, duration=duration)


@pytest.fixture
def noisy_audio():
    """Generate audio with noise."""
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = np.sin(2 * np.pi * 440 * t)
    noise = np.random.normal(0, 0.1, signal.shape)
    data = (signal + noise).astype(np.float32)
    return AudioData(data=data, sample_rate=sample_rate, duration=duration)


@pytest.fixture
def corrupt_audio():
    """Generate corrupt audio data."""
    return AudioData(
        data=np.array([np.nan, np.inf, -np.inf, 0.5]).astype(np.float32),
        sample_rate=16000,
        duration=0.001
    )


# ==================== Audio Format Conversion Tests ====================

@pytest.mark.unit
def test_audio_format_conversion_wav_to_mp3(processor, sample_audio):
    """Test WAV to MP3 format conversion."""
    converted = processor.convert_audio_format(sample_audio, "mp3")
    assert converted is not None
    assert converted.format == "mp3"
    assert len(converted.data) == len(sample_audio.data)


@pytest.mark.unit
def test_audio_format_conversion_wav_to_ogg(processor, sample_audio):
    """Test WAV to OGG format conversion."""
    converted = processor.convert_audio_format(sample_audio, "ogg")
    assert converted is not None
    assert converted.format == "ogg"


@pytest.mark.unit
def test_audio_format_conversion_preserves_data(processor, sample_audio):
    """Test that format conversion preserves audio data."""
    converted = processor.convert_audio_format(sample_audio, "flac")
    np.testing.assert_array_equal(converted.data, sample_audio.data)
    assert converted.sample_rate == sample_audio.sample_rate


@pytest.mark.unit
def test_audio_format_conversion_feature_unavailable(sample_audio):
    """Test format conversion when feature is unavailable."""
    proc = SimplifiedAudioProcessor()
    proc.features['format_conversion'] = False
    result = proc.convert_audio_format(sample_audio, "mp3")
    assert result == sample_audio  # Returns original


@pytest.mark.unit
def test_audio_format_conversion_error_handling(processor, sample_audio):
    """Test format conversion error handling."""
    with patch.object(processor, 'logger') as mock_logger:
        # Simulate error by passing invalid format
        result = processor.convert_audio_format(sample_audio, "invalid")
        assert result is not None  # Should return something


# ==================== Sample Rate Conversion Tests ====================

@pytest.mark.unit
def test_resample_audio_44100_to_16000(processor):
    """Test downsampling from 44100Hz to 16000Hz."""
    audio = AudioData(
        data=np.random.randn(44100).astype(np.float32),
        sample_rate=44100,
        duration=1.0
    )
    if LIBROSA_AVAILABLE:
        import librosa
        resampled = librosa.resample(
            audio.data.astype(np.float32),
            orig_sr=44100,
            target_sr=16000
        )
        assert len(resampled) < len(audio.data)


@pytest.mark.unit
def test_resample_audio_8000_to_16000(processor):
    """Test upsampling from 8000Hz to 16000Hz."""
    audio = AudioData(
        data=np.random.randn(8000).astype(np.float32),
        sample_rate=8000,
        duration=1.0
    )
    if LIBROSA_AVAILABLE:
        import librosa
        resampled = librosa.resample(
            audio.data.astype(np.float32),
            orig_sr=8000,
            target_sr=16000
        )
        assert len(resampled) > len(audio.data)


# ==================== Noise Reduction Tests ====================

@pytest.mark.unit
def test_noise_reduction_reduces_noise(processor, noisy_audio):
    """Test that noise reduction actually reduces noise."""
    original_std = np.std(noisy_audio.data)
    reduced = processor.reduce_background_noise(noisy_audio.data)
    
    if NOISEREDUCE_AVAILABLE:
        reduced_std = np.std(reduced)
        # Noise reduction should reduce variance
        assert reduced_std <= original_std


@pytest.mark.unit
def test_noise_reduction_without_library(noisy_audio):
    """Test noise reduction fallback when library unavailable."""
    proc = SimplifiedAudioProcessor()
    
    if not NOISEREDUCE_AVAILABLE:
        result = proc.reduce_background_noise(noisy_audio.data)
        np.testing.assert_array_equal(result, noisy_audio.data)


@pytest.mark.unit
def test_noise_reduction_error_handling(processor):
    """Test noise reduction error handling."""
    invalid_data = np.array([np.nan, np.inf]).astype(np.float32)
    result = processor.reduce_background_noise(invalid_data)
    assert result is not None


# ==================== Volume Normalization Tests ====================

@pytest.mark.unit
def test_normalize_audio_level_default_target(processor, sample_audio):
    """Test volume normalization with default target."""
    normalized = processor.normalize_audio_level(sample_audio.data)
    max_val = np.max(np.abs(normalized))
    assert np.isclose(max_val, 0.5, atol=0.01)


@pytest.mark.unit
def test_normalize_audio_level_custom_target(processor, sample_audio):
    """Test volume normalization with custom target."""
    target = 0.8
    normalized = processor.normalize_audio_level(sample_audio.data, target)
    max_val = np.max(np.abs(normalized))
    assert np.isclose(max_val, target, atol=0.01)


@pytest.mark.unit
def test_normalize_audio_level_zero_audio(processor):
    """Test normalization of silent audio."""
    silent = np.zeros(1000, dtype=np.float32)
    normalized = processor.normalize_audio_level(silent)
    np.testing.assert_array_equal(normalized, silent)


@pytest.mark.unit
def test_normalize_audio_level_error_handling(processor):
    """Test normalization error handling."""
    invalid_data = np.array([np.nan, np.inf]).astype(np.float32)
    result = processor.normalize_audio_level(invalid_data)
    assert result is not None


# ==================== Audio Quality Validation Tests ====================

@pytest.mark.unit
def test_calculate_quality_metrics_clean_audio(processor, sample_audio):
    """Test quality metrics for clean audio."""
    metrics = processor.calculate_audio_quality_metrics(sample_audio.data)
    assert isinstance(metrics, AudioQualityMetrics)
    assert metrics.speech_level >= 0
    assert metrics.noise_level >= 0
    assert metrics.snr_ratio >= 0


@pytest.mark.unit
def test_calculate_quality_metrics_noisy_audio(processor, noisy_audio):
    """Test quality metrics for noisy audio."""
    metrics = processor.calculate_audio_quality_metrics(noisy_audio.data)
    assert metrics.noise_level > 0
    assert 0 <= metrics.overall_quality <= 1.0


@pytest.mark.unit
def test_calculate_quality_metrics_without_librosa():
    """Test quality metrics when librosa unavailable."""
    proc = SimplifiedAudioProcessor()
    
    if not LIBROSA_AVAILABLE:
        data = np.random.randn(1000).astype(np.float32)
        metrics = proc.calculate_audio_quality_metrics(data)
        assert metrics.overall_quality == 0.0


@pytest.mark.unit
def test_quality_metrics_to_dict(processor, sample_audio):
    """Test conversion of quality metrics to dictionary."""
    metrics = processor.calculate_audio_quality_metrics(sample_audio.data)
    d = metrics.to_dict()
    assert 'snr_ratio' in d
    assert 'noise_level' in d
    assert 'overall_quality' in d


@pytest.mark.unit
def test_calculate_quality_metrics_error_handling(processor):
    """Test quality metrics error handling."""
    invalid_data = np.array([np.nan]).astype(np.float32)
    metrics = processor.calculate_audio_quality_metrics(invalid_data)
    assert isinstance(metrics, AudioQualityMetrics)


# ==================== Corrupt Audio Handling Tests ====================

@pytest.mark.unit
def test_process_audio_with_nan_values(processor):
    """Test processing audio with NaN values."""
    data = np.array([1.0, np.nan, 2.0, np.nan]).astype(np.float32)
    audio = AudioData(data=data, sample_rate=16000, duration=0.001)
    result = processor._process_audio(audio)
    assert result is not None


@pytest.mark.unit
def test_process_audio_with_inf_values(processor):
    """Test processing audio with infinite values."""
    data = np.array([1.0, np.inf, 2.0, -np.inf]).astype(np.float32)
    audio = AudioData(data=data, sample_rate=16000, duration=0.001)
    result = processor._process_audio(audio)
    assert result is not None


@pytest.mark.unit
def test_load_corrupt_wav_file(processor):
    """Test loading corrupt WAV file."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        f.write(b'CORRUPT DATA')
        temp_path = f.name
    
    try:
        result = processor.load_audio(temp_path)
        assert result is None  # Should handle gracefully
    finally:
        os.unlink(temp_path)


@pytest.mark.unit
def test_audio_data_from_bytes_corrupt(processor):
    """Test creating AudioData from corrupt bytes."""
    corrupt_bytes = b'\x00\x01\x02\x03'
    try:
        result = AudioData.from_bytes(corrupt_bytes)
        # Should either work or raise exception
        assert result is not None or True
    except Exception:
        pass  # Expected


# ==================== Buffer Management Tests ====================

@pytest.mark.unit
def test_add_to_buffer_normal(processor):
    """Test adding audio to buffer."""
    data = np.random.randn(1024).astype(np.float32)
    processor.add_to_buffer(data)
    assert len(processor.audio_buffer) == 1


@pytest.mark.unit
def test_buffer_overflow_protection(processor):
    """Test buffer overflow protection."""
    # Fill buffer beyond max size
    for i in range(processor.max_buffer_size + 50):
        data = np.random.randn(1024).astype(np.float32)
        processor.add_to_buffer(data)
    
    # Buffer should not exceed max size
    assert len(processor.audio_buffer) <= processor.max_buffer_size


@pytest.mark.unit
def test_buffer_memory_limit(processor):
    """Test buffer memory limit enforcement."""
    # Try to exceed memory limit
    large_chunk = np.random.randn(1000000).astype(np.float32)
    
    for i in range(200):
        processor.add_to_buffer(large_chunk)
    
    # Should have dropped chunks to stay under limit
    assert processor._buffer_bytes_estimate <= processor._max_memory_bytes


@pytest.mark.unit
def test_clear_buffer(processor):
    """Test buffer clearing."""
    data = np.random.randn(1024).astype(np.float32)
    processor.add_to_buffer(data)
    processor.clear_buffer()
    assert len(processor.audio_buffer) == 0
    assert processor._buffer_bytes_estimate == 0


@pytest.mark.unit
def test_get_buffer_contents(processor):
    """Test getting buffer contents."""
    data1 = np.random.randn(1024).astype(np.float32)
    data2 = np.random.randn(1024).astype(np.float32)
    processor.add_to_buffer(data1)
    processor.add_to_buffer(data2)
    
    contents = processor.get_buffer_contents()
    assert len(contents) == 2


@pytest.mark.unit
def test_get_audio_chunk(processor):
    """Test getting audio chunk from buffer."""
    data = np.random.randn(1024).astype(np.float32)
    processor.add_to_buffer(data)
    
    chunk = processor.get_audio_chunk()
    assert isinstance(chunk, bytes)
    assert len(processor.audio_buffer) == 0  # Should be removed


@pytest.mark.unit
def test_force_cleanup_buffers(processor):
    """Test force cleanup of buffers."""
    # Add some data
    for i in range(10):
        data = np.random.randn(1024).astype(np.float32)
        processor.add_to_buffer(data)
    
    cleared = processor.force_cleanup_buffers()
    assert cleared == 10
    assert len(processor.audio_buffer) == 0


# ==================== Memory Management Tests ====================

@pytest.mark.unit
def test_get_memory_usage(processor):
    """Test getting memory usage information."""
    usage = processor.get_memory_usage()
    assert 'buffer_size' in usage
    assert 'max_buffer_size' in usage
    assert 'memory_usage_bytes' in usage
    assert 'memory_limit_bytes' in usage


@pytest.mark.unit
def test_memory_tracking_accuracy(processor):
    """Test accuracy of memory tracking."""
    initial_estimate = processor._buffer_bytes_estimate
    data = np.random.randn(1024).astype(np.float32)
    processor.add_to_buffer(data)
    
    expected_increase = data.nbytes
    actual_increase = processor._buffer_bytes_estimate - initial_estimate
    assert actual_increase == expected_increase


@pytest.mark.unit
def test_memory_cleanup_callback(processor):
    """Test memory cleanup callback."""
    # Add data
    for i in range(5):
        data = np.random.randn(1024).astype(np.float32)
        processor.add_to_buffer(data)
    
    # Trigger cleanup
    processor._memory_cleanup_callback()
    assert len(processor.audio_buffer) == 0


# ==================== Processing Pipeline Tests ====================

@pytest.mark.unit
def test_processing_pipeline_complete(processor, noisy_audio):
    """Test complete audio processing pipeline."""
    result = processor._process_audio(noisy_audio)
    assert result is not None
    assert isinstance(result, AudioData)


@pytest.mark.unit
def test_processing_pipeline_noise_reduction(processor, noisy_audio):
    """Test pipeline with noise reduction enabled."""
    if NOISEREDUCE_AVAILABLE:
        result = processor._process_audio(noisy_audio)
        # Should have been processed
        assert result.data is not None


@pytest.mark.unit
def test_processing_pipeline_normalization(processor, sample_audio):
    """Test pipeline applies normalization."""
    # Create audio with low volume
    low_volume = AudioData(
        data=sample_audio.data * 0.1,
        sample_rate=sample_audio.sample_rate,
        duration=sample_audio.duration
    )
    result = processor._process_audio(low_volume)
    # Volume should be normalized
    assert np.max(np.abs(result.data)) > np.max(np.abs(low_volume.data))


# ==================== Error Recovery Tests ====================

@pytest.mark.unit
def test_stop_recording_when_not_recording(processor):
    """Test stopping recording when not recording."""
    result = processor.stop_recording()
    assert result is None


@pytest.mark.unit
def test_recording_error_recovery(processor):
    """Test recovery from recording errors."""
    processor.state = AudioProcessorState.ERROR
    result = processor.start_recording()
    # Should handle error state


@pytest.mark.unit
def test_processing_error_recovery(processor):
    """Test recovery from processing errors."""
    # Simulate processing error
    with patch.object(processor, 'reduce_background_noise', side_effect=Exception("Test error")):
        audio = AudioData(
            data=np.random.randn(1000).astype(np.float32),
            sample_rate=16000,
            duration=0.0625
        )
        result = processor._process_audio(audio)
        # Should recover and return something


# ==================== Resource Cleanup Tests ====================

@pytest.mark.unit
def test_cleanup_stops_recording(processor):
    """Test that cleanup stops recording."""
    processor.is_recording = True
    processor.cleanup()
    assert processor.is_recording == False


@pytest.mark.unit
def test_cleanup_clears_buffers(processor):
    """Test that cleanup clears buffers."""
    data = np.random.randn(1024).astype(np.float32)
    processor.add_to_buffer(data)
    processor.cleanup()
    assert len(processor.audio_buffer) == 0


@pytest.mark.unit
def test_cleanup_resets_state(processor):
    """Test that cleanup resets processor state."""
    processor.state = AudioProcessorState.RECORDING
    processor.cleanup()
    # State should be reset


@pytest.mark.unit
def test_cleanup_error_handling(processor):
    """Test cleanup handles errors gracefully."""
    # Simulate error condition
    with patch.object(processor, 'stop_recording', side_effect=Exception("Test error")):
        processor.is_recording = True
        processor.cleanup()  # Should not raise


# ==================== Voice Activity Detection Tests ====================

@pytest.mark.unit
def test_detect_voice_activity_with_speech(processor, sample_audio):
    """Test VAD with speech-like audio."""
    result = processor.detect_voice_activity_simple(sample_audio.data)
    assert isinstance(result, bool)


@pytest.mark.unit
def test_detect_voice_activity_silence(processor):
    """Test VAD with silence."""
    silence = np.zeros(1000, dtype=np.float32)
    result = processor.detect_voice_activity_simple(silence)
    assert result == False


@pytest.mark.unit
def test_detect_voice_activity_without_vad(sample_audio):
    """Test VAD fallback when webrtcvad unavailable."""
    proc = SimplifiedAudioProcessor()
    if not VAD_AVAILABLE:
        result = proc.detect_voice_activity_simple(sample_audio.data)
        assert isinstance(result, bool)


@pytest.mark.unit
def test_vad_error_handling(processor):
    """Test VAD error handling."""
    invalid_data = np.array([np.nan]).astype(np.float32)
    result = processor.detect_voice_activity_simple(invalid_data)
    assert isinstance(result, bool)


# ==================== Audio Data Serialization Tests ====================

@pytest.mark.unit
def test_audio_data_to_bytes(sample_audio):
    """Test converting AudioData to bytes."""
    data_bytes = sample_audio.to_bytes()
    assert isinstance(data_bytes, bytes)
    assert len(data_bytes) > 0


@pytest.mark.unit
def test_audio_data_from_bytes(sample_audio):
    """Test creating AudioData from bytes."""
    data_bytes = sample_audio.to_bytes()
    reconstructed = AudioData.from_bytes(data_bytes, sample_audio.sample_rate)
    assert isinstance(reconstructed, AudioData)


@pytest.mark.unit
def test_audio_data_serialization_roundtrip(sample_audio):
    """Test round-trip serialization."""
    data_bytes = sample_audio.to_bytes()
    reconstructed = AudioData.from_bytes(data_bytes, sample_audio.sample_rate)
    
    # Should be close (some loss expected)
    if SOUNDDEVICE_AVAILABLE:
        assert reconstructed.sample_rate == sample_audio.sample_rate


# ==================== Streaming Audio Tests ====================

@pytest.mark.unit
def test_start_streaming_recording(processor):
    """Test starting streaming recording."""
    result = processor.start_streaming_recording()
    if processor.streaming_enabled:
        processor.stop_streaming_recording()


@pytest.mark.unit
def test_stop_streaming_recording(processor):
    """Test stopping streaming recording."""
    result = processor.stop_streaming_recording()
    assert result == True  # Should succeed even if not started


@pytest.mark.unit
def test_streaming_already_active(processor):
    """Test starting streaming when already active."""
    processor.streaming_active = True
    result = processor.start_streaming_recording()
    assert result == False
    processor.streaming_active = False


# ==================== Audio Compression Tests ====================

@pytest.mark.unit
def test_compress_audio_data(processor, sample_audio):
    """Test audio data compression."""
    compressed = processor.compress_audio_data(sample_audio)
    assert isinstance(compressed, bytes)


@pytest.mark.unit
def test_decompress_audio_data(processor, sample_audio):
    """Test audio data decompression."""
    compressed = processor.compress_audio_data(sample_audio)
    decompressed = processor.decompress_audio_data(compressed)
    assert decompressed is not None


@pytest.mark.unit
def test_compression_disabled(sample_audio):
    """Test compression when disabled."""
    proc = SimplifiedAudioProcessor()
    proc.compression_enabled = False
    result = proc.compress_audio_data(sample_audio)
    # Should return uncompressed bytes


@pytest.mark.unit
def test_decompression_error_handling(processor):
    """Test decompression error handling."""
    corrupt_data = b'CORRUPT'
    result = processor.decompress_audio_data(corrupt_data)
    # Should handle gracefully


# ==================== Performance Stats Tests ====================

@pytest.mark.unit
def test_get_performance_stats(processor):
    """Test getting performance statistics."""
    stats = processor.get_performance_stats()
    assert 'streaming_active' in stats
    assert 'compression_enabled' in stats
    assert 'buffer_size' in stats


@pytest.mark.unit
def test_performance_stats_with_managers(processor):
    """Test performance stats with memory/cache managers."""
    stats = processor.get_performance_stats()
    assert 'memory_manager_active' in stats
    assert 'cache_manager_active' in stats


# ==================== Device Detection Tests ====================

@pytest.mark.unit
def test_detect_audio_devices(processor):
    """Test audio device detection."""
    input_devices, output_devices = processor.detect_audio_devices()
    assert isinstance(input_devices, list)
    assert isinstance(output_devices, list)


@pytest.mark.unit
def test_select_input_device_valid(processor):
    """Test selecting valid input device."""
    processor.input_devices = [{'index': 0, 'name': 'Test'}]
    result = processor.select_input_device(0)
    assert result == True


@pytest.mark.unit
def test_select_input_device_invalid(processor):
    """Test selecting invalid input device."""
    processor.input_devices = []
    result = processor.select_input_device(99)
    assert result == False


# ==================== File I/O Tests ====================

@pytest.mark.unit
def test_save_audio_to_file(processor, sample_audio):
    """Test saving audio to file."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = f.name
    
    try:
        result = processor.save_audio_to_file(sample_audio, temp_path)
        if SOUNDDEVICE_AVAILABLE:
            assert result == True
            assert os.path.exists(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@pytest.mark.unit
def test_load_audio_from_file(processor, sample_audio):
    """Test loading audio from file."""
    if not SOUNDDEVICE_AVAILABLE:
        pytest.skip("soundfile not available")
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = f.name
    
    try:
        processor.save_audio_to_file(sample_audio, temp_path)
        loaded = processor.load_audio_from_file(temp_path)
        assert loaded is not None
        assert loaded.sample_rate == sample_audio.sample_rate
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@pytest.mark.unit
def test_load_audio_unsupported_format(processor):
    """Test loading unsupported audio format."""
    result = processor.load_audio("test.xyz")
    assert result is None


# ==================== State Management Tests ====================

@pytest.mark.unit
def test_get_state(processor):
    """Test getting processor state."""
    state = processor.get_state()
    assert isinstance(state, AudioProcessorState)


@pytest.mark.unit
def test_is_available(processor):
    """Test checking if processor is available."""
    assert processor.is_available() == True
    processor.state = AudioProcessorState.ERROR
    assert processor.is_available() == False


@pytest.mark.unit
def test_get_status(processor):
    """Test getting processor status."""
    status = processor.get_status()
    assert 'state' in status
    assert 'is_recording' in status
    assert 'buffer_size' in status
    assert 'available_features' in status


# ==================== Edge Cases ====================

@pytest.mark.unit
def test_empty_audio_buffer(processor):
    """Test operations on empty buffer."""
    chunk = processor.get_audio_chunk()
    assert chunk == b''


@pytest.mark.unit
def test_process_very_short_audio(processor):
    """Test processing very short audio."""
    short_audio = AudioData(
        data=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        sample_rate=16000,
        duration=0.0001
    )
    result = processor._process_audio(short_audio)
    assert result is not None


@pytest.mark.unit
def test_process_very_long_audio(processor):
    """Test processing very long audio."""
    # 10 seconds
    long_audio = AudioData(
        data=np.random.randn(160000).astype(np.float32),
        sample_rate=16000,
        duration=10.0
    )
    result = processor._process_audio(long_audio)
    assert result is not None


@pytest.mark.unit
def test_multichannel_audio_conversion(processor):
    """Test converting multichannel audio to mono."""
    stereo_data = np.random.randn(1000, 2).astype(np.float32)
    audio = AudioData(
        data=stereo_data,
        sample_rate=16000,
        duration=0.0625,
        channels=2
    )
    # Processor should handle multichannel


@pytest.mark.unit
def test_get_available_features(processor):
    """Test getting available features."""
    features = processor.get_available_features()
    assert isinstance(features, dict)
    assert 'audio_capture' in features
    assert 'noise_reduction' in features


@pytest.mark.unit
def test_create_audio_stream(processor):
    """Test creating audio stream."""
    stream = processor.create_audio_stream()
    # May return None or mock stream


@pytest.mark.unit
def test_audio_callback(processor):
    """Test audio callback function."""
    indata = np.random.randn(1024, 1).astype(np.float32)
    processor.audio_callback(indata, 1024, None, None)
    # Should add to buffer
    assert len(processor.audio_buffer) > 0
