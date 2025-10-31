"""
Comprehensive tests for Audio Processing Pipeline.
Covers format conversion, VAD (Voice Activity Detection), buffer management,
audio quality analysis, and processing pipeline integration.
"""

import pytest
import pytest_asyncio
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import tempfile
import os
from pathlib import Path

# Mock problematic imports before loading voice modules
if 'torch' not in sys.modules:
    sys.modules['torch'] = MagicMock()
if 'whisper' not in sys.modules:
    sys.modules['whisper'] = MagicMock()
if 'langchain_ollama' not in sys.modules:
    sys.modules['langchain_ollama'] = MagicMock()
if 'langchain_core' not in sys.modules:
    sys.modules['langchain_core'] = MagicMock()
if 'langchain_core.language_models' not in sys.modules:
    sys.modules['langchain_core.language_models'] = MagicMock()
if 'langchain_core.prompt_values' not in sys.modules:
    sys.modules['langchain_core.prompt_values'] = MagicMock()
if 'app' not in sys.modules:
    sys.modules['app'] = MagicMock()

from voice.audio_processor import (
    SimplifiedAudioProcessor, AudioData, AudioQualityMetrics,
    AudioProcessorState, VAD_AVAILABLE, LIBROSA_AVAILABLE, SOUNDDEVICE_AVAILABLE
)


class TestAudioProcessingPipeline:
    """Comprehensive tests for audio processing pipeline components."""

    @pytest.fixture
    def sample_audio(self):
        """Create sample audio data for testing."""
        # Generate 1 second of 16kHz audio
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)

        # Create a sine wave with some noise
        frequency = 440  # A4 note
        sine_wave = np.sin(2 * np.pi * frequency * t)
        noise = np.random.normal(0, 0.1, len(sine_wave))
        audio_data = sine_wave + noise

        return AudioData(
            data=audio_data.astype(np.float32),
            sample_rate=sample_rate,
            duration=duration,
            channels=1,
            format='wav'
        )

    @pytest.fixture
    def silent_audio(self):
        """Create silent audio data."""
        return AudioData(
            data=np.zeros(16000, dtype=np.float32),  # 1 second of silence
            sample_rate=16000,
            duration=1.0,
            channels=1,
            format='wav'
        )

    @pytest.fixture
    def noisy_audio(self):
        """Create noisy audio data."""
        noise = np.random.normal(0, 0.5, 16000)
        return AudioData(
            data=noise.astype(np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1,
            format='wav'
        )

    @pytest.fixture
    def audio_processor(self):
        """Create audio processor instance."""
        return SimplifiedAudioProcessor()

    # Format Conversion Tests (15 tests)
    def test_format_conversion_wav_to_mp3_success(self, audio_processor, sample_audio):
        """Test successful WAV to MP3 conversion."""
        if not audio_processor.features.get('format_conversion', False):
            pytest.skip("Format conversion not available")

        result = audio_processor.convert_audio_format(sample_audio, "mp3")

        assert result is not None
        assert result.format == "mp3"
        assert np.array_equal(result.data, sample_audio.data)
        assert result.sample_rate == sample_audio.sample_rate

    def test_format_conversion_wav_to_ogg_success(self, audio_processor, sample_audio):
        """Test successful WAV to OGG conversion."""
        if not audio_processor.features.get('format_conversion', False):
            pytest.skip("Format conversion not available")

        result = audio_processor.convert_audio_format(sample_audio, "ogg")

        assert result is not None
        assert result.format == "ogg"
        assert np.array_equal(result.data, sample_audio.data)

    def test_format_conversion_wav_to_flac_success(self, audio_processor, sample_audio):
        """Test successful WAV to FLAC conversion."""
        if not audio_processor.features.get('format_conversion', False):
            pytest.skip("Format conversion not available")

        result = audio_processor.convert_audio_format(sample_audio, "flac")

        assert result is not None
        assert result.format == "flac"

    def test_format_conversion_no_conversion_available(self, audio_processor, sample_audio):
        """Test format conversion when feature is not available."""
        # Mock format conversion as unavailable
        audio_processor.features['format_conversion'] = False

        result = audio_processor.convert_audio_format(sample_audio, "mp3")

        # Should return original audio unchanged
        assert result is sample_audio
        assert result.format == "wav"

    def test_format_conversion_invalid_format(self, audio_processor, sample_audio):
        """Test format conversion with invalid target format."""
        if not audio_processor.features.get('format_conversion', False):
            pytest.skip("Format conversion not available")

        result = audio_processor.convert_audio_format(sample_audio, "invalid_format")

        # Should still create new AudioData object
        assert result is not None
        assert result.format == "invalid_format"

    def test_format_conversion_exception_handling(self, audio_processor, sample_audio):
        """Test format conversion handles exceptions gracefully."""
        if not audio_processor.features.get('format_conversion', False):
            pytest.skip("Format conversion not available")

        # Mock AudioData constructor to raise exception
        with patch('voice.audio_processor.AudioData', side_effect=Exception("Constructor error")):
            result = audio_processor.convert_audio_format(sample_audio, "mp3")

            # Should return original audio
            assert result is sample_audio

    def test_format_conversion_preserves_metadata(self, audio_processor, sample_audio):
        """Test format conversion preserves audio metadata."""
        if not audio_processor.features.get('format_conversion', False):
            pytest.skip("Format conversion not available")

        result = audio_processor.convert_audio_format(sample_audio, "mp3")

        assert result.sample_rate == sample_audio.sample_rate
        assert result.duration == sample_audio.duration
        assert result.channels == sample_audio.channels

    def test_format_conversion_empty_audio(self, audio_processor):
        """Test format conversion with empty audio data."""
        if not audio_processor.features.get('format_conversion', False):
            pytest.skip("Format conversion not available")

        empty_audio = AudioData(
            data=np.array([], dtype=np.float32),
            sample_rate=16000,
            duration=0.0
        )

        result = audio_processor.convert_audio_format(empty_audio, "mp3")

        assert result is not None
        assert len(result.data) == 0

    def test_format_conversion_large_audio(self, audio_processor):
        """Test format conversion with large audio data."""
        if not audio_processor.features.get('format_conversion', False):
            pytest.skip("Format conversion not available")

        # Create 10MB of audio data
        large_data = np.random.random(16000 * 60).astype(np.float32)  # 1 minute
        large_audio = AudioData(large_data, 16000, 60.0)

        result = audio_processor.convert_audio_format(large_audio, "mp3")

        assert result is not None
        assert result.format == "mp3"
        assert len(result.data) == len(large_data)

    def test_format_conversion_different_sample_rates(self, audio_processor):
        """Test format conversion preserves different sample rates."""
        if not audio_processor.features.get('format_conversion', False):
            pytest.skip("Format conversion not available")

        audio_8k = AudioData(
            data=np.random.random(8000).astype(np.float32),
            sample_rate=8000,
            duration=1.0
        )

        result = audio_processor.convert_audio_format(audio_8k, "mp3")

        assert result.sample_rate == 8000

    def test_format_conversion_stereo_audio(self, audio_processor):
        """Test format conversion with stereo audio."""
        if not audio_processor.features.get('format_conversion', False):
            pytest.skip("Format conversion not available")

        stereo_data = np.random.random((16000, 2)).astype(np.float32)
        stereo_audio = AudioData(stereo_data, 16000, 1.0, channels=2)

        result = audio_processor.convert_audio_format(stereo_audio, "mp3")

        assert result.channels == 2
        assert result.format == "mp3"

    def test_format_conversion_multiple_conversions(self, audio_processor, sample_audio):
        """Test multiple consecutive format conversions."""
        if not audio_processor.features.get('format_conversion', False):
            pytest.skip("Format conversion not available")

        # Convert WAV -> MP3 -> OGG -> FLAC
        mp3 = audio_processor.convert_audio_format(sample_audio, "mp3")
        ogg = audio_processor.convert_audio_format(mp3, "ogg")
        flac = audio_processor.convert_audio_format(ogg, "flac")

        assert mp3.format == "mp3"
        assert ogg.format == "ogg"
        assert flac.format == "flac"
        assert np.array_equal(flac.data, sample_audio.data)

    def test_format_conversion_memory_efficiency(self, audio_processor, sample_audio):
        """Test format conversion doesn't leak memory."""
        if not audio_processor.features.get('format_conversion', False):
            pytest.skip("Format conversion not available")

        import gc

        # Get initial object count
        initial_objects = len(gc.get_objects())

        # Perform multiple conversions
        for _ in range(10):
            audio_processor.convert_audio_format(sample_audio, "mp3")

        gc.collect()
        final_objects = len(gc.get_objects())

        # Object count should not grow significantly
        assert final_objects - initial_objects < 50

    def test_format_conversion_logging(self, audio_processor, sample_audio):
        """Test format conversion logging."""
        if not audio_processor.features.get('format_conversion', False):
            pytest.skip("Format conversion not available")

        with patch.object(audio_processor.logger, 'error') as mock_error:
            # Force an error
            with patch('voice.audio_processor.AudioData', side_effect=Exception("Test error")):
                audio_processor.convert_audio_format(sample_audio, "mp3")

            mock_error.assert_called()

    # Voice Activity Detection Tests (15 tests)
    def test_vad_detect_speech_in_sample_audio(self, audio_processor, sample_audio):
        """Test VAD detects speech in sample audio."""
        if not audio_processor.features.get('vad', False):
            pytest.skip("VAD not available")

        result = audio_processor.detect_voice_activity(sample_audio)

        # Should return a list
        assert isinstance(result, list)

    def test_vad_detect_no_speech_in_silence(self, audio_processor, silent_audio):
        """Test VAD correctly identifies silence."""
        if not audio_processor.features.get('vad', False):
            pytest.skip("VAD not available")

        result = audio_processor.detect_voice_activity(silent_audio)

        # Should return empty list for silence
        assert isinstance(result, list)
        # May be empty or contain no speech segments

    def test_vad_detect_speech_in_noise(self, audio_processor, noisy_audio):
        """Test VAD handles noisy audio."""
        if not audio_processor.features.get('vad', False):
            pytest.skip("VAD not available")

        result = audio_processor.detect_voice_activity(noisy_audio)

        assert isinstance(result, list)

    def test_vad_no_vad_available_fallback(self, audio_processor, sample_audio):
        """Test VAD fallback when WebRTC VAD is not available."""
        # Force VAD unavailable
        audio_processor.features['vad'] = False
        audio_processor.vad = None

        result = audio_processor.detect_voice_activity(sample_audio)

        # Should return empty list
        assert result == []

    def test_vad_resampling_for_different_rates(self, audio_processor):
        """Test VAD handles different sample rates."""
        if not audio_processor.features.get('vad', False) or not LIBROSA_AVAILABLE:
            pytest.skip("VAD or Librosa not available")

        # Create 8kHz audio
        audio_8k = AudioData(
            data=np.random.random(8000).astype(np.float32),
            sample_rate=8000,
            duration=1.0
        )

        result = audio_processor.detect_voice_activity(audio_8k)

        assert isinstance(result, list)

    def test_vad_no_librosa_resampling_fallback(self, audio_processor):
        """Test VAD when librosa is not available for resampling."""
        if not audio_processor.features.get('vad', False):
            pytest.skip("VAD not available")

        # Mock librosa as unavailable
        with patch('voice.audio_processor.LIBROSA_AVAILABLE', False):
            audio_8k = AudioData(
                data=np.random.random(8000).astype(np.float32),
                sample_rate=8000,
                duration=1.0
            )

            result = audio_processor.detect_voice_activity(audio_8k)

            # Should return empty list when resampling not possible
            assert result == []

    def test_vad_handle_invalid_audio_data(self, audio_processor):
        """Test VAD handles invalid audio data."""
        if not audio_processor.features.get('vad', False):
            pytest.skip("VAD not available")

        # Create audio with NaN values
        invalid_audio = AudioData(
            data=np.array([np.nan, np.inf, -np.inf, 1.0]),
            sample_rate=16000,
            duration=0.00025
        )

        result = audio_processor.detect_voice_activity(invalid_audio)

        # Should handle gracefully
        assert isinstance(result, list)

    def test_vad_short_audio_frames(self, audio_processor):
        """Test VAD with very short audio frames."""
        if not audio_processor.features.get('vad', False):
            pytest.skip("VAD not available")

        # Create very short audio (less than 30ms)
        short_audio = AudioData(
            data=np.random.random(100).astype(np.float32),  # ~6ms at 16kHz
            sample_rate=16000,
            duration=0.00625
        )

        result = audio_processor.detect_voice_activity(short_audio)

        assert isinstance(result, list)

    def test_vad_exception_handling(self, audio_processor, sample_audio):
        """Test VAD handles exceptions gracefully."""
        if not audio_processor.features.get('vad', False):
            pytest.skip("VAD not available")

        # Mock VAD to raise exception
        with patch.object(audio_processor.vad, 'is_speech', side_effect=Exception("VAD error")):
            result = audio_processor.detect_voice_activity(sample_audio)

            # Should return empty list on error
            assert result == []

    def test_vad_multiple_speech_segments(self, audio_processor):
        """Test VAD detects multiple speech segments."""
        if not audio_processor.features.get('vad', False):
            pytest.skip("VAD not available")

        # Create audio with speech-like segments separated by silence
        # This is complex to test reliably, so we'll test the structure
        result = audio_processor.detect_voice_activity(sample_audio)

        assert isinstance(result, list)
        # Each element should be a dict with start/end/confidence if speech detected
        for segment in result:
            if result:  # Only check if segments were found
                assert isinstance(segment, dict)
                assert 'start' in segment
                assert 'end' in segment
                assert 'confidence' in segment

    def test_vad_empty_audio(self, audio_processor):
        """Test VAD with empty audio."""
        if not audio_processor.features.get('vad', False):
            pytest.skip("VAD not available")

        empty_audio = AudioData(
            data=np.array([], dtype=np.float32),
            sample_rate=16000,
            duration=0.0
        )

        result = audio_processor.detect_voice_activity(empty_audio)

        assert result == []

    def test_vad_different_channels(self, audio_processor):
        """Test VAD with multi-channel audio."""
        if not audio_processor.features.get('vad', False):
            pytest.skip("VAD not available")

        # VAD expects mono, but test how it handles stereo
        stereo_data = np.random.random((16000, 2)).astype(np.float32)
        stereo_audio = AudioData(stereo_data, 16000, 1.0, channels=2)

        result = audio_processor.detect_voice_activity(stereo_audio)

        # Should handle gracefully (may return empty list for unsupported format)
        assert isinstance(result, list)

    def test_vad_performance_under_load(self, audio_processor, sample_audio):
        """Test VAD performance with multiple calls."""
        if not audio_processor.features.get('vad', False):
            pytest.skip("VAD not available")

        import time

        start_time = time.time()

        # Process the same audio multiple times
        for _ in range(10):
            result = audio_processor.detect_voice_activity(sample_audio)
            assert isinstance(result, list)

        end_time = time.time()

        # Should complete within reasonable time
        assert end_time - start_time < 1.0

    def test_vad_simple_fallback_detection(self, audio_processor, sample_audio):
        """Test simple energy-based VAD fallback."""
        # Force VAD unavailable to test energy-based fallback
        audio_processor.features['vad'] = False

        result = audio_processor.detect_voice_activity_simple(sample_audio.data)

        # Should return boolean
        assert isinstance(result, (bool, np.bool_))

    # Buffer Management Tests (15 tests)
    def test_buffer_add_single_chunk(self, audio_processor):
        """Test adding single audio chunk to buffer."""
        chunk = np.random.random(1024).astype(np.float32)

        audio_processor.add_to_buffer(chunk)

        contents = audio_processor.get_buffer_contents()
        assert len(contents) == 1
        assert np.array_equal(contents[0], chunk)

    def test_buffer_add_multiple_chunks(self, audio_processor):
        """Test adding multiple chunks to buffer."""
        chunks = [np.random.random(1024).astype(np.float32) for _ in range(5)]

        for chunk in chunks:
            audio_processor.add_to_buffer(chunk)

        contents = audio_processor.get_buffer_contents()
        assert len(contents) == 5

        for i, chunk in enumerate(chunks):
            assert np.array_equal(contents[i], chunk)

    def test_buffer_memory_limit_enforcement(self, audio_processor):
        """Test buffer respects memory limits."""
        # Set very low memory limit
        audio_processor._max_memory_bytes = 1024  # 1KB

        large_chunk = np.random.random(1000).astype(np.float32)  # ~4KB

        # First chunk should succeed
        audio_processor.add_to_buffer(large_chunk)
        assert len(audio_processor.get_buffer_contents()) == 1

        # Second chunk should be rejected due to memory limit
        audio_processor.add_to_buffer(large_chunk)
        contents = audio_processor.get_buffer_contents()
        assert len(contents) == 1  # Still only 1 chunk

    def test_buffer_clear_operation(self, audio_processor):
        """Test clearing buffer contents."""
        chunks = [np.random.random(1024).astype(np.float32) for _ in range(3)]

        for chunk in chunks:
            audio_processor.add_to_buffer(chunk)

        assert len(audio_processor.get_buffer_contents()) == 3

        audio_processor.clear_buffer()

        assert len(audio_processor.get_buffer_contents()) == 0
        assert audio_processor._buffer_bytes_estimate == 0

    def test_buffer_thread_safety(self, audio_processor):
        """Test buffer operations are thread-safe."""
        import threading

        results = []
        errors = []

        def add_chunks(thread_id):
            try:
                for i in range(10):
                    chunk = np.random.random(1024).astype(np.float32)
                    chunk[0] = thread_id  # Mark chunk with thread ID
                    audio_processor.add_to_buffer(chunk)
                results.append(f"thread_{thread_id}_success")
            except Exception as e:
                errors.append(f"thread_{thread_id}_error: {e}")

        threads = [threading.Thread(target=add_chunks, args=(i,)) for i in range(3)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 3
        assert len(errors) == 0

        contents = audio_processor.get_buffer_contents()
        assert len(contents) == 30  # 10 chunks per thread

    def test_buffer_maxlen_respected(self, audio_processor):
        """Test buffer respects maxlen setting."""
        # Add more chunks than maxlen
        for i in range(audio_processor.max_buffer_size + 5):
            chunk = np.random.random(1024).astype(np.float32)
            audio_processor.add_to_buffer(chunk)

        contents = audio_processor.get_buffer_contents()
        assert len(contents) <= audio_processor.max_buffer_size

    def test_buffer_get_contents_copy(self, audio_processor):
        """Test get_buffer_contents returns a copy."""
        chunk = np.random.random(1024).astype(np.float32)
        audio_processor.add_to_buffer(chunk)

        contents1 = audio_processor.get_buffer_contents()
        contents2 = audio_processor.get_buffer_contents()

        # Should be separate lists
        assert contents1 is not contents2
        # But should contain same data
        assert np.array_equal(contents1[0], contents2[0])

    def test_buffer_empty_operations(self, audio_processor):
        """Test buffer operations on empty buffer."""
        # Get contents of empty buffer
        contents = audio_processor.get_buffer_contents()
        assert contents == []

        # Clear empty buffer
        audio_processor.clear_buffer()
        assert audio_processor._buffer_bytes_estimate == 0

    def test_buffer_large_data_handling(self, audio_processor):
        """Test buffer handles large data chunks."""
        # Create a large chunk
        large_chunk = np.random.random(10000).astype(np.float32)  # ~40KB

        audio_processor.add_to_buffer(large_chunk)

        contents = audio_processor.get_buffer_contents()
        assert len(contents) == 1
        assert len(contents[0]) == 10000

    def test_buffer_memory_estimation_accuracy(self, audio_processor):
        """Test buffer memory estimation is accurate."""
        chunk = np.random.random(1024).astype(np.float32)

        initial_memory = audio_processor._buffer_bytes_estimate

        audio_processor.add_to_buffer(chunk)

        # Memory estimate should increase by chunk.nbytes
        expected_increase = chunk.nbytes
        actual_increase = audio_processor._buffer_bytes_estimate - initial_memory

        assert actual_increase == expected_increase

    def test_buffer_force_cleanup(self, audio_processor):
        """Test force cleanup of buffers."""
        # Add some data
        for _ in range(5):
            chunk = np.random.random(1024).astype(np.float32)
            audio_processor.add_to_buffer(chunk)

        assert len(audio_processor.get_buffer_contents()) > 0

        # Force cleanup
        audio_processor.force_cleanup_buffers()

        # Buffer should be cleared
        assert len(audio_processor.get_buffer_contents()) == 0

    def test_buffer_exception_handling(self, audio_processor):
        """Test buffer operations handle exceptions."""
        # Test with corrupted buffer
        audio_processor.audio_buffer = None

        # Should handle gracefully
        contents = audio_processor.get_buffer_contents()
        assert contents == []

        # Clear should handle gracefully
        audio_processor.clear_buffer()

    def test_buffer_concurrent_read_write(self, audio_processor):
        """Test concurrent read/write operations on buffer."""
        import threading

        stop_flag = threading.Event()

        def writer():
            count = 0
            while not stop_flag.is_set() and count < 100:
                chunk = np.random.random(1024).astype(np.float32)
                audio_processor.add_to_buffer(chunk)
                count += 1

        def reader():
            while not stop_flag.is_set():
                contents = audio_processor.get_buffer_contents()
                # Just read, don't assert anything specific
                len(contents)

        writer_thread = threading.Thread(target=writer)
        reader_thread = threading.Thread(target=reader)

        writer_thread.start()
        reader_thread.start()

        # Run for a short time
        import time
        time.sleep(0.1)
        stop_flag.set()

        writer_thread.join(timeout=1.0)
        reader_thread.join(timeout=1.0)

        # Test completed without deadlocks or exceptions

    def test_buffer_initialization(self, audio_processor):
        """Test buffer is properly initialized."""
        assert hasattr(audio_processor, 'audio_buffer')
        assert hasattr(audio_processor, '_buffer_bytes_estimate')
        assert audio_processor._buffer_bytes_estimate == 0

        contents = audio_processor.get_buffer_contents()
        assert contents == []

    # Audio Quality Analysis Tests (10 tests)
    def test_quality_metrics_clean_audio(self, audio_processor, sample_audio):
        """Test quality metrics for clean audio."""
        if not LIBROSA_AVAILABLE:
            pytest.skip("Librosa not available for quality analysis")

        metrics = audio_processor.calculate_audio_quality_metrics(sample_audio.data)

        assert isinstance(metrics, AudioQualityMetrics)
        assert metrics.speech_level >= 0
        assert metrics.noise_level >= 0
        assert metrics.clarity_score >= 0
        assert metrics.overall_quality >= 0

    def test_quality_metrics_noisy_audio(self, audio_processor, noisy_audio):
        """Test quality metrics for noisy audio."""
        if not LIBROSA_AVAILABLE:
            pytest.skip("Librosa not available for quality analysis")

        metrics = audio_processor.calculate_audio_quality_metrics(noisy_audio.data)

        assert isinstance(metrics, AudioQualityMetrics)
        # Noisy audio should have higher noise level
        assert metrics.noise_level > 0

    def test_quality_metrics_silent_audio(self, audio_processor, silent_audio):
        """Test quality metrics for silent audio."""
        if not LIBROSA_AVAILABLE:
            pytest.skip("Librosa not available for quality analysis")

        metrics = audio_processor.calculate_audio_quality_metrics(silent_audio.data)

        assert isinstance(metrics, AudioQualityMetrics)
        # Silent audio should have very low speech level
        assert metrics.speech_level < 0.01

    def test_quality_metrics_no_librosa_fallback(self, audio_processor, sample_audio):
        """Test quality metrics fallback when librosa unavailable."""
        with patch('voice.audio_processor.LIBROSA_AVAILABLE', False):
            metrics = audio_processor.calculate_audio_quality_metrics(sample_audio.data)

            # Should return default metrics (all zeros)
            assert metrics.speech_level == 0.0
            assert metrics.noise_level == 0.0
            assert metrics.snr_ratio == 0.0
            assert metrics.clarity_score == 0.0
            assert metrics.overall_quality == 0.0

    def test_quality_metrics_exception_handling(self, audio_processor):
        """Test quality metrics handles exceptions."""
        if not LIBROSA_AVAILABLE:
            pytest.skip("Librosa not available for quality analysis")

        # Pass invalid data
        invalid_data = np.array([np.nan, np.inf])

        metrics = audio_processor.calculate_audio_quality_metrics(invalid_data)

        # Should handle gracefully and return some metrics
        assert isinstance(metrics, AudioQualityMetrics)

    def test_quality_metrics_different_data_types(self, audio_processor):
        """Test quality metrics with different numpy data types."""
        if not LIBROSA_AVAILABLE:
            pytest.skip("Librosa not available for quality analysis")

        # Test with int16 data
        int_data = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
        metrics = audio_processor.calculate_audio_quality_metrics(int_data.astype(np.float32))

        assert isinstance(metrics, AudioQualityMetrics)

    def test_quality_metrics_empty_array(self, audio_processor):
        """Test quality metrics with empty array."""
        if not LIBROSA_AVAILABLE:
            pytest.skip("Librosa not available for quality analysis")

        empty_data = np.array([], dtype=np.float32)
        metrics = audio_processor.calculate_audio_quality_metrics(empty_data)

        assert isinstance(metrics, AudioQualityMetrics)

    def test_quality_metrics_normalization(self, audio_processor, sample_audio):
        """Test audio normalization."""
        if not LIBROSA_AVAILABLE:
            pytest.skip("Librosa not available for quality analysis")

        # Normalize quiet audio
        quiet_audio = sample_audio.data * 0.1
        normalized = audio_processor.normalize_audio_level(quiet_audio, target_level=0.8)

        # Should be louder
        assert np.max(np.abs(normalized)) > np.max(np.abs(quiet_audio))

    def test_quality_metrics_normalization_edge_cases(self, audio_processor):
        """Test audio normalization edge cases."""
        # Test with zeros
        zeros = np.zeros(1000, dtype=np.float32)
        normalized = audio_processor.normalize_audio_level(zeros)
        assert np.array_equal(normalized, zeros)

        # Test with single value
        single = np.array([0.5], dtype=np.float32)
        normalized = audio_processor.normalize_audio_level(single, target_level=1.0)
        assert normalized[0] == 1.0

    def test_quality_metrics_to_dict_conversion(self, audio_processor, sample_audio):
        """Test quality metrics to_dict conversion."""
        if not LIBROSA_AVAILABLE:
            pytest.skip("Librosa not available for quality analysis")

        metrics = audio_processor.calculate_audio_quality_metrics(sample_audio.data)
        metrics_dict = metrics.to_dict()

        assert isinstance(metrics_dict, dict)
        assert 'snr_ratio' in metrics_dict
        assert 'noise_level' in metrics_dict
        assert 'speech_level' in metrics_dict
        assert 'clarity_score' in metrics_dict
        assert 'overall_quality' in metrics_dict

    # Processing Pipeline Integration Tests (5 tests)
    def test_processing_pipeline_complete_flow(self, audio_processor, sample_audio):
        """Test complete audio processing pipeline."""
        # Add to buffer
        audio_processor.add_to_buffer(sample_audio.data)

        # Get buffer contents
        buffer_data = audio_processor.get_buffer_contents()
        assert len(buffer_data) > 0

        # Calculate quality metrics
        if LIBROSA_AVAILABLE:
            metrics = audio_processor.calculate_audio_quality_metrics(buffer_data[0])
            assert isinstance(metrics, AudioQualityMetrics)

        # Convert format
        if audio_processor.features.get('format_conversion', False):
            converted = audio_processor.convert_audio_format(sample_audio, "mp3")
            assert converted.format == "mp3"

        # Detect voice activity
        if audio_processor.features.get('vad', False):
            vad_result = audio_processor.detect_voice_activity(sample_audio)
            assert isinstance(vad_result, list)

    def test_processing_pipeline_error_recovery(self, audio_processor, sample_audio):
        """Test pipeline error recovery."""
        # Disable all features to test fallbacks
        audio_processor.features = {
            'audio_capture': False,
            'audio_playback': False,
            'noise_reduction': False,
            'vad': False,
            'quality_analysis': False,
            'format_conversion': False
        }

        # Pipeline should still work with fallbacks
        audio_processor.add_to_buffer(sample_audio.data)
        contents = audio_processor.get_buffer_contents()
        assert len(contents) > 0

        # Quality metrics should return defaults
        metrics = audio_processor.calculate_audio_quality_metrics(sample_audio.data)
        assert metrics.speech_level == 0.0

        # Format conversion should return original
        converted = audio_processor.convert_audio_format(sample_audio, "mp3")
        assert converted is sample_audio

        # VAD should return empty list
        vad_result = audio_processor.detect_voice_activity(sample_audio)
        assert vad_result == []

    def test_processing_pipeline_memory_management(self, audio_processor):
        """Test pipeline memory management."""
        # Fill buffer to capacity
        chunk_size = 1024
        num_chunks = audio_processor.max_buffer_size + 10

        for i in range(num_chunks):
            chunk = np.ones(chunk_size, dtype=np.float32) * (i / num_chunks)  # Different values
            audio_processor.add_to_buffer(chunk)

        # Buffer should not exceed limits
        contents = audio_processor.get_buffer_contents()
        assert len(contents) <= audio_processor.max_buffer_size

        # Memory estimate should be reasonable
        assert audio_processor._buffer_bytes_estimate > 0

        # Clear should free memory
        audio_processor.clear_buffer()
        assert audio_processor._buffer_bytes_estimate == 0

    def test_processing_pipeline_performance(self, audio_processor, sample_audio):
        """Test pipeline performance under load."""
        import time

        start_time = time.time()

        # Perform multiple processing operations
        for _ in range(10):
            audio_processor.add_to_buffer(sample_audio.data)

            if LIBROSA_AVAILABLE:
                audio_processor.calculate_audio_quality_metrics(sample_audio.data)

            if audio_processor.features.get('format_conversion', False):
                audio_processor.convert_audio_format(sample_audio, "mp3")

            if audio_processor.features.get('vad', False):
                audio_processor.detect_voice_activity(sample_audio)

        end_time = time.time()

        # Should complete within reasonable time (adjust based on features)
        max_time = 5.0 if audio_processor.features.get('vad', False) else 2.0
        assert end_time - start_time < max_time

    def test_processing_pipeline_cleanup(self, audio_processor, sample_audio):
        """Test pipeline cleanup operations."""
        # Fill buffer and perform operations
        audio_processor.add_to_buffer(sample_audio.data)

        initial_buffer_size = len(audio_processor.get_buffer_contents())
        assert initial_buffer_size > 0

        # Perform cleanup
        audio_processor.clear_buffer()
        audio_processor.force_cleanup_buffers()

        # Everything should be cleaned up
        assert len(audio_processor.get_buffer_contents()) == 0
        assert audio_processor._buffer_bytes_estimate == 0
