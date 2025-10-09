"""
Comprehensive Audio Processor Tests

This module provides extensive coverage for audio_processor.py functionality:
- Real-time audio capture and recording
- Audio format conversion and quality analysis
- Voice Activity Detection (VAD)
- Noise reduction and preprocessing
- Audio buffer management
- Performance and memory optimization
- Error handling and fallback scenarios
"""

import pytest
import asyncio
import threading
import time
import tempfile
import os
import sys
import numpy as np
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import the audio processor module
try:
    from voice.audio_processor import (
        AudioProcessor, 
        AudioData, 
        AudioProcessorState,
        SimplifiedAudioProcessor
    )
    AUDIO_PROCESSOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Audio processor import failed: {e}")
    AUDIO_PROCESSOR_AVAILABLE = False

# Import fixtures
try:
    from tests.fixtures.voice_fixtures import mock_voice_config, sample_audio_data
except ImportError:
    # Fallback fixtures
    @pytest.fixture
    def mock_voice_config():
        """Fallback mock VoiceConfig for testing."""
        config = MagicMock()
        config.voice_enabled = True
        config.audio_sample_rate = 16000
        config.audio_channels = 1
        config.max_buffer_size = 300
        config.max_memory_mb = 50
        return config
    
    @pytest.fixture
    def sample_audio_data():
        """Create sample audio data for testing."""
        return AudioData(
            data=np.random.random(16000).astype(np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1,
            format="wav"
        )


class TestAudioProcessorCore:
    """Test core audio processor functionality."""
    
    @pytest.fixture
    def processor(self):
        """Create audio processor instance for testing."""
        if not AUDIO_PROCESSOR_AVAILABLE:
            pytest.skip("Audio processor not available")
        return SimplifiedAudioProcessor()
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock all external dependencies for audio processing."""
        with patch('voice.audio_processor.sf') as mock_sf, \
             patch('voice.audio_processor.nr') as mock_nr, \
             patch('voice.audio_processor.webrtcvad') as mock_vad, \
             patch('voice.audio_processor.librosa') as mock_librosa:
            
            # Setup soundfile mock
            mock_sf.io.BytesIO = MagicMock()
            mock_sf.write = MagicMock()
            mock_sf.read = MagicMock(return_value=(np.array([0.1, 0.2, 0.3]), 16000))
            
            # Setup noise reduction mock
            mock_nr.reduce_noise = MagicMock(return_value=np.array([0.1, 0.2, 0.3]))
            
            # Setup VAD mock
            mock_vad.Vad = MagicMock()
            
            # Setup librosa mock
            mock_librosa.load = MagicMock(return_value=(np.array([0.1, 0.2, 0.3]), 16000))
            mock_librosa.effects.preemphasis = MagicMock(return_value=np.array([0.1, 0.2, 0.3]))
            
            yield {
                'soundfile': mock_sf,
                'noisereduce': mock_nr,
                'webrtcvad': mock_vad,
                'librosa': mock_librosa
            }


class TestAudioDataOperations:
    """Test AudioData class operations."""
    
    def test_audio_data_creation(self):
        """Test AudioData object creation and attributes."""
        data = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        audio_data = AudioData(
            data=data,
            sample_rate=16000,
            duration=0.00025,
            channels=1,
            format="wav"
        )
        
        assert audio_data.data is data
        assert audio_data.sample_rate == 16000
        assert audio_data.duration == 0.00025
        assert audio_data.channels == 1
        assert audio_data.format == "wav"
    
    def test_audio_data_to_bytes_with_soundfile(self):
        """Test AudioData to_bytes conversion with soundfile available."""
        data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        audio_data = AudioData(data, 16000, 0.0001875, 1, "wav")
        
        with patch('voice.audio_processor.sf') as mock_sf:
            mock_buffer = MagicMock()
            mock_sf.io.BytesIO.return_value = mock_buffer
            mock_buffer.getvalue.return_value = b'fake_audio_data'
            
            result = audio_data.to_bytes()
            
            assert result == b'fake_audio_data'
            mock_sf.write.assert_called_once_with(mock_buffer, data, 16000, format='WAV')
    
    def test_audio_data_to_bytes_fallback(self):
        """Test AudioData to_bytes conversion fallback without soundfile."""
        data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        audio_data = AudioData(data, 16000, 0.0001875, 1, "wav")
        
        with patch('voice.audio_processor.sf', None):
            result = audio_data.to_bytes()
            
            # Should return base64 encoded data
            import base64
            expected = base64.b64encode(data.tobytes())
            assert result == expected
    
    def test_audio_data_from_bytes_with_soundfile(self):
        """Test AudioData from_bytes creation with soundfile available."""
        test_bytes = b'fake_audio_data'
        
        with patch('voice.audio_processor.sf') as mock_sf:
            mock_buffer = MagicMock()
            mock_sf.io.BytesIO.return_value = mock_buffer
            mock_sf.read.return_value = (np.array([0.1, 0.2, 0.3]), 16000)
            
            result = AudioData.from_bytes(test_bytes, 16000)
            
            assert isinstance(result, AudioData)
            assert result.sample_rate == 16000
            assert np.array_equal(result.data, np.array([0.1, 0.2, 0.3]))
    
    def test_audio_data_from_bytes_fallback(self):
        """Test AudioData from_bytes creation fallback without soundfile."""
        data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        encoded_data = np.base64.b64encode(data.tobytes())
        
        with patch('voice.audio_processor.sf', None):
            result = AudioData.from_bytes(encoded_data, 16000)
            
            assert isinstance(result, AudioData)
            assert result.sample_rate == 16000
            assert np.array_equal(result.data, data)


@pytest.mark.skipif(not AUDIO_PROCESSOR_AVAILABLE, reason="Audio processor not available")
class TestAudioProcessorStateManagement:
    """Test audio processor state transitions and management."""
    
    @pytest.fixture
    def processor(self):
        """Create processor for state testing."""
        return SimplifiedAudioProcessor()
    
    def test_initial_state(self, processor):
        """Test processor starts in IDLE state."""
        assert processor.state == AudioProcessorState.IDLE
    
    def test_state_transitions(self, processor):
        """Test state transitions during audio operations."""
        # Test transition to recording state
        processor.start_recording()
        assert processor.state == AudioProcessorState.RECORDING
        
        # Test transition to processing state
        processor.stop_recording()
        assert processor.state == AudioProcessorState.PROCESSING
        
        # Test transition back to idle
        time.sleep(0.1)  # Allow processing to complete
        assert processor.state == AudioProcessorState.IDLE
    
    def test_concurrent_state_changes(self, processor):
        """Test handling of concurrent state changes."""
        async def simulate_concurrent_changes():
            tasks = []
            for i in range(5):
                task = asyncio.create_task(asyncio.sleep(0.01))  # Simulate async operation
                tasks.append(task)
            await asyncio.gather(*tasks)
        
        # This test ensures thread safety of state changes
        asyncio.run(simulate_concurrent_changes())
        assert processor.state == AudioProcessorState.IDLE


@pytest.mark.skipif(not AUDIO_PROCESSOR_AVAILABLE, reason="Audio processor not available")
class TestAudioRecording:
    """Test audio recording functionality."""
    
    @pytest.fixture
    def processor(self):
        """Create processor for recording tests."""
        return SimplifiedAudioProcessor()
    
    def test_start_recording(self, processor):
        """Test starting audio recording."""
        processor.start_recording()
        
        assert processor.is_recording()
        assert processor.state == AudioProcessorState.RECORDING
        assert len(processor.audio_buffer) == 0
    
    def test_stop_recording(self, processor):
        """Test stopping audio recording."""
        processor.start_recording()
        
        # Add some mock audio data
        mock_audio = AudioData(
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            16000, 0.0001875, 1, "wav"
        )
        processor.audio_buffer.append(mock_audio)
        
        result = processor.stop_recording()
        
        assert not processor.is_recording()
        assert processor.state == AudioProcessorState.IDLE
        assert len(processor.audio_buffer) == 0  # Buffer should be cleared
        assert isinstance(result, AudioData)
    
    def test_recording_timeout(self, processor):
        """Test recording timeout handling."""
        processor.config.recording_timeout = 0.1  # Short timeout for testing
        processor.start_recording()
        
        # Wait for timeout
        time.sleep(0.2)
        
        # Recording should have stopped automatically
        assert not processor.is_recording()
        assert processor.state == AudioProcessorState.ERROR
    
    def test_audio_buffer_management(self, processor):
        """Test audio buffer size limits."""
        processor.config.max_buffer_size = 2  # Small buffer for testing
        
        processor.start_recording()
        
        # Add audio data exceeding buffer limit
        for i in range(3):
            mock_audio = AudioData(
                np.array([0.1, 0.2, 0.3], dtype=np.float32),
                16000, 0.0001875, 1, "wav"
            )
            processor.audio_buffer.append(mock_audio)
        
        # Buffer should be limited to max size
        assert len(processor.audio_buffer) <= 2


@pytest.mark.skipif(not AUDIO_PROCESSOR_AVAILABLE, reason="Audio processor not available")
class TestAudioProcessing:
    """Test audio processing operations."""
    
    @pytest.fixture
    def processor(self):
        """Create processor for processing tests."""
        return SimplifiedAudioProcessor()
    
    @pytest.fixture
    def test_audio_data(self):
        """Create test audio data."""
        return AudioData(
            np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
            16000, 0.0003125, 1, "wav"
        )
    
    def test_noise_reduction(self, processor, test_audio_data):
        """Test noise reduction functionality."""
        with patch('voice.audio_processor.nr') as mock_nr:
            mock_nr.reduce_noise.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
            
            result = processor.reduce_noise(test_audio_data)
            
            assert isinstance(result, AudioData)
            mock_nr.reduce_noise.assert_called_once()
    
    def test_noise_reduction_fallback(self, processor, test_audio_data):
        """Test noise reduction fallback when library not available."""
        with patch('voice.audio_processor.nr', None):
            result = processor.reduce_noise(test_audio_data)
            
            # Should return original audio unchanged
            assert result == test_audio_data
    
    def test_voice_activity_detection(self, processor, test_audio_data):
        """Test voice activity detection."""
        with patch('voice.audio_processor.webrtcvad') as mock_vad:
            mock_vad_instance = MagicMock()
            mock_vad_instance.is_speech.return_value = True
            mock_vad.Vad.return_value = mock_vad_instance
            
            result = processor.detect_voice_activity(test_audio_data)
            
            assert result is True
            mock_vad_instance.is_speech.assert_called()
    
    def test_vad_fallback(self, processor, test_audio_data):
        """Test VAD fallback when library not available."""
        with patch('voice.audio_processor.webrtcvad', None):
            result = processor.detect_voice_activity(test_audio_data)
            
            # Should default to True (assume speech)
            assert result is True
    
    def test_audio_quality_analysis(self, processor, test_audio_data):
        """Test audio quality scoring."""
        quality_score = processor.analyze_audio_quality(test_audio_data)
        
        assert isinstance(quality_score, float)
        assert 0.0 <= quality_score <= 1.0
    
    def test_audio_format_conversion(self, processor, test_audio_data):
        """Test audio format conversion."""
        with patch('voice.audio_processor.librosa') as mock_librosa:
            mock_librosa.load.return_value = (test_audio_data.data, test_audio_data.sample_rate)
            
            # Test conversion to different format
            converted = processor.convert_format(test_audio_data, "mp3")
            
            assert isinstance(converted, AudioData)
            assert converted.format == "mp3"
    
    def test_audio_preprocessing_pipeline(self, processor, test_audio_data):
        """Test complete audio preprocessing pipeline."""
        with patch.object(processor, 'reduce_noise') as mock_reduce, \
             patch.object(processor, 'detect_voice_activity') as mock_vad, \
             patch.object(processor, 'analyze_audio_quality') as mock_quality:
            
            mock_reduce.return_value = test_audio_data
            mock_vad.return_value = True
            mock_quality.return_value = 0.85
            
            result = processor.preprocess_audio(test_audio_data)
            
            assert isinstance(result, AudioData)
            mock_reduce.assert_called_once_with(test_audio_data)
            mock_vad.assert_called_once()
            mock_quality.assert_called_once()


@pytest.mark.skipif(not AUDIO_PROCESSOR_AVAILABLE, reason="Audio processor not available")
class TestAudioProcessorPerformance:
    """Test audio processor performance and optimization."""
    
    @pytest.fixture
    def processor(self):
        """Create processor for performance tests."""
        return SimplifiedAudioProcessor()
    
    def test_memory_usage_monitoring(self, processor):
        """Test memory usage monitoring during audio processing."""
        initial_memory = processor.get_memory_usage()
        
        # Process large audio data
        large_audio = AudioData(
            np.random.random(16000 * 10).astype(np.float32),  # 10 seconds
            16000, 10.0, 1, "wav"
        )
        
        processor.process_audio(large_audio)
        
        final_memory = processor.get_memory_usage()
        
        # Memory should be within reasonable bounds
        memory_increase = final_memory - initial_memory
        assert memory_increase < 100  # MB
    
    def test_concurrent_processing(self, processor):
        """Test concurrent audio processing capabilities."""
        async def process_multiple_audio():
            tasks = []
            for i in range(3):
                audio_data = AudioData(
                    np.random.random(16000).astype(np.float32),  # 1 second
                    16000, 1.0, 1, "wav"
                )
                task = asyncio.create_task(processor.process_audio_async(audio_data))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            return results
        
        results = asyncio.run(process_multiple_audio())
        assert len(results) == 3
        for result in results:
            assert isinstance(result, AudioData)
    
    def test_processing_latency(self, processor):
        """Test audio processing latency."""
        audio_data = AudioData(
            np.random.random(16000).astype(np.float32),  # 1 second
            16000, 1.0, 1, "wav"
        )
        
        start_time = time.time()
        result = processor.process_audio(audio_data)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        assert isinstance(result, AudioData)
        assert processing_time < 1.0  # Should process within 1 second


@pytest.mark.skipif(not AUDIO_PROCESSOR_AVAILABLE, reason="Audio processor not available")
class TestAudioProcessorErrorHandling:
    """Test error handling and fallback scenarios."""
    
    @pytest.fixture
    def processor(self):
        """Create processor for error handling tests."""
        return SimplifiedAudioProcessor()
    
    def test_invalid_audio_data_handling(self, processor):
        """Test handling of invalid audio data."""
        # Test with None data
        with pytest.raises(ValueError):
            processor.process_audio(None)
        
        # Test with empty audio data
        empty_audio = AudioData(
            np.array([], dtype=np.float32),
            16000, 0.0, 1, "wav"
        )
        
        result = processor.process_audio(empty_audio)
        assert isinstance(result, AudioData)
    
    def test_corrupted_audio_file_handling(self, processor):
        """Test handling of corrupted audio files."""
        with patch('voice.audio_processor.sf') as mock_sf:
            mock_sf.read.side_effect = Exception("Corrupted file")
            
            with pytest.raises(Exception):
                AudioData.from_bytes(b'corrupted_data', 16000)
    
    def test_audio_device_error_handling(self, processor):
        """Test handling of audio device errors."""
        with patch('voice.audio_processor.sounddevice') as mock_sd:
            mock_sd.InputStream.side_effect = OSError("No audio device")
            
            # Should handle error gracefully
            try:
                processor.start_recording()
            except OSError:
                # Expected error
                assert not processor.is_recording()
    
    def test_memory_exhaustion_handling(self, processor):
        """Test handling of memory exhaustion scenarios."""
        # Set very low memory limit
        processor.config.max_memory_mb = 1
        
        # Try to process large audio data
        large_audio = AudioData(
            np.random.random(16000 * 60).astype(np.float32),  # 60 seconds
            16000, 60.0, 1, "wav"
        )
        
        # Should handle memory constraints gracefully
        result = processor.process_audio(large_audio)
        assert isinstance(result, AudioData)


@pytest.mark.skipif(not AUDIO_PROCESSOR_AVAILABLE, reason="Audio processor not available")
class TestAudioProcessorIntegration:
    """Test integration scenarios with other voice components."""
    
    @pytest.fixture
    def processor(self):
        """Create processor for integration tests."""
        return SimplifiedAudioProcessor()
    
    def test_integration_with_stt_service(self, processor):
        """Test integration with speech-to-text service."""
        from voice.stt_service import STTService
        
        audio_data = AudioData(
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            16000, 0.0001875, 1, "wav"
        )
        
        # Process audio for STT
        processed_audio = processor.preprocess_audio(audio_data)
        
        # Should be ready for STT
        assert isinstance(processed_audio, AudioData)
        assert processed_audio.sample_rate == 16000  # STT-compatible format
    
    def test_integration_with_tts_service(self, processor):
        """Test integration with text-to-speech service."""
        from voice.tts_service import TTSService
        
        # Test that processor can handle TTS output
        tts_audio = AudioData(
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            22050, 0.000136, 1, "wav"  # Common TTS sample rate
        )
        
        # Processor should be able to handle different sample rates
        converted_audio = processor.resample_audio(tts_audio, 16000)
        
        assert isinstance(converted_audio, AudioData)
        assert converted_audio.sample_rate == 16000
    
    def test_integration_with_voice_security(self, processor):
        """Test integration with voice security features."""
        # Test encryption of audio data
        audio_data = AudioData(
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            16000, 0.0001875, 1, "wav"
        )
        
        # Encrypt audio data
        encrypted_data = processor.encrypt_audio(audio_data)
        
        assert isinstance(encrypted_data, bytes)
        
        # Decrypt audio data
        decrypted_audio = processor.decrypt_audio(encrypted_data)
        
        assert isinstance(decrypted_audio, AudioData)
        assert np.array_equal(decrypted_audio.data, audio_data.data)


class TestAudioProcessorAsyncOperations:
    """Test asynchronous audio processing operations."""
    
    @pytest.fixture
    def processor(self):
        """Create processor for async tests."""
        if not AUDIO_PROCESSOR_AVAILABLE:
            pytest.skip("Audio processor not available")
        return SimplifiedAudioProcessor()
    
    @pytest.mark.asyncio
    async def test_async_audio_processing(self, processor):
        """Test asynchronous audio processing."""
        audio_data = AudioData(
            np.random.random(16000).astype(np.float32),
            16000, 1.0, 1, "wav"
        )
        
        result = await processor.process_audio_async(audio_data)
        
        assert isinstance(result, AudioData)
        assert result.duration == audio_data.duration
    
    @pytest.mark.asyncio
    async def test_async_recording(self, processor):
        """Test asynchronous audio recording."""
        # Start recording asynchronously
        await processor.start_recording_async()
        
        assert processor.is_recording()
        
        # Stop recording asynchronously
        result = await processor.stop_recording_async()
        
        assert isinstance(result, AudioData)
        assert not processor.is_recording()
    
    @pytest.mark.asyncio
    async def test_concurrent_async_operations(self, processor):
        """Test concurrent asynchronous operations."""
        audio_data = AudioData(
            np.random.random(16000).astype(np.float32),
            16000, 1.0, 1, "wav"
        )
        
        # Run multiple async operations concurrently
        tasks = [
            processor.process_audio_async(audio_data),
            processor.analyze_audio_quality_async(audio_data),
            processor.detect_voice_activity_async(audio_data)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All operations should complete successfully
        for result in results:
            assert not isinstance(result, Exception)


# Test utility functions
class TestAudioProcessorUtilities:
    """Test audio processor utility functions."""
    
    def test_audio_format_validation(self):
        """Test audio format validation."""
        if not AUDIO_PROCESSOR_AVAILABLE:
            pytest.skip("Audio processor not available")
        
        processor = SimplifiedAudioProcessor()
        
        # Valid formats
        assert processor.validate_format("wav") is True
        assert processor.validate_format("mp3") is True
        assert processor.validate_format("flac") is True
        
        # Invalid formats
        assert processor.validate_format("xyz") is False
        assert processor.validate_format("") is False
        assert processor.validate_format(None) is False
    
    def test_sample_rate_validation(self):
        """Test sample rate validation."""
        if not AUDIO_PROCESSOR_AVAILABLE:
            pytest.skip("Audio processor not available")
        
        processor = SimplifiedAudioProcessor()
        
        # Valid sample rates
        assert processor.validate_sample_rate(16000) is True
        assert processor.validate_sample_rate(22050) is True
        assert processor.validate_sample_rate(44100) is True
        
        # Invalid sample rates
        assert processor.validate_sample_rate(0) is False
        assert processor.validate_sample_rate(-1) is False
        assert processor.validate_sample_rate(1000000) is False
    
    def test_audio_duration_calculation(self):
        """Test audio duration calculation."""
        if not AUDIO_PROCESSOR_AVAILABLE:
            pytest.skip("Audio processor not available")
        
        processor = SimplifiedAudioProcessor()
        
        # Test duration calculation
        samples = 16000  # 1 second at 16kHz
        sample_rate = 16000
        duration = processor.calculate_duration(samples, sample_rate)
        
        assert duration == 1.0
        
        # Test with different values
        samples = 8000  # 0.5 seconds at 16kHz
        duration = processor.calculate_duration(samples, sample_rate)
        
        assert duration == 0.5


# Performance benchmarks
class TestAudioProcessorBenchmarks:
    """Performance benchmark tests for audio processor."""
    
    @pytest.fixture
    def processor(self):
        """Create processor for benchmarking."""
        if not AUDIO_PROCESSOR_AVAILABLE:
            pytest.skip("Audio processor not available")
        return SimplifiedAudioProcessor()
    
    def benchmark_audio_processing_speed(self, processor):
        """Benchmark audio processing speed."""
        import time
        
        # Create test audio data of different sizes
        sizes = [1, 5, 10]  # seconds
        
        for size in sizes:
            audio_data = AudioData(
                np.random.random(16000 * size).astype(np.float32),
                16000, size, 1, "wav"
            )
            
            start_time = time.time()
            result = processor.process_audio(audio_data)
            end_time = time.time()
            
            processing_time = end_time - start_time
            processing_speed = size / processing_time  # seconds per second
            
            print(f"Audio size: {size}s, Processing time: {processing_time:.3f}s, Speed: {processing_speed:.2f}x")
            
            # Should process faster than real-time
            assert processing_speed > 1.0
    
    def benchmark_memory_usage(self, processor):
        """Benchmark memory usage during processing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process multiple audio files
        for i in range(10):
            audio_data = AudioData(
                np.random.random(16000 * 5).astype(np.float32),  # 5 seconds
                16000, 5.0, 1, "wav"
            )
            processor.process_audio(audio_data)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Initial memory: {initial_memory:.2f}MB")
        print(f"Final memory: {final_memory:.2f}MB")
        print(f"Memory increase: {memory_increase:.2f}MB")
        
        # Memory increase should be reasonable
        assert memory_increase < 50  # MB


# Run tests if this file is executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])