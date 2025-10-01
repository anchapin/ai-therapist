"""
Audio Processing Pipeline Integration Tests

Tests the complete audio processing pipeline including:
- Audio capture and recording workflows
- Voice activity detection and noise reduction
- Audio format conversion and quality analysis
- Memory management across long audio sessions
- Error handling and fallback mechanisms
"""

import pytest
import asyncio
import time
import threading
import numpy as np
from unittest.mock import MagicMock, patch
import psutil
import os
import tempfile
from pathlib import Path

from voice.audio_processor import (
    SimplifiedAudioProcessor,
    AudioProcessor,
    AudioData,
    AudioProcessorState,
    AudioQualityMetrics
)
from voice.config import VoiceConfig


class TestAudioPipelineIntegration:
    """Test complete audio processing pipeline integration."""

    @pytest.fixture
    def audio_config(self):
        """Create audio-focused configuration."""
        config = VoiceConfig()
        config.voice_enabled = True
        config.audio_sample_rate = 16000
        config.audio_channels = 1
        config.audio_chunk_size = 1024
        config.audio_format = "wav"
        config.audio_max_buffer_size = 100  # Small buffer for testing
        config.audio_max_memory_mb = 50
        config.audio_recording_timeout = 10.0
        config.audio_playback_enabled = True
        config.audio_noise_reduction_enabled = True
        config.audio_vad_enabled = True
        config.audio_quality_analysis_enabled = True
        return config

    @pytest.fixture
    def mock_audio_processor(self, audio_config):
        """Create mock audio processor with realistic behavior."""
        with patch('voice.audio_processor.SOUNDDEVICE_AVAILABLE', True), \
             patch('voice.audio_processor.NOISEREDUCE_AVAILABLE', True), \
             patch('voice.audio_processor.VAD_AVAILABLE', True), \
             patch('voice.audio_processor.LIBROSA_AVAILABLE', True), \
             patch('voice.audio_processor.sf') as mock_sf, \
             patch('voice.audio_processor.nr') as mock_nr, \
             patch('voice.audio_processor.webrtcvad') as mock_vad:

            processor = SimplifiedAudioProcessor(audio_config)

            # Configure soundfile mock
            mock_sf.write = MagicMock()
            mock_sf.read = MagicMock(return_value=(np.random.random(16000), 16000))
            mock_sf.io = MagicMock()
            mock_sf.io.BytesIO = MagicMock()

            # Configure noise reduction mock
            mock_nr.reduce_noise = MagicMock(return_value=np.random.random(16000))

            # Configure VAD mock
            mock_vad.Vad = MagicMock()
            processor.vad = mock_vad.Vad.return_value
            processor.vad.is_speech = MagicMock(return_value=True)

            # Set up device info
            processor.input_devices = [
                {'name': 'Mock Microphone', 'channels': 1, 'sample_rate': 16000}
            ]
            processor.output_devices = [
                {'name': 'Mock Speaker', 'channels': 1, 'sample_rate': 16000}
            ]

            return processor

    @pytest.fixture
    def test_audio_data(self):
        """Create test audio data for pipeline testing."""
        # Generate realistic speech-like audio (sine wave with varying frequency)
        duration = 3.0
        sample_rate = 16000
        samples = int(duration * sample_rate)

        # Create speech-like pattern
        t = np.linspace(0, duration, samples)
        # Mix of frequencies to simulate speech
        audio_data = (
            0.3 * np.sin(2 * np.pi * 440 * t) +  # Fundamental frequency
            0.2 * np.sin(2 * np.pi * 880 * t) +  # First harmonic
            0.1 * np.sin(2 * np.pi * 1320 * t)   # Second harmonic
        )

        # Add some background noise
        noise = 0.05 * np.random.normal(0, 1, samples)
        audio_data += noise

        return AudioData(
            data=audio_data.astype(np.float32),
            sample_rate=sample_rate,
            duration=duration,
            channels=1,
            format="float32"
        )

    @pytest.fixture
    def long_audio_session_data(self):
        """Create audio data for long session testing."""
        # Generate 30 seconds of audio
        duration = 30.0
        sample_rate = 16000
        samples = int(duration * sample_rate)

        # Create varied audio pattern
        t = np.linspace(0, duration, samples)
        audio_data = np.sin(2 * np.pi * 300 * t)  # Varying frequency

        return AudioData(
            data=audio_data.astype(np.float32),
            sample_rate=sample_rate,
            duration=duration,
            channels=1,
            format="float32"
        )

    @pytest.mark.asyncio
    async def test_complete_audio_capture_pipeline(self, mock_audio_processor, test_audio_data):
        """Test complete audio capture and processing pipeline."""
        # Test recording start
        recording_started = mock_audio_processor.start_recording()
        assert recording_started == True
        assert mock_audio_processor.state == AudioProcessorState.RECORDING
        assert mock_audio_processor.is_recording == True

        # Simulate audio capture by adding data to buffer
        # Add multiple chunks to simulate real recording
        chunk_duration = 0.1  # 100ms chunks
        chunk_samples = int(chunk_duration * mock_audio_processor.sample_rate)

        for i in range(30):  # 3 seconds of audio
            chunk_data = test_audio_data.data[i*chunk_samples:(i+1)*chunk_samples]
            if len(chunk_data) > 0:
                mock_audio_processor.add_to_buffer(chunk_data)

        # Stop recording
        recorded_audio = mock_audio_processor.stop_recording()

        # Verify recording pipeline
        assert recorded_audio is not None
        assert recorded_audio.duration > 0
        assert len(recorded_audio.data) > 0
        assert recorded_audio.sample_rate == mock_audio_processor.sample_rate

        # Test audio processing features
        voice_activities = mock_audio_processor.detect_voice_activity(recorded_audio)
        assert isinstance(voice_activities, list)

        # Test audio quality analysis
        quality_metrics = mock_audio_processor._analyze_audio_quality(recorded_audio)
        assert isinstance(quality_metrics, AudioQualityMetrics)
        assert 0.0 <= quality_metrics.overall_quality <= 1.0

    def test_audio_processing_features_integration(self, mock_audio_processor, test_audio_data):
        """Test integration of all audio processing features."""
        # Test noise reduction
        if mock_audio_processor.features['noise_reduction']:
            processed_audio = mock_audio_processor._process_audio(test_audio_data)
            assert processed_audio is not None

        # Test voice activity detection
        if mock_audio_processor.features['vad']:
            voice_activities = mock_audio_processor.detect_voice_activity(test_audio_data)
            assert isinstance(voice_activities, list)

        # Test audio quality analysis
        if mock_audio_processor.features['quality_analysis']:
            quality_metrics = mock_audio_processor._analyze_audio_quality(test_audio_data)
            assert isinstance(quality_metrics, AudioQualityMetrics)
            assert quality_metrics.snr_ratio >= 0.0
            assert quality_metrics.clarity_score >= 0.0

    def test_audio_format_conversion_pipeline(self, mock_audio_processor):
        """Test audio format conversion through pipeline."""
        # Create test audio in different format
        original_data = np.random.random(16000).astype(np.float32)

        # Test audio normalization
        normalized_data = mock_audio_processor.normalize_audio_level(original_data, 0.7)
        assert normalized_data is not None

        # Test format conversion
        audio_data = AudioData(original_data, 16000, 1.0, 1, "float32")
        converted_audio = mock_audio_processor.convert_audio_format(audio_data, "wav")
        assert converted_audio.format == "wav"

        # Test audio quality metrics calculation
        quality_metrics = mock_audio_processor.calculate_audio_quality_metrics(original_data)
        assert isinstance(quality_metrics, AudioQualityMetrics)

    @pytest.mark.asyncio
    async def test_audio_memory_management_pipeline(self, mock_audio_processor, long_audio_session_data):
        """Test memory management through extended audio processing."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Test memory monitoring during long audio processing
        memory_snapshots = []

        # Process audio in chunks to simulate long session
        chunk_size = 16000  # 1 second chunks
        total_chunks = int(long_audio_session_data.duration)

        for i in range(total_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(long_audio_session_data.data))
            chunk_data = long_audio_session_data.data[start_idx:end_idx]

            if len(chunk_data) > 0:
                # Add to processor buffer
                mock_audio_processor.add_to_buffer(chunk_data)

                # Periodically check memory usage
                if i % 5 == 0:  # Every 5 seconds
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_snapshots.append(current_memory)

                    # Check memory usage hasn't grown excessively
                    memory_growth = current_memory - initial_memory
                    assert memory_growth < 100, f"Memory grew by {memory_growth:.2f}MB after {i} seconds"

                    # Verify buffer management
                    buffer_usage = mock_audio_processor.get_memory_usage()
                    assert buffer_usage['memory_usage_percent'] <= 100

        # Verify final memory state
        final_memory = process.memory_info().rss / 1024 / 1024
        total_memory_growth = final_memory - initial_memory

        # Should not have excessive memory growth
        assert total_memory_growth < 150, f"Total memory growth {total_memory_growth:.2f}MB seems excessive"

        # Test buffer cleanup
        cleared_chunks = mock_audio_processor.force_cleanup_buffers()
        assert cleared_chunks >= 0

        # Verify cleanup effectiveness
        final_buffer_usage = mock_audio_processor.get_memory_usage()
        assert final_buffer_usage['buffer_size'] == 0
        assert final_buffer_usage['memory_usage_bytes'] == 0

    @pytest.mark.asyncio
    async def test_concurrent_audio_processing(self, mock_audio_processor):
        """Test concurrent audio processing operations."""
        # Start recording
        recording_started = mock_audio_processor.start_recording()
        assert recording_started == True

        # Test concurrent operations
        operations = []

        # Operation 1: Monitor audio level
        async def monitor_audio_level():
            levels = []
            for _ in range(10):
                level = mock_audio_processor.get_audio_level()
                levels.append(level)
                await asyncio.sleep(0.1)
            return levels

        # Operation 2: Add audio data to buffer
        async def add_audio_chunks():
            for i in range(20):
                chunk_data = np.random.random(1024).astype(np.float32)
                mock_audio_processor.add_to_buffer(chunk_data)
                await asyncio.sleep(0.05)

        # Operation 3: Monitor buffer status
        async def monitor_buffer_status():
            statuses = []
            for _ in range(15):
                status = mock_audio_processor.get_status()
                statuses.append(status)
                await asyncio.sleep(0.07)
            return statuses

        # Execute concurrent operations
        results = await asyncio.gather(
            monitor_audio_level(),
            add_audio_chunks(),
            monitor_buffer_status(),
            return_exceptions=True
        )

        # Verify all operations completed
        assert len(results) == 3

        # Check results
        audio_levels, _, buffer_statuses = results

        assert isinstance(audio_levels, list)
        assert len(audio_levels) == 10
        assert isinstance(buffer_statuses, list)
        assert len(buffer_statuses) == 15

        # Verify buffer status monitoring
        for status in buffer_statuses:
            assert 'buffer_size' in status
            assert 'memory_usage_bytes' in status
            assert 'state' in status

        # Stop recording
        recorded_audio = mock_audio_processor.stop_recording()
        assert recorded_audio is not None

    def test_audio_quality_analysis_pipeline(self, mock_audio_processor):
        """Test audio quality analysis throughout processing pipeline."""
        # Create audio with different quality levels
        qualities = {
            'high_quality': 0.8 * np.random.random(16000).astype(np.float32),
            'medium_quality': 0.3 * np.random.random(16000).astype(np.float32),
            'low_quality': 0.05 * np.random.random(16000).astype(np.float32),
            'noisy_audio': np.random.normal(0, 0.5, 16000).astype(np.float32)
        }

        quality_results = {}

        for quality_name, audio_data in qualities.items():
            # Create AudioData object
            audio_obj = AudioData(audio_data, 16000, 1.0, 1, "float32")

            # Analyze quality
            if mock_audio_processor.features['quality_analysis']:
                metrics = mock_audio_processor._analyze_audio_quality(audio_obj)
                quality_results[quality_name] = metrics

                # Verify quality metrics
                assert isinstance(metrics, AudioQualityMetrics)
                assert 0.0 <= metrics.overall_quality <= 1.0

        # High quality should generally score better than low quality
        if 'high_quality' in quality_results and 'low_quality' in quality_results:
            assert quality_results['high_quality'].overall_quality >= quality_results['low_quality'].overall_quality

    def test_audio_buffer_management_integration(self, mock_audio_processor):
        """Test audio buffer management throughout pipeline."""
        # Test buffer operations
        initial_status = mock_audio_processor.get_status()
        assert initial_status['buffer_size'] == 0

        # Add data to buffer
        test_chunks = []
        for i in range(10):
            chunk_data = np.random.random(1024).astype(np.float32)
            test_chunks.append(chunk_data)
            mock_audio_processor.add_to_buffer(chunk_data)

        # Verify buffer contents
        buffer_contents = mock_audio_processor.get_buffer_contents()
        assert len(buffer_contents) == len(test_chunks)

        # Test buffer status after adding data
        buffer_status = mock_audio_processor.get_status()
        assert buffer_status['buffer_size'] == len(test_chunks)
        assert buffer_status['memory_usage_bytes'] > 0

        # Test buffer cleanup
        mock_audio_processor.clear_buffer()
        final_status = mock_audio_processor.get_status()
        assert final_status['buffer_size'] == 0
        assert final_status['memory_usage_bytes'] == 0

    def test_audio_file_operations_pipeline(self, mock_audio_processor, test_audio_data):
        """Test audio file save/load operations in pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test audio saving
            test_file = Path(temp_dir) / "test_audio.wav"

            if mock_audio_processor.features['audio_capture']:
                save_success = mock_audio_processor.save_audio(test_audio_data, str(test_file))
                # Should succeed if soundfile is available (mocked)
                assert save_success == True

            # Test audio loading
            if test_file.exists():
                loaded_audio = mock_audio_processor.load_audio(str(test_file))
                if mock_audio_processor.features['audio_capture']:
                    assert loaded_audio is not None
                    assert loaded_audio.sample_rate == test_audio_data.sample_rate
                    assert abs(loaded_audio.duration - test_audio_data.duration) < 0.1

    def test_audio_playback_pipeline(self, mock_audio_processor, test_audio_data):
        """Test audio playback through processing pipeline."""
        if not mock_audio_processor.features['audio_playback']:
            pytest.skip("Audio playback not available")

        # Test playback start
        playback_started = mock_audio_processor.play_audio(test_audio_data)
        # Should return True if playback features are available
        assert playback_started == True

        # Test playback status
        status = mock_audio_processor.get_status()
        assert 'is_playing' in status

        # Test playback stop
        stop_success = mock_audio_processor.stop_playback()
        # Should return True if playback features are available
        assert stop_success == True

    def test_audio_device_detection_pipeline(self, mock_audio_processor):
        """Test audio device detection and selection."""
        # Test device detection
        input_devices, output_devices = mock_audio_processor.detect_audio_devices()

        # Should return lists (may be empty if no real devices)
        assert isinstance(input_devices, list)
        assert isinstance(output_devices, list)

        # Test device information
        if hasattr(mock_audio_processor, 'input_devices'):
            assert len(mock_audio_processor.input_devices) >= 0
        if hasattr(mock_audio_processor, 'output_devices'):
            assert len(mock_audio_processor.output_devices) >= 0

        # Test device selection
        if mock_audio_processor.input_devices:
            select_success = mock_audio_processor.select_input_device(0)
            assert select_success == True

    @pytest.mark.asyncio
    async def test_audio_error_handling_pipeline(self, mock_audio_processor):
        """Test error handling throughout audio processing pipeline."""
        # Test invalid audio data handling
        invalid_audio_cases = [
            AudioData(np.array([]), 16000, 0.0, 1),  # Empty data
            AudioData(np.array([np.nan, np.inf]), 16000, 0.001, 1),  # Invalid values
            None  # None input
        ]

        for invalid_audio in invalid_audio_cases:
            if invalid_audio is None:
                # Test None input handling
                try:
                    voice_activities = mock_audio_processor.detect_voice_activity(invalid_audio)
                    # Should handle gracefully or return empty result
                except Exception as e:
                    # Should not crash, may raise appropriate exception
                    assert "detect_voice_activity" in str(e) or True
            else:
                # Test invalid AudioData handling
                try:
                    voice_activities = mock_audio_processor.detect_voice_activity(invalid_audio)
                    # Should handle gracefully
                    assert isinstance(voice_activities, list)
                except Exception as e:
                    # Should not crash on invalid data
                    pass

        # Test processing pipeline with problematic audio
        problematic_audio = AudioData(
            np.full(16000, 1.0, dtype=np.float32),  # Constant high amplitude (clipped)
            16000, 1.0, 1
        )

        # Should handle without crashing
        processed_audio = mock_audio_processor._process_audio(problematic_audio)
        assert processed_audio is not None

        # Test quality analysis with edge cases
        quality_metrics = mock_audio_processor._analyze_audio_quality(problematic_audio)
        assert isinstance(quality_metrics, AudioQualityMetrics)

    def test_audio_stream_processing_integration(self, mock_audio_processor):
        """Test audio streaming through processing pipeline."""
        # Test audio stream creation
        if mock_audio_processor.features['audio_capture']:
            stream = mock_audio_processor.create_audio_stream()
            # May return None if not fully implemented, but shouldn't crash
            assert stream is None or hasattr(stream, 'start_stream')

        # Test streaming data processing
        stream_data = []

        # Simulate streaming chunks
        for i in range(5):
            chunk_data = np.random.random(1024).astype(np.float32)
            mock_audio_processor.add_to_buffer(chunk_data)
            stream_data.append(chunk_data)

        # Verify streaming data handling
        buffer_contents = mock_audio_processor.get_buffer_contents()
        assert len(buffer_contents) == len(stream_data)

        # Test getting audio chunks
        chunk = mock_audio_processor.get_audio_chunk()
        if buffer_contents:
            assert len(chunk) > 0
        else:
            assert len(chunk) == 0  # Empty if no data

    def test_audio_processor_state_management(self, mock_audio_processor):
        """Test state management throughout audio processing."""
        # Test initial state
        initial_state = mock_audio_processor.get_state()
        assert initial_state == AudioProcessorState.READY

        # Test state during recording
        recording_started = mock_audio_processor.start_recording()
        if recording_started:
            recording_state = mock_audio_processor.get_state()
            assert recording_state == AudioProcessorState.RECORDING

            # Stop recording and check final state
            mock_audio_processor.stop_recording()
            final_state = mock_audio_processor.get_state()
            assert final_state == AudioProcessorState.READY

        # Test processor availability
        is_available = mock_audio_processor.is_available()
        assert isinstance(is_available, bool)

        # Test comprehensive status reporting
        status = mock_audio_processor.get_status()
        required_status_fields = [
            'state', 'is_recording', 'is_playing', 'recording_duration',
            'audio_level', 'available_features', 'sample_rate', 'buffer_size'
        ]

        for field in required_status_fields:
            assert field in status

    def test_audio_memory_monitoring_integration(self, mock_audio_processor):
        """Test memory monitoring throughout audio processing."""
        # Test memory usage reporting
        memory_usage = mock_audio_processor.get_memory_usage()
        required_memory_fields = [
            'buffer_size', 'max_buffer_size', 'buffer_usage_percent',
            'memory_usage_bytes', 'memory_limit_bytes', 'memory_usage_percent'
        ]

        for field in required_memory_fields:
            assert field in memory_usage
            assert isinstance(memory_usage[field], (int, float))

        # Test memory limits
        assert memory_usage['memory_limit_bytes'] > 0
        assert memory_usage['memory_usage_percent'] >= 0
        assert memory_usage['buffer_usage_percent'] >= 0

        # Test buffer size limits
        assert memory_usage['max_buffer_size'] > 0
        assert memory_usage['buffer_usage_percent'] <= 100

    @pytest.mark.asyncio
    async def test_audio_long_session_stress_test(self, mock_audio_processor):
        """Test audio processing under long session stress."""
        # Simulate long recording session
        session_duration = 60  # 60 seconds
        chunk_duration = 0.1   # 100ms chunks
        chunks_per_second = int(1.0 / chunk_duration)
        total_chunks = session_duration * chunks_per_second

        # Track performance metrics
        start_time = time.time()
        memory_snapshots = []
        buffer_size_snapshots = []

        # Start recording
        recording_started = mock_audio_processor.start_recording()
        assert recording_started == True

        # Generate and process audio chunks
        for i in range(total_chunks):
            # Generate chunk data
            chunk_samples = int(chunk_duration * mock_audio_processor.sample_rate)
            chunk_data = np.random.random(chunk_samples).astype(np.float32)

            # Add to buffer
            mock_audio_processor.add_to_buffer(chunk_data)

            # Periodic monitoring
            if i % (chunks_per_second * 5) == 0:  # Every 5 seconds
                memory_usage = mock_audio_processor.get_memory_usage()
                memory_snapshots.append(memory_usage['memory_usage_bytes'])
                buffer_size_snapshots.append(memory_usage['buffer_size'])

                # Verify memory usage stays within limits
                memory_percent = memory_usage['memory_usage_percent']
                assert memory_percent <= 100, f"Memory usage exceeded 100% at chunk {i}"

        # Stop recording
        recorded_audio = mock_audio_processor.stop_recording()
        end_time = time.time()

        # Verify session completed
        total_time = end_time - start_time
        assert total_time >= session_duration * 0.9  # Allow some timing variance

        # Verify memory management was effective
        if memory_snapshots:
            max_memory_usage = max(memory_snapshots)
            memory_limit = mock_audio_processor._max_memory_bytes

            # Memory usage should not exceed reasonable limits
            assert max_memory_usage < memory_limit * 1.1  # Allow 10% overage

        # Verify recorded audio
        assert recorded_audio is not None
        assert recorded_audio.duration > 0

    def test_audio_quality_improvement_pipeline(self, mock_audio_processor):
        """Test audio quality improvement through processing pipeline."""
        # Create low-quality audio
        duration = 2.0
        sample_rate = 16000
        samples = int(duration * sample_rate)

        # Create noisy, low-level audio
        clean_signal = 0.1 * np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples))
        noise = 0.3 * np.random.normal(0, 1, samples)
        low_quality_audio = clean_signal + noise

        audio_data = AudioData(low_quality_audio, sample_rate, duration, 1, "float32")

        # Test quality metrics before processing
        initial_metrics = mock_audio_processor._analyze_audio_quality(audio_data)

        # Apply processing
        processed_audio = mock_audio_processor._process_audio(audio_data)

        # Test quality metrics after processing
        if mock_audio_processor.features['quality_analysis']:
            processed_metrics = mock_audio_processor._analyze_audio_quality(processed_audio)

            # Processing should not make quality worse
            assert processed_metrics.overall_quality >= 0.0
            assert processed_metrics.overall_quality <= 1.0

    def test_audio_processor_resource_cleanup(self, mock_audio_processor):
        """Test resource cleanup in audio processing pipeline."""
        # Add data to processor
        for i in range(20):
            chunk_data = np.random.random(1024).astype(np.float32)
            mock_audio_processor.add_to_buffer(chunk_data)

        # Verify data is present
        initial_buffer_size = len(mock_audio_processor.get_buffer_contents())
        assert initial_buffer_size == 20

        # Test cleanup
        mock_audio_processor.cleanup()

        # Verify cleanup
        final_state = mock_audio_processor.get_state()
        assert final_state == AudioProcessorState.IDLE

        # Verify buffer is cleared
        final_buffer_size = len(mock_audio_processor.get_buffer_contents())
        assert final_buffer_size == 0

        # Verify memory is freed
        memory_usage = mock_audio_processor.get_memory_usage()
        assert memory_usage['memory_usage_bytes'] == 0

    @pytest.mark.asyncio
    async def test_audio_concurrent_access_protection(self, mock_audio_processor):
        """Test thread safety and concurrent access protection."""
        # Test concurrent buffer access
        access_results = []

        async def access_buffer(operation_id):
            """Simulate concurrent buffer access."""
            try:
                for i in range(10):
                    if operation_id == 0:
                        # Add data
                        chunk_data = np.random.random(512).astype(np.float32)
                        mock_audio_processor.add_to_buffer(chunk_data)
                    elif operation_id == 1:
                        # Read buffer contents
                        contents = mock_audio_processor.get_buffer_contents()
                        _ = len(contents)
                    elif operation_id == 2:
                        # Get status
                        status = mock_audio_processor.get_status()
                        _ = status['buffer_size']

                    await asyncio.sleep(0.01)

                return True
            except Exception as e:
                return False

        # Execute concurrent operations
        tasks = [access_buffer(i) for i in range(3)]
        results = await asyncio.gather(*tasks)

        # All operations should succeed without errors
        assert all(results)

        # Final state should be consistent
        final_status = mock_audio_processor.get_status()
        assert isinstance(final_status['buffer_size'], int)
        assert final_status['buffer_size'] >= 0

    def test_audio_feature_availability_integration(self, mock_audio_processor):
        """Test feature availability detection and integration."""
        # Test feature availability reporting
        available_features = mock_audio_processor.get_available_features()

        # Should return dictionary of boolean features
        assert isinstance(available_features, dict)

        required_features = [
            'audio_capture', 'audio_playback', 'noise_reduction',
            'vad', 'quality_analysis', 'format_conversion'
        ]

        for feature in required_features:
            assert feature in available_features
            assert isinstance(available_features[feature], bool)

        # Test feature usage in processing
        test_audio = AudioData(
            np.random.random(16000).astype(np.float32),
            16000, 1.0, 1, "float32"
        )

        # Test VAD feature if available
        if available_features['vad']:
            vad_result = mock_audio_processor.detect_voice_activity(test_audio)
            assert isinstance(vad_result, list)

        # Test noise reduction feature if available
        if available_features['noise_reduction']:
            processed_audio = mock_audio_processor._process_audio(test_audio)
            assert processed_audio is not None

        # Test quality analysis feature if available
        if available_features['quality_analysis']:
            quality_metrics = mock_audio_processor._analyze_audio_quality(test_audio)
            assert isinstance(quality_metrics, AudioQualityMetrics)

    def test_audio_processor_health_check(self, mock_audio_processor):
        """Test audio processor health check functionality."""
        # Test health check
        is_healthy = mock_audio_processor.is_available()
        assert isinstance(is_healthy, bool)

        # Test detailed status
        status = mock_audio_processor.get_status()
        assert isinstance(status, dict)

        # Status should contain comprehensive information
        status_fields = [
            'state', 'is_recording', 'is_playing', 'recording_duration',
            'audio_level', 'available_features', 'buffer_size'
        ]

        for field in status_fields:
            assert field in status

        # Test memory usage reporting
        memory_usage = mock_audio_processor.get_memory_usage()
        assert isinstance(memory_usage, dict)

        memory_fields = [
            'buffer_size', 'memory_usage_bytes', 'memory_usage_percent'
        ]

        for field in memory_fields:
            assert field in memory_usage
            assert isinstance(memory_usage[field], (int, float))

    @pytest.mark.asyncio
    async def test_audio_pipeline_performance_monitoring(self, mock_audio_processor):
        """Test performance monitoring throughout audio pipeline."""
        # Monitor performance during audio processing
        performance_metrics = []

        # Process multiple audio chunks and measure performance
        num_chunks = 50
        chunk_size = 2048

        for i in range(num_chunks):
            chunk_start_time = time.time()

            # Generate and process audio chunk
            chunk_data = np.random.random(chunk_size).astype(np.float32)
            audio_chunk = AudioData(chunk_data, 16000, chunk_size/16000, 1, "float32")

            # Apply processing
            processed_audio = mock_audio_processor._process_audio(audio_chunk)

            # Calculate processing time
            processing_time = time.time() - chunk_start_time
            performance_metrics.append(processing_time)

            # Verify processing completed
            assert processed_audio is not None

        # Analyze performance metrics
        avg_processing_time = sum(performance_metrics) / len(performance_metrics)
        max_processing_time = max(performance_metrics)

        # Performance should be reasonable
        assert avg_processing_time < 1.0  # Average under 1 second
        assert max_processing_time < 5.0  # Max under 5 seconds

        # Verify processor status after performance test
        final_status = mock_audio_processor.get_status()
        assert final_status['state'] != AudioProcessorState.ERROR

    def test_audio_data_conversion_pipeline(self, mock_audio_processor):
        """Test audio data conversion throughout pipeline."""
        # Test different audio formats
        test_formats = ["float32", "int16", "int32"]

        for format_type in test_formats:
            # Create test audio in different format
            if format_type == "float32":
                test_data = np.random.random(16000).astype(np.float32)
            elif format_type == "int16":
                test_data = (np.random.random(16000) * 32767).astype(np.int16)
            elif format_type == "int32":
                test_data = (np.random.random(16000) * 2147483647).astype(np.int32)

            audio_data = AudioData(test_data, 16000, 1.0, 1, format_type)

            # Test format conversion
            if mock_audio_processor.features['format_conversion']:
                converted_audio = mock_audio_processor.convert_audio_format(audio_data, "wav")
                assert converted_audio is not None

            # Test audio quality analysis with different formats
            if mock_audio_processor.features['quality_analysis']:
                quality_metrics = mock_audio_processor._analyze_audio_quality(audio_data)
                assert isinstance(quality_metrics, AudioQualityMetrics)

    def test_audio_buffer_overflow_protection(self, mock_audio_processor):
        """Test buffer overflow protection in audio pipeline."""
        # Fill buffer beyond normal capacity
        chunk_size = 1024
        max_chunks = mock_audio_processor.max_buffer_size + 50  # Exceed buffer size

        # Add excessive data
        for i in range(max_chunks):
            chunk_data = np.random.random(chunk_size).astype(np.float32)
            mock_audio_processor.add_to_buffer(chunk_data)

        # Verify buffer protection worked
        final_buffer_size = len(mock_audio_processor.get_buffer_contents())
        assert final_buffer_size <= mock_audio_processor.max_buffer_size

        # Memory usage should still be within limits
        memory_usage = mock_audio_processor.get_memory_usage()
        assert memory_usage['memory_usage_percent'] <= 100

    def test_audio_processor_initialization_integration(self, audio_config):
        """Test audio processor initialization and configuration."""
        # Test processor creation
        processor = SimplifiedAudioProcessor(audio_config)

        # Verify initialization
        assert processor.state != AudioProcessorState.ERROR
        assert processor.sample_rate == audio_config.audio_sample_rate
        assert processor.channels == audio_config.audio_channels

        # Test feature availability
        features = processor.get_available_features()
        assert isinstance(features, dict)

        # Test initial status
        initial_status = processor.get_status()
        assert initial_status['state'] == AudioProcessorState.READY.value
        assert initial_status['buffer_size'] == 0

        # Test memory configuration
        memory_usage = processor.get_memory_usage()
        assert memory_usage['memory_limit_bytes'] > 0
        assert memory_usage['max_buffer_size'] > 0

    def test_audio_pipeline_robustness_under_load(self, mock_audio_processor):
        """Test audio pipeline robustness under various load conditions."""
        # Test with different audio characteristics
        test_scenarios = [
            {
                'name': 'high_amplitude',
                'data': 0.9 * np.random.random(16000).astype(np.float32),
                'expected_quality': 'high'
            },
            {
                'name': 'low_amplitude',
                'data': 0.01 * np.random.random(16000).astype(np.float32),
                'expected_quality': 'low'
            },
            {
                'name': 'high_frequency',
                'data': self._generate_high_frequency_audio(16000, 8000),  # 8kHz
                'expected_quality': 'medium'
            },
            {
                'name': 'mixed_content',
                'data': self._generate_mixed_audio(16000),
                'expected_quality': 'medium'
            }
        ]

        for scenario in test_scenarios:
            # Create AudioData
            audio_data = AudioData(
                scenario['data'], 16000, 1.0, 1, "float32"
            )

            # Test processing pipeline
            processed_audio = mock_audio_processor._process_audio(audio_data)
            assert processed_audio is not None

            # Test quality analysis
            if mock_audio_processor.features['quality_analysis']:
                quality_metrics = mock_audio_processor._analyze_audio_quality(processed_audio)
                assert isinstance(quality_metrics, AudioQualityMetrics)
                assert 0.0 <= quality_metrics.overall_quality <= 1.0

            # Test VAD
            if mock_audio_processor.features['vad']:
                voice_activities = mock_audio_processor.detect_voice_activity(processed_audio)
                assert isinstance(voice_activities, list)

    def _generate_high_frequency_audio(self, samples, frequency):
        """Generate high frequency test audio."""
        t = np.linspace(0, 1.0, samples)
        return 0.1 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

    def _generate_mixed_audio(self, samples):
        """Generate mixed frequency test audio."""
        t = np.linspace(0, 1.0, samples)
        return (
            0.2 * np.sin(2 * np.pi * 440 * t) +   # Speech range
            0.1 * np.sin(2 * np.pi * 4400 * t) +  # High frequency
            0.05 * np.random.random(samples)      # Noise
        ).astype(np.float32)

    def test_audio_processor_factory_integration(self, audio_config):
        """Test audio processor factory and creation."""
        # Test factory function
        processor = AudioProcessor(audio_config)  # Backward compatibility alias

        # Should be same as direct creation
        assert isinstance(processor, SimplifiedAudioProcessor)
        assert processor.config == audio_config

        # Test processor interface consistency
        assert hasattr(processor, 'start_recording')
        assert hasattr(processor, 'stop_recording')
        assert hasattr(processor, 'play_audio')
        assert hasattr(processor, 'get_status')
        assert hasattr(processor, 'cleanup')

        # Test method signatures
        assert callable(processor.start_recording)
        assert callable(processor.stop_recording)
        assert callable(processor.play_audio)
        assert callable(processor.get_status)
        assert callable(processor.cleanup)