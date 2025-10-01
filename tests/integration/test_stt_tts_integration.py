"""
STT/TTS Service Integration Tests

Tests the integration between Speech-to-Text and Text-to-Speech services,
focusing on multi-provider fallback, error handling, and performance optimization.
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import MagicMock, patch, AsyncMock
import numpy as np
from typing import Dict, List, Any
import psutil
import os

from voice.stt_service import STTService, STTResult
from voice.tts_service import TTSService, TTSResult, EmotionType
from voice.config import VoiceConfig
from voice.audio_processor import AudioData


class TestSTTTTSIntegration:
    """Test STT and TTS service integration."""

    @pytest.fixture
    def stt_config(self):
        """Create STT-focused configuration."""
        config = VoiceConfig()
        config.voice_enabled = True
        config.stt_confidence_threshold = 0.7
        config.stt_language = "en-US"
        config.stt_timeout = 10.0
        config.stt_max_retries = 2
        config.stt_fallback_enabled = True
        return config

    @pytest.fixture
    def tts_config(self):
        """Create TTS-focused configuration."""
        config = VoiceConfig()
        config.voice_enabled = True
        config.tts_voice = "alloy"
        config.tts_speed = 1.0
        config.tts_pitch = 1.0
        config.tts_model = "tts-1"
        config.default_voice_profile = "calm_therapist"
        config.tts_fallback_enabled = True
        return config

    @pytest.fixture
    def mock_stt_service(self, stt_config):
        """Create mock STT service with realistic behavior."""
        with patch('voice.stt_service.openai') as mock_openai, \
             patch('voice.stt_service.whisper') as mock_whisper:

            service = STTService(stt_config)

            # Configure OpenAI mock
            mock_openai.Audio = MagicMock()
            mock_openai.Audio.transcribe = MagicMock(return_value={
                'text': 'Mock transcription from OpenAI',
                'language': 'en',
                'duration': 2.0,
                'segments': []
            })

            # Configure Whisper mock
            mock_whisper_model = MagicMock()
            mock_whisper_model.transcribe = MagicMock(return_value={
                'text': 'Mock transcription from Whisper',
                'language': 'en',
                'segments': []
            })
            mock_whisper.load_model = MagicMock(return_value=mock_whisper_model)

            # Initialize service
            service.openai_client = mock_openai
            service.whisper_model = mock_whisper_model

            return service

    @pytest.fixture
    def mock_tts_service(self, tts_config):
        """Create mock TTS service with realistic behavior."""
        with patch('voice.tts_service.openai.OpenAI') as mock_openai_client, \
             patch('voice.tts_service.elevenlabs') as mock_elevenlabs:

            service = TTSService(tts_config)

            # Configure OpenAI mock
            mock_client_instance = MagicMock()
            mock_client_instance.audio.speech.create = MagicMock()
            mock_client_instance.audio.speech.create.return_value.content = b'mock_audio_data_16bit'
            mock_openai_client.return_value = mock_client_instance

            # Configure ElevenLabs mock
            mock_elevenlabs.generate = MagicMock(return_value=[b'mock_audio_chunk1', b'mock_audio_chunk2'])
            mock_elevenlabs.VoiceSettings = MagicMock()

            # Initialize service
            service.openai_client = mock_client_instance

            return service

    @pytest.fixture
    def comprehensive_voice_config(self):
        """Create comprehensive voice configuration for integration testing."""
        config = VoiceConfig()
        config.voice_enabled = True
        config.voice_input_enabled = True
        config.voice_output_enabled = True
        config.voice_commands_enabled = True
        config.security_enabled = True
        config.encryption_enabled = True

        # STT configuration
        config.stt_confidence_threshold = 0.7
        config.stt_language = "en-US"
        config.stt_timeout = 10.0
        config.stt_max_retries = 2
        config.stt_fallback_enabled = True

        # TTS configuration
        config.tts_voice = "alloy"
        config.tts_speed = 1.0
        config.tts_pitch = 1.0
        config.tts_model = "tts-1"
        config.default_voice_profile = "calm_therapist"
        config.tts_fallback_enabled = True

        return config

    @pytest.mark.asyncio
    async def test_stt_tts_basic_integration(self, mock_stt_service, mock_tts_service):
        """Test basic STT to TTS integration."""
        # Test audio data
        mock_audio = AudioData(
            np.random.randint(-32768, 32767, 16000, dtype=np.int16),
            16000, 1.0, 1
        )

        # Process STT
        stt_result = await mock_stt_service.transcribe_audio(mock_audio, "openai")

        # Verify STT result
        assert stt_result is not None
        assert stt_result.text == 'Mock transcription from OpenAI'
        assert stt_result.provider == 'openai'
        assert stt_result.confidence > 0

        # Generate TTS from STT result
        tts_result = await mock_tts_service.synthesize_speech(
            stt_result.text,
            voice_profile="calm_therapist",
            provider="openai"
        )

        # Verify TTS result
        assert tts_result is not None
        assert tts_result.text == stt_result.text
        assert tts_result.provider == 'openai'
        assert tts_result.audio_data is not None
        assert len(tts_result.audio_data.data) > 0

    @pytest.mark.asyncio
    async def test_multi_provider_stt_fallback(self, stt_config):
        """Test STT service fallback between multiple providers."""
        with patch('voice.stt_service.openai') as mock_openai, \
             patch('voice.stt_service.whisper') as mock_whisper:

            service = STTService(stt_config)

            # Configure primary provider to fail
            mock_openai.Audio = MagicMock()
            mock_openai.Audio.transcribe = MagicMock(side_effect=Exception("OpenAI API unavailable"))

            # Configure fallback provider to succeed
            mock_whisper_model = MagicMock()
            mock_whisper_model.transcribe = MagicMock(return_value={
                'text': 'Fallback transcription successful',
                'language': 'en',
                'segments': []
            })
            mock_whisper.load_model = MagicMock(return_value=mock_whisper_model)

            service.openai_client = mock_openai
            service.whisper_model = mock_whisper_model

            # Test audio
            mock_audio = AudioData(
                np.random.randint(-32768, 32767, 16000, dtype=np.int16),
                16000, 1.0, 1
            )

            # Should succeed with fallback
            stt_result = await service.transcribe_audio(mock_audio, "openai")

            assert stt_result is not None
            assert stt_result.text == 'Fallback transcription successful'
            assert stt_result.provider == 'whisper'  # Should use fallback

    @pytest.mark.asyncio
    async def test_multi_provider_tts_fallback(self, tts_config):
        """Test TTS service fallback between multiple providers."""
        with patch('voice.tts_service.openai.OpenAI') as mock_openai_client, \
             patch('voice.tts_service.elevenlabs') as mock_elevenlabs:

            service = TTSService(tts_config)

            # Configure primary provider to fail
            mock_client_instance = MagicMock()
            mock_client_instance.audio.speech.create = MagicMock(side_effect=Exception("OpenAI API unavailable"))
            mock_openai_client.return_value = mock_client_instance

            # Configure fallback provider to succeed
            mock_elevenlabs.generate = MagicMock(return_value=[b'mock_audio_chunk1', b'mock_audio_chunk2'])
            mock_elevenlabs.VoiceSettings = MagicMock()

            service.openai_client = mock_client_instance

            # Should succeed with fallback
            tts_result = await service.synthesize_speech(
                "Test message for fallback",
                provider="openai"  # Request primary but get fallback
            )

            assert tts_result is not None
            assert tts_result.text == 'Test message for fallback'

    @pytest.mark.asyncio
    async def test_concurrent_stt_tts_processing(self, stt_config, tts_config):
        """Test concurrent STT and TTS processing."""
        with patch('voice.stt_service.openai') as mock_openai_stt, \
             patch('voice.tts_service.openai.OpenAI') as mock_openai_tts:

            stt_service = STTService(stt_config)
            tts_service = TTSService(tts_config)

            # Configure STT service
            mock_openai_stt.Audio = MagicMock()
            mock_openai_stt.Audio.transcribe = MagicMock(return_value={
                'text': f'Message {0}',
                'language': 'en',
                'duration': 2.0,
                'segments': []
            })
            stt_service.openai_client = mock_openai_stt

            # Configure TTS service
            mock_client_instance = MagicMock()
            mock_client_instance.audio.speech.create = MagicMock()
            mock_client_instance.audio.speech.create.return_value.content = b'mock_audio_data'
            mock_openai_tts.return_value = mock_client_instance
            tts_service.openai_client = mock_client_instance

            # Test concurrent processing
            num_concurrent = 10
            tasks = []

            for i in range(num_concurrent):
                # Create unique audio for each request
                mock_audio = AudioData(
                    np.random.randint(-32768, 32767, 16000, dtype=np.int16),
                    16000, 1.0, 1
                )

                # Create concurrent task
                task = asyncio.create_task(self._process_stt_tts_pair(
                    stt_service, tts_service, mock_audio, i
                ))
                tasks.append(task)

            # Wait for all tasks
            results = await asyncio.gather(*tasks)

            # Verify all processed successfully
            assert len(results) == num_concurrent
            assert all(result is not None for result in results)

            # Check performance
            stt_stats = stt_service.get_statistics()
            tts_stats = tts_service.get_statistics()

            assert stt_stats['request_count'] == num_concurrent
            assert tts_stats['request_count'] == num_concurrent

    async def _process_stt_tts_pair(self, stt_service, tts_service, audio, index):
        """Process one STT-TTS pair."""
        # STT processing
        stt_result = await stt_service.transcribe_audio(audio)

        # TTS processing
        tts_result = await tts_service.synthesize_speech(
            stt_result.text,
            voice_profile="calm_therapist"
        )

        return {
            'stt_text': stt_result.text,
            'tts_audio_length': len(tts_result.audio_data.data)
        }

    @pytest.mark.asyncio
    async def test_stt_tts_error_recovery(self, stt_config, tts_config):
        """Test error recovery in STT-TTS pipeline."""
        with patch('voice.stt_service.openai') as mock_openai_stt, \
             patch('voice.tts_service.openai.OpenAI') as mock_openai_tts:

            stt_service = STTService(stt_config)
            tts_service = TTSService(tts_config)

            # Configure intermittent failures
            failure_count = {'stt': 0, 'tts': 0}

            def failing_stt_transcribe(*args, **kwargs):
                failure_count['stt'] += 1
                if failure_count['stt'] % 3 == 0:  # Fail every 3rd request
                    raise Exception("Intermittent STT failure")
                return {
                    'text': f'Successful transcription {failure_count["stt"]}',
                    'language': 'en',
                    'duration': 2.0,
                    'segments': []
                }

            def failing_tts_create(*args, **kwargs):
                failure_count['tts'] += 1
                if failure_count['tts'] % 3 == 0:  # Fail every 3rd request
                    raise Exception("Intermittent TTS failure")
                mock_response = MagicMock()
                mock_response.content = b'mock_audio_data'
                return mock_response

            # Configure services
            mock_openai_stt.Audio = MagicMock()
            mock_openai_stt.Audio.transcribe = MagicMock(side_effect=failing_stt_transcribe)
            stt_service.openai_client = mock_openai_stt

            mock_client_instance = MagicMock()
            mock_client_instance.audio.speech.create = MagicMock(side_effect=failing_tts_create)
            mock_openai_tts.return_value = mock_client_instance
            tts_service.openai_client = mock_client_instance

            # Test error recovery
            success_count = 0
            total_requests = 10

            for i in range(total_requests):
                try:
                    mock_audio = AudioData(
                        np.random.randint(-32768, 32767, 16000, dtype=np.int16),
                        16000, 1.0, 1
                    )

                    # Process STT
                    stt_result = await stt_service.transcribe_audio(mock_audio)

                    # Process TTS
                    tts_result = await tts_service.synthesize_speech(
                        stt_result.text,
                        voice_profile="calm_therapist"
                    )

                    if stt_result and tts_result:
                        success_count += 1

                except Exception as e:
                    # Expected for some requests due to intermittent failures
                    pass

            # Should have some successful requests despite failures
            assert success_count > 0
            assert success_count < total_requests  # Some should have failed

    @pytest.mark.asyncio
    async def test_stt_tts_memory_management(self, stt_config, tts_config):
        """Test memory management during extended STT-TTS processing."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        with patch('voice.stt_service.openai') as mock_openai_stt, \
             patch('voice.tts_service.openai.OpenAI') as mock_openai_tts:

            stt_service = STTService(stt_config)
            tts_service = TTSService(tts_config)

            # Configure services
            mock_openai_stt.Audio = MagicMock()
            mock_openai_stt.Audio.transcribe = MagicMock(return_value={
                'text': 'Memory management test',
                'language': 'en',
                'duration': 2.0,
                'segments': []
            })
            stt_service.openai_client = mock_openai_stt

            mock_client_instance = MagicMock()
            mock_client_instance.audio.speech.create = MagicMock()
            mock_client_instance.audio.speech.create.return_value.content = b'mock_audio_data'
            mock_openai_tts.return_value = mock_client_instance
            tts_service.openai_client = mock_client_instance

            # Process many requests to test memory management
            num_requests = 100

            for i in range(num_requests):
                mock_audio = AudioData(
                    np.random.randint(-32768, 32767, 16000, dtype=np.int16),
                    16000, 1.0, 1
                )

                # STT processing
                stt_result = await stt_service.transcribe_audio(mock_audio)

                # TTS processing
                tts_result = await tts_service.synthesize_speech(
                    stt_result.text,
                    voice_profile="calm_therapist"
                )

                # Verify results
                assert stt_result is not None
                assert tts_result is not None

                # Periodically check memory usage
                if i % 20 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_growth = current_memory - initial_memory

                    # Memory shouldn't grow excessively (allow up to 100MB growth)
                    assert memory_growth < 100, f"Memory grew by {memory_growth:.2f}MB after {i} requests"

            # Final memory check
            final_memory = process.memory_info().rss / 1024 / 1024
            total_memory_growth = final_memory - initial_memory

            # Should not have excessive memory growth
            assert total_memory_growth < 150, f"Total memory growth {total_memory_growth".2f"}MB seems excessive"

    @pytest.mark.asyncio
    async def test_stt_tts_performance_benchmarks(self, stt_config, tts_config):
        """Test STT-TTS performance under various loads."""
        with patch('voice.stt_service.openai') as mock_openai_stt, \
             patch('voice.tts_service.openai.OpenAI') as mock_openai_tts:

            stt_service = STTService(stt_config)
            tts_service = TTSService(tts_config)

            # Configure services
            mock_openai_stt.Audio = MagicMock()
            mock_openai_stt.Audio.transcribe = MagicMock(return_value={
                'text': 'Performance benchmark test',
                'language': 'en',
                'duration': 2.0,
                'segments': []
            })
            stt_service.openai_client = mock_openai_stt

            mock_client_instance = MagicMock()
            mock_client_instance.audio.speech.create = MagicMock()
            mock_client_instance.audio.speech.create.return_value.content = b'mock_audio_data'
            mock_openai_tts.return_value = mock_client_instance
            tts_service.openai_client = mock_client_instance

            # Test different load levels
            load_levels = [1, 5, 10, 20]

            for num_concurrent in load_levels:
                start_time = time.time()

                # Create concurrent tasks
                tasks = []
                for i in range(num_concurrent):
                    mock_audio = AudioData(
                        np.random.randint(-32768, 32767, 16000, dtype=np.int16),
                        16000, 1.0, 1
                    )

                    task = asyncio.create_task(self._process_stt_tts_pair(
                        stt_service, tts_service, mock_audio, i
                    ))
                    tasks.append(task)

                # Wait for completion
                results = await asyncio.gather(*tasks)
                end_time = time.time()

                total_time = end_time - start_time
                avg_time_per_request = total_time / num_concurrent

                # Performance assertions
                assert total_time > 0
                assert avg_time_per_request < 2.0  # Should be under 2 seconds per request

                # Verify all requests succeeded
                assert len(results) == num_concurrent
                assert all(result is not None for result in results)

    @pytest.mark.asyncio
    async def test_stt_tts_caching_integration(self, stt_config, tts_config):
        """Test caching integration between STT and TTS services."""
        with patch('voice.stt_service.openai') as mock_openai_stt, \
             patch('voice.tts_service.openai.OpenAI') as mock_openai_tts:

            stt_service = STTService(stt_config)
            tts_service = TTSService(tts_config)

            # Configure services
            mock_openai_stt.Audio = MagicMock()
            stt_service.openai_client = mock_openai_stt

            mock_client_instance = MagicMock()
            mock_client_instance.audio.speech.create = MagicMock()
            mock_client_instance.audio.speech.create.return_value.content = b'mock_audio_data'
            mock_openai_tts.return_value = mock_client_instance
            tts_service.openai_client = mock_client_instance

            # Test same audio processed multiple times
            same_audio = AudioData(
                np.random.randint(-32768, 32767, 16000, dtype=np.int16),
                16000, 1.0, 1
            )

            # Process same audio multiple times
            results = []
            for i in range(5):
                stt_result = await stt_service.transcribe_audio(same_audio)
                tts_result = await tts_service.synthesize_speech(
                    stt_result.text,
                    voice_profile="calm_therapist"
                )
                results.append((stt_result, tts_result))

            # All results should be identical due to caching
            first_stt, first_tts = results[0]
            for stt_result, tts_result in results[1:]:
                assert stt_result.text == first_stt.text
                assert tts_result.text == first_tts.text

            # Check cache statistics
            stt_stats = stt_service.get_statistics()
            tts_stats = tts_service.get_statistics()

            # Should have processed multiple requests
            assert stt_stats['request_count'] == 5
            assert tts_stats['request_count'] == 5

    @pytest.mark.asyncio
    async def test_stt_tts_emotion_handling(self, tts_config):
        """Test emotion handling in TTS service."""
        with patch('voice.tts_service.openai.OpenAI') as mock_openai_tts:

            service = TTSService(tts_config)

            # Configure TTS service
            mock_client_instance = MagicMock()
            mock_client_instance.audio.speech.create = MagicMock()
            mock_client_instance.audio.speech.create.return_value.content = b'mock_audio_data'
            mock_openai_tts.return_value = mock_client_instance
            service.openai_client = mock_client_instance

            # Test different emotions
            emotions = [
                EmotionType.CALM,
                EmotionType.EMPATHETIC,
                EmotionType.PROFESSIONAL,
                EmotionType.ENCOURAGING,
                EmotionType.SUPPORTIVE
            ]

            results = []
            for emotion in emotions:
                tts_result = await service.synthesize_speech(
                    "I understand you're feeling anxious",
                    voice_profile="calm_therapist",
                    emotion=emotion
                )
                results.append(tts_result)

                # Verify emotion was applied
                assert tts_result is not None
                assert tts_result.emotion == emotion.value

            # All requests should succeed
            assert len(results) == len(emotions)

            # Check service statistics
            stats = service.get_statistics()
            assert stats['request_count'] == len(emotions)

    @pytest.mark.asyncio
    async def test_stt_tts_streaming_integration(self, stt_config, tts_config):
        """Test streaming integration between STT and TTS."""
        with patch('voice.stt_service.openai') as mock_openai_stt, \
             patch('voice.tts_service.openai.OpenAI') as mock_openai_tts, \
             patch('voice.tts_service.elevenlabs') as mock_elevenlabs:

            stt_service = STTService(stt_config)
            tts_service = TTSService(tts_config)

            # Configure STT service
            mock_openai_stt.Audio = MagicMock()
            mock_openai_stt.Audio.transcribe = MagicMock(return_value={
                'text': 'Streaming test message',
                'language': 'en',
                'duration': 2.0,
                'segments': []
            })
            stt_service.openai_client = mock_openai_stt

            # Configure TTS service for streaming
            mock_client_instance = MagicMock()
            mock_openai_tts.return_value = mock_client_instance
            tts_service.openai_client = mock_client_instance

            # Mock streaming response
            mock_elevenlabs.generate_stream = MagicMock(return_value=[
                b'chunk1', b'chunk2', b'chunk3', b'chunk4'
            ])
            mock_elevenlabs.VoiceSettings = MagicMock()

            # Test streaming TTS
            text = "This is a test of streaming text-to-speech synthesis"
            audio_chunks = []

            async for chunk in tts_service.synthesize_stream(
                text,
                voice_profile="calm_therapist",
                provider="openai"
            ):
                audio_chunks.append(chunk)

            # Verify streaming worked
            assert len(audio_chunks) > 0

            # Each chunk should be valid AudioData
            for chunk in audio_chunks:
                assert chunk is not None
                assert len(chunk.data) > 0
                assert chunk.sample_rate > 0

    def test_stt_tts_service_health_integration(self, stt_config, tts_config):
        """Test health check integration between STT and TTS services."""
        with patch('voice.stt_service.openai') as mock_openai_stt, \
             patch('voice.tts_service.openai.OpenAI') as mock_openai_tts:

            stt_service = STTService(stt_config)
            tts_service = TTSService(tts_config)

            # Configure services
            mock_openai_stt.Audio = MagicMock()
            mock_openai_stt.Audio.transcribe = MagicMock(return_value={
                'text': 'Health check test',
                'language': 'en',
                'duration': 1.0,
                'segments': []
            })
            stt_service.openai_client = mock_openai_stt

            mock_client_instance = MagicMock()
            mock_client_instance.audio.speech.create = MagicMock()
            mock_client_instance.audio.speech.create.return_value.content = b'mock_audio_data'
            mock_openai_tts.return_value = mock_client_instance
            tts_service.openai_client = mock_client_instance

            # Test service health
            stt_healthy = stt_service.test_service("openai")
            tts_healthy = tts_service.test_voice_profile("calm_therapist")

            assert stt_healthy == True
            assert tts_healthy == True

            # Test service statistics integration
            stt_stats = stt_service.get_statistics()
            tts_stats = tts_service.get_statistics()

            # Both should have statistics
            assert 'request_count' in stt_stats
            assert 'request_count' in tts_stats
            assert 'error_count' in stt_stats
            assert 'error_count' in tts_stats

    @pytest.mark.asyncio
    async def test_stt_tts_data_flow_integrity(self, stt_config, tts_config):
        """Test data flow integrity through STT-TTS pipeline."""
        with patch('voice.stt_service.openai') as mock_openai_stt, \
             patch('voice.tts_service.openai.OpenAI') as mock_openai_tts:

            stt_service = STTService(stt_config)
            tts_service = TTSService(tts_config)

            # Configure services
            original_text = "I'm feeling anxious and need help with coping strategies"
            mock_openai_stt.Audio = MagicMock()
            mock_openai_stt.Audio.transcribe = MagicMock(return_value={
                'text': original_text,
                'language': 'en',
                'duration': 3.0,
                'segments': [
                    {'text': 'I\'m feeling anxious', 'start': 0.0, 'end': 1.5},
                    {'text': 'and need help with coping strategies', 'start': 1.5, 'end': 3.0}
                ]
            })
            stt_service.openai_client = mock_openai_stt

            mock_client_instance = MagicMock()
            mock_client_instance.audio.speech.create = MagicMock()
            mock_client_instance.audio.speech.create.return_value.content = b'mock_audio_data'
            mock_openai_tts.return_value = mock_client_instance
            tts_service.openai_client = mock_client_instance

            # Test complete pipeline
            mock_audio = AudioData(
                np.random.randint(-32768, 32767, 48000, dtype=np.int16),  # 3 seconds
                16000, 3.0, 1
            )

            # 1. STT Processing
            stt_result = await stt_service.transcribe_audio(mock_audio)

            # Verify STT data integrity
            assert stt_result.text == original_text
            assert stt_result.confidence > 0
            assert stt_result.duration == 3.0
            assert len(stt_result.word_timestamps) == 2

            # 2. TTS Processing
            tts_result = await tts_service.synthesize_speech(
                stt_result.text,
                voice_profile="empathetic_guide",
                emotion=EmotionType.EMPATHETIC
            )

            # Verify TTS data integrity
            assert tts_result.text == original_text
            assert tts_result.voice_profile == "empathetic_guide"
            assert tts_result.emotion == "empathetic"
            assert tts_result.audio_data is not None
            assert tts_result.duration > 0
            assert tts_result.processing_time > 0

            # 3. Verify end-to-end data preservation
            assert tts_result.text == stt_result.text  # Text should be preserved

    @pytest.mark.asyncio
    async def test_stt_tts_therapy_keyword_detection(self, stt_config):
        """Test therapy keyword detection in STT service."""
        with patch('voice.stt_service.openai') as mock_openai_stt:

            service = STTService(stt_config)

            # Configure STT service with therapy keywords
            mock_openai_stt.Audio = MagicMock()
            stt_service.openai_client = mock_openai_stt

            # Test different therapy scenarios
            therapy_scenarios = [
                {
                    'text': "I'm feeling depressed and need help",
                    'expected_keywords': ['depression'],
                    'expected_crisis': False
                },
                {
                    'text': "I want to kill myself and end it all",
                    'expected_keywords': ['suicide', 'kill myself', 'end it all'],
                    'expected_crisis': True
                },
                {
                    'text': "Can you help me with anxiety and panic attacks?",
                    'expected_keywords': ['anxiety', 'panic'],
                    'expected_crisis': False
                },
                {
                    'text': "I'm struggling with PTSD and trauma",
                    'expected_keywords': ['trauma'],
                    'expected_crisis': False
                }
            ]

            for scenario in therapy_scenarios:
                # Mock STT response
                mock_openai_stt.Audio.transcribe = MagicMock(return_value={
                    'text': scenario['text'],
                    'language': 'en',
                    'duration': 2.0,
                    'segments': []
                })

                mock_audio = AudioData(
                    np.random.randint(-32768, 32767, 32000, dtype=np.int16),
                    16000, 2.0, 1
                )

                # Process STT
                stt_result = await service.transcribe_audio(mock_audio)

                # Verify therapy keyword detection
                assert stt_result is not None

                # Check detected keywords
                for keyword in scenario['expected_keywords']:
                    assert keyword in stt_result.therapy_keywords_detected

                # Check crisis detection
                assert stt_result.is_crisis == scenario['expected_crisis']

                # Check sentiment analysis
                assert stt_result.sentiment is not None
                assert 'score' in stt_result.sentiment

    def test_stt_tts_configuration_integration(self, comprehensive_voice_config):
        """Test configuration integration between STT and TTS services."""
        # Create both services with same config
        stt_service = STTService(comprehensive_voice_config)
        tts_service = TTSService(comprehensive_voice_config)

        # Verify configuration consistency
        assert stt_service.config.voice_enabled == tts_service.config.voice_enabled
        assert stt_service.config.security_enabled == tts_service.config.security_enabled
        assert stt_service.config.encryption_enabled == tts_service.config.encryption_enabled

        # Test voice profile consistency
        stt_voice_profile = stt_service.get_preferred_provider()
        tts_voice_profile = tts_service.get_preferred_provider()

        # Both should be configured
        assert stt_voice_profile != "none"
        assert tts_voice_profile != "none"

        # Test service availability
        stt_available = stt_service.is_available()
        tts_available = tts_service.is_available()

        # Both should be available (even if mocked)
        assert stt_available == True
        assert tts_available == True

    def test_stt_tts_cleanup_integration(self, stt_config, tts_config):
        """Test cleanup integration between STT and TTS services."""
        stt_service = STTService(stt_config)
        tts_service = TTSService(tts_config)

        # Add some data to caches
        stt_service.cache = {'test_key': 'test_value'}
        tts_service.audio_cache = {'test_key': 'test_value'}

        # Verify cleanup
        stt_service.cleanup()
        tts_service.cleanup()

        # Caches should be cleared
        assert len(stt_service.cache) == 0
        assert len(tts_service.audio_cache) == 0

        # Service instances should be cleaned up
        assert stt_service.openai_client is None
        assert stt_service.whisper_model is None
        assert tts_service.openai_client is None

    @pytest.mark.asyncio
    async def test_stt_tts_error_boundary_integration(self, stt_config, tts_config):
        """Test error boundary handling in STT-TTS integration."""
        with patch('voice.stt_service.openai') as mock_openai_stt, \
             patch('voice.tts_service.openai.OpenAI') as mock_openai_tts:

            stt_service = STTService(stt_config)
            tts_service = TTSService(tts_config)

            # Configure services to fail
            mock_openai_stt.Audio = MagicMock()
            mock_openai_stt.Audio.transcribe = MagicMock(side_effect=Exception("STT service error"))

            mock_client_instance = MagicMock()
            mock_client_instance.audio.speech.create = MagicMock(side_effect=Exception("TTS service error"))
            mock_openai_tts.return_value = mock_client_instance

            stt_service.openai_client = mock_openai_stt
            tts_service.openai_client = mock_client_instance

            # Test error handling
            mock_audio = AudioData(
                np.random.randint(-32768, 32767, 16000, dtype=np.int16),
                16000, 1.0, 1
            )

            # STT should handle errors gracefully
            with pytest.raises(RuntimeError):  # Should raise when no fallback available
                await stt_service.transcribe_audio(mock_audio)

            # TTS should handle errors gracefully
            with pytest.raises(Exception):  # Should raise TTS error
                await tts_service.synthesize_speech("Test message")

            # Check error statistics
            stt_stats = stt_service.get_statistics()
            tts_stats = tts_service.get_statistics()

            assert stt_stats['error_count'] > 0
            assert tts_stats['error_count'] > 0

    def test_stt_tts_resource_monitoring(self, stt_config, tts_config):
        """Test resource monitoring in STT-TTS integration."""
        stt_service = STTService(stt_config)
        tts_service = TTSService(tts_config)

        # Test resource monitoring
        stt_stats = stt_service.get_statistics()
        tts_stats = tts_service.get_statistics()

        # Both should provide resource statistics
        required_stats = [
            'request_count', 'error_count', 'success_rate',
            'average_processing_time', 'available_providers'
        ]

        for stat in required_stats:
            assert stat in stt_stats
            assert stat in tts_stats

        # Statistics should be reasonable
        assert stt_stats['request_count'] >= 0
        assert tts_stats['request_count'] >= 0
        assert 0.0 <= stt_stats['success_rate'] <= 1.0
        assert stt_stats['average_processing_time'] >= 0.0
        assert tts_stats['average_processing_time'] >= 0.0

    @pytest.mark.asyncio
    async def test_stt_tts_concurrent_provider_stress_test(self, stt_config, tts_config):
        """Stress test concurrent provider usage in STT-TTS integration."""
        with patch('voice.stt_service.openai') as mock_openai_stt, \
             patch('voice.tts_service.openai.OpenAI') as mock_openai_tts:

            stt_service = STTService(stt_config)
            tts_service = TTSService(tts_config)

            # Configure services
            mock_openai_stt.Audio = MagicMock()
            stt_service.openai_client = mock_openai_stt

            mock_client_instance = MagicMock()
            mock_openai_tts.return_value = mock_client_instance
            tts_service.openai_client = mock_client_instance

            # Simulate heavy concurrent load
            num_concurrent_requests = 50
            request_counter = {'count': 0}

            def stt_side_effect(*args, **kwargs):
                request_counter['count'] += 1
                return {
                    'text': f'Stress test message {request_counter["count"]}',
                    'language': 'en',
                    'duration': 2.0,
                    'segments': []
                }

            def tts_side_effect(*args, **kwargs):
                mock_response = MagicMock()
                mock_response.content = b'mock_audio_data'
                return mock_response

            mock_openai_stt.Audio.transcribe = MagicMock(side_effect=stt_side_effect)
            mock_client_instance.audio.speech.create = MagicMock(side_effect=tts_side_effect)

            # Execute concurrent requests
            tasks = []
            for i in range(num_concurrent_requests):
                mock_audio = AudioData(
                    np.random.randint(-32768, 32767, 16000, dtype=np.int16),
                    16000, 1.0, 1
                )

                task = asyncio.create_task(self._process_stt_tts_pair(
                    stt_service, tts_service, mock_audio, i
                ))
                tasks.append(task)

            # Wait for all to complete
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            end_time = time.time()

            total_time = end_time - start_time

            # Performance verification
            assert len(results) == num_concurrent_requests
            assert total_time < 30.0  # Should complete within 30 seconds

            # Verify all requests processed
            stt_stats = stt_service.get_statistics()
            tts_stats = tts_service.get_statistics()

            assert stt_stats['request_count'] == num_concurrent_requests
            assert tts_stats['request_count'] == num_concurrent_requests

            # Check average processing time is reasonable
            assert stt_stats['average_processing_time'] < 5.0
            assert tts_stats['average_processing_time'] < 5.0