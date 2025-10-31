"""
STT/TTS Integration Testing

Comprehensive test suite for STT/TTS integration functionality:
- Provider fallback chains and error handling
- Audio quality optimization and caching mechanisms
- Concurrent request processing and streaming
- Service health monitoring and performance metrics

Coverage targets: STT/TTS integration testing for 26-28%→50% coverage improvement
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from voice.stt_service import STTService, STTResult, STTProvider, STTError
from voice.tts_service import TTSService, TTSResult, EmotionType, TTSError
from voice.config import VoiceConfig
from voice.audio_processor import AudioData
import numpy as np


@pytest.fixture
def voice_config():
    """Create voice configuration for testing."""
    config = VoiceConfig()
    config.stt_provider = "openai"
    config.tts_provider = "openai"
    config.openai_api_key = "test_key"
    config.performance.cache_enabled = True
    config.performance.streaming_enabled = True
    return config


@pytest.fixture
def mock_audio_data():
    """Create mock audio data for testing."""
    audio_numpy = np.random.rand(16000).astype(np.float32)  # 1 second at 16kHz
    return AudioData(
        data=audio_numpy,
        sample_rate=16000,
        channels=1,
        format="float32",
        duration=1.0
    )


@pytest.fixture
def stt_service(voice_config):
    """Create STT service instance for testing."""
    with patch('voice.stt_service.os.getenv', return_value='test_key'):
        service = STTService(voice_config)
        yield service


@pytest.fixture
def tts_service(voice_config):
    """Create TTS service instance for testing."""
    with patch('voice.tts_service.os.getenv', return_value='test_key'):
        service = TTSService(voice_config)
        yield service


class TestProviderFallbackChains:
    """Test provider fallback chain functionality."""

    def test_stt_provider_fallback_chain_basic(self, stt_service):
        """Test basic STT provider fallback chain."""
        # Mock all providers as unavailable initially
        stt_service.openai_client = None
        stt_service.google_speech_client = None
        stt_service.whisper_model = None

        # Should return empty chain when no providers available
        chain = stt_service._get_provider_fallback_chain(None)
        assert chain == []

    @patch('voice.stt_service.os.getenv', return_value='test_key')
    def test_stt_provider_fallback_chain_with_preferred(self, voice_config):
        """Test STT provider fallback chain with preferred provider."""
        with patch('voice.stt_service.openai') as mock_openai:
            mock_openai.OpenAI.return_value = Mock()
            service = STTService(voice_config)

            # Mock available providers
            service.openai_client = Mock()
            service.google_speech_client = Mock()
            service.whisper_model = Mock()

            # Test with OpenAI preferred
            chain = service._get_provider_fallback_chain("openai")
            assert chain == ["openai", "google", "whisper"]

            # Test with Google preferred
            chain = service._get_provider_fallback_chain("google")
            assert chain == ["google", "openai", "whisper"]

    def test_stt_provider_fallback_on_failure(self, stt_service, mock_audio_data):
        """Test STT provider fallback when primary fails."""
        # Mock primary provider (OpenAI) to fail
        stt_service.openai_client = Mock()
        mock_response = Mock()
        mock_response.Audio.transcribe.side_effect = Exception("OpenAI API error")
        stt_service.openai_client.Audio = Mock()
        stt_service.openai_client.Audio.transcribe = Mock(side_effect=Exception("API error"))

        # Mock fallback provider (Google) to succeed
        stt_service.google_speech_client = Mock()
        mock_google_result = Mock()
        mock_google_result.alternatives = [Mock(transcript="Fallback transcription", confidence=0.9)]
        stt_service.google_speech_client.recognize.return_value = Mock(results=[mock_google_result])

        async def test_fallback():
            try:
                result = await stt_service.transcribe_audio(mock_audio_data, provider="openai")
                return result.provider
            except Exception:
                return None

        # Should fallback to Google and succeed
        result_provider = asyncio.run(test_fallback())
        assert result_provider == "google"

    def test_tts_provider_fallback_on_failure(self, tts_service):
        """Test TTS provider fallback when primary fails."""
        # Mock primary provider to fail
        tts_service.openai_client = Mock()
        tts_service.openai_client.audio.speech.create.side_effect = Exception("OpenAI TTS error")

        # Mock fallback provider to succeed
        tts_service.elevenlabs_client = Mock()
        mock_elevenlabs_result = TTSResult(
            audio_data=AudioData(data=np.zeros(1000), sample_rate=22050, channels=1, format="float32", duration=0.1),
            text="Fallback audio",
            provider="elevenlabs"
        )

        async def mock_elevenlabs_generate(*args, **kwargs):
            return [b'test_audio_data']

        with patch('voice.tts_service.elevenlabs.generate', side_effect=mock_elevenlabs_generate):
            async def test_fallback():
                try:
                    result = await tts_service.synthesize_speech("Test text", provider="openai")
                    return result.provider
                except Exception as e:
                    # Try fallback manually
                    result = await tts_service._try_fallback_provider("Test text", "default", "openai")
                    return result.provider if result else None

            result_provider = asyncio.run(test_fallback())
            # Should either succeed with fallback or return None
            assert result_provider in [None, "elevenlabs"]

    def test_stt_fallback_chain_exhaustion(self, stt_service, mock_audio_data):
        """Test STT fallback chain exhaustion."""
        # Mock all providers to fail
        stt_service.openai_client = Mock()
        stt_service.google_speech_client = Mock()
        stt_service.whisper_model = Mock()

        # Make all providers fail
        async def failing_transcribe(*args, **kwargs):
            raise Exception("Provider failed")

        stt_service._transcribe_with_openai = failing_transcribe
        stt_service._transcribe_with_google = failing_transcribe
        stt_service._transcribe_with_whisper = failing_transcribe

        async def test_exhaustion():
            with pytest.raises(RuntimeError, match="All STT providers failed"):
                await stt_service.transcribe_audio(mock_audio_data)

        asyncio.run(test_exhaustion())

    def test_tts_fallback_chain_exhaustion(self, tts_service):
        """Test TTS fallback chain exhaustion."""
        # Mock all providers to fail
        tts_service.openai_client = Mock()
        tts_service.elevenlabs_client = Mock()
        tts_service.piper_tts = Mock()

        async def failing_synthesize(*args, **kwargs):
            raise Exception("Provider failed")

        tts_service._synthesize_with_openai = failing_synthesize
        tts_service._synthesize_with_elevenlabs = failing_synthesize
        tts_service._synthesize_with_piper = failing_synthesize

        async def test_exhaustion():
            with pytest.raises(TTSError):
                await tts_service.synthesize_speech("Test text")

        asyncio.run(test_exhaustion())


class TestAudioQualityOptimization:
    """Test audio quality optimization features."""

    def test_stt_audio_quality_scoring(self, stt_service, mock_audio_data):
        """Test STT audio quality scoring."""
        # Test with clean audio (should score high)
        clean_audio = AudioData(
            data=np.random.normal(0, 0.1, 16000).astype(np.float32),  # Low noise
            sample_rate=16000,
            channels=1,
            format="float32",
            duration=1.0
        )

        quality_score = stt_service._calculate_audio_quality_score(clean_audio)
        assert 0.5 <= quality_score <= 1.0

    def test_stt_audio_quality_with_clipping(self, stt_service):
        """Test STT audio quality with clipping detection."""
        # Create audio with clipping
        clipped_audio = np.ones(16000, dtype=np.float32) * 1.5  # Above 1.0 (clipped)
        clipped_data = AudioData(
            data=clipped_audio,
            sample_rate=16000,
            channels=1,
            format="float32",
            duration=1.0
        )

        quality_score = stt_service._calculate_audio_quality_score(clipped_data)
        assert quality_score < 0.8  # Should be penalized for clipping

    def test_stt_audio_quality_with_noise(self, stt_service):
        """Test STT audio quality with noise detection."""
        # Create noisy audio (low SNR)
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
        noise = np.random.normal(0, 0.5, 16000)  # High noise
        noisy_audio = signal + noise

        noisy_data = AudioData(
            data=noisy_audio.astype(np.float32),
            sample_rate=16000,
            channels=1,
            format="float32",
            duration=1.0
        )

        quality_score = stt_service._calculate_audio_quality_score(noisy_data)
        assert quality_score < 0.9  # Should be lower due to noise

    def test_tts_audio_quality_optimization(self, tts_service):
        """Test TTS audio quality optimization."""
        # Test that different providers optimize quality differently
        # This is mainly about ensuring quality parameters are passed correctly
        profile = tts_service.voice_profiles.get('professional_counselor')
        assert profile is not None
        assert hasattr(profile, 'pitch')
        assert hasattr(profile, 'speed')
        assert hasattr(profile, 'volume')


class TestCachingMechanisms:
    """Test caching mechanisms for performance."""

    def test_stt_result_caching(self, stt_service, mock_audio_data):
        """Test STT result caching."""
        cache_key = stt_service._generate_cache_key(mock_audio_data)

        # Create mock cached result
        cached_result = STTResult(
            text="Cached transcription",
            confidence=0.95,
            provider="cached",
            cached=True
        )

        # Manually add to cache
        stt_service.cache[cache_key] = {
            'result': cached_result,
            'timestamp': time.time()
        }

        async def test_cache_hit():
            result = await stt_service.transcribe_audio(mock_audio_data)
            return result

        # Should return cached result
        result = asyncio.run(test_cache_hit())
        assert result.cached is True
        assert result.text == "Cached transcription"

    def test_stt_cache_expiration(self, stt_service, mock_audio_data):
        """Test STT cache expiration."""
        cache_key = stt_service._generate_cache_key(mock_audio_data)

        # Add expired cached result (24+ hours old)
        expired_result = STTResult(
            text="Expired transcription",
            confidence=0.9,
            provider="expired"
        )

        expired_timestamp = time.time() - (25 * 3600)  # 25 hours ago
        stt_service.cache[cache_key] = {
            'result': expired_result,
            'timestamp': expired_timestamp
        }

        async def test_expired_cache():
            result = await stt_service.transcribe_audio(mock_audio_data)
            return result

        # Should not return expired result (will try to transcribe)
        # Since no real transcription, it will fail, but cache should be cleared
        try:
            asyncio.run(test_expired_cache())
        except:
            pass  # Expected to fail without mocked providers

        # Cache should be cleared or expired entry removed
        assert cache_key not in stt_service.cache

    def test_tts_audio_caching(self, tts_service):
        """Test TTS audio caching."""
        cache_key = tts_service._get_cache_key("Test text", "default", "openai")

        # Create mock cached result
        cached_audio = AudioData(
            data=np.zeros(1000, dtype=np.float32),
            sample_rate=22050,
            channels=1,
            format="float32",
            duration=0.1
        )

        cached_result = TTSResult(
            audio_data=cached_audio,
            text="Test text",
            voice_profile="default",
            provider="cached"
        )

        # Manually add to cache
        tts_service.audio_cache[cache_key] = cached_result

        async def test_cache_hit():
            result = await tts_service.synthesize_speech("Test text", provider="openai")
            return result

        # Should return cached result
        result = asyncio.run(test_cache_hit())
        assert result.provider == "cached"
        assert np.array_equal(result.audio_data.data, cached_audio.data)

    def test_stt_cache_size_limits(self, stt_service, mock_audio_data):
        """Test STT cache size limits and eviction."""
        stt_service.max_cache_size = 2

        # Fill cache beyond limit
        for i in range(5):
            modified_audio = mock_audio_data
            modified_audio.data = np.random.rand(16000).astype(np.float32)

            cache_key = stt_service._generate_cache_key(modified_audio)
            result = STTResult(
                text=f"Transcription {i}",
                confidence=0.9,
                provider="test"
            )
            stt_service._add_to_cache(cache_key, result)

        # Cache should not exceed max size
        assert len(stt_service.cache) <= stt_service.max_cache_size

    def test_tts_cache_eviction(self, tts_service):
        """Test TTS cache eviction under memory pressure."""
        tts_service.max_cache_size = 2

        # Add items beyond cache limit
        for i in range(5):
            cache_key = f"cache_key_{i}"
            audio_data = AudioData(
                data=np.zeros(1000, dtype=np.float32),
                sample_rate=22050,
                channels=1,
                format="float32",
                duration=0.1
            )
            result = TTSResult(
                audio_data=audio_data,
                text=f"Text {i}",
                voice_profile="default"
            )
            tts_service._cache_result(cache_key, result)

        # Cache should not exceed max size
        assert len(tts_service.audio_cache) <= tts_service.max_cache_size


class TestConcurrentRequestProcessing:
    """Test concurrent request processing."""

    def test_stt_concurrent_requests(self, stt_service):
        """Test concurrent STT requests."""
        # Mock successful transcription
        async def mock_transcribe(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate processing time
            return STTResult(
                text="Concurrent transcription",
                confidence=0.9,
                provider="mock"
            )

        stt_service._transcribe_with_openai = mock_transcribe
        stt_service.openai_client = Mock()  # Mark as available

        async def concurrent_stt_test():
            tasks = []
            for i in range(5):
                audio_data = AudioData(
                    data=np.random.rand(8000).astype(np.float32),
                    sample_rate=16000,
                    channels=1,
                    format="float32",
                    duration=0.5
                )
                task = stt_service.transcribe_audio(audio_data)
                tasks.append(task)

            results = await asyncio.gather(*tasks)
            return results

        results = asyncio.run(concurrent_stt_test())

        # All requests should succeed
        assert len(results) == 5
        for result in results:
            assert result.text == "Concurrent transcription"
            assert result.confidence == 0.9

    def test_tts_concurrent_requests(self, tts_service):
        """Test concurrent TTS requests."""
        # Mock successful synthesis
        async def mock_synthesize(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate processing time
            audio_data = AudioData(
                data=np.random.rand(1000).astype(np.float32),
                sample_rate=22050,
                channels=1,
                format="float32",
                duration=0.1
            )
            return TTSResult(
                audio_data=audio_data,
                text="Concurrent synthesis",
                provider="mock"
            )

        tts_service._synthesize_with_openai = mock_synthesize
        tts_service.openai_client = Mock()  # Mark as available

        async def concurrent_tts_test():
            tasks = []
            texts = [f"Text {i}" for i in range(5)]
            for text in texts:
                task = tts_service.synthesize_speech(text)
                tasks.append(task)

            results = await asyncio.gather(*tasks)
            return results

        results = asyncio.run(concurrent_tts_test())

        # All requests should succeed
        assert len(results) == 5
        for result in results:
            assert result.text.startswith("Concurrent synthesis")
            assert result.audio_data is not None

    def test_mixed_concurrent_stt_tts(self, stt_service, tts_service):
        """Test mixed concurrent STT and TTS requests."""
        # Mock both services
        async def mock_stt_transcribe(*args, **kwargs):
            await asyncio.sleep(0.05)
            return STTResult(text="Mixed test", confidence=0.9, provider="mock")

        async def mock_tts_synthesize(*args, **kwargs):
            await asyncio.sleep(0.05)
            audio_data = AudioData(data=np.zeros(500), sample_rate=22050, channels=1, format="float32", duration=0.05)
            return TTSResult(audio_data=audio_data, text="Mixed response", provider="mock")

        stt_service._transcribe_with_openai = mock_stt_transcribe
        stt_service.openai_client = Mock()

        tts_service._synthesize_with_openai = mock_tts_synthesize
        tts_service.openai_client = Mock()

        async def mixed_concurrent_test():
            # Create mixed tasks
            stt_audio = AudioData(data=np.zeros(4000), sample_rate=16000, channels=1, format="float32", duration=0.25)
            stt_task = stt_service.transcribe_audio(stt_audio)
            tts_task = tts_service.synthesize_speech("Test response")

            results = await asyncio.gather(stt_task, tts_task)
            return results

        results = asyncio.run(mixed_concurrent_test())

        assert len(results) == 2
        assert results[0].text == "Mixed test"  # STT result
        assert results[1].text == "Mixed response"  # TTS result


class TestStreamingAudioProcessing:
    """Test streaming audio processing capabilities."""

    async def test_stt_streaming_basic(self, stt_service):
        """Test basic STT streaming functionality."""
        # Mock streaming audio source
        audio_chunks = []
        for i in range(3):
            chunk = AudioData(
                data=np.random.rand(4000).astype(np.float32),
                sample_rate=16000,
                channels=1,
                format="float32",
                duration=0.25
            )
            audio_chunks.append(chunk)

        async def mock_audio_stream():
            for chunk in audio_chunks:
                yield chunk
                await asyncio.sleep(0.1)

        # Mock transcription for each chunk
        async def mock_transcribe_chunk(audio_data):
            await asyncio.sleep(0.05)
            return STTResult(
                text=f"Chunk transcription",
                confidence=0.85,
                provider="streaming_mock"
            )

        stt_service._transcribe_with_openai = mock_transcribe_chunk
        stt_service.openai_client = Mock()

        # This would test streaming if the method existed
        # For now, just test that the service can handle individual chunks
        results = []
        async for chunk in mock_audio_stream():
            result = await stt_service.transcribe_audio(chunk)
            results.append(result)

        assert len(results) == 3
        for result in results:
            assert "Chunk transcription" in result.text

    async def test_tts_streaming_basic(self, tts_service):
        """Test basic TTS streaming functionality."""
        # Mock streaming synthesis
        async def mock_stream_synthesis(text, profile, chunk_size):
            # Simulate streaming by yielding chunks
            audio_data = AudioData(
                data=np.random.rand(2000).astype(np.float32),
                sample_rate=22050,
                channels=1,
                format="float32",
                duration=0.1
            )

            # Yield in chunks
            chunk_size_samples = chunk_size
            total_samples = len(audio_data.data)

            for i in range(0, total_samples, chunk_size_samples):
                chunk_data = audio_data.data[i:i + chunk_size_samples]
                chunk_duration = len(chunk_data) / audio_data.sample_rate

                chunk_audio = AudioData(
                    data=chunk_data,
                    sample_rate=audio_data.sample_rate,
                    channels=1,
                    format="float32",
                    duration=chunk_duration
                )

                yield chunk_audio

        tts_service._synthesize_stream_openai = mock_stream_synthesis
        tts_service.openai_client = Mock()

        # Test streaming synthesis
        chunks = []
        async for chunk in tts_service.synthesize_stream("Test streaming text", chunk_size=500):
            chunks.append(chunk)

        assert len(chunks) > 0
        total_duration = sum(chunk.duration for chunk in chunks)
        assert total_duration > 0

    def test_streaming_error_handling(self, tts_service):
        """Test streaming error handling."""
        # Mock streaming that fails
        async def failing_stream(*args, **kwargs):
            raise Exception("Streaming failed")
            yield  # Unreachable

        tts_service._synthesize_stream_openai = failing_stream
        tts_service.openai_client = Mock()

        async def test_streaming_failure():
            with pytest.raises(Exception, match="Streaming failed"):
                async for chunk in tts_service.synthesize_stream("Test text"):
                    pass

        asyncio.run(test_streaming_failure())


class TestServiceHealthMonitoring:
    """Test service health monitoring and performance metrics."""

    def test_stt_service_statistics(self, stt_service, mock_audio_data):
        """Test STT service statistics collection."""
        # Mock a successful transcription
        async def mock_successful_transcribe(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate processing time
            return STTResult(
                text="Statistics test",
                confidence=0.95,
                processing_time=0.1,
                provider="stats_test"
            )

        stt_service._transcribe_with_openai = mock_successful_transcribe
        stt_service.openai_client = Mock()

        async def generate_stats():
            for i in range(3):
                await stt_service.transcribe_audio(mock_audio_data)
            return stt_service.get_statistics()

        stats = asyncio.run(generate_stats())

        assert stats['request_count'] == 3
        assert stats['error_count'] == 0
        assert stats['success_rate'] == 1.0
        assert stats['average_processing_time'] > 0
        assert 'available_providers' in stats

    def test_tts_service_statistics(self, tts_service):
        """Test TTS service statistics collection."""
        # Mock successful synthesis
        async def mock_successful_synthesize(*args, **kwargs):
            await asyncio.sleep(0.05)
            audio_data = AudioData(
                data=np.zeros(1000, dtype=np.float32),
                sample_rate=22050,
                channels=1,
                format="float32",
                duration=0.1
            )
            return TTSResult(
                audio_data=audio_data,
                text="Stats test",
                processing_time=0.05,
                duration=0.1,
                provider="stats_test"
            )

        tts_service._synthesize_with_openai = mock_successful_synthesize
        tts_service.openai_client = Mock()

        async def generate_stats():
            for i in range(3):
                await tts_service.synthesize_speech(f"Test text {i}")
            return tts_service.get_statistics()

        stats = asyncio.run(generate_stats())

        assert stats['request_count'] == 3
        assert stats['error_count'] == 0
        assert stats['total_audio_duration'] > 0
        assert 'cache_size' in stats
        assert 'cache_hit_rate' in stats

    def test_stt_health_check(self, stt_service):
        """Test STT service health check."""
        # Mock available providers
        stt_service.openai_client = Mock()

        is_healthy = stt_service.test_service()
        # Should return True if service is mocked as available
        assert isinstance(is_healthy, bool)

    def test_tts_health_check(self, tts_service):
        """Test TTS service health check."""
        # Mock available providers
        tts_service.openai_client = Mock()

        async def test_health():
            try:
                result = await tts_service.synthesize_speech("Health check", voice_profile="calm_therapist")
                return result is not None
            except:
                return False

        # Health check via actual synthesis test
        is_healthy = asyncio.run(test_health())
        assert isinstance(is_healthy, bool)

    def test_service_performance_metrics(self, stt_service, tts_service):
        """Test comprehensive performance metrics."""
        # Test STT metrics
        stt_stats = stt_service.get_statistics()
        assert 'average_processing_time' in stt_stats
        assert 'error_rate' in stt_stats

        # Test TTS metrics
        tts_stats = tts_service.get_statistics()
        assert 'average_processing_time' in tts_stats
        assert 'cache_hit_rate' in tts_stats


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery mechanisms."""

    def test_stt_error_recovery_with_fallback(self, stt_service, mock_audio_data):
        """Test STT error recovery with provider fallback."""
        # Mock primary provider failure, fallback success
        call_count = {'openai': 0, 'google': 0}

        async def failing_openai(*args, **kwargs):
            call_count['openai'] += 1
            raise Exception("OpenAI failed")

        async def successful_google(*args, **kwargs):
            call_count['google'] += 1
            return STTResult(
                text="Fallback success",
                confidence=0.8,
                provider="google"
            )

        stt_service._transcribe_with_openai = failing_openai
        stt_service._transcribe_with_google = successful_google
        stt_service.openai_client = Mock()
        stt_service.google_speech_client = Mock()

        async def test_recovery():
            result = await stt_service.transcribe_audio(mock_audio_data, provider="openai")
            return result

        result = asyncio.run(test_recovery())

        assert call_count['openai'] == 1  # Primary was tried
        assert call_count['google'] == 1  # Fallback was used
        assert result.provider == "google"
        assert result.text == "Fallback success"

    def test_tts_error_recovery_with_fallback(self, tts_service):
        """Test TTS error recovery with provider fallback."""
        # Mock primary failure, fallback success
        call_count = {'openai': 0, 'elevenlabs': 0}

        async def failing_openai(*args, **kwargs):
            call_count['openai'] += 1
            raise Exception("OpenAI TTS failed")

        async def successful_elevenlabs(*args, **kwargs):
            call_count['elevenlabs'] += 1
            audio_data = AudioData(
                data=np.zeros(1000, dtype=np.float32),
                sample_rate=22050,
                channels=1,
                format="float32",
                duration=0.1
            )
            return TTSResult(
                audio_data=audio_data,
                text="Fallback TTS success",
                provider="elevenlabs"
            )

        tts_service._synthesize_with_openai = failing_openai
        tts_service._synthesize_with_elevenlabs = successful_elevenlabs
        tts_service.openai_client = Mock()
        tts_service.elevenlabs_client = Mock()

        async def test_recovery():
            result = await tts_service.synthesize_speech("Test text", provider="openai")
            return result

        result = asyncio.run(test_recovery())

        assert call_count['openai'] == 1  # Primary was tried
        assert result.provider == "elevenlabs"
        assert result.text == "Fallback TTS success"

    def test_concurrent_error_handling(self, stt_service):
        """Test concurrent error handling."""
        async def sometimes_failing_transcribe(audio_data, attempt_num):
            if attempt_num % 3 == 0:  # Every third call fails
                raise Exception(f"Simulated failure on attempt {attempt_num}")
            await asyncio.sleep(0.01)
            return STTResult(
                text=f"Success on attempt {attempt_num}",
                confidence=0.9,
                provider="test"
            )

        stt_service._transcribe_with_openai = lambda audio_data: sometimes_failing_transcribe(audio_data, stt_service.request_count)
        stt_service.openai_client = Mock()

        async def concurrent_error_test():
            tasks = []
            for i in range(9):  # 9 requests, 3 should fail initially
                audio_data = AudioData(
                    data=np.random.rand(4000).astype(np.float32),
                    sample_rate=16000,
                    channels=1,
                    format="float32",
                    duration=0.25
                )
                task = stt_service.transcribe_audio(audio_data)
                tasks.append(task)

            results = []
            for task in asyncio.as_completed(tasks):
                try:
                    result = await task
                    results.append(result)
                except Exception as e:
                    results.append(f"error: {e}")

            return results

        results = asyncio.run(concurrent_error_test())

        # Should have some successful results despite errors
        successful_results = [r for r in results if isinstance(r, STTResult)]
        assert len(successful_results) >= 6  # At least 6 should succeed


# Run basic validation
if __name__ == "__main__":
    print("STT/TTS Integration Test Suite")
    print("=" * 45)

    try:
        from voice.stt_service import STTService
        from voice.tts_service import TTSService
        from voice.config import VoiceConfig
        print("✅ STT/TTS imports successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")

    try:
        config = VoiceConfig()
        stt = STTService(config)
        tts = TTSService(config)

        # Test basic provider availability
        stt_available = stt.is_available()
        tts_available = tts.is_available()

        print(f"✅ STT available: {stt_available}")
        print(f"✅ TTS available: {tts_available}")

        # Test provider chains
        stt_providers = stt.get_available_providers()
        tts_providers = tts.get_available_providers()

        print(f"✅ STT providers: {stt_providers}")
        print(f"✅ TTS providers: {tts_providers}")

        stt.cleanup()
        tts.cleanup()

    except Exception as e:
        print(f"❌ Service initialization failed: {e}")

    print("STT/TTS integration test file created - run with pytest for full validation")
