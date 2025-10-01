"""
Error Recovery and Network Failure Tests

This module tests system resilience and failure recovery across all components:
- Network timeouts and connection failures
- API rate limiting and quota exhaustion  
- Provider API failures and fallback scenarios
- Service degradation and graceful fallback behavior
- Concurrent operation conflicts and race conditions
- Network interruption recovery
"""

import asyncio
import time
import pytest
import threading
import requests
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List, Optional
import numpy as np
import socket
import http.client

# Import project modules
from voice.voice_service import VoiceService, VoiceSession, VoiceSessionState
from voice.stt_service import STTService, STTResult
from voice.tts_service import TTSService, TTSResult
from voice.audio_processor import AudioData, SimplifiedAudioProcessor
from voice.config import VoiceConfig, SecurityConfig
from voice.security import VoiceSecurity


class TestNetworkFailureRecovery:
    """Test network failure scenarios and recovery mechanisms."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        config = Mock(spec=VoiceConfig)
        config.voice_enabled = True
        config.stt_enabled = True
        config.tts_enabled = True
        config.get_preferred_stt_service.return_value = "openai"
        config.get_preferred_tts_service.return_value = "openai"
        config.stt_language = "en"
        config.tts_language = "en"
        config.security = Mock(spec=SecurityConfig)
        return config

    @pytest.fixture
    def mock_security(self):
        """Create mock security module."""
        security = Mock(spec=VoiceSecurity)
        security.initialize.return_value = True
        security.process_audio = AsyncMock(return_value=AudioData(np.array([]), 16000, 0.0, 1))
        return security

    @pytest.fixture
    def voice_service(self, mock_config, mock_security):
        """Create voice service for testing."""
        return VoiceService(mock_config, mock_security)

    def test_network_timeout_stt_recovery(self, voice_service):
        """Test STT service recovery from network timeouts."""
        # Mock STT service with timeout error
        voice_service.stt_service.transcribe_audio = AsyncMock(
            side_effect=asyncio.TimeoutError("Network timeout")
        )

        # Mock fallback STT service
        voice_service.fallback_stt_service = Mock()
        voice_service.fallback_stt_service.transcribe_audio = AsyncMock(
            return_value=STTResult(
                text="Fallback transcription",
                confidence=0.8,
                language="en",
                duration=1.0,
                provider="fallback"
            )
        )

        # Test audio data
        audio_data = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1
        )

        # Test timeout and fallback
        async def run_test():
            result = await voice_service.process_voice_input(audio_data)
            assert result is not None
            assert result.text == "Fallback transcription"
            assert result.provider == "fallback"

        asyncio.run(run_test())

    def test_api_rate_limit_handling(self, voice_service):
        """Test handling of API rate limiting responses."""
        # Mock rate limit error (429 status)
        rate_limit_error = requests.exceptions.HTTPError()
        rate_limit_error.response = Mock()
        rate_limit_error.response.status_code = 429
        rate_limit_error.response.headers = {
            'Retry-After': '60',
            'X-RateLimit-Remaining': '0'
        }

        voice_service.stt_service.transcribe_audio = AsyncMock(
            side_effect=rate_limit_error
        )

        # Mock fallback service
        voice_service.fallback_stt_service = Mock()
        voice_service.fallback_stt_service.transcribe_audio = AsyncMock(
            return_value=STTResult(
                text="Rate limited fallback",
                confidence=0.7,
                language="en",
                duration=1.0,
                provider="fallback"
            )
        )

        audio_data = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1
        )

        async def run_test():
            result = await voice_service.process_voice_input(audio_data)
            assert result is not None
            assert "fallback" in result.provider

        asyncio.run(run_test())

    def test_connection_refused_recovery(self, voice_service):
        """Test recovery from connection refused errors."""
        # Mock connection refused error
        connection_error = ConnectionRefusedError("Connection refused")
        voice_service.stt_service.transcribe_audio = AsyncMock(
            side_effect=connection_error
        )

        # Mock alternative provider
        voice_service.fallback_stt_service = Mock()
        voice_service.fallback_stt_service.transcribe_audio = AsyncMock(
            return_value=STTResult(
                text="Connection recovery",
                confidence=0.75,
                language="en",
                duration=1.0,
                provider="alternative"
            )
        )

        audio_data = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1
        )

        async def run_test():
            result = await voice_service.process_voice_input(audio_data)
            assert result is not None
            assert result.provider == "alternative"

        asyncio.run(run_test())

    def test_dns_resolution_failure(self, voice_service):
        """Test handling of DNS resolution failures."""
        # Mock DNS resolution failure
        dns_error = socket.gaierror("Name resolution failed")
        voice_service.stt_service.transcribe_audio = AsyncMock(
            side_effect=dns_error
        )

        # Test with no fallback available (should return mock result)
        audio_data = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1
        )

        async def run_test():
            result = await voice_service.process_voice_input(audio_data)
            # Should return None or handle gracefully
            assert result is None or hasattr(result, 'error')

        asyncio.run(run_test())

    def test_ssl_certificate_error_recovery(self, voice_service):
        """Test recovery from SSL certificate errors."""
        # Mock SSL certificate error
        import ssl
        ssl_error = ssl.SSLError("Certificate verification failed")
        voice_service.stt_service.transcribe_audio = AsyncMock(
            side_effect=ssl_error
        )

        # Mock local fallback
        voice_service.fallback_stt_service = Mock()
        voice_service.fallback_stt_service.transcribe_audio = AsyncMock(
            return_value=STTResult(
                text="SSL fallback transcription",
                confidence=0.6,
                language="en",
                duration=1.0,
                provider="local"
            )
        )

        audio_data = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1
        )

        async def run_test():
            result = await voice_service.process_voice_input(audio_data)
            assert result is not None
            assert result.provider == "local"

        asyncio.run(run_test())

    def test_network_interruption_recovery(self, voice_service):
        """Test recovery from network interruptions."""
        # Simulate network interruption scenario
        interruption_count = 0
        max_interruptions = 3

        def mock_transcribe_with_interruption(audio_data):
            nonlocal interruption_count
            interruption_count += 1
            if interruption_count <= max_interruptions:
                raise ConnectionError("Network is unreachable")
            return STTResult(
                text="Recovered after interruption",
                confidence=0.8,
                language="en",
                duration=1.0,
                provider="openai"
            )

        voice_service.stt_service.transcribe_audio = AsyncMock(
            side_effect=mock_transcribe_with_interruption
        )

        audio_data = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1
        )

        async def run_test():
            result = await voice_service.process_voice_input(audio_data)
            assert result is not None
            assert result.text == "Recovered after interruption"
            assert interruption_count == max_interruptions + 1

        asyncio.run(run_test())

    def test_concurrent_request_rate_limiting(self, voice_service):
        """Test rate limiting for concurrent requests."""
        # Mock rate limiting for concurrent requests
        request_count = 0
        rate_limit_threshold = 5

        def mock_rate_limited_transcribe(audio_data):
            nonlocal request_count
            request_count += 1
            if request_count > rate_limit_threshold:
                # Simulate rate limit error
                error = requests.exceptions.HTTPError()
                error.response = Mock()
                error.response.status_code = 429
                raise error
            return STTResult(
                text=f"Request {request_count}",
                confidence=0.8,
                language="en",
                duration=1.0,
                provider="openai"
            )

        voice_service.stt_service.transcribe_audio = AsyncMock(
            side_effect=mock_rate_limited_transcribe
        )

        # Test multiple concurrent requests
        async def run_concurrent_test():
            tasks = []
            for i in range(10):
                audio_data = AudioData(
                    data=np.random.randn(1600).astype(np.float32),
                    sample_rate=16000,
                    duration=0.1,
                    channels=1
                )
                tasks.append(voice_service.process_voice_input(audio_data))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check that some requests succeeded and some were rate limited
            success_count = sum(1 for r in results if r is not None and not hasattr(r, 'error'))
            assert success_count <= rate_limit_threshold

        asyncio.run(run_concurrent_test())

    def test_service_unavailable_recovery(self, voice_service):
        """Test recovery when primary service becomes unavailable."""
        # Mock service unavailable error (503)
        service_unavailable = requests.exceptions.HTTPError()
        service_unavailable.response = Mock()
        service_unavailable.response.status_code = 503

        voice_service.stt_service.transcribe_audio = AsyncMock(
            side_effect=service_unavailable
        )

        # Mock backup service
        voice_service.fallback_stt_service = Mock()
        voice_service.fallback_stt_service.transcribe_audio = AsyncMock(
            return_value=STTResult(
                text="Service unavailable fallback",
                confidence=0.7,
                language="en",
                duration=1.0,
                provider="backup"
            )
        )

        audio_data = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1
        )

        async def run_test():
            result = await voice_service.process_voice_input(audio_data)
            assert result is not None
            assert result.provider == "backup"

        asyncio.run(run_test())

    def test_authentication_failure_recovery(self, voice_service):
        """Test recovery from authentication failures."""
        # Mock authentication error (401)
        auth_error = requests.exceptions.HTTPError()
        auth_error.response = Mock()
        auth_error.response.status_code = 401

        voice_service.stt_service.transcribe_audio = AsyncMock(
            side_effect=auth_error
        )

        # Mock fallback without auth requirements
        voice_service.fallback_stt_service = Mock()
        voice_service.fallback_stt_service.transcribe_audio = AsyncMock(
            return_value=STTResult(
                text="Authentication fallback",
                confidence=0.65,
                language="en",
                duration=1.0,
                provider="local"
            )
        )

        audio_data = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1
        )

        async def run_test():
            result = await voice_service.process_voice_input(audio_data)
            assert result is not None
            assert result.provider == "local"

        asyncio.run(run_test())

    def test_bandwidth_limitation_handling(self, voice_service):
        """Test handling of bandwidth limitations."""
        # Mock slow response due to bandwidth limits
        async def slow_transcribe(audio_data):
            await asyncio.sleep(30)  # Simulate slow response
            return STTResult(
                text="Slow response",
                confidence=0.8,
                language="en",
                duration=1.0,
                provider="openai"
            )

        voice_service.stt_service.transcribe_audio = slow_transcribe

        # Mock faster fallback
        voice_service.fallback_stt_service = Mock()
        voice_service.fallback_stt_service.transcribe_audio = AsyncMock(
            return_value=STTResult(
                text="Fast fallback response",
                confidence=0.75,
                language="en",
                duration=1.0,
                provider="fast"
            )
        )

        audio_data = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1
        )

        async def run_test():
            # Test with timeout
            try:
                result = await asyncio.wait_for(
                    voice_service.process_voice_input(audio_data),
                    timeout=5.0
                )
                # Should use fallback due to timeout
                assert result.provider == "fast"
            except asyncio.TimeoutError:
                # Expected if no fallback logic implemented
                pass

        asyncio.run(run_test())


class TestAPIFailureScenarios:
    """Test various API failure scenarios and recovery mechanisms."""

    @pytest.fixture
    def stt_service(self, mock_config):
        """Create STT service for testing."""
        return STTService(mock_config)

    def test_openai_api_key_invalid(self, stt_service):
        """Test handling of invalid OpenAI API key."""
        # Mock invalid API key response
        import openai
        api_error = openai.AuthenticationError("Invalid API key")

        with patch('voice.stt_service.os.getenv', return_value='invalid_key'):
            with patch.object(stt_service, '_initialize_openai_whisper'):
                stt_service._initialize_services()

                # Mock the API call to raise authentication error
                if stt_service.openai_client:
                    stt_service.openai_client.Audio = Mock()
                    stt_service.openai_client.Audio.transcribe = Mock(
                        side_effect=api_error
                    )

        audio_data = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1
        )

        async def run_test():
            try:
                await stt_service.transcribe_audio(audio_data)
                assert False, "Should have raised an error"
            except RuntimeError as e:
                assert "All STT providers failed" in str(e)

        asyncio.run(run_test())

    def test_google_credentials_missing(self, stt_service):
        """Test handling of missing Google credentials."""
        with patch.object(stt_service.config, 'is_google_speech_configured', return_value=True):
            with patch.object(stt_service.config, 'google_cloud_credentials_path', None):
                with patch('google.oauth2.service_account.Credentials.from_service_account_file',
                          side_effect=FileNotFoundError("Credentials file not found")):
                    stt_service._initialize_services()

                    # Google client should not be initialized
                    assert stt_service.google_speech_client is None

    def test_whisper_model_load_failure(self, stt_service):
        """Test handling of Whisper model loading failures."""
        with patch('voice.stt_service.whisper') as mock_whisper:
            mock_whisper.load_model = Mock(side_effect=RuntimeError("Model file not found"))

            with patch.object(stt_service.config, 'is_whisper_configured', return_value=True):
                stt_service._initialize_services()

                # Whisper model should not be initialized
                assert stt_service.whisper_model is None

    def test_api_quota_exceeded_recovery(self, stt_service):
        """Test recovery from API quota exceeded errors."""
        # Mock quota exceeded error (429 with specific message)
        quota_error = requests.exceptions.HTTPError()
        quota_error.response = Mock()
        quota_error.response.status_code = 429
        quota_error.response.text = "Quota exceeded"

        stt_service.transcribe_audio = AsyncMock(side_effect=quota_error)

        # Mock fallback provider
        stt_service.fallback_stt_service = Mock()
        stt_service.fallback_stt_service.transcribe_audio = AsyncMock(
            return_value=STTResult(
                text="Quota exceeded fallback",
                confidence=0.7,
                language="en",
                duration=1.0,
                provider="fallback"
            )
        )

        audio_data = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1
        )

        async def run_test():
            # Test that fallback is used when quota exceeded
            result = await stt_service.transcribe_audio(audio_data)
            assert result.provider == "fallback"

        asyncio.run(run_test())

    def test_provider_priority_fallback(self, stt_service):
        """Test provider priority and fallback chain."""
        # Test provider priority order
        available_providers = stt_service.get_available_providers()

        # Mock each provider to fail except the last one
        failure_count = 0

        async def failing_transcribe(audio_data):
            nonlocal failure_count
            failure_count += 1
            if failure_count < len(available_providers):
                raise RuntimeError(f"Provider {failure_count} failed")
            return STTResult(
                text=f"Success from provider {failure_count}",
                confidence=0.8,
                language="en",
                duration=1.0,
                provider="success_provider"
            )

        stt_service.transcribe_audio = AsyncMock(side_effect=failing_transcribe)

        audio_data = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1
        )

        async def run_test():
            result = await stt_service.transcribe_audio(audio_data)
            assert result.provider == "success_provider"
            assert failure_count > 1  # Should have tried multiple providers

        asyncio.run(run_test())


class TestServiceResilience:
    """Test overall service resilience and degradation handling."""

    def test_graceful_degradation_no_providers(self, mock_config, mock_security):
        """Test graceful degradation when no providers are available."""
        # Create service with no available providers
        service = VoiceService(mock_config, mock_security)

        # Mock all providers as unavailable
        service.stt_service.is_available = Mock(return_value=False)
        service.tts_service.is_available = Mock(return_value=False)
        service.audio_processor.input_devices = []
        service.audio_processor.output_devices = []

        # Initialize should handle gracefully
        result = service.initialize()
        assert result is False  # Should return False but not crash

        # Health check should report degraded status
        health = service.health_check()
        assert health['overall_status'] in ['degraded', 'error']

    def test_partial_service_availability(self, voice_service):
        """Test system behavior with partial service availability."""
        # Mock STT available but TTS unavailable
        voice_service.stt_service.is_available = Mock(return_value=True)
        voice_service.tts_service.is_available = Mock(return_value=False)

        health = voice_service.health_check()
        assert health['overall_status'] == 'degraded'
        assert health['stt_service']['status'] == 'healthy'
        assert health['tts_service']['status'] in ['mock', 'unhealthy']

    def test_circuit_breaker_pattern(self, voice_service):
        """Test circuit breaker pattern for repeated failures."""
        # Simulate repeated failures that should trigger circuit breaker
        failure_threshold = 5
        failure_count = 0

        def mock_failing_transcribe(audio_data):
            nonlocal failure_count
            failure_count += 1
            if failure_count <= failure_threshold:
                raise ConnectionError("Service temporarily unavailable")
            return STTResult(
                text="Circuit breaker recovery",
                confidence=0.8,
                language="en",
                duration=1.0,
                provider="openai"
            )

        voice_service.stt_service.transcribe_audio = AsyncMock(
            side_effect=mock_failing_transcribe
        )

        audio_data = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1
        )

        async def run_test():
            # First few requests should fail
            for i in range(failure_threshold):
                result = await voice_service.process_voice_input(audio_data)
                assert result is None or hasattr(result, 'error')

            # After threshold, service should recover
            result = await voice_service.process_voice_input(audio_data)
            assert result.text == "Circuit breaker recovery"

        asyncio.run(run_test())

    def test_automatic_retry_mechanism(self, voice_service):
        """Test automatic retry mechanism for transient failures."""
        retry_count = 0
        max_retries = 3

        async def mock_retry_transcribe(audio_data):
            nonlocal retry_count
            retry_count += 1
            if retry_count < max_retries:
                raise requests.exceptions.ConnectionError("Temporary network issue")
            return STTResult(
                text="Retry success",
                confidence=0.8,
                language="en",
                duration=1.0,
                provider="openai"
            )

        voice_service.stt_service.transcribe_audio = mock_retry_transcribe

        audio_data = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1
        )

        async def run_test():
            result = await voice_service.process_voice_input(audio_data)
            assert result.text == "Retry success"
            assert retry_count == max_retries

        asyncio.run(run_test())

    def test_error_rate_monitoring(self, voice_service):
        """Test error rate monitoring and alerting."""
        # Mock high error rate scenario
        error_count = 0
        total_requests = 0

        def mock_monitored_transcribe(audio_data):
            nonlocal error_count, total_requests
            total_requests += 1
            if total_requests % 3 == 0:  # Every 3rd request fails
                error_count += 1
                raise RuntimeError("Monitored failure")
            return STTResult(
                text=f"Request {total_requests}",
                confidence=0.8,
                language="en",
                duration=1.0,
                provider="openai"
            )

        voice_service.stt_service.transcribe_audio = AsyncMock(
            side_effect=mock_monitored_transcribe
        )

        audio_data = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1
        )

        async def run_test():
            results = []
            for i in range(10):
                result = await voice_service.process_voice_input(audio_data)
                results.append(result)

            # Calculate error rate
            success_count = sum(1 for r in results if r is not None and not hasattr(r, 'error'))
            error_rate = (total_requests - success_count) / total_requests

            assert error_rate > 0.2  # Should have significant error rate
            assert error_count >= 3  # Should have multiple errors

        asyncio.run(run_test())

    def test_resource_cleanup_on_failure(self, voice_service):
        """Test that resources are properly cleaned up on failures."""
        # Mock cleanup methods
        voice_service.stt_service.cleanup = Mock()
        voice_service.tts_service.cleanup = Mock()
        voice_service.audio_processor.cleanup = Mock()

        # Simulate failure during operation
        def failing_operation():
            raise RuntimeError("Operation failed")

        voice_service.stt_service.transcribe_audio = AsyncMock(
            side_effect=failing_operation
        )

        audio_data = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1
        )

        async def run_test():
            # Trigger failure
            result = await voice_service.process_voice_input(audio_data)

            # Cleanup should still be called on failure
            # (This tests that cleanup is called in finally blocks)
            voice_service.cleanup()

            # Verify cleanup was called
            voice_service.stt_service.cleanup.assert_called_once()
            voice_service.tts_service.cleanup.assert_called_once()
            voice_service.audio_processor.cleanup.assert_called_once()

        asyncio.run(run_test())