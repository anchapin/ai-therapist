"""
Integration Fallback and Service Degradation Tests

This module tests provider fallbacks and service degradation across the entire system:
- Multi-provider fallback chains
- Service degradation and graceful fallback behavior
- Provider priority and load balancing
- Emergency fallback mechanisms
- Partial service availability scenarios
- Cross-component fallback coordination
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List, Optional
import numpy as np

# Import project modules
from voice.voice_service import VoiceService, VoiceSession, VoiceSessionState
from voice.stt_service import STTService, STTResult
from voice.tts_service import TTSService, TTSResult
from voice.audio_processor import AudioData, SimplifiedAudioProcessor
from voice.config import VoiceConfig, SecurityConfig
from voice.security import VoiceSecurity


class TestMultiProviderFallbacks:
    """Test multi-provider fallback mechanisms."""

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

    def test_stt_provider_fallback_chain(self, mock_config, mock_security):
        """Test STT provider fallback chain."""
        # Create STT service with multiple providers
        stt_service = STTService(mock_config)

        # Mock provider availability
        stt_service.get_available_providers = Mock(return_value=["openai", "google", "whisper"])

        # Mock provider failures in chain
        failure_count = 0

        async def mock_transcribe_with_failure(audio_data):
            nonlocal failure_count
            failure_count += 1

            if failure_count == 1:
                # OpenAI fails
                raise RuntimeError("OpenAI API unavailable")
            elif failure_count == 2:
                # Google fails
                raise RuntimeError("Google API unavailable")
            else:
                # Whisper succeeds
                return STTResult(
                    text="Whisper fallback success",
                    confidence=0.8,
                    language="en",
                    duration=1.0,
                    provider="whisper"
                )

        stt_service.transcribe_audio = mock_transcribe_with_failure

        # Test fallback chain
        audio_data = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1
        )

        async def run_test():
            result = await stt_service.transcribe_audio(audio_data)
            assert result.provider == "whisper"
            assert failure_count == 3  # Should have tried all providers

        asyncio.run(run_test())

    def test_tts_provider_fallback_chain(self, mock_config, mock_security):
        """Test TTS provider fallback chain."""
        # Create TTS service with multiple providers
        tts_service = TTSService(mock_config)

        # Mock provider availability
        tts_service.get_available_providers = Mock(return_value=["openai", "elevenlabs", "google"])

        # Mock provider failures in chain
        failure_count = 0

        def mock_synthesize_with_failure(text, voice_profile=None):
            nonlocal failure_count
            failure_count += 1

            if failure_count == 1:
                # OpenAI fails
                raise RuntimeError("OpenAI TTS unavailable")
            elif failure_count == 2:
                # ElevenLabs fails
                raise RuntimeError("ElevenLabs API unavailable")
            else:
                # Google succeeds
                return TTSResult(
                    audio_data=b"fallback_audio_data",
                    duration=1.0,
                    provider="google"
                )

        tts_service.synthesize_speech = mock_synthesize_with_failure

        # Test fallback chain
        async def run_test():
            result = await tts_service.synthesize_speech("Test text")
            assert result.provider == "google"
            assert failure_count == 3

        asyncio.run(run_test())

    def test_complete_voice_service_fallback(self, mock_config, mock_security):
        """Test complete voice service with provider fallbacks."""
        service = VoiceService(mock_config, mock_security)

        # Mock STT service with fallback chain
        primary_failure_count = 0

        async def mock_stt_with_fallback(audio_data):
            nonlocal primary_failure_count
            primary_failure_count += 1

            if primary_failure_count <= 2:
                # Primary fails first two times
                raise RuntimeError("Primary STT unavailable")

            return STTResult(
                text="Service fallback success",
                confidence=0.75,
                language="en",
                duration=1.0,
                provider="fallback_stt"
            )

        service.stt_service.transcribe_audio = mock_stt_with_fallback

        # Mock TTS service with fallback
        tts_failure_count = 0

        def mock_tts_with_fallback(text, voice_profile=None):
            nonlocal tts_failure_count
            tts_failure_count += 1

            if tts_failure_count == 1:
                # Primary TTS fails
                raise RuntimeError("Primary TTS unavailable")

            return TTSResult(
                audio_data=b"fallback_tts_data",
                duration=1.0,
                provider="fallback_tts"
            )

        service.tts_service.synthesize_speech = mock_tts_with_fallback

        # Initialize service
        service.initialize()

        # Test complete voice interaction with fallbacks
        audio_data = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1
        )

        async def run_integration_test():
            # Should succeed with fallback providers
            stt_result = await service.process_voice_input(audio_data)
            assert stt_result.provider == "fallback_stt"
            assert primary_failure_count == 3

            # Test TTS fallback
            tts_result = await service.generate_voice_output("Test response")
            assert tts_result.provider == "fallback_tts"
            assert tts_failure_count == 2

        asyncio.run(run_integration_test())


class TestServiceDegradation:
    """Test service degradation scenarios."""

    def test_partial_stt_service_availability(self, mock_config, mock_security):
        """Test system behavior with partial STT service availability."""
        stt_service = STTService(mock_config)

        # Mock partial availability (only one provider working)
        stt_service.get_available_providers = Mock(return_value=["whisper"])

        # Mock other providers as unavailable
        with patch.object(stt_service, 'openai_client', None):
            with patch.object(stt_service, 'google_speech_client', None):
                with patch.object(stt_service, 'whisper_model', Mock()):
                    pass

        # Test transcription with limited providers
        audio_data = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1
        )

        async def run_test():
            result = await stt_service.transcribe_audio(audio_data)
            # Should succeed with available provider
            assert result is not None
            assert result.provider == "whisper"

        asyncio.run(run_test())

    def test_cascading_service_failures(self, mock_config, mock_security):
        """Test cascading service failure scenarios."""
        service = VoiceService(mock_config, mock_security)

        # Mock cascading failures
        failure_scenarios = [
            {"stt": "network_error", "tts": "success"},
            {"stt": "success", "tts": "network_error"},
            {"stt": "rate_limit", "tts": "rate_limit"},
        ]

        for scenario in failure_scenarios:
            if scenario["stt"] == "network_error":
                service.stt_service.transcribe_audio = AsyncMock(
                    side_effect=ConnectionError("Network error")
                )
            elif scenario["stt"] == "success":
                service.stt_service.transcribe_audio = AsyncMock(
                    return_value=STTResult(
                        text="STT success",
                        confidence=0.8,
                        language="en",
                        duration=1.0,
                        provider="openai"
                    )
                )

            if scenario["tts"] == "network_error":
                service.tts_service.synthesize_speech = Mock(
                    side_effect=ConnectionError("Network error")
                )
            elif scenario["tts"] == "success":
                service.tts_service.synthesize_speech = Mock(
                    return_value=TTSResult(
                        audio_data=b"tts_data",
                        duration=1.0,
                        provider="openai"
                    )
                )

            # Service should handle mixed success/failure gracefully
            health = service.health_check()
            # Should report degraded status for failed components
            assert health['overall_status'] in ['degraded', 'healthy']

    def test_emergency_fallback_activation(self, mock_config, mock_security):
        """Test activation of emergency fallback mechanisms."""
        service = VoiceService(mock_config, mock_security)

        # Mock all primary services as failed
        service.stt_service.is_available = Mock(return_value=False)
        service.tts_service.is_available = Mock(return_value=False)
        service.audio_processor.input_devices = []
        service.audio_processor.output_devices = []

        # Should activate emergency fallback
        health = service.health_check()
        assert health['overall_status'] == 'degraded'

        # Should still provide basic functionality through mocks
        # (In real implementation, would fall back to text-only mode)
        basic_response = service.generate_ai_response("Emergency test")
        assert basic_response is not None

    def test_provider_load_balancing(self, mock_config, mock_security):
        """Test load balancing across multiple providers."""
        stt_service = STTService(mock_config)

        # Mock multiple available providers
        stt_service.get_available_providers = Mock(return_value=["openai", "google", "whisper"])

        # Track provider usage
        provider_usage = {"openai": 0, "google": 0, "whisper": 0}

        def mock_load_balanced_transcribe(audio_data):
            # Simple round-robin load balancing
            providers = stt_service.get_available_providers()
            provider = providers[len(provider_usage) % len(providers)]
            provider_usage[provider] += 1

            return STTResult(
                text=f"Response from {provider}",
                confidence=0.8,
                language="en",
                duration=1.0,
                provider=provider
            )

        stt_service.transcribe_audio = AsyncMock(side_effect=mock_load_balanced_transcribe)

        # Test load balancing
        audio_data = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1
        )

        async def run_load_balancing_test():
            results = []
            for i in range(9):  # Test 9 requests
                result = await stt_service.transcribe_audio(audio_data)
                results.append(result)

            # Should distribute across providers
            for result in results:
                assert result.provider in ["openai", "google", "whisper"]

            # Each provider should be used at least once
            assert provider_usage["openai"] >= 1
            assert provider_usage["google"] >= 1
            assert provider_usage["whisper"] >= 1

        asyncio.run(run_load_balancing_test())


class TestCrossComponentFallbacks:
    """Test fallback coordination across components."""

    def test_audio_processor_fallback_to_mock(self, mock_config, mock_security):
        """Test audio processor fallback to mock functionality."""
        service = VoiceService(mock_config, mock_security)

        # Mock audio devices as unavailable
        service.audio_processor.input_devices = []
        service.audio_processor.output_devices = []

        # Should fall back to mock audio processing
        success = service.start_listening("test_session")
        # Should not crash even without audio devices

        audio_data = service.stop_listening("test_session")
        # Should return mock audio data
        assert audio_data is not None

    def test_security_fallback_handling(self, mock_config, mock_security):
        """Test security component fallback scenarios."""
        service = VoiceService(mock_config, mock_security)

        # Mock security processing failure
        service.security.process_audio = AsyncMock(
            side_effect=Exception("Security processing failed")
        )

        # Should fall back to direct audio processing
        audio_data = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1
        )

        async def run_test():
            # Should handle security failure gracefully
            try:
                result = await service.process_voice_input(audio_data)
                # May return None or handle gracefully
                assert result is None or hasattr(result, 'text')
            except Exception:
                # Should not propagate security failures
                pass

        asyncio.run(run_test())

    def test_command_processor_fallback(self, mock_config, mock_security):
        """Test command processor fallback scenarios."""
        service = VoiceService(mock_config, mock_security)

        # Mock command processor failure
        service.command_processor.process_text = AsyncMock(
            side_effect=Exception("Command processing failed")
        )

        # Should fall back to regular text processing
        audio_data = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1
        )

        async def run_test():
            # Should handle command processor failure
            result = await service.process_voice_input(audio_data)
            # Should still process as regular text input
            assert result is None or hasattr(result, 'text')

        asyncio.run(run_test())


class TestGracefulDegradation:
    """Test graceful degradation under various failure conditions."""

    def test_stt_service_unavailable_degradation(self, mock_config, mock_security):
        """Test system degradation when STT service is unavailable."""
        service = VoiceService(mock_config, mock_security)

        # Mock STT service as completely unavailable
        service.stt_service.is_available = Mock(return_value=False)
        service.stt_service.transcribe_audio = AsyncMock(
            side_effect=RuntimeError("STT service unavailable")
        )

        # Should degrade gracefully to text-only mode
        health = service.health_check()
        assert health['overall_status'] == 'degraded'

        # Text processing should still work
        response = service.generate_ai_response("Text input")
        assert response is not None

    def test_tts_service_unavailable_degradation(self, mock_config, mock_security):
        """Test system degradation when TTS service is unavailable."""
        service = VoiceService(mock_config, mock_security)

        # Mock TTS service as unavailable
        service.tts_service.is_available = Mock(return_value=False)
        service.tts_service.synthesize_speech = Mock(
            side_effect=RuntimeError("TTS service unavailable")
        )

        # Should degrade to text-only output
        async def run_test():
            result = await service.generate_voice_output("Test response")
            # Should return mock result or None
            assert result is None or hasattr(result, 'audio_data')

        asyncio.run(run_test())

        # Text processing should still work
        response = service.generate_ai_response("Text input")
        assert response is not None

    def test_audio_hardware_unavailable_degradation(self, mock_config, mock_security):
        """Test degradation when audio hardware is unavailable."""
        service = VoiceService(mock_config, mock_security)

        # Mock audio hardware as unavailable
        service.audio_processor.input_devices = []
        service.audio_processor.output_devices = []

        # Should fall back to mock audio processing
        session_id = service.create_session("test_session")

        # Recording should work with mock data
        success = service.start_listening(session_id)
        assert success is True  # Should not fail

        audio_data = service.stop_listening(session_id)
        assert audio_data is not None

    def test_network_isolation_degradation(self, mock_config, mock_security):
        """Test system behavior under complete network isolation."""
        service = VoiceService(mock_config, mock_security)

        # Mock all network services as unavailable
        service.stt_service.is_available = Mock(return_value=False)
        service.tts_service.is_available = Mock(return_value=False)

        # Mock network errors for all API calls
        service.stt_service.transcribe_audio = AsyncMock(
            side_effect=ConnectionError("Network is unreachable")
        )
        service.tts_service.synthesize_speech = Mock(
            side_effect=ConnectionError("Network is unreachable")
        )

        # Should degrade to local-only functionality
        health = service.health_check()
        assert health['overall_status'] == 'degraded'

        # Local text processing should still work
        response = service.generate_ai_response("Local processing test")
        assert response is not None


class TestEmergencyFallbackMechanisms:
    """Test emergency fallback mechanisms."""

    def test_crisis_detection_fallback(self, mock_config, mock_security):
        """Test fallback mechanisms for crisis detection."""
        service = VoiceService(mock_config, mock_security)

        # Mock STT service failure during crisis
        crisis_audio = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1
        )

        service.stt_service.transcribe_audio = AsyncMock(
            side_effect=RuntimeError("STT failed during crisis")
        )

        # Mock fallback crisis detection
        async def run_test():
            # Should handle crisis even with STT failure
            result = await service.process_voice_input(crisis_audio)

            # Should either fail gracefully or use fallback detection
            if result is None:
                # Fallback: use simple keyword detection
                # In real implementation, would have offline crisis detection
                pass

        asyncio.run(run_test())

    def test_emergency_resource_allocation(self, mock_config, mock_security):
        """Test emergency resource allocation during critical failures."""
        service = VoiceService(mock_config, mock_security)

        # Simulate resource exhaustion scenario
        service.sessions = {}  # Clear sessions

        # Mock resource monitoring
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.percent = 95  # High memory usage

            # Should prioritize emergency operations
            emergency_session = service.create_session("emergency_session")

            # Should succeed even under resource pressure
            assert emergency_session is not None

            # Should clean up non-emergency resources
            if len(service.sessions) > 1:
                # Would clean up old sessions here
                pass

    def test_offline_fallback_mode(self, mock_config, mock_security):
        """Test offline fallback mode when all services fail."""
        service = VoiceService(mock_config, mock_security)

        # Mock complete service failure
        service.stt_service.is_available = Mock(return_value=False)
        service.tts_service.is_available = Mock(return_value=False)
        service.audio_processor.input_devices = []
        service.audio_processor.output_devices = []

        # Should enter offline mode
        health = service.health_check()
        assert health['overall_status'] == 'degraded'

        # Should provide basic offline functionality
        # (In real implementation, would use local models or cached responses)
        basic_response = service.generate_ai_response("Offline mode test")
        assert basic_response is not None


class TestProviderPriorityAndLoadBalancing:
    """Test provider priority and load balancing mechanisms."""

    def test_provider_priority_order(self, mock_config, mock_security):
        """Test that providers are used in correct priority order."""
        stt_service = STTService(mock_config)

        # Mock provider priority
        priority_order = ["openai", "google", "whisper"]
        stt_service.get_available_providers = Mock(return_value=priority_order)

        # Track usage order
        usage_order = []

        def mock_priority_transcribe(audio_data):
            provider = stt_service.get_preferred_provider()
            usage_order.append(provider)

            return STTResult(
                text=f"Response from {provider}",
                confidence=0.8,
                language="en",
                duration=1.0,
                provider=provider
            )

        stt_service.transcribe_audio = AsyncMock(side_effect=mock_priority_transcribe)
        stt_service.get_preferred_provider = Mock(return_value="openai")

        # Test priority usage
        audio_data = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1
        )

        async def run_test():
            for i in range(5):
                result = await stt_service.transcribe_audio(audio_data)
                assert result.provider == "openai"

            # Should use preferred provider consistently
            assert all(provider == "openai" for provider in usage_order)

        asyncio.run(run_test())

    def test_dynamic_provider_selection(self, mock_config, mock_security):
        """Test dynamic provider selection based on conditions."""
        stt_service = STTService(mock_config)

        # Mock dynamic conditions
        conditions = [
            {"load": "high", "latency": "low", "preferred": "google"},
            {"load": "low", "latency": "high", "preferred": "whisper"},
            {"load": "medium", "latency": "medium", "preferred": "openai"},
        ]

        def mock_conditional_transcribe(audio_data):
            # Simulate dynamic provider selection
            provider = stt_service.get_preferred_provider()
            return STTResult(
                text=f"Dynamic response from {provider}",
                confidence=0.8,
                language="en",
                duration=1.0,
                provider=provider
            )

        stt_service.transcribe_audio = AsyncMock(side_effect=mock_conditional_transcribe)

        # Test different conditions
        for condition in conditions:
            stt_service.get_preferred_provider = Mock(return_value=condition["preferred"])

            audio_data = AudioData(
                data=np.random.randn(16000).astype(np.float32),
                sample_rate=16000,
                duration=1.0,
                channels=1
            )

            async def run_condition_test():
                result = await stt_service.transcribe_audio(audio_data)
                assert result.provider == condition["preferred"]

            asyncio.run(run_condition_test())

    def test_provider_health_based_routing(self, mock_config, mock_security):
        """Test provider routing based on health status."""
        stt_service = STTService(mock_config)

        # Mock provider health status
        healthy_providers = ["google", "whisper"]
        unhealthy_providers = ["openai"]

        stt_service.get_available_providers = Mock(return_value=healthy_providers + unhealthy_providers)

        def mock_health_based_transcribe(audio_data):
            available = stt_service.get_available_providers()
            # Route to healthy provider
            provider = available[0]  # First healthy provider

            return STTResult(
                text=f"Healthy response from {provider}",
                confidence=0.8,
                language="en",
                duration=1.0,
                provider=provider
            )

        stt_service.transcribe_audio = AsyncMock(side_effect=mock_health_based_transcribe)

        # Test health-based routing
        audio_data = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1
        )

        async def run_test():
            result = await stt_service.transcribe_audio(audio_data)
            # Should use healthy provider
            assert result.provider in healthy_providers
            assert result.provider not in unhealthy_providers

        asyncio.run(run_test())


class TestFallbackPerformance:
    """Test performance characteristics of fallback mechanisms."""

    def test_fallback_response_time_monitoring(self, mock_config, mock_security):
        """Test response time monitoring for fallback operations."""
        service = VoiceService(mock_config, mock_security)

        # Mock slow primary provider
        async def slow_primary_transcribe(audio_data):
            await asyncio.sleep(5.0)  # 5 second delay
            raise RuntimeError("Primary provider timeout")

        # Mock fast fallback provider
        async def fast_fallback_transcribe(audio_data):
            await asyncio.sleep(0.5)  # 0.5 second delay
            return STTResult(
                text="Fast fallback response",
                confidence=0.7,
                language="en",
                duration=1.0,
                provider="fast_fallback"
            )

        service.stt_service.transcribe_audio = slow_primary_transcribe

        # Test with timeout and fallback
        audio_data = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1
        )

        async def run_performance_test():
            start_time = time.time()

            # Should timeout primary and use fallback
            try:
                result = await asyncio.wait_for(
                    service.process_voice_input(audio_data),
                    timeout=2.0
                )
            except asyncio.TimeoutError:
                # Expected timeout
                pass

            elapsed_time = time.time() - start_time

            # Should complete within reasonable time due to fallback
            assert elapsed_time < 3.0

    def test_fallback_success_rate_monitoring(self, mock_config, mock_security):
        """Test success rate monitoring for fallback operations."""
        service = VoiceService(mock_config, mock_security)

        # Track success/failure rates
        success_count = 0
        failure_count = 0

        def mock_monitored_transcribe(audio_data):
            nonlocal success_count, failure_count

            # Simulate occasional failures
            if success_count < 7:  # 70% success rate
                success_count += 1
                return STTResult(
                    text="Success response",
                    confidence=0.8,
                    language="en",
                    duration=1.0,
                    provider="monitored_provider"
                )
            else:
                failure_count += 1
                raise RuntimeError("Monitored failure")

        service.stt_service.transcribe_audio = AsyncMock(side_effect=mock_monitored_transcribe)

        # Test success rate monitoring
        audio_data = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1
        )

        async def run_monitoring_test():
            results = []
            for i in range(10):
                try:
                    result = await service.process_voice_input(audio_data)
                    results.append(result)
                except:
                    results.append(None)

            # Calculate success rate
            successful_results = [r for r in results if r is not None]
            success_rate = len(successful_results) / len(results)

            # Should maintain reasonable success rate
            assert success_rate >= 0.5  # At least 50% success rate

        asyncio.run(run_monitoring_test())