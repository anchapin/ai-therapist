"""
Comprehensive tests for VoiceService error recovery, exception handling, and resilience.
Covers failure scenarios, recovery mechanisms, and graceful degradation.
"""

import pytest
import pytest_asyncio
import asyncio
import time
import threading
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
import sys
from datetime import datetime, timedelta

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

from voice.voice_service import VoiceService, VoiceSession, VoiceSessionState
from voice.config import VoiceConfig
from voice.audio_processor import AudioData


class TestVoiceServiceErrorRecovery:
    """Tests for VoiceService error recovery and resilience."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock voice configuration."""
        config = Mock(spec=VoiceConfig)
        config.stt_enabled = True
        config.tts_enabled = True
        config.commands_enabled = True
        config.security_enabled = True
        config.max_session_duration = 3600
        config.audio_sample_rate = 16000
        config.audio_channels = 1
        config.session_timeout = 300
        config.default_voice_profile = "calm_therapist"
        config.voice_enabled = True
        return config

    @pytest.fixture
    def mock_security(self):
        """Create a mock security service."""
        security = Mock()
        security.initialize = Mock(return_value=True)
        security.encrypt_audio = AsyncMock(return_value=b'encrypted_audio')
        security.decrypt_audio = AsyncMock(return_value=b'decrypted_audio')
        security.sanitize_transcription = Mock(return_value="sanitized text")
        security.is_healthy = True
        security.health_check = Mock(return_value={'status': 'healthy', 'issues': []})
        return security

    @pytest.fixture
    def voice_service(self, mock_config, mock_security):
        """Create a VoiceService instance with mocked dependencies."""
        with patch('voice.voice_service.SimplifiedAudioProcessor'), \
             patch('voice.voice_service.STTService'), \
             patch('voice.voice_service.TTSService'), \
             patch('voice.voice_service.VoiceCommandProcessor'):

            service = VoiceService(mock_config, mock_security)
            service.session_repo = Mock()
            service.voice_data_repo = Mock()
            service.audit_repo = Mock()
            service.consent_repo = Mock()
            service._db_initialized = True

            # Mock component health checks
            service.audio_processor.health_check = Mock(return_value={'status': 'healthy', 'issues': []})
            service.stt_service.health_check = Mock(return_value={'status': 'healthy', 'issues': []})
            service.tts_service.health_check = Mock(return_value={'status': 'healthy', 'issues': []})
            service.command_processor.health_check = Mock(return_value={'status': 'healthy', 'issues': []})

            return service

    # Initialization Error Recovery
    def test_initialization_recovery_after_config_error(self, voice_service, mock_config):
        """Test recovery after configuration error during initialization."""
        # Make config invalid
        mock_config.voice_enabled = None

        success = voice_service.initialize()
        assert success is False

        # Reset config and retry
        mock_config.voice_enabled = True
        success = voice_service.initialize()
        assert success is True

        voice_service.cleanup()

    def test_initialization_recovery_partial_component_failure(self, voice_service):
        """Test initialization recovers when some components fail."""
        # Make STT service fail initialization
        voice_service.stt_service.is_available = Mock(return_value=False)

        success = voice_service.initialize()
        # Should still initialize even if STT is unavailable
        assert success is True

        voice_service.cleanup()

    def test_initialization_database_connection_failure(self, voice_service):
        """Test initialization handles database connection failure."""
        voice_service._initialize_database_repositories = Mock(side_effect=Exception("DB connect failed"))

        success = voice_service.initialize()
        # Should initialize without database
        assert success is True
        assert voice_service._db_initialized is False

        voice_service.cleanup()

    # Runtime Error Recovery
    def test_runtime_recovery_stt_service_failure(self, voice_service):
        """Test recovery when STT service fails during operation."""
        voice_service.initialize()
        session_id = voice_service.create_session()

        # Make STT fail
        voice_service.stt_service.transcribe_audio = Mock(side_effect=Exception("STT down"))

        audio_data = AudioData(b'test', 16000, 1)
        result = asyncio.run(voice_service.process_voice_input(audio_data, session_id))

        # Should return None or handle gracefully
        assert result is None or hasattr(result, 'text')

        voice_service.cleanup()

    def test_runtime_recovery_tts_service_failure(self, voice_service):
        """Test recovery when TTS service fails during operation."""
        voice_service.initialize()
        session_id = voice_service.create_session()

        # Make TTS fail
        voice_service.tts_service.synthesize_speech = Mock(side_effect=Exception("TTS down"))

        result = asyncio.run(voice_service.generate_voice_output("Test text", session_id))

        # Should return mock result on failure
        assert result is not None
        assert hasattr(result, 'audio_data')

        voice_service.cleanup()

    def test_runtime_recovery_audio_processor_failure(self, voice_service):
        """Test recovery when audio processor fails."""
        voice_service.initialize()
        session_id = voice_service.create_session()

        # Make audio processor fail
        voice_service.audio_processor.start_recording = Mock(side_effect=Exception("Audio device error"))

        result = voice_service.start_listening(session_id)

        assert result is False
        session = voice_service.get_session(session_id)
        assert session.state == VoiceSessionState.ERROR

        voice_service.cleanup()

    # Network and Connectivity Errors
    def test_network_error_recovery_stt_timeout(self, voice_service):
        """Test recovery from STT service timeout."""
        voice_service.initialize()
        session_id = voice_service.create_session()

        # Simulate timeout
        async def timeout_stt(audio_data):
            await asyncio.sleep(30)  # Long delay
            return Mock(text="delayed response")

        voice_service.stt_service.transcribe_audio = timeout_stt

        # Should handle timeout gracefully
        audio_data = AudioData(b'test', 16000, 1)
        result = asyncio.run(voice_service.process_voice_input(audio_data, session_id))

        # Implementation should handle or timeout
        assert result is None or hasattr(result, 'text')

        voice_service.cleanup()

    def test_network_error_recovery_fallback_stt_activation(self, voice_service):
        """Test fallback STT service activation on primary failure."""
        voice_service.initialize()
        session_id = voice_service.create_session()

        # Setup fallback STT
        fallback_result = Mock()
        fallback_result.text = "Fallback transcription"
        voice_service.fallback_stt_service = Mock()
        voice_service.fallback_stt_service.transcribe_audio = Mock(return_value=fallback_result)

        # Make primary STT fail
        voice_service.stt_service.transcribe_audio = Mock(side_effect=Exception("Primary STT failed"))

        audio_data = AudioData(b'test', 16000, 1)
        result = asyncio.run(voice_service.process_voice_input(audio_data, session_id))

        assert result is not None
        assert result.text == "Fallback transcription"

        voice_service.cleanup()

    # Memory and Resource Errors
    def test_memory_error_recovery_large_audio_processing(self, voice_service):
        """Test recovery from memory errors during large audio processing."""
        voice_service.initialize()
        session_id = voice_service.create_session()

        # Create very large audio data
        large_audio = AudioData(b'x' * (10 * 1024 * 1024), 16000, 1)  # 10MB

        # Make processing fail with memory error
        voice_service.stt_service.transcribe_audio = Mock(side_effect=MemoryError("Out of memory"))

        result = asyncio.run(voice_service.process_voice_input(large_audio, session_id))

        # Should handle memory error gracefully
        assert result is None

        voice_service.cleanup()

    def test_resource_error_recovery_file_handle_exhaustion(self, voice_service):
        """Test recovery when file handles are exhausted."""
        voice_service.initialize()

        # Simulate file handle exhaustion
        voice_service.audio_processor.start_recording = Mock(side_effect=OSError("Too many open files"))

        session_id = voice_service.create_session()
        result = voice_service.start_listening(session_id)

        assert result is False
        session = voice_service.get_session(session_id)
        assert session.state == VoiceSessionState.ERROR

        voice_service.cleanup()

    # Concurrent Error Scenarios
    def test_concurrent_error_recovery_multiple_failures(self, voice_service):
        """Test recovery when multiple operations fail concurrently."""
        voice_service.initialize()

        results = []
        errors = []

        def failing_operation(session_num):
            try:
                session_id = voice_service.create_session(f"user_{session_num}")

                # Make multiple services fail
                voice_service.stt_service.transcribe_audio = Mock(side_effect=Exception("STT failed"))
                voice_service.tts_service.synthesize_speech = Mock(side_effect=Exception("TTS failed"))

                # Try operations
                audio_data = AudioData(b'test', 16000, 1)
                stt_result = asyncio.run(voice_service.process_voice_input(audio_data, session_id))
                tts_result = asyncio.run(voice_service.generate_voice_output("test", session_id))

                results.append({
                    'session': session_num,
                    'stt_result': stt_result,
                    'tts_result': tts_result
                })

            except Exception as e:
                errors.append(f"session_{session_num}: {e}")

        threads = [threading.Thread(target=failing_operation, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should handle all failures gracefully
        assert len(results) == 5
        assert len(errors) == 0

        voice_service.cleanup()

    # State Machine Error Recovery
    def test_state_machine_error_recovery_invalid_transitions(self, voice_service):
        """Test recovery from invalid state transitions."""
        voice_service.initialize()
        session_id = voice_service.create_session()

        session = voice_service.get_session(session_id)

        # Force invalid state
        session.state = "invalid_state"

        # Try to perform operations - should handle gracefully
        result = voice_service.start_listening(session_id)
        # Implementation should handle invalid state

        voice_service.cleanup()

    def test_state_machine_error_recovery_corrupted_session(self, voice_service):
        """Test recovery when session data is corrupted."""
        voice_service.initialize()
        session_id = voice_service.create_session()

        session = voice_service.get_session(session_id)

        # Corrupt session data
        session.conversation_history = "corrupted_string_instead_of_list"
        session.metadata = None

        # Operations should handle corruption gracefully
        result = voice_service.add_conversation_entry(session_id, 'user', 'test')
        # Should handle gracefully

        voice_service.cleanup()

    # Callback Error Handling
    def test_callback_error_recovery_failing_callbacks(self, voice_service):
        """Test recovery when callbacks fail."""
        voice_service.initialize()
        session_id = voice_service.create_session()

        # Set failing callbacks
        def failing_text_callback(session_id, text):
            raise Exception("Callback error")

        def failing_audio_callback(audio_data):
            raise Exception("Audio callback error")

        voice_service.on_text_received = failing_text_callback
        voice_service.on_audio_played = failing_audio_callback

        # Operations should continue despite callback failures
        audio_data = AudioData(b'test', 16000, 1)
        result = asyncio.run(voice_service.process_voice_input(audio_data, session_id))

        # Should still process successfully
        assert result is not None

        voice_service.cleanup()

    def test_callback_error_recovery_none_callbacks(self, voice_service):
        """Test handling when callbacks are None."""
        voice_service.initialize()
        session_id = voice_service.create_session()

        # Explicitly set callbacks to None
        voice_service.on_text_received = None
        voice_service.on_audio_played = None
        voice_service.on_command_executed = None
        voice_service.on_error = None

        # Operations should work without callbacks
        audio_data = AudioData(b'test', 16000, 1)
        result = asyncio.run(voice_service.process_voice_input(audio_data, session_id))

        assert result is not None

        voice_service.cleanup()

    # Database Error Recovery
    def test_database_error_recovery_connection_loss(self, voice_service):
        """Test recovery when database connection is lost during operation."""
        voice_service.initialize()
        session_id = voice_service.create_session("test_user")

        # Simulate database connection loss
        voice_service.session_repo.save = Mock(side_effect=Exception("Connection lost"))
        voice_service.session_repo.find_by_id = Mock(side_effect=Exception("Connection lost"))

        # Operations should continue with in-memory fallback
        result = voice_service.add_conversation_entry(session_id, 'user', 'test')
        assert result is True

        voice_service.cleanup()

    def test_database_error_recovery_transaction_rollback(self, voice_service):
        """Test recovery from database transaction failures."""
        voice_service.initialize()

        # Simulate transaction failure
        voice_service.session_repo.save = Mock(side_effect=Exception("Transaction failed"))

        session_id = voice_service.create_session("test_user")

        # Should still work in memory
        assert session_id in voice_service.sessions

        voice_service.cleanup()

    # Security Error Recovery
    def test_security_error_recovery_encryption_failure(self, voice_service):
        """Test recovery when audio encryption fails."""
        voice_service.initialize()
        session_id = voice_service.create_session()

        # Make encryption fail
        voice_service.security.encrypt_audio = AsyncMock(side_effect=Exception("Encryption failed"))

        # Operations should continue without encryption
        audio_data = AudioData(b'test', 16000, 1)
        result = asyncio.run(voice_service.process_voice_input(audio_data, session_id))

        assert result is not None

        voice_service.cleanup()

    def test_security_error_recovery_pii_masking_failure(self, voice_service):
        """Test recovery when PII masking fails."""
        voice_service.initialize()

        # Disable PII protection
        voice_service.pii_protection = None

        session_id = voice_service.create_session()

        # Should work without PII masking
        result = voice_service.add_conversation_entry(session_id, 'user', 'My SSN is 123-45-6789')
        assert result is True

        voice_service.cleanup()

    # Configuration Error Recovery
    def test_config_error_recovery_runtime_config_change(self, voice_service, mock_config):
        """Test recovery when configuration changes at runtime."""
        voice_service.initialize()

        # Change config at runtime
        mock_config.max_session_duration = -1  # Invalid

        # Operations should continue with previous valid config
        session_id = voice_service.create_session()
        assert session_id is not None

        voice_service.cleanup()

    def test_config_error_recovery_missing_config_values(self, voice_service, mock_config):
        """Test recovery when config values are missing."""
        voice_service.initialize()

        # Remove required config attributes
        del mock_config.audio_sample_rate

        # Should handle missing config gracefully
        session_id = voice_service.create_session()
        assert session_id is not None

        voice_service.cleanup()

    # Threading Error Recovery
    def test_threading_error_recovery_worker_thread_crash(self, voice_service):
        """Test recovery when worker thread crashes."""
        voice_service.initialize()

        # Inject crash into worker loop
        original_loop = voice_service._process_voice_queue

        async def crashing_loop():
            raise SystemExit("Thread crash")

        with patch.object(voice_service, '_process_voice_queue', side_effect=crashing_loop):
            time.sleep(0.1)

        # Service should handle thread crash
        assert voice_service.is_running is False

    def test_threading_error_recovery_event_loop_failure(self, voice_service):
        """Test recovery when event loop fails."""
        voice_service.initialize()

        # Corrupt event loop
        voice_service._event_loop = None

        # Should recreate event loop on next operation
        time.sleep(0.1)

        # Event loop should be recreated
        assert voice_service._event_loop is not None

        voice_service.cleanup()

    # Performance Degradation Recovery
    def test_performance_degradation_recovery_high_load(self, voice_service):
        """Test recovery from performance degradation under high load."""
        voice_service.initialize()

        start_time = time.time()

        # Create high load
        for i in range(50):
            session_id = voice_service.create_session(f"user_{i}")
            for j in range(10):
                voice_service.add_conversation_entry(session_id, 'user', f'message {j}')

        end_time = time.time()

        # Should complete within reasonable time
        assert end_time - start_time < 5.0

        # All sessions should exist
        assert len(voice_service.sessions) == 50

        voice_service.cleanup()

    def test_performance_degradation_recovery_memory_pressure(self, voice_service):
        """Test recovery under memory pressure."""
        voice_service.initialize()

        # Create memory pressure with large conversation histories
        session_id = voice_service.create_session()

        for i in range(1000):
            large_message = f"Message {i} " * 100  # Large message
            voice_service.add_conversation_entry(session_id, 'user', large_message)

        # Should handle memory pressure gracefully
        history = voice_service.get_conversation_history(session_id)
        assert len(history) == 1000

        voice_service.cleanup()

    # External Service Failure Recovery
    def test_external_service_recovery_api_rate_limiting(self, voice_service):
        """Test recovery from API rate limiting."""
        voice_service.initialize()
        session_id = voice_service.create_session()

        # Simulate rate limiting
        voice_service.stt_service.transcribe_audio = Mock(side_effect=Exception("Rate limit exceeded"))

        audio_data = AudioData(b'test', 16000, 1)
        result = asyncio.run(voice_service.process_voice_input(audio_data, session_id))

        # Should handle rate limiting gracefully
        assert result is None

        voice_service.cleanup()

    def test_external_service_recovery_service_unavailable(self, voice_service):
        """Test recovery when external service is completely unavailable."""
        voice_service.initialize()
        session_id = voice_service.create_session()

        # Make all STT services unavailable
        voice_service.stt_service.transcribe_audio = Mock(side_effect=Exception("Service unavailable"))
        voice_service.fallback_stt_service = None

        audio_data = AudioData(b'test', 16000, 1)
        result = asyncio.run(voice_service.process_voice_input(audio_data, session_id))

        # Should handle unavailability gracefully
        assert result is None

        voice_service.cleanup()

    # Graceful Degradation Tests
    def test_graceful_degradation_partial_service_failure(self, voice_service):
        """Test graceful degradation when some services fail."""
        voice_service.initialize()

        # Disable TTS but keep STT working
        voice_service.tts_service.synthesize_speech = Mock(side_effect=Exception("TTS unavailable"))

        session_id = voice_service.create_session()

        # STT should still work
        mock_stt_result = Mock()
        mock_stt_result.text = "Hello"
        voice_service.stt_service.transcribe_audio = Mock(return_value=mock_stt_result)

        audio_data = AudioData(b'test', 16000, 1)
        stt_result = asyncio.run(voice_service.process_voice_input(audio_data, session_id))

        # TTS should fail gracefully
        tts_result = asyncio.run(voice_service.generate_voice_output("test", session_id))

        assert stt_result is not None
        assert tts_result is not None  # Should return mock result

        voice_service.cleanup()

    def test_graceful_degradation_fallback_mechanisms(self, voice_service):
        """Test fallback mechanisms for graceful degradation."""
        voice_service.initialize()
        session_id = voice_service.create_session()

        # Setup fallback STT
        fallback_result = Mock()
        fallback_result.text = "Fallback text"
        voice_service.fallback_stt_service = Mock()
        voice_service.fallback_stt_service.transcribe_audio = Mock(return_value=fallback_result)

        # Primary STT fails
        voice_service.stt_service.transcribe_audio = Mock(side_effect=Exception("Primary failed"))

        audio_data = AudioData(b'test', 16000, 1)
        result = asyncio.run(voice_service.process_voice_input(audio_data, session_id))

        # Should fall back successfully
        assert result is not None
        assert result.text == "Fallback text"

        voice_service.cleanup()

    # Recovery Time Testing
    def test_recovery_time_initialization_retry(self, voice_service, mock_security):
        """Test time taken for initialization retry after failure."""
        # Make first initialization fail
        mock_security.initialize = Mock(side_effect=Exception("Init failed"))

        start_time = time.time()
        success1 = voice_service.initialize()
        fail_time = time.time()

        # Make second initialization succeed
        mock_security.initialize = Mock(return_value=True)
        success2 = voice_service.initialize()
        success_time = time.time()

        assert success1 is False
        assert success2 is True

        # Recovery should be reasonably fast
        recovery_time = success_time - fail_time
        assert recovery_time < 1.0

        voice_service.cleanup()

    def test_recovery_time_runtime_service_restoration(self, voice_service):
        """Test time taken to restore service at runtime."""
        voice_service.initialize()
        session_id = voice_service.create_session()

        # Make service fail
        voice_service.stt_service.transcribe_audio = Mock(side_effect=Exception("Service down"))

        start_time = time.time()
        audio_data = AudioData(b'test', 16000, 1)
        result1 = asyncio.run(voice_service.process_voice_input(audio_data, session_id))
        fail_time = time.time()

        # Restore service
        mock_result = Mock()
        mock_result.text = "Restored"
        voice_service.stt_service.transcribe_audio = Mock(return_value=mock_result)

        result2 = asyncio.run(voice_service.process_voice_input(audio_data, session_id))
        restore_time = time.time()

        assert result1 is None
        assert result2 is not None

        # Restoration should be fast
        recovery_time = restore_time - fail_time
        assert recovery_time < 0.1

        voice_service.cleanup()
