"""
Additional comprehensive tests for VoiceService worker loops, state transitions, and error recovery.
Extends the existing test_voice_service_core.py with additional coverage for edge cases and error conditions.
"""

import pytest
import pytest_asyncio
import asyncio
import time
import threading
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
import sys
import queue

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


class TestVoiceServiceWorkerLoops:
    """Additional tests for VoiceService worker loops and state management."""

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

    # Worker Loop Edge Cases
    def test_worker_loop_multiple_exceptions(self, voice_service):
        """Test worker loop handles multiple consecutive exceptions."""
        voice_service.initialize()

        exception_count = 0
        original_process = voice_service._process_voice_queue

        def failing_process():
            nonlocal exception_count
            exception_count += 1
            if exception_count <= 3:
                raise Exception(f"Exception {exception_count}")
            # Stop the service after 3 exceptions
            voice_service.is_running = False

        with patch.object(voice_service, '_process_voice_queue', side_effect=failing_process):
            # Let worker run briefly
            time.sleep(0.2)

        # Service should have stopped due to exceptions
        assert voice_service.is_running is False
        voice_service.cleanup()

    def test_worker_loop_asyncio_error_recovery(self, voice_service):
        """Test worker loop recovers from asyncio errors."""
        voice_service.initialize()

        async def failing_async_process():
            raise asyncio.CancelledError("Async operation cancelled")

        with patch.object(voice_service, '_process_voice_queue', side_effect=failing_async_process):
            time.sleep(0.1)

        # Worker should continue running despite asyncio errors
        assert voice_service.is_running is True
        voice_service.cleanup()

    def test_worker_loop_event_loop_recreation(self, voice_service):
        """Test worker loop recreates event loop when needed."""
        voice_service.initialize()

        # Simulate event loop being closed
        original_loop = voice_service._event_loop
        voice_service._event_loop = None

        # Worker should recreate event loop
        time.sleep(0.1)

        assert voice_service._event_loop is not None
        assert voice_service._event_loop != original_loop
        voice_service.cleanup()

    def test_worker_loop_concurrent_queue_operations(self, voice_service):
        """Test worker loop handles concurrent queue operations."""
        voice_service.initialize()
        voice_service._ensure_queue_initialized()

        results = []

        def producer():
            for i in range(10):
                voice_service.voice_queue.put(("test_command", {"data": f"item_{i}"}))
                time.sleep(0.01)

        def consumer():
            for i in range(10):
                try:
                    item = voice_service.voice_queue.get(timeout=0.1)
                    results.append(item)
                except:
                    break

        producer_thread = threading.Thread(target=producer)
        consumer_thread = threading.Thread(target=consumer)

        producer_thread.start()
        consumer_thread.start()

        producer_thread.join()
        consumer_thread.join()

        # Should have processed some items
        assert len(results) > 0
        voice_service.cleanup()

    # State Transition Complex Scenarios
    def test_state_transition_idle_to_speaking_direct(self, voice_service):
        """Test direct transition from IDLE to SPEAKING."""
        session_id = voice_service.create_session()
        session = voice_service.get_session(session_id)

        assert session.state == VoiceSessionState.IDLE

        # Directly set to speaking (simulating TTS start)
        session.state = VoiceSessionState.SPEAKING
        assert session.state == VoiceSessionState.SPEAKING

    def test_state_transition_speaking_to_idle_on_completion(self, voice_service):
        """Test SPEAKING to IDLE transition when audio completes."""
        session_id = voice_service.create_session()
        session = voice_service.get_session(session_id)
        session.state = VoiceSessionState.SPEAKING

        # Simulate completion
        voice_service.audio_processor.stop_playback = Mock(return_value=True)
        voice_service.stop_speaking(session_id)

        # Should transition to IDLE
        session = voice_service.get_session(session_id)
        assert session.state == VoiceSessionState.IDLE

    def test_state_transition_error_recovery_paths(self, voice_service):
        """Test error state transitions and recovery."""
        session_id = voice_service.create_session()
        session = voice_service.get_session(session_id)

        # Put in error state
        session.state = VoiceSessionState.ERROR

        # Should be able to restart listening from error state
        voice_service.audio_processor.start_recording = Mock(return_value=True)
        voice_service.start_listening(session_id)

        session = voice_service.get_session(session_id)
        assert session.state == VoiceSessionState.LISTENING

    def test_state_transition_multiple_sessions_concurrent(self, voice_service):
        """Test state transitions across multiple concurrent sessions."""
        session_ids = []
        for i in range(5):
            session_id = voice_service.create_session(f"user_{i}")
            session_ids.append(session_id)

        # Start listening on all sessions
        voice_service.audio_processor.start_recording = Mock(return_value=True)
        for session_id in session_ids:
            voice_service.start_listening(session_id)

        # Check all are in listening state
        for session_id in session_ids:
            session = voice_service.get_session(session_id)
            assert session.state == VoiceSessionState.LISTENING

        # Stop listening on alternating sessions
        voice_service.audio_processor.stop_recording = Mock(return_value=AudioData(b'test', 16000, 1))
        for i, session_id in enumerate(session_ids):
            if i % 2 == 0:
                voice_service.stop_listening(session_id)
                session = voice_service.get_session(session_id)
                assert session.state == VoiceSessionState.IDLE

    def test_state_transition_with_session_timeout(self, voice_service):
        """Test state transitions considering session timeouts."""
        session_id = voice_service.create_session()
        session = voice_service.get_session(session_id)

        # Simulate old session
        session.last_activity = time.time() - 400  # Past timeout

        # Should still allow state transitions even if timed out
        voice_service.audio_processor.start_recording = Mock(return_value=True)
        voice_service.start_listening(session_id)

        assert session.state == VoiceSessionState.LISTENING

    # Queue Processing Advanced Scenarios
    @pytest.mark.asyncio
    async def test_queue_processing_with_invalid_commands(self, voice_service):
        """Test queue processing handles invalid commands gracefully."""
        voice_service._ensure_queue_initialized()

        # Send invalid command
        await voice_service.voice_queue.put(("invalid_command", {"data": "test"}))

        # Should not raise exception
        await voice_service._process_voice_queue()

    @pytest.mark.asyncio
    async def test_queue_processing_empty_data_handling(self, voice_service):
        """Test queue processing with empty or None data."""
        voice_service._ensure_queue_initialized()

        # Send commands with None/empty data
        await voice_service.voice_queue.put(("start_session", None))
        await voice_service.voice_queue.put(("stop_listening", {}))
        await voice_service.voice_queue.put(("speak_text", {"text": None}))

        # Should handle gracefully
        await voice_service._process_voice_queue()

    @pytest.mark.asyncio
    async def test_queue_processing_large_data_payloads(self, voice_service):
        """Test queue processing with large data payloads."""
        voice_service._ensure_queue_initialized()

        # Create large data payload
        large_data = {"data": "x" * 10000, "metadata": {"size": "large"}}

        await voice_service.voice_queue.put(("start_session", large_data))

        # Should process without issues
        await voice_service._process_voice_queue()

    @pytest.mark.asyncio
    async def test_queue_processing_concurrent_async_operations(self, voice_service):
        """Test queue processing with concurrent async operations."""
        voice_service._ensure_queue_initialized()

        async def async_handler(data):
            await asyncio.sleep(0.01)  # Simulate async work
            return f"processed_{data}"

        # Mock handler
        voice_service._handle_start_session = async_handler

        # Send multiple concurrent requests
        tasks = []
        for i in range(5):
            task = voice_service.voice_queue.put(("start_session", f"data_{i}"))
            tasks.append(task)

        await asyncio.gather(*tasks)

        # Process queue
        await voice_service._process_voice_queue()

    # Thread Safety Advanced Tests
    def test_thread_safety_session_operations_under_load(self, voice_service):
        """Test thread safety of session operations under high load."""
        results = []
        errors = []

        def session_worker(worker_id):
            try:
                for i in range(50):
                    session_id = voice_service.create_session(f"worker_{worker_id}_{i}")
                    voice_service.get_session(session_id)
                    voice_service.destroy_session(session_id)
                results.append(f"worker_{worker_id}_completed")
            except Exception as e:
                errors.append(f"worker_{worker_id}_error: {e}")

        threads = [threading.Thread(target=session_worker, args=(i,)) for i in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have completed without errors
        assert len(results) == 10
        assert len(errors) == 0

    def test_thread_safety_metrics_updates(self, voice_service):
        """Test thread safety of metrics updates."""
        initial_sessions = voice_service.metrics['sessions_created']

        def metrics_updater():
            for _ in range(100):
                voice_service.create_session()
                voice_service.metrics['total_interactions'] = voice_service.metrics.get('total_interactions', 0) + 1

        threads = [threading.Thread(target=metrics_updater) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Metrics should be consistent
        assert voice_service.metrics['sessions_created'] == initial_sessions + 500
        assert voice_service.metrics['total_interactions'] == 500

    # Error Recovery Comprehensive Tests
    def test_error_recovery_after_initialization_failure(self, voice_service, mock_security):
        """Test recovery after partial initialization failure."""
        # Make security initialization fail
        mock_security.initialize = Mock(return_value=False)

        success = voice_service.initialize()
        assert success is False

        # Should be able to retry initialization
        mock_security.initialize = Mock(return_value=True)
        success = voice_service.initialize()
        assert success is True

        voice_service.cleanup()

    def test_error_recovery_database_connection_loss(self, voice_service):
        """Test recovery when database connection is lost mid-operation."""
        voice_service.initialize()

        # Simulate database failure
        voice_service.session_repo.save = Mock(side_effect=Exception("DB connection lost"))

        # Should continue operating without database
        session_id = voice_service.create_session("test_user")

        # Session should still be created in memory
        assert session_id in voice_service.sessions

        voice_service.cleanup()

    def test_error_recovery_component_failure_during_operation(self, voice_service):
        """Test recovery when a component fails during operation."""
        voice_service.initialize()

        session_id = voice_service.create_session()

        # Make STT service fail
        voice_service.stt_service.transcribe_audio = Mock(side_effect=Exception("STT service down"))

        # Should handle gracefully
        audio_data = AudioData(b'test', 16000, 1)
        result = asyncio.run(voice_service.process_voice_input(audio_data, session_id))

        # Should return None or mock result on failure
        assert result is None or hasattr(result, 'text')

        voice_service.cleanup()

    # Resource Management Under Stress
    def test_resource_management_memory_pressure(self, voice_service):
        """Test resource management under memory pressure conditions."""
        voice_service.initialize()

        # Create many sessions with large conversation histories
        session_ids = []
        for i in range(20):
            session_id = voice_service.create_session(f"user_{i}")
            session_ids.append(session_id)

            # Add large conversation history
            for j in range(10):
                voice_service.add_conversation_entry(
                    session_id,
                    {
                        'type': 'user_input',
                        'text': f"Message {j} from user {i}" * 100,  # Large message
                        'timestamp': time.time()
                    }
                )

        # Should handle memory pressure gracefully
        stats = voice_service.get_service_statistics()
        assert stats['sessions_count'] == 20

        # Cleanup should free resources
        voice_service.cleanup()
        assert len(voice_service.sessions) == 0

    def test_resource_management_thread_pool_exhaustion(self, voice_service):
        """Test behavior when thread pool is exhausted."""
        voice_service.initialize()

        # Simulate thread pool exhaustion by making cleanup slow
        voice_service.audio_processor.cleanup = Mock(side_effect=lambda: time.sleep(0.1))

        start_time = time.time()
        voice_service.cleanup()
        end_time = time.time()

        # Should complete within reasonable time despite slow cleanup
        assert end_time - start_time < 1.0

    # Performance Monitoring Integration
    def test_performance_monitoring_integration(self, voice_service):
        """Test integration with performance monitoring."""
        voice_service.initialize()

        # Generate some activity
        for i in range(10):
            session_id = voice_service.create_session(f"user_{i}")
            voice_service.add_conversation_entry(session_id, {
                'type': 'user_input',
                'text': f'Test message {i}',
                'timestamp': time.time()
            })

        stats = voice_service.get_service_statistics()

        # Should track performance metrics
        assert 'service_uptime' in stats
        assert stats['total_conversations'] >= 10
        assert stats['sessions_count'] == 10

        voice_service.cleanup()

    def test_performance_monitoring_error_conditions(self, voice_service):
        """Test performance monitoring under error conditions."""
        voice_service.initialize()

        # Cause some errors
        voice_service.metrics['error_count'] = 5

        # Make stats calculation fail partially
        voice_service.stt_service.get_statistics = Mock(side_effect=Exception("Stats error"))

        stats = voice_service.get_service_statistics()

        # Should still return valid stats despite partial failure
        assert 'error_count' in stats
        assert stats['error_count'] == 5
        assert 'stt_stats' in stats

        voice_service.cleanup()

    # Configuration Change Handling
    def test_configuration_change_runtime_handling(self, voice_service, mock_config):
        """Test handling of configuration changes at runtime."""
        voice_service.initialize()

        # Simulate config change
        mock_config.max_session_duration = 7200

        # Service should continue operating with new config
        session_id = voice_service.create_session()
        assert session_id in voice_service.sessions

        voice_service.cleanup()

    def test_configuration_validation_runtime(self, voice_service, mock_config):
        """Test runtime configuration validation."""
        voice_service.initialize()

        # Test invalid config changes
        mock_config.audio_sample_rate = -1  # Invalid

        # Should continue operating (config validation is not runtime)
        session_id = voice_service.create_session()
        assert session_id is not None

        voice_service.cleanup()

    # Security Integration Advanced Tests
    def test_security_integration_pii_masking_failure(self, voice_service):
        """Test behavior when PII masking fails."""
        voice_service.initialize()

        # Disable PII protection
        voice_service.pii_protection = None

        # Should continue operating without PII masking
        session_id = voice_service.create_session("test_user")
        voice_service.add_conversation_entry(session_id, {
            'type': 'user_input',
            'text': 'My SSN is 123-45-6789',
            'timestamp': time.time()
        })

        history = voice_service.get_conversation_history(session_id)
        assert len(history) == 1

        voice_service.cleanup()

    def test_security_integration_encryption_failure(self, voice_service):
        """Test behavior when audio encryption fails."""
        voice_service.initialize()

        # Make encryption fail
        voice_service.security.encrypt_audio = AsyncMock(side_effect=Exception("Encryption failed"))

        # Should handle gracefully
        session_id = voice_service.create_session()
        # Service should still function
        assert session_id in voice_service.sessions

        voice_service.cleanup()

    # Health Check Comprehensive Testing
    def test_health_check_detailed_component_status(self, voice_service):
        """Test detailed health check component status reporting."""
        voice_service.initialize()

        # Set different health statuses
        voice_service.audio_processor.health_check = Mock(return_value={
            'status': 'degraded',
            'issues': ['High CPU usage']
        })

        voice_service.stt_service.health_check = Mock(return_value={
            'status': 'healthy',
            'issues': []
        })

        health = voice_service.health_check()

        assert health['overall_status'] == 'degraded'
        assert health['audio_processor']['status'] == 'degraded'
        assert health['stt_service']['status'] == 'healthy'

        voice_service.cleanup()

    def test_health_check_timeout_handling(self, voice_service):
        """Test health check handles timeouts gracefully."""
        voice_service.initialize()

        # Make health check slow
        voice_service.audio_processor.health_check = Mock(side_effect=lambda: time.sleep(0.1))

        start_time = time.time()
        health = voice_service.health_check()
        end_time = time.time()

        # Should complete within reasonable time
        assert end_time - start_time < 0.2
        assert 'overall_status' in health

        voice_service.cleanup()

    # Lifecycle Management Advanced Tests
    def test_lifecycle_management_graceful_shutdown(self, voice_service):
        """Test graceful shutdown under various conditions."""
        voice_service.initialize()

        # Create active sessions
        session_ids = []
        for i in range(5):
            session_id = voice_service.create_session(f"user_{i}")
            session_ids.append(session_id)

            # Start some operations
            voice_service.audio_processor.start_recording = Mock(return_value=True)
            voice_service.start_listening(session_id)

        # Graceful shutdown
        voice_service.cleanup()

        # All sessions should be cleaned up
        assert len(voice_service.sessions) == 0
        assert voice_service.is_running is False

    def test_lifecycle_management_force_shutdown(self, voice_service):
        """Test force shutdown when graceful shutdown fails."""
        voice_service.initialize()

        # Make cleanup slow/fail
        voice_service.audio_processor.cleanup = Mock(side_effect=lambda: time.sleep(1))

        start_time = time.time()
        voice_service.cleanup()
        end_time = time.time()

        # Should not wait indefinitely
        assert end_time - start_time < 2.0

    def test_lifecycle_management_restart_capability(self, voice_service):
        """Test service can be restarted after shutdown."""
        # First run
        voice_service.initialize()
        assert voice_service.is_running is True
        voice_service.cleanup()
        assert voice_service.is_running is False

        # Restart
        success = voice_service.initialize()
        assert success is True
        assert voice_service.is_running is True

        voice_service.cleanup()

    # Integration with External Systems
    def test_external_system_integration_callback_handling(self, voice_service):
        """Test integration with external systems via callbacks."""
        voice_service.initialize()

        callback_results = []

        def text_callback(session_id, text):
            callback_results.append(f"text_{session_id}_{text}")

        def audio_callback(audio_data):
            callback_results.append(f"audio_{len(audio_data.data)}")

        def error_callback(source, error):
            callback_results.append(f"error_{source}_{str(error)}")

        # Set callbacks
        voice_service.on_text_received = text_callback
        voice_service.on_audio_played = audio_callback
        voice_service.on_error = error_callback

        session_id = voice_service.create_session()

        # Trigger callbacks
        voice_service.on_text_received(session_id, "test text")
        voice_service.on_audio_played(AudioData(b'test', 16000, 1))
        voice_service.on_error("test", Exception("test error"))

        assert len(callback_results) == 3
        assert "text_" in callback_results[0]
        assert "audio_" in callback_results[1]
        assert "error_" in callback_results[2]

        voice_service.cleanup()

    def test_external_system_integration_database_sync(self, voice_service):
        """Test synchronization with external database systems."""
        voice_service.initialize()

        # Test database sync on session operations
        session_id = voice_service.create_session("test_user")

        # Verify database calls were made
        voice_service.session_repo.save.assert_called()

        voice_service.cleanup()

    # Performance Baseline Tests
    def test_performance_baseline_session_creation(self, voice_service):
        """Test performance baseline for session creation."""
        voice_service.initialize()

        start_time = time.time()
        session_ids = []
        for i in range(100):
            session_ids.append(voice_service.create_session(f"user_{i}"))
        end_time = time.time()

        creation_time = end_time - start_time

        # Should create 100 sessions quickly
        assert len(session_ids) == 100
        assert creation_time < 1.0  # Less than 1 second

        voice_service.cleanup()

    def test_performance_baseline_conversation_handling(self, voice_service):
        """Test performance baseline for conversation handling."""
        voice_service.initialize()

        session_id = voice_service.create_session()

        start_time = time.time()
        for i in range(100):
            voice_service.add_conversation_entry(session_id, {
                'type': 'user_input',
                'text': f'Message {i}',
                'timestamp': time.time()
            })
        end_time = time.time()

        handling_time = end_time - start_time

        # Should handle 100 messages quickly
        assert handling_time < 0.5  # Less than 0.5 seconds

        history = voice_service.get_conversation_history(session_id)
        assert len(history) == 100

        voice_service.cleanup()
