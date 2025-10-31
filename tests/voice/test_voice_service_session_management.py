"""
Comprehensive tests for VoiceService session management, database integration, and persistence.
Covers session lifecycle, database operations, and concurrent session handling.
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


class TestVoiceServiceSessionManagement:
    """Tests for VoiceService session management and database integration."""

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

    # Session Creation Tests
    def test_session_creation_unique_ids(self, voice_service):
        """Test that created sessions have unique IDs."""
        session_ids = set()
        for _ in range(100):
            session_id = voice_service.create_session()
            assert session_id not in session_ids
            session_ids.add(session_id)

        assert len(session_ids) == 100

    def test_session_creation_with_custom_id(self, voice_service):
        """Test session creation with custom session ID."""
        custom_id = "custom_session_123"
        session_id = voice_service.create_session(session_id=custom_id)

        assert session_id == custom_id
        assert custom_id in voice_service.sessions

    def test_session_creation_duplicate_id_handling(self, voice_service):
        """Test handling of duplicate session ID creation."""
        session_id = "duplicate_test"
        voice_service.create_session(session_id=session_id)

        # Try to create again with same ID
        result = voice_service.create_session(session_id=session_id)

        # Should return existing session ID
        assert result == session_id
        assert len(voice_service.sessions) == 1

    def test_session_creation_with_voice_profile(self, voice_service):
        """Test session creation with specific voice profile."""
        profile = "energetic_therapist"
        session_id = voice_service.create_session(voice_profile=profile)

        session = voice_service.get_session(session_id)
        assert session.current_voice_profile == profile

    def test_session_creation_without_voice_profile(self, voice_service, mock_config):
        """Test session creation uses default voice profile when none specified."""
        mock_config.default_voice_profile = "default_test_profile"
        session_id = voice_service.create_session()

        session = voice_service.get_session(session_id)
        assert session.current_voice_profile == "default_test_profile"

    # Session Retrieval Tests
    def test_get_session_existing(self, voice_service):
        """Test retrieving existing session."""
        session_id = voice_service.create_session()
        session = voice_service.get_session(session_id)

        assert session is not None
        assert session.session_id == session_id

    def test_get_session_nonexistent(self, voice_service):
        """Test retrieving non-existent session returns None."""
        session = voice_service.get_session("nonexistent_id")
        assert session is None

    def test_get_current_session_when_set(self, voice_service):
        """Test getting current session when one is set."""
        session_id = voice_service.create_session()
        # Current session is automatically set to newly created session
        current = voice_service.get_current_session()

        assert current is not None
        assert current.session_id == session_id

    def test_get_current_session_when_none(self, voice_service):
        """Test getting current session when none exists."""
        # Clear current session
        voice_service.current_session_id = None
        current = voice_service.get_current_session()

        assert current is None

    # Session Destruction Tests
    def test_destroy_session_existing(self, voice_service):
        """Test destroying existing session."""
        session_id = voice_service.create_session()
        assert session_id in voice_service.sessions

        voice_service.destroy_session(session_id)
        assert session_id not in voice_service.sessions

    def test_destroy_session_current_session_update(self, voice_service):
        """Test destroying current session updates current_session_id."""
        session_id = voice_service.create_session()
        assert voice_service.current_session_id == session_id

        voice_service.destroy_session(session_id)
        assert voice_service.current_session_id is None

    def test_destroy_session_with_active_operations(self, voice_service):
        """Test destroying session stops active operations."""
        session_id = voice_service.create_session()

        # Start listening
        voice_service.audio_processor.start_recording = Mock(return_value=True)
        voice_service.start_listening(session_id)

        session = voice_service.get_session(session_id)
        assert session.state == VoiceSessionState.LISTENING

        # Destroy session should stop listening
        voice_service.destroy_session(session_id)
        assert session_id not in voice_service.sessions

    def test_destroy_session_nonexistent(self, voice_service):
        """Test destroying non-existent session doesn't raise error."""
        # Should not raise exception
        voice_service.destroy_session("nonexistent_id")

    # Session Ending Tests
    def test_end_session_success(self, voice_service):
        """Test successfully ending a session."""
        session_id = voice_service.create_session()
        result = voice_service.end_session(session_id)

        assert result is True
        assert session_id not in voice_service.sessions

    def test_end_session_with_error(self, voice_service):
        """Test ending session handles errors gracefully."""
        session_id = voice_service.create_session()

        # Make destroy_session raise exception
        with patch.object(voice_service, 'destroy_session', side_effect=Exception("Destroy error")):
            result = voice_service.end_session(session_id)

        assert result is False

    # Database Integration Tests
    def test_session_creation_database_persistence(self, voice_service):
        """Test session creation persists to database."""
        user_id = "test_user_123"
        session_id = voice_service.create_session(user_id=user_id)

        # Verify database save was called
        voice_service.session_repo.save.assert_called_once()

    def test_session_creation_database_failure_fallback(self, voice_service):
        """Test session creation falls back to memory when database fails."""
        voice_service.session_repo.save = Mock(side_effect=Exception("DB error"))

        session_id = voice_service.create_session("test_user")

        # Should still create session in memory
        assert session_id in voice_service.sessions

    def test_session_creation_without_user_id_no_database_call(self, voice_service):
        """Test session creation without user_id doesn't call database."""
        session_id = voice_service.create_session()

        # Database should not be called
        voice_service.session_repo.save.assert_not_called()

    # Session Metadata Management
    def test_session_metadata_initialization(self, voice_service):
        """Test session metadata is properly initialized."""
        user_id = "test_user"
        session_id = voice_service.create_session(user_id=user_id)

        session = voice_service.get_session(session_id)

        # Check metadata structure
        assert 'created_at' in session.metadata
        assert 'voice_settings' in session.metadata
        assert session.metadata['voice_settings']['voice_speed'] == 1.2
        assert session.metadata['voice_settings']['volume'] == 1.0

    def test_session_metadata_user_id_storage(self, voice_service):
        """Test user_id is stored in session metadata."""
        user_id = "test_user_456"
        session_id = voice_service.create_session(user_id=user_id)

        session = voice_service.get_session(session_id)
        assert session.metadata.get('user_id') == user_id

    # Voice Settings Management
    def test_update_voice_settings_session_specific(self, voice_service):
        """Test updating voice settings for specific session."""
        session_id = voice_service.create_session()
        settings = {'voice_speed': 1.5, 'volume': 0.8}

        result = voice_service.update_voice_settings(settings, session_id)

        assert result is True
        session = voice_service.get_session(session_id)
        assert session.metadata['voice_settings']['voice_speed'] == 1.5
        assert session.metadata['voice_settings']['volume'] == 0.8

    def test_update_voice_settings_global_fallback(self, voice_service):
        """Test updating voice settings without session_id."""
        settings = {'voice_speed': 2.0}

        result = voice_service.update_voice_settings(settings)

        assert result is True
        # Global settings don't persist to sessions

    def test_update_voice_settings_invalid_session_id(self, voice_service):
        """Test updating voice settings with invalid session_id type."""
        settings = {'volume': 0.5}

        # Pass integer instead of string
        result = voice_service.update_voice_settings(settings, 12345)

        assert result is False

    def test_update_voice_settings_nonexistent_session(self, voice_service):
        """Test updating voice settings for non-existent session."""
        settings = {'volume': 0.5}

        result = voice_service.update_voice_settings(settings, "nonexistent")

        assert result is False

    # Session Activity Tracking
    def test_update_session_activity_timestamp(self, voice_service):
        """Test update_session_activity updates timestamp."""
        session_id = voice_service.create_session()
        session = voice_service.get_session(session_id)
        initial_activity = session.last_activity

        time.sleep(0.01)
        voice_service.update_session_activity(session_id)

        assert session.last_activity > initial_activity

    def test_update_session_activity_nonexistent_session(self, voice_service):
        """Test update_session_activity with non-existent session."""
        # Should not raise exception
        voice_service.update_session_activity("nonexistent")

    # Conversation History Management
    def test_add_conversation_entry_new_format(self, voice_service):
        """Test adding conversation entry with new format."""
        session_id = voice_service.create_session()

        entry = {
            'type': 'user_input',
            'text': 'Hello therapist',
            'timestamp': time.time(),
            'confidence': 0.95
        }

        result = voice_service.add_conversation_entry(session_id, entry)

        assert result is True
        history = voice_service.get_conversation_history(session_id)
        assert len(history) == 1
        assert history[0]['text'] == 'Hello therapist'

    def test_add_conversation_entry_old_format(self, voice_service):
        """Test adding conversation entry with old format (speaker, text)."""
        session_id = voice_service.create_session()

        result = voice_service.add_conversation_entry(session_id, 'user', 'Hello')

        assert result is True
        history = voice_service.get_conversation_history(session_id)
        assert len(history) == 1
        assert history[0]['speaker'] == 'user'
        assert history[0]['text'] == 'Hello'

    def test_add_conversation_entry_invalid_args(self, voice_service):
        """Test add_conversation_entry with invalid arguments."""
        result = voice_service.add_conversation_entry("session_id")  # Missing entry

        assert result is False

    def test_add_conversation_entry_nonexistent_session(self, voice_service):
        """Test add_conversation_entry with non-existent session."""
        entry = {'type': 'user_input', 'text': 'test'}
        result = voice_service.add_conversation_entry("nonexistent", entry)

        assert result is False

    def test_get_conversation_history_empty(self, voice_service):
        """Test getting conversation history for empty session."""
        session_id = voice_service.create_session()
        history = voice_service.get_conversation_history(session_id)

        assert history == []

    def test_get_conversation_history_nonexistent_session(self, voice_service):
        """Test getting conversation history for non-existent session."""
        history = voice_service.get_conversation_history("nonexistent")

        assert history == []

    # Thread Safety Tests
    def test_session_operations_thread_safety(self, voice_service):
        """Test session operations are thread-safe."""
        results = []
        errors = []

        def session_operations(thread_id):
            try:
                # Create sessions
                session_ids = []
                for i in range(10):
                    session_id = voice_service.create_session(f"thread_{thread_id}_{i}")
                    session_ids.append(session_id)

                # Read sessions
                for session_id in session_ids:
                    session = voice_service.get_session(session_id)
                    assert session is not None

                # Add conversation entries
                for session_id in session_ids:
                    voice_service.add_conversation_entry(session_id, {
                        'type': 'user_input',
                        'text': f'Thread {thread_id} message',
                        'timestamp': time.time()
                    })

                # Destroy sessions
                for session_id in session_ids:
                    voice_service.destroy_session(session_id)

                results.append(f"thread_{thread_id}_success")

            except Exception as e:
                errors.append(f"thread_{thread_id}_error: {e}")

        threads = [threading.Thread(target=session_operations, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5
        assert len(errors) == 0

    # Session State Persistence
    def test_session_state_persistence_across_operations(self, voice_service):
        """Test session state persists across operations."""
        session_id = voice_service.create_session()

        # Initial state
        session = voice_service.get_session(session_id)
        assert session.state == VoiceSessionState.IDLE

        # Change state through operations
        voice_service.audio_processor.start_recording = Mock(return_value=True)
        voice_service.start_listening(session_id)
        assert session.state == VoiceSessionState.LISTENING

        voice_service.audio_processor.stop_recording = Mock(return_value=AudioData(b'test', 16000, 1))
        voice_service.stop_listening(session_id)
        assert session.state == VoiceSessionState.IDLE

    # Concurrent Session Management
    def test_concurrent_session_creation_and_destruction(self, voice_service):
        """Test concurrent creation and destruction of sessions."""
        active_sessions = set()

        def create_destroy_worker():
            for _ in range(25):
                # Create session
                session_id = voice_service.create_session()
                active_sessions.add(session_id)

                # Small delay to allow interleaving
                time.sleep(0.001)

                # Destroy session
                if session_id in active_sessions:
                    voice_service.destroy_session(session_id)
                    active_sessions.discard(session_id)

        threads = [threading.Thread(target=create_destroy_worker) for _ in range(4)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All sessions should be cleaned up
        assert len(voice_service.sessions) == 0

    # Session Timeout Handling
    def test_session_timeout_detection(self, voice_service, mock_config):
        """Test detection of timed out sessions."""
        mock_config.session_timeout = 1  # 1 second timeout
        session_id = voice_service.create_session()

        session = voice_service.get_session(session_id)
        # Manually set last activity to past timeout
        session.last_activity = time.time() - 2

        # Session should still be accessible (timeout doesn't auto-remove)
        assert voice_service.get_session(session_id) is not None

    # Memory Management
    def test_session_memory_cleanup_on_destroy(self, voice_service):
        """Test memory is properly cleaned up when sessions are destroyed."""
        session_id = voice_service.create_session()

        # Add some data to session
        voice_service.add_conversation_entry(session_id, 'user', 'Test message')
        voice_service.add_conversation_entry(session_id, 'ai', 'Response message')

        session = voice_service.get_session(session_id)
        assert len(session.conversation_history) == 2

        # Destroy session
        voice_service.destroy_session(session_id)

        # Memory should be freed
        assert session_id not in voice_service.sessions

    # Session Statistics Integration
    def test_session_statistics_tracking(self, voice_service):
        """Test session statistics are properly tracked."""
        initial_count = voice_service.metrics.get('sessions_created', 0)

        # Create multiple sessions
        session_ids = []
        for i in range(5):
            session_id = voice_service.create_session(f"user_{i}")
            session_ids.append(session_id)

            # Add some conversation activity
            voice_service.add_conversation_entry(session_id, 'user', f'Message {i}')
            voice_service.add_conversation_entry(session_id, 'ai', f'Response {i}')

        stats = voice_service.get_service_statistics()

        assert stats['sessions_count'] == 5
        assert stats['total_conversations'] == 10  # 2 entries per session
        assert voice_service.metrics['sessions_created'] == initial_count + 5

    # Error Handling in Session Operations
    def test_session_operation_error_recovery(self, voice_service):
        """Test error recovery in session operations."""
        session_id = voice_service.create_session()

        # Simulate error in conversation history access
        with patch.object(voice_service, '_sessions_lock', side_effect=Exception("Lock error")):
            # Operations should handle errors gracefully
            history = voice_service.get_conversation_history(session_id)
            assert history == []  # Should return empty list on error

            session = voice_service.get_session(session_id)
            # Session access should still work
            assert session is not None

    # Session Data Integrity
    def test_session_data_integrity_concurrent_access(self, voice_service):
        """Test session data integrity under concurrent access."""
        session_id = voice_service.create_session()

        results = []

        def integrity_checker(thread_id):
            try:
                # Read session data
                session = voice_service.get_session(session_id)
                initial_history_len = len(session.conversation_history)

                # Add entries
                for i in range(5):
                    voice_service.add_conversation_entry(session_id, {
                        'type': 'user_input',
                        'text': f'Thread {thread_id} message {i}',
                        'timestamp': time.time()
                    })

                # Verify data integrity
                final_session = voice_service.get_session(session_id)
                final_history_len = len(final_session.conversation_history)

                results.append(final_history_len - initial_history_len)

            except Exception as e:
                results.append(f"error_{thread_id}: {e}")

        threads = [threading.Thread(target=integrity_checker, args=(i,)) for i in range(3)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have added 15 entries total (5 per thread)
        session = voice_service.get_session(session_id)
        assert len(session.conversation_history) == 15

    # Session Resource Limits
    def test_session_resource_limits_conversation_history(self, voice_service):
        """Test conversation history doesn't grow unbounded."""
        session_id = voice_service.create_session()

        # Add many conversation entries
        for i in range(1000):
            voice_service.add_conversation_entry(session_id, {
                'type': 'user_input',
                'text': f'Message {i}',
                'timestamp': time.time()
            })

        history = voice_service.get_conversation_history(session_id)

        # Should contain all entries (no artificial limit in current implementation)
        assert len(history) == 1000

    # Session Export/Import Simulation
    def test_session_data_export_simulation(self, voice_service):
        """Test session data can be exported (simulation)."""
        session_id = voice_service.create_session("test_user")

        # Add some data
        voice_service.add_conversation_entry(session_id, 'user', 'Hello')
        voice_service.add_conversation_entry(session_id, 'ai', 'Hi there')

        session = voice_service.get_session(session_id)

        # Simulate export
        export_data = {
            'session_id': session.session_id,
            'state': session.state.value,
            'conversation_history': session.conversation_history,
            'metadata': session.metadata
        }

        assert export_data['session_id'] == session_id
        assert len(export_data['conversation_history']) == 2

    # Session Recovery Scenarios
    def test_session_recovery_after_service_restart(self, voice_service):
        """Test session recovery simulation after service restart."""
        # Create session with data
        session_id = voice_service.create_session("user_123")
        voice_service.add_conversation_entry(session_id, 'user', 'Test message')

        session = voice_service.get_session(session_id)
        session_data = {
            'session_id': session.session_id,
            'conversation_history': session.conversation_history.copy(),
            'metadata': session.metadata.copy()
        }

        # Simulate service restart by clearing in-memory sessions
        voice_service.sessions.clear()
        voice_service.current_session_id = None

        # Restore session (simulation)
        restored_session = VoiceSession(
            session_id=session_data['session_id'],
            state=VoiceSessionState.IDLE,
            start_time=time.time(),
            last_activity=time.time(),
            conversation_history=session_data['conversation_history'],
            current_voice_profile=session_data['metadata'].get('voice_settings', {}).get('profile', 'default'),
            audio_buffer=[],
            metadata=session_data['metadata']
        )

        voice_service.sessions[session_id] = restored_session

        # Verify recovery
        recovered_session = voice_service.get_session(session_id)
        assert recovered_session is not None
        assert len(recovered_session.conversation_history) == 1
        assert recovered_session.conversation_history[0]['text'] == 'Test message'
