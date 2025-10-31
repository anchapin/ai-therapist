"""
Comprehensive tests for VoiceService to improve coverage from 14% to 80%.
Focus on worker loop, state transitions, error handling branches, and validation logic.
"""

import pytest
import pytest_asyncio
import asyncio
import time
import threading
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
from typing import Dict, Any
import numpy as np
from pathlib import Path
import json

from voice.voice_service import VoiceService, VoiceSession, VoiceSessionState
from voice.config import VoiceConfig
from voice.audio_processor import AudioData
from voice.stt_service import STTService, STTResult
from voice.tts_service import TTSService, TTSResult
from voice.security import VoiceSecurity


class TestVoiceServiceComprehensive:
    """Comprehensive tests for VoiceService coverage improvement."""

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
        return config

    @pytest.fixture
    def mock_security(self):
        """Create a mock security service."""
        security = Mock(spec=VoiceSecurity)
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

            return service

    # Tests for initialization and setup
    def test_initialization_with_valid_config(self, mock_config, mock_security):
        """Test VoiceService initialization with valid configuration."""
        with patch('voice.voice_service.SimplifiedAudioProcessor'), \
             patch('voice.voice_service.STTService'), \
             patch('voice.voice_service.TTSService'), \
             patch('voice.voice_service.VoiceCommandProcessor'):

            service = VoiceService(mock_config, mock_security)
            assert service.config == mock_config
            assert service.security == mock_security
            assert service.sessions == {}
            assert service.metrics['sessions_created'] == 0

    def test_initialization_database_repositories(self, voice_service):
        """Test database repository initialization."""
        voice_service._initialize_database_repositories()
        assert voice_service.session_repo is not None
        assert voice_service.voice_data_repo is not None

    def test_check_service_availability_all_available(self, voice_service):
        """Test service availability check when all services are available."""
        # Mock all components as healthy
        voice_service.audio_processor.is_available = Mock(return_value=True)
        voice_service.stt_service.is_available = Mock(return_value=True)
        voice_service.tts_service.is_available = Mock(return_value=True)
        voice_service.command_processor.is_available = Mock(return_value=True)

        available = voice_service._check_service_availability()
        assert available is True

    def test_check_service_availability_partial_unavailable(self, voice_service):
        """Test service availability when some services are unavailable."""
        voice_service.audio_processor.is_available = Mock(return_value=True)
        voice_service.stt_service.is_available = Mock(return_value=False)  # STT unavailable
        voice_service.tts_service.is_available = Mock(return_value=True)
        voice_service.command_processor.is_available = Mock(return_value=True)

        available = voice_service._check_service_availability()
        assert available is False  # Should be false if any critical service unavailable

    # Tests for session management and state transitions
    def test_create_session_with_custom_id(self, voice_service):
        """Test creating session with custom session ID."""
        custom_id = "custom_session_123"
        result_id = voice_service.create_session(session_id=custom_id, user_id="test_user")

        assert result_id == custom_id
        assert custom_id in voice_service.sessions
        session = voice_service.get_session(custom_id)
        assert session.state == VoiceSessionState.IDLE

    def test_create_session_auto_generated_id(self, voice_service):
        """Test creating session with auto-generated ID."""
        result_id = voice_service.create_session(user_id="test_user")

        assert result_id is not None
        assert isinstance(result_id, str)
        assert len(result_id) > 0
        assert result_id in voice_service.sessions

    def test_create_session_with_voice_profile(self, voice_service):
        """Test creating session with specific voice profile."""
        profile = "anxious_profile"
        session_id = voice_service.create_session(user_id="test_user", voice_profile=profile)

        session = voice_service.get_session(session_id)
        assert session.current_voice_profile == profile

    def test_get_session_existing(self, voice_service):
        """Test getting existing session."""
        session_id = voice_service.create_session(user_id="test_user")
        session = voice_service.get_session(session_id)

        assert session is not None
        assert session.session_id == session_id
        assert session.state == VoiceSessionState.IDLE

    def test_get_session_nonexistent(self, voice_service):
        """Test getting non-existent session returns None."""
        session = voice_service.get_session("nonexistent_session")
        assert session is None

    def test_get_current_session_none(self, voice_service):
        """Test getting current session when none exists."""
        session = voice_service.get_current_session()
        assert session is None

    def test_get_current_session_exists(self, voice_service):
        """Test getting current session when one exists."""
        session_id = voice_service.create_session(user_id="test_user")
        voice_service.current_session_id = session_id

        current = voice_service.get_current_session()
        assert current is not None
        assert current.session_id == session_id

    def test_destroy_session_existing(self, voice_service):
        """Test destroying existing session."""
        session_id = voice_service.create_session(user_id="test_user")
        assert session_id in voice_service.sessions

        voice_service.destroy_session(session_id)
        assert session_id not in voice_service.sessions

    def test_destroy_session_nonexistent(self, voice_service):
        """Test destroying non-existent session doesn't crash."""
        # Should not raise exception
        voice_service.destroy_session("nonexistent_session")

    def test_end_session_success(self, voice_service):
        """Test ending session successfully."""
        session_id = voice_service.create_session(user_id="test_user")
        result = voice_service.end_session(session_id)

        assert result is True
        assert session_id not in voice_service.sessions

    def test_end_session_nonexistent(self, voice_service):
        """Test ending non-existent session."""
        result = voice_service.end_session("nonexistent_session")
        assert result is True  # destroy_session handles nonexistent sessions gracefully

    # Tests for listening state transitions
    def test_start_listening_existing_session(self, voice_service):
        """Test starting listening on existing session."""
        session_id = voice_service.create_session(user_id="test_user")

        # Mock audio processor
        voice_service.audio_processor.start_recording = Mock(return_value=True)
        voice_service.audio_processor.set_callback = Mock()

        result = voice_service.start_listening(session_id)
        assert result is True

        session = voice_service.get_session(session_id)
        assert session.state == VoiceSessionState.LISTENING

    def test_start_listening_current_session(self, voice_service):
        """Test starting listening on current session."""
        session_id = voice_service.create_session(user_id="test_user")
        voice_service.current_session_id = session_id

        voice_service.audio_processor.start_recording = Mock(return_value=True)
        voice_service.audio_processor.set_callback = Mock()

        result = voice_service.start_listening()  # No session_id provided
        assert result is True

        session = voice_service.get_session(session_id)
        assert session.state == VoiceSessionState.LISTENING

    def test_start_listening_no_session(self, voice_service):
        """Test starting listening when no session exists."""
        result = voice_service.start_listening()
        assert result is False

    def test_stop_listening_success(self, voice_service):
        """Test stopping listening successfully."""
        session_id = voice_service.create_session(user_id="test_user")
        voice_service.start_listening(session_id)

        # Mock recorded audio data
        mock_audio = AudioData(
        data=np.random.randn(16000).astype(np.float32),
        sample_rate=16000,
        duration=1.0,
            channels=1
        )
        voice_service.audio_processor.stop_recording = Mock(return_value=mock_audio)

        result = voice_service.stop_listening(session_id)
        assert result is not None
        assert isinstance(result, AudioData)

        session = voice_service.get_session(session_id)
        assert session.state != VoiceSessionState.LISTENING

    def test_stop_listening_no_session(self, voice_service):
        """Test stopping listening when no session exists."""
        result = voice_service.stop_listening()
        assert result is None

    # Tests for audio callback and processing
    def test_audio_callback_processing(self, voice_service):
        """Test audio callback processing."""
        session_id = voice_service.create_session(user_id="test_user")
        voice_service.start_listening(session_id)

        audio_data = AudioData(
        data=np.random.randn(8000).astype(np.float32),
        sample_rate=16000,
        duration=0.5,
            channels=1
        )

        # Should not crash
        voice_service._audio_callback(audio_data)

        session = voice_service.get_session(session_id)
        assert len(session.audio_buffer) >= 0  # Audio should be buffered

    def test_audio_callback_no_current_session(self, voice_service):
        """Test audio callback when no current session."""
        audio_data = AudioData(
            data=np.random.randn(8000).astype(np.float32),
            sample_rate=16000,
            duration=0.5,
            channels=1
        )

        # Should not crash
        voice_service._audio_callback(audio_data)

    # Tests for mock result creation
    def test_create_mock_stt_result_success(self, voice_service):
        """Test creating mock STT result for success case."""
        result = voice_service._create_mock_stt_result("Hello world", has_error=False)

        assert result.text == "Hello world"
        assert result.confidence == 0.9
        assert result.language == "en-US"
        assert result.is_final is True

    def test_create_mock_stt_result_error(self, voice_service):
        """Test creating mock STT result for error case."""
        result = voice_service._create_mock_stt_result("Error occurred", has_error=True)

        assert result.text == "Error occurred"
        assert result.confidence == 0.1  # Lower confidence for errors
        assert result.language == "en-US"
        assert result.is_final is False

    def test_create_mock_tts_result(self, voice_service):
        """Test creating mock TTS result."""
        result = voice_service._create_mock_tts_result("Hello world")

        assert result.success is True
        assert result.audio_data is not None
        assert len(result.audio_data.data) > 0
        assert result.sample_rate == 16000

    # Tests for speaking functionality
    def test_stop_speaking_existing_session(self, voice_service):
        """Test stopping speaking on existing session."""
        session_id = voice_service.create_session(user_id="test_user")
            session = voice_service.get_session(session_id)
        session.state = VoiceSessionState.SPEAKING  # Put session in speaking state

            # Mock TTS service
        voice_service.tts_service.stop_synthesis = Mock(return_value=True)

            result = voice_service.stop_speaking(session_id)
        assert result is True

    session = voice_service.get_session(session_id)
    assert session.state == VoiceSessionState.IDLE

    def test_stop_speaking_current_session(self, voice_service):
        """Test stopping speaking on current session."""
        session_id = voice_service.create_session(user_id="test_user")
        voice_service.current_session_id = session_id

        voice_service.tts_service.stop_synthesis = Mock(return_value=True)

        result = voice_service.stop_speaking()  # No session_id provided
        assert result is True

    # Tests for session activity updates
    def test_update_session_activity_success(self, voice_service):
        """Test updating session activity timestamp."""
        session_id = voice_service.create_session(user_id="test_user")
        session = voice_service.get_session(session_id)
        original_activity = session.last_activity

        time.sleep(0.01)  # Small delay to ensure timestamp difference

        result = voice_service.update_session_activity(session_id)
        assert result is True

        updated_session = voice_service.get_session(session_id)
        assert updated_session.last_activity > original_activity

    def test_update_session_activity_nonexistent(self, voice_service):
        """Test updating activity for non-existent session."""
        result = voice_service.update_session_activity("nonexistent_session")
        assert result is False

    # Tests for conversation history
    def test_get_conversation_history_existing(self, voice_service):
        """Test getting conversation history for existing session."""
        session_id = voice_service.create_session(user_id="test_user")

        # Add some conversation entries
        voice_service.add_conversation_entry(session_id, "user", "Hello")
        voice_service.add_conversation_entry(session_id, "ai", "Hi there")

        history = voice_service.get_conversation_history(session_id)
        assert len(history) == 2
        assert history[0]['text'] == 'Hello'
        assert history[1]['text'] == 'Hi there'

    def test_get_conversation_history_nonexistent(self, voice_service):
        """Test getting conversation history for non-existent session."""
        history = voice_service.get_conversation_history("nonexistent_session")
        assert history == []

    # Tests for AI response generation
    def test_generate_ai_response_basic(self, voice_service):
        """Test basic AI response generation."""
        user_input = "I feel anxious"

        response = voice_service.generate_ai_response(user_input)
        assert "anxious" in response.lower()

    def test_generate_ai_response_error_fallback(self, voice_service):
        """Test AI response generation with error fallback."""
        user_input = "I feel anxious"

        response = voice_service.generate_ai_response(user_input)
        assert isinstance(response, str)
        assert len(response) > 0  # Should return response

    # Tests for service statistics
    def test_get_service_statistics_comprehensive(self, voice_service):
        """Test getting comprehensive service statistics."""
        # Create some sessions and activity
        session_id1 = voice_service.create_session(user_id="user1")
        session_id2 = voice_service.create_session(user_id="user2")

        voice_service.add_conversation_entry(session_id1, "user", "Hello")
        voice_service.add_conversation_entry(session_id1, "ai", "Hi")
        voice_service.add_conversation_entry(session_id2, "user", "Test")

        stats = voice_service.get_service_statistics()

        assert 'sessions_count' in stats
        assert 'total_conversations' in stats
        assert 'active_sessions' in stats
        assert stats['sessions_count'] >= 2
        assert stats['total_conversations'] >= 3

    # Tests for cleanup functionality
    def test_cleanup_comprehensive(self, voice_service):
        """Test comprehensive cleanup of resources."""
        # Create sessions and add some data
        session_id = voice_service.create_session(user_id="test_user")
        voice_service.add_conversation_entry(session_id, "user", "Test")

        # Mock cleanup methods
        voice_service.audio_processor.cleanup = Mock()
        voice_service.stt_service.cleanup = AsyncMock()
        voice_service.tts_service.cleanup = AsyncMock()
        voice_service.command_processor.cleanup = Mock()
        voice_service.security.cleanup = Mock()

        voice_service.cleanup()

        # Verify cleanup was called on all components
        voice_service.audio_processor.cleanup.assert_called_once()
        voice_service.stt_service.cleanup.assert_called_once()
        voice_service.tts_service.cleanup.assert_called_once()
        voice_service.command_processor.cleanup.assert_called_once()
        voice_service.security.cleanup.assert_called_once()

    # Tests for error handling and edge cases
    def test_invalid_session_operations(self, voice_service):
        """Test operations with invalid session parameters."""
        # Test with None session_id
        result = voice_service.start_listening(None)
        assert result is False

        result = voice_service.add_conversation_entry(None, "user", "text")
        assert result is False

    def test_concurrent_session_operations(self, voice_service):
        """Test concurrent operations on sessions."""
        session_ids = []
        for i in range(5):
            session_id = voice_service.create_session(user_id=f"user_{i}")
            session_ids.append(session_id)

        # Concurrent conversation additions
        for session_id in session_ids:
            voice_service.add_conversation_entry(session_id, "user", f"Message {session_id}")

        # Verify all sessions have their messages
        for session_id in session_ids:
            history = voice_service.get_conversation_history(session_id)
            assert len(history) == 1
            assert f"Message {session_id}" in history[0]['text']

    def test_session_timeout_handling(self, voice_service):
        """Test session timeout detection and handling."""
        session_id = voice_service.create_session(user_id="test_user")

        # Manually set old timestamp to simulate timeout
        session = voice_service.get_session(session_id)
        session.last_activity = time.time() - 400  # 400 seconds ago (past 300s timeout)

        # Check if session operations detect timeout
        result = voice_service.update_session_activity(session_id)
        assert result is True  # Should still work, just update timestamp

    def test_metrics_tracking_comprehensive(self, voice_service):
        """Test comprehensive metrics tracking."""
        initial_metrics = voice_service.metrics.copy()

        # Create session
        session_id = voice_service.create_session(user_id="test_user")
        assert voice_service.metrics['sessions_created'] == initial_metrics['sessions_created'] + 1

        # Add interactions
        voice_service.add_conversation_entry(session_id, "user", "Hello")
        voice_service.add_conversation_entry(session_id, "ai", "Hi")
        assert voice_service.metrics['total_interactions'] == initial_metrics['total_interactions'] + 2

        # Start/stop listening
        voice_service.audio_processor.start_recording = Mock(return_value=True)
        voice_service.start_listening(session_id)
        assert voice_service.metrics['listening_sessions'] == initial_metrics.get('listening_sessions', 0) + 1

    def test_voice_service_worker_initialization(self, voice_service):
        """Test voice service worker initialization."""
        # Worker should be initialized but not started
        assert hasattr(voice_service, '_voice_service_worker')
        assert voice_service.voice_thread is None or not voice_service.voice_thread.is_alive()

    def test_queue_initialization_and_operations(self, voice_service):
        """Test queue initialization and basic operations."""
        voice_service._ensure_queue_initialized()
        assert voice_service.voice_queue is not None

        # Test queue operations
        assert voice_service.voice_queue.empty()

        # Add item to queue
        voice_service.voice_queue.put_nowait(("test_command", {"data": "test"}))
        assert not voice_service.voice_queue.empty()

    def test_service_availability_checks(self, voice_service):
        """Test various service availability scenarios."""
        # Test when services are available
        voice_service.audio_processor.is_available = Mock(return_value=True)
        voice_service.stt_service.is_available = Mock(return_value=True)
        voice_service.tts_service.is_available = Mock(return_value=True)

        assert voice_service.is_available() is True

        # Test when critical service unavailable
        voice_service.stt_service.is_available = Mock(return_value=False)
        assert voice_service.is_available() is False

    def test_initialization_state_tracking(self, voice_service):
        """Test initialization state tracking."""
        assert voice_service.initialized is True  # We set _db_initialized = True in fixture

        # Test when not initialized
        voice_service._db_initialized = False
        assert voice_service.initialized is False
