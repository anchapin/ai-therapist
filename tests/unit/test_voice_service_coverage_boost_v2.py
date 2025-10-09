"""
Simplified targeted unit tests for voice_service.py to boost coverage.
Focuses on core business logic functions that are currently uncovered.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import threading
import time
import numpy as np

# Import with robust error handling
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from voice.voice_service import VoiceService, VoiceSession, VoiceSessionState
    from voice.config import VoiceConfig
    from voice.security import VoiceSecurity
    from voice.audio_processor import AudioData
except ImportError as e:
    pytest.skip(f"voice_service module not available: {e}", allow_module_level=True)


class TestVoiceServiceCoreCoverage:
    """Targeted unit tests to boost voice_service.py coverage."""
    
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
        config.voice_command_timeout = 30000
        config.voice_wake_word = "hey therapist"
        return config
    
    @pytest.fixture
    def mock_security(self):
        """Create a mock security instance."""
        security = Mock(spec=VoiceSecurity)
        security.encrypt_audio_data.return_value = b"encrypted_audio"
        security.decrypt_audio_data.return_value = b"decrypted_audio"
        return security
    
    @pytest.fixture
    def voice_service(self, mock_config, mock_security):
        """Create a VoiceService with mocked dependencies."""
        with patch('voice.voice_service.SimplifiedAudioProcessor'), \
             patch('voice.voice_service.STTService'), \
             patch('voice.voice_service.TTSService'):
            
            service = VoiceService(mock_config, mock_security)
            return service
    
    def test_get_session_existing(self, voice_service):
        """Test getting an existing session."""
        # Create a session first
        session_id = voice_service.create_session("user123")
        
        # Get the session
        retrieved_session = voice_service.get_session(session_id)
        
        assert retrieved_session is not None
        assert retrieved_session.session_id == session_id
        assert retrieved_session.state == VoiceSessionState.IDLE
    
    def test_get_session_nonexistent(self, voice_service):
        """Test getting a non-existent session."""
        result = voice_service.get_session("nonexistent_session")
        assert result is None
    
    def test_get_current_session_with_active_session(self, voice_service):
        """Test getting current session when one is active."""
        # Create a session
        session_id = voice_service.create_session("user123")
        
        # Get current session
        current_session = voice_service.get_current_session()
        
        assert current_session is not None
        assert current_session.session_id == session_id
    
    def test_get_current_session_no_active_session(self, voice_service):
        """Test getting current session when none is active."""
        current_session = voice_service.get_current_session()
        assert current_session is None
    
    def test_destroy_session_existing(self, voice_service):
        """Test destroying an existing session."""
        # Create a session
        session_id = voice_service.create_session("user123")
        assert session_id in voice_service.sessions
        
        # Destroy the session
        voice_service.destroy_session(session_id)
        
        assert session_id not in voice_service.sessions
    
    def test_destroy_session_nonexistent(self, voice_service):
        """Test destroying a non-existent session."""
        # Should not raise an error
        voice_service.destroy_session("nonexistent_session")
    
    def test_get_service_statistics_with_sessions(self, voice_service):
        """Test getting service statistics with sessions."""
        # Create some sessions
        voice_service.create_session("user1")
        voice_service.create_session("user2")
        
        # Get statistics
        stats = voice_service.get_service_statistics()
        
        assert isinstance(stats, dict)
        assert 'sessions_count' in stats
        assert 'active_sessions' in stats
        assert stats['sessions_count'] == 2
        assert 'service_uptime' in stats
        assert 'error_count' in stats
    
    def test_get_service_statistics_empty(self, voice_service):
        """Test getting service statistics when no sessions exist."""
        stats = voice_service.get_service_statistics()
        
        assert isinstance(stats, dict)
        assert 'sessions_count' in stats
        assert 'active_sessions' in stats
        assert stats['sessions_count'] == 0
        assert 'service_uptime' in stats
        assert 'error_count' in stats
    
    def test_session_state_transitions(self, voice_service):
        """Test session state transitions."""
        session_id = voice_service.create_session("user123")
        session = voice_service.get_session(session_id)
        
        # Initial state should be IDLE
        assert session.state == VoiceSessionState.IDLE
        
        # Test state change
        session.state = VoiceSessionState.LISTENING
        assert session.state == VoiceSessionState.LISTENING
        
        session.state = VoiceSessionState.SPEAKING
        assert session.state == VoiceSessionState.SPEAKING
        
        session.state = VoiceSessionState.PROCESSING
        assert session.state == VoiceSessionState.PROCESSING
    
    def test_session_metadata_management(self, voice_service):
        """Test session metadata management."""
        session_id = voice_service.create_session("user123")
        session = voice_service.get_session(session_id)
        
        # Add metadata
        session.metadata['test_key'] = 'test_value'
        session.metadata['user_preference'] = 'dark_mode'
        
        # Verify metadata
        assert session.metadata['test_key'] == 'test_value'
        assert session.metadata['user_preference'] == 'dark_mode'
        assert 'created_at' in session.metadata
    
    def test_conversation_history_management(self, voice_service):
        """Test conversation history in sessions."""
        session_id = voice_service.create_session("user123")
        session = voice_service.get_session(session_id)
        
        # Add conversation entries
        session.conversation_history.append({
            'role': 'user',
            'content': 'Hello',
            'timestamp': '2024-01-01T12:00:00'
        })
        session.conversation_history.append({
            'role': 'assistant', 
            'content': 'Hi there!',
            'timestamp': '2024-01-01T12:00:01'
        })
        
        # Verify history
        assert len(session.conversation_history) == 2
        assert session.conversation_history[0]['role'] == 'user'
        assert session.conversation_history[1]['role'] == 'assistant'
    
    def test_audio_buffer_management(self, voice_service):
        """Test audio buffer in sessions."""
        session_id = voice_service.create_session("user123")
        session = voice_service.get_session(session_id)
        
        # Create mock audio data
        audio_data = AudioData(
            data=np.array([1, 2, 3], dtype=np.float32),
            sample_rate=16000,
            duration=0.1,
            channels=1,
            format='wav'
        )
        
        # Add to buffer
        session.audio_buffer.append(audio_data)
        
        # Verify buffer
        assert len(session.audio_buffer) == 1
        assert session.audio_buffer[0] == audio_data
    
    def test_session_thread_safety(self, voice_service):
        """Test thread safety of session operations."""
        session_id = voice_service.create_session("user123")
        
        def access_session():
            for i in range(5):
                session = voice_service.get_session(session_id)
                assert session is not None
                assert session.session_id == session_id
                time.sleep(0.001)
        
        # Create multiple threads accessing the same session
        threads = [threading.Thread(target=access_session) for _ in range(3)]
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Session should still exist and be valid
        session = voice_service.get_session(session_id)
        assert session is not None
        assert session.session_id == session_id
    
    def test_end_session_basic(self, voice_service):
        """Test ending a session."""
        # Create a session
        session_id = voice_service.create_session("user123")
        assert session_id in voice_service.sessions
        
        # End the session
        result = voice_service.end_session(session_id)
        assert result is True
    
    def test_end_session_nonexistent(self, voice_service):
        """Test ending a non-existent session."""
        result = voice_service.end_session("nonexistent_session")
        # end_session returns True even for non-existent sessions (by design)
        # The session just doesn't exist after the call
        assert result is True
        # Verify session doesn't exist
        assert voice_service.get_session("nonexistent_session") is None
    
    def test_health_check_basic(self, voice_service):
        """Test basic health check."""
        health = voice_service.health_check()
        
        assert isinstance(health, dict)
        assert 'overall_status' in health
        # Health should have basic structure
        assert health['overall_status'] in ['healthy', 'degraded', 'unhealthy']