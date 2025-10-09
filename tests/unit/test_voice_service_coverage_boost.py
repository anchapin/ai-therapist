"""
Targeted unit tests for voice_service.py to boost coverage.
Focuses on core business logic functions that are currently uncovered.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import threading
import time

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
    
    def test_get_service_statistics(self, voice_service):
        """Test getting service statistics."""
        # Create some sessions
        voice_service.create_session("user1")
        voice_service.create_session("user2")
        
        # Get statistics
        stats = voice_service.get_service_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_sessions' in stats
        assert 'active_sessions' in stats
        assert 'sessions_created' in stats
        assert stats['total_sessions'] == 2
        assert stats['active_sessions'] == 2
    
    def test_get_service_statistics_empty(self, voice_service):
        """Test getting service statistics when no sessions exist."""
        stats = voice_service.get_service_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_sessions' in stats
        assert 'active_sessions' in stats
        # These might be 0 or not present, so just check structure
        assert isinstance(stats['total_sessions'], int)
    
    def test_No set_current_session method exists(self, voice_service):
        """Test setting the current session."""
        # Create sessions
        session1_id = voice_service.create_session("user1")
        session2_id = voice_service.create_session("user2")
        
        # Set current session
        voice_service.No set_current_session method exists(session2_id)
        
        # Verify current session
        current_session = voice_service.get_current_session()
        assert current_session.session_id == session2_id
    
    def test_No set_current_session method exists_nonexistent(self, voice_service):
        """Test setting current session to non-existent session."""
        # Should not raise an error
        voice_service.No set_current_session method exists("nonexistent_session")
        
        # Current session should be None
        current_session = voice_service.get_current_session()
        assert current_session is None
    
    def test_No cleanup_expired_sessions method exists(self, voice_service):
        """Test cleaning up expired sessions."""
        # Create a session
        session_id = voice_service.create_session("user123")
        
        # Manually expire the session
        session = voice_service.get_session(session_id)
        session.created_at = datetime.now() - timedelta(hours=2)  # 2 hours ago
        session.expires_at = datetime.now() - timedelta(hours=1)   # 1 hour ago
        
        # Run cleanup
        voice_service.No cleanup_expired_sessions method exists()
        
        # Session should be removed
        assert session_id not in voice_service.sessions
    
    def test_update_session_activity(self, voice_service):
        """Test updating session activity."""
        # Create a session
        session_id = voice_service.create_session("user123")
        original_activity = voice_service.get_session(session_id).last_activity
        
        # Wait a bit and update activity
        time.sleep(0.01)
        voice_service.update_session_activity(session_id)
        
        # Activity should be updated
        updated_activity = voice_service.get_session(session_id).last_activity
        assert updated_activity > original_activity
    
    def test_update_session_activity_nonexistent(self, voice_service):
        """Test updating activity for non-existent session."""
        # Should not raise an error
        voice_service.update_session_activity("nonexistent_session")
    
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
            'timestamp': datetime.now().isoformat()
        })
        session.conversation_history.append({
            'role': 'assistant', 
            'content': 'Hi there!',
            'timestamp': datetime.now().isoformat()
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
            data=b'test_audio_data',
            sample_rate=16000,
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
        
        def update_session():
            for i in range(10):
                voice_service.update_session_activity(session_id)
                time.sleep(0.001)
        
        # Create multiple threads updating the same session
        threads = [threading.Thread(target=update_session) for _ in range(5)]
        
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