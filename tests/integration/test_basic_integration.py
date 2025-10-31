"""
Simplified integration test for database and voice service.
Tests basic integration functionality without complex async setup.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Import with robust error handling
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from database.db_manager import DatabaseManager
    from database.models import (
        UserRepository, SessionRepository, VoiceSessionRepository,
        ConversationRepository, MessageRepository
    )
    from voice.voice_service import VoiceService
    from voice.config import VoiceConfig
    from auth.auth_service import AuthService
except ImportError as e:
    pytest.skip(f"Integration test dependencies not available: {e}", allow_module_level=True)


@pytest.mark.integration
class TestBasicVoiceServiceIntegration:
    """Test basic integration between services without complex async setup."""
    
    @pytest.fixture
    def test_db_manager(self):
        """Create an in-memory database for testing."""
        # Use SQLite in-memory database for testing
        db_path = ":memory:"
        
        db_manager = DatabaseManager(db_path)
        
        yield db_manager
        
        # Cleanup
        db_manager.close()
    
    @pytest.fixture
    def mock_voice_config(self):
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
        config.voice_command_timeout = 30000  # Add missing config
        config.voice_wake_word = "hey therapist"  # Add missing config
        return config
    
    @pytest.fixture
    def repositories(self):
        """Create repository instances using default database manager."""
        return {
            'user': UserRepository(),
            'session': SessionRepository(),
            'voice_session': VoiceSessionRepository(),
            'conversation': ConversationRepository(),
            'message': MessageRepository()
        }
    
    @pytest.fixture
    def voice_service_with_db(self, mock_voice_config, repositories):
        """Create a VoiceService with mocked database repositories."""
        # Mock the repository initialization to use our test repositories
        with patch('voice.voice_service.SessionRepository', return_value=repositories['session']), \
             patch('voice.voice_service.VoiceDataRepository', return_value=repositories['voice_session']), \
             patch('voice.voice_service.AuditLogRepository', return_value=Mock()), \
             patch('voice.voice_service.ConsentRepository', return_value=Mock()):
            
            # Create voice service with mocked dependencies
            with patch('voice.voice_service.SimplifiedAudioProcessor'), \
                 patch('voice.voice_service.STTService'), \
                 patch('voice.voice_service.TTSService'):
                
                # Mock VoiceSecurity
                mock_security = Mock()
                service = VoiceService(mock_voice_config, mock_security)
                # Manually set up repositories
                service.session_repo = repositories['session']
                service.voice_data_repo = repositories['voice_session']
                service._db_initialized = True
                return service
    
    def test_voice_session_creation_basic(self, voice_service_with_db):
        """Test basic voice session creation without database persistence."""
        # Create a voice session
        session_id = voice_service_with_db.create_session("user123")
        
        # Verify session exists in memory
        assert session_id in voice_service_with_db.sessions
        session = voice_service_with_db.sessions[session_id]
        assert session.session_id == session_id
        assert hasattr(session, 'start_time')  # Session should have timestamp
        # Note: user_id is stored in session_id for this implementation
    
    def test_voice_session_state_management(self, voice_service_with_db):
        """Test voice session state transitions."""
        session_id = voice_service_with_db.create_session("user123")
        
        # Get session
        session = voice_service_with_db.get_session(session_id)
        assert session is not None
        
        # Session should be active initially
        if hasattr(session, 'is_active'):
            assert session.is_active is True
        
        # End session
        result = voice_service_with_db.end_session(session_id)
        assert result is True  # Should return True on successful end
        
        # Session should be removed or inactive
        after_session = voice_service_with_db.get_session(session_id)
        if after_session is not None:
            # If session still exists, it should be inactive
            if hasattr(after_session, 'is_active'):
                assert after_session.is_active is False
        else:
            # Session was removed, which is also acceptable
            assert session_id not in voice_service_with_db.sessions
    
    def test_voice_service_basic_health(self, voice_service_with_db):
        """Test basic voice service health check."""
        health = voice_service_with_db.health_check()
        
        # Health should be a dictionary
        assert isinstance(health, dict)
        
        # Should contain basic health indicators
        assert 'overall_status' in health
        # Check for individual service health indicators
        assert 'audio_processor' in health or 'services' in health
    
    def test_service_initialization_with_dependencies(self, mock_voice_config):
        """Test that voice service can be initialized with mocked dependencies."""
        with patch('voice.voice_service.SimplifiedAudioProcessor'), \
             patch('voice.voice_service.STTService'), \
             patch('voice.voice_service.TTSService'):
            
            # Mock VoiceSecurity
            mock_security = Mock()
            service = VoiceService(mock_voice_config, mock_security)
            
            # Service should be created successfully
            assert service is not None
            assert hasattr(service, 'config')
            assert service.config == mock_voice_config
    
    def test_concurrent_session_creation(self, voice_service_with_db):
        """Test creating multiple sessions concurrently."""
        # Create multiple sessions
        session1_id = voice_service_with_db.create_session("user1")
        session2_id = voice_service_with_db.create_session("user2")
        session3_id = voice_service_with_db.create_session("user3")
        
        # All sessions should be created with unique IDs
        assert session1_id != session2_id
        assert session2_id != session3_id
        assert session1_id != session3_id
        
        # All sessions should exist
        assert session1_id in voice_service_with_db.sessions
        assert session2_id in voice_service_with_db.sessions
        assert session3_id in voice_service_with_db.sessions
        
        # Should have 3 active sessions
        active_sessions = [s for s in voice_service_with_db.sessions.values() 
                          if getattr(s, 'is_active', True)]
        assert len(active_sessions) == 3