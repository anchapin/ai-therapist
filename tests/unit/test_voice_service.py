"""
Comprehensive unit tests for voice/voice_service.py module.
Tests core voice service functionality with proper mocking and isolation.
"""

import pytest
import pytest_asyncio
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from pathlib import Path

# Import the module to test with robust error handling
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from voice.voice_service import VoiceService, VoiceSession, VoiceSessionState
    from voice.config import VoiceConfig, VoiceProfile
    from voice.audio_processor import AudioData
    from voice.stt_service import STTResult
    from voice.tts_service import TTSResult
    from voice.commands import VoiceCommand
    class VoiceError(Exception):
        pass
except ImportError as e:
    pytest.skip(f"voice.voice_service module not available: {e}", allow_module_level=True)


class TestVoiceService:
    """Test VoiceService core functionality."""
    
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
        return config
    
    @pytest.fixture
    def mock_audio_processor(self):
        """Create a mock audio processor."""
        processor = Mock()
        processor.start_capture = AsyncMock()
        processor.stop_capture = AsyncMock()
        processor.get_audio_data = AsyncMock(return_value=AudioData(b'test_audio', 16000, 1))
        processor.is_capturing = False
        return processor
    
    @pytest.fixture
    def mock_stt_service(self):
        """Create a mock STT service."""
        stt = Mock()
        stt.transcribe_audio = AsyncMock(return_value=STTResult("hello world", 0.95, 1.0))
        stt.is_healthy = True
        return stt
    
    @pytest.fixture
    def mock_tts_service(self):
        """Create a mock TTS service."""
        tts = Mock()
        tts.synthesize_speech = AsyncMock(return_value=TTSResult(b'synthesized_audio', 'audio/wav', 1.5))
        tts.is_healthy = True
        return tts
    
    @pytest.fixture
    def mock_security(self):
        """Create a mock security service."""
        security = Mock()
        security.encrypt_audio = AsyncMock(return_value=b'encrypted_audio')
        security.decrypt_audio = AsyncMock(return_value=b'decrypted_audio')
        security.sanitize_transcription = Mock(return_value="sanitized text")
        security.is_healthy = True
        return security
    
    @pytest.fixture
    def mock_command_processor(self):
        """Create a mock command processor."""
        processor = Mock()
        processor.process_command = AsyncMock(return_value={"command": "test", "action": "respond"})
        processor.detect_command = Mock(return_value=None)
        return processor
    
    @pytest.fixture
    def voice_service(self, mock_config, mock_audio_processor, mock_stt_service, 
                     mock_tts_service, mock_security, mock_command_processor):
        """Create a VoiceService instance with mocked dependencies."""
        with patch('voice.voice_service.SimplifiedAudioProcessor', return_value=mock_audio_processor), \
             patch('voice.voice_service.STTService', return_value=mock_stt_service), \
             patch('voice.voice_service.TTSService', return_value=mock_tts_service), \
             patch('voice.voice_service.VoiceCommandProcessor', return_value=mock_command_processor):
            
            service = VoiceService(mock_config, mock_security)
            service.initialize()
            return service
    
    def test_voice_service_initialization(self, voice_service):
        """Test voice service initialization."""
        assert voice_service.config is not None
        assert voice_service.audio_processor is not None
        assert voice_service.stt_service is not None
        assert voice_service.tts_service is not None
        assert voice_service.security is not None
        assert voice_service.command_processor is not None
        assert voice_service.initialized is True
    
    def test_create_voice_session(self, voice_service):
        """Test creating a voice session."""
        session_id = voice_service.create_session("user123")
        
        assert session_id is not None
        assert session_id in voice_service.sessions
        session = voice_service.sessions[session_id]
        assert hasattr(session, 'state')
        assert hasattr(session, 'start_time')
        assert hasattr(session, 'metadata')
    
    def test_create_voice_session_with_custom_config(self, voice_service):
        """Test creating a voice session with custom configuration."""
        profile = "custom_voice"
        
        session_id = voice_service.create_session("user123", voice_profile=profile)
        
        assert session_id is not None
        assert session_id in voice_service.sessions
        session = voice_service.sessions[session_id]
        assert session.current_voice_profile == profile
    
    def test_get_existing_session(self, voice_service):
        """Test getting an existing session."""
        session_id = voice_service.create_session("user123")
        session = voice_service.get_session(session_id)
        
        assert session is not None
        assert session.session_id == session_id
        assert hasattr(session, 'state')
        assert hasattr(session, 'metadata')
    
    def test_get_nonexistent_session(self, voice_service):
        """Test getting a non-existent session."""
        retrieved = voice_service.get_session("nonexistent")
        assert retrieved is None
    
    def test_process_voice_input_success(self, voice_service):
        """Test successful voice input processing."""
        session_id = voice_service.create_session("user123")
        audio_data = AudioData(b'test_audio', 16000, 1)
        
        result = voice_service.process_voice_input(session_id, audio_data)
        
        assert result is not None
    
    def test_process_voice_input_session_not_found(self, voice_service):
        """Test processing voice input with non-existent session."""
        audio_data = AudioData(b'test_audio', 16000, 1)
        
        with pytest.raises(VoiceError, match="Session not found"):
            voice_service.process_voice_input("nonexistent", audio_data)
    
    async def test_process_voice_input_stt_failure(self, voice_service, mock_stt_service):
        """Test processing voice input when STT fails."""
        mock_stt_service.transcribe_audio = AsyncMock(side_effect=Exception("STT failed"))
        session = voice_service.create_session("user123")
        audio_data = AudioData(b'test_audio', 16000, 1)
        
        result = voice_service.process_voice_input(session.session_id, audio_data)
        
        assert result is not None
        assert 'error' in result
        assert 'transcription' not in result
    
    async def test_generate_voice_response_success(self, voice_service):
        """Test successful voice response generation."""
        session = voice_service.create_session("user123")
        
        result = voice_service.generate_voice_response(session.session_id, "Hello world")
        
        assert result is not None
        assert 'audio_data' in result
        assert 'format' in result
        assert 'duration' in result
        assert 'processing_time' in result
    
    async def test_generate_voice_response_session_not_found(self, voice_service):
        """Test generating voice response with non-existent session."""
        with pytest.raises(VoiceError, match="Session not found"):
            voice_service.generate_voice_response("nonexistent", "Hello")
    
    async def test_generate_voice_response_tts_failure(self, voice_service, mock_tts_service):
        """Test generating voice response when TTS fails."""
        mock_tts_service.synthesize_speech = AsyncMock(side_effect=Exception("TTS failed"))
        session = voice_service.create_session("user123")
        
        result = voice_service.generate_voice_response(session.session_id, "Hello")
        
        assert result is not None
        assert 'error' in result
        assert 'audio_data' not in result
    
    async def test_process_voice_command_detected(self, voice_service, mock_command_processor):
        """Test processing when a voice command is detected."""
        mock_command_processor.detect_command = Mock(return_value=VoiceCommand("emergency", {}))
        mock_command_processor.process_command = AsyncMock(return_value={"action": "emergency_response"})
        
        session = voice_service.create_session("user123")
        audio_data = AudioData(b'test_audio', 16000, 1)
        
        result = voice_service.process_voice_input(session.session_id, audio_data)
        
        assert 'command' in result
        assert result['command']['action'] == "emergency_response"
    
    async def test_end_voice_session_success(self, voice_service):
        """Test successfully ending a voice session."""
        session = voice_service.create_session("user123")
        
        result = voice_service.end_session(session.session_id)
        
        assert result is True
        assert session.session_id not in voice_service.sessions
        assert not session.is_active
    
    async def test_end_voice_session_not_found(self, voice_service):
        """Test ending a non-existent voice session."""
        result = voice_service.end_session("nonexistent")
        assert result is False
    
    async def test_cleanup_expired_sessions(self, voice_service):
        """Test cleanup of expired sessions."""
        # Create a session and manually expire it
        session = voice_service.create_session("user123")
        session.created_at = datetime.now() - timedelta(hours=2)
        session.expires_at = datetime.now() - timedelta(hours=1)
        
        voice_service.cleanup_expired_sessions()
        
        assert session.session_id not in voice_service.sessions
    
    async def test_get_session_statistics(self, voice_service):
        """Test getting session statistics."""
        voice_service.create_session("user123")
        voice_service.create_session("user456")
        
        stats = voice_service.get_session_statistics()
        
        assert 'total_sessions' in stats
        assert 'active_sessions' in stats
        assert 'expired_sessions' in stats
        assert stats['total_sessions'] >= 2
    
    async def test_health_check_all_healthy(self, voice_service, mock_stt_service, mock_tts_service, mock_security):
        """Test health check when all services are healthy."""
        mock_stt_service.is_healthy = True
        mock_tts_service.is_healthy = True
        mock_security.is_healthy = True
        
        health = voice_service.health_check()
        
        assert health['overall'] is True
        assert health['stt_service'] is True
        assert health['tts_service'] is True
        assert health['security'] is True
    
    async def test_health_check_some_unhealthy(self, voice_service, mock_stt_service):
        """Test health check when some services are unhealthy."""
        mock_stt_service.is_healthy = False
        
        health = voice_service.health_check()
        
        assert health['overall'] is False
        assert health['stt_service'] is False
    
    async def test_concurrent_session_limit(self, voice_service):
        """Test enforcement of concurrent session limits."""
        # Create multiple sessions for the same user
        sessions = []
        for i in range(5):  # Assuming limit is lower than 5
            try:
                session = voice_service.create_session("user123")
                sessions.append(session)
            except VoiceError:
                break
        
        # Should have created at least one session but not all
        assert len(sessions) >= 1
        assert len(sessions) < 5
        
        # Clean up
        for session in sessions:
            voice_service.end_session(session.session_id)


class TestVoiceSession:
    """Test VoiceSession functionality."""
    
    def test_voice_session_creation(self):
        """Test voice session creation."""
        from voice.voice_service import VoiceSession, VoiceSessionState
        
        session = VoiceSession(
            session_id="session456",
            state=VoiceSessionState.IDLE,
            start_time=time.time(),
            last_activity=time.time(),
            conversation_history=[],
            current_voice_profile="default",
            audio_buffer=[],
            metadata={}
        )
        
        assert session.session_id == "session456"
        assert session.state == VoiceSessionState.IDLE
        assert session.conversation_history == []
        assert session.current_voice_profile == "default"
        assert session.audio_buffer == []
        assert isinstance(session.metadata, dict)
    
    def test_voice_session_metadata(self):
        """Test voice session metadata handling."""
        from voice.voice_service import VoiceSession, VoiceSessionState
        
        start_time = time.time()
        session = VoiceSession(
            session_id="session456",
            state=VoiceSessionState.IDLE,
            start_time=start_time,
            last_activity=start_time,
            conversation_history=[],
            current_voice_profile="default",
            audio_buffer=[],
            metadata={}
        )
        
        # Check that created_at is added
        assert 'created_at' in session.metadata
        assert session.metadata['created_at'] == start_time
        
        # Check that voice_settings are added
        assert 'voice_settings' in session.metadata
        assert 'voice_speed' in session.metadata['voice_settings']
        assert session.metadata['voice_settings']['voice_speed'] == 1.2
    
    def test_voice_session_state_transitions(self):
        """Test voice session state transitions."""
        from voice.voice_service import VoiceSession, VoiceSessionState
        
        session = VoiceSession(
            session_id="session456",
            state=VoiceSessionState.IDLE,
            start_time=time.time(),
            last_activity=time.time(),
            conversation_history=[],
            current_voice_profile="default",
            audio_buffer=[],
            metadata={}
        )
        
        # Test state changes
        session.state = VoiceSessionState.LISTENING
        assert session.state == VoiceSessionState.LISTENING
        
        session.state = VoiceSessionState.PROCESSING
        assert session.state == VoiceSessionState.PROCESSING
        
        session.state = VoiceSessionState.SPEAKING
        assert session.state == VoiceSessionState.SPEAKING


class TestVoiceCommand:
    """Test VoiceCommand functionality."""
    
    def test_voice_command_creation(self):
        """Test voice command creation."""
        from voice.commands import VoiceCommand, CommandCategory
        
        command = VoiceCommand(
            command_type="test_command",
            category=CommandCategory.SESSION,
            patterns=["test", "example"],
            action=lambda x: {"result": "test_executed"},
            description="Test command",
            confidence_threshold=0.8
        )
        
        assert command.command_type == "test_command"
        assert command.category == CommandCategory.SESSION
        assert "test" in command.patterns
        assert "example" in command.patterns
        assert command.description == "Test command"
        assert command.confidence_threshold == 0.8
    
    def test_voice_command_execution(self):
        """Test voice command execution."""
        from voice.commands import VoiceCommand, CommandCategory
        
        test_action = Mock(return_value={"executed": True})
        command = VoiceCommand(
            command_type="test_command",
            category=CommandCategory.SESSION,
            patterns=["test"],
            action=test_action,
            description="Test command"
        )
        
        result = command.action({"text": "test command"})
        
        assert result["executed"] is True
        test_action.assert_called_once_with({"text": "test command"})