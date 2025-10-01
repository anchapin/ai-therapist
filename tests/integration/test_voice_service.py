"""
Integration tests for voice service functionality.

Tests SPEECH_PRD.md requirements:
- End-to-end voice conversation testing
- Service integration testing
- Multi-provider fallback testing
- Voice command integration
- Crisis response integration
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
import tempfile
import json

from voice.voice_service import VoiceService
from voice.config import VoiceConfig
from voice.security import VoiceSecurity
from voice.stt_service import STTResult
from voice.tts_service import TTSResult


class TestVoiceServiceIntegration:
    """Test voice service integration."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = VoiceConfig()
        config.voice_enabled = True
        config.voice_input_enabled = True
        config.voice_output_enabled = True
        config.voice_commands_enabled = True
        config.recording_timeout = 5.0
        return config

    @pytest.fixture
    def security(self, config):
        """Create security instance."""
        return VoiceSecurity(config)

    @pytest.fixture
    def voice_service(self, config, security):
        """Create VoiceService instance for testing."""
        with patch('voice.voice_service.SimplifiedAudioProcessor') as mock_audio_processor, \
             patch('voice.voice_service.STTService') as mock_stt_service, \
             patch('voice.voice_service.TTSService') as mock_tts_service, \
             patch('voice.voice_service.VoiceCommandProcessor') as mock_command_processor, \
             patch.object(security, '_check_consent_status', return_value=True), \
             patch.object(security, '_verify_security_requirements', return_value=True):

            service = VoiceService(config, security)

            # Configure mocked components with proper methods
            service.audio_processor = mock_audio_processor.return_value
            service.stt_service = mock_stt_service.return_value
            service.tts_service = mock_tts_service.return_value
            service.command_processor = mock_command_processor.return_value

            # Configure audio processor mocks
            service.audio_processor.initialize.return_value = True
            service.audio_processor.cleanup.return_value = None
            service.audio_processor.get_audio_chunk.return_value = b'mock_audio_data'
            service.audio_processor.detect_voice_activity.return_value = True
            service.audio_processor.play_audio.return_value = True

            # Configure STT service mocks
            service.stt_service.initialize.return_value = True
            service.stt_service.cleanup.return_value = None
            service.stt_service.is_available.return_value = True

            # Configure TTS service mocks
            service.tts_service.initialize.return_value = True
            service.tts_service.cleanup.return_value = None
            service.tts_service.is_available.return_value = True
            service.tts_service.synthesize_speech.return_value = None

            # Configure command processor mocks
            service.command_processor.initialize.return_value = True
            service.command_processor.process_text = AsyncMock(return_value=None)
            service.command_processor.execute_command = AsyncMock(return_value={'success': True})

            return service

    def test_service_initialization(self, voice_service, config, security):
        """Test voice service initialization."""
        assert voice_service.config == config
        assert voice_service.security == security
        assert voice_service.initialized == False

    def test_service_initialization_process(self, voice_service):
        """Test complete service initialization."""
        # Mock successful initialization
        voice_service.audio_processor = MagicMock()
        voice_service.stt_service = MagicMock()
        voice_service.tts_service = MagicMock()
        voice_service.command_processor = MagicMock()

        # Mock successful initialization calls
        voice_service.audio_processor.initialize.return_value = True
        voice_service.stt_service.initialize.return_value = True
        voice_service.tts_service.initialize.return_value = True
        voice_service.command_processor.initialize.return_value = True

        result = voice_service.initialize()

        assert result == True
        assert voice_service.initialized == True

    def test_session_creation(self, voice_service):
        """Test session creation."""
        session_id = voice_service.create_session()

        assert session_id is not None
        assert len(session_id) > 0
        assert session_id in voice_service.sessions

        # Verify session structure
        session = voice_service.sessions[session_id]
        assert hasattr(session, 'metadata')
        assert 'created_at' in session.metadata
        assert hasattr(session, 'last_activity')
        assert hasattr(session, 'conversation_history')
        assert 'voice_settings' in session.metadata

    def test_session_management(self, voice_service):
        """Test session management."""
        # Create session
        session_id = voice_service.create_session()
        assert voice_service.get_session(session_id) is not None

        # Update session activity
        voice_service.update_session_activity(session_id)
        session = voice_service.get_session(session_id)
        assert session.last_activity is not None

        # End session
        voice_service.end_session(session_id)
        assert session_id not in voice_service.sessions

    @pytest.mark.asyncio
    async def test_voice_input_processing(self, voice_service, mock_audio_data):
        """Test voice input processing pipeline."""
        session_id = voice_service.create_session()

        # Create proper STT result mock with expected attributes
        mock_stt_result = MagicMock(spec=STTResult)
        mock_stt_result.text = "I need help with anxiety"
        mock_stt_result.confidence = 0.95
        mock_stt_result.is_crisis = False
        mock_stt_result.is_command = False
        mock_stt_result.alternatives = []
        mock_stt_result.provider = 'openai'
        mock_stt_result.processing_time = 1.0

        # Mock audio processing
        voice_service.audio_processor.get_audio_chunk.return_value = mock_audio_data['data']
        voice_service.audio_processor.detect_voice_activity.return_value = True

        # Mock STT service to return proper result, not MagicMock
        voice_service.stt_service.transcribe_audio.return_value = mock_stt_result

        # Process voice input
        result = await voice_service.process_voice_input(session_id, mock_audio_data['data'])

        assert result is not None
        assert result.text == "I need help with anxiety"
        assert result.confidence == 0.95
        assert result.is_crisis == False

    @pytest.mark.asyncio
    async def test_voice_output_generation(self, voice_service):
        """Test voice output generation."""
        session_id = voice_service.create_session()

        # Create proper TTS result mock with expected attributes
        mock_tts_result = MagicMock(spec=TTSResult)
        mock_tts_result.audio_data = b'synthesized_audio'
        mock_tts_result.duration = 2.5
        mock_tts_result.provider = 'openai'
        mock_tts_result.voice = 'alloy'
        mock_tts_result.format = 'wav'
        mock_tts_result.sample_rate = 22050

        # Mock TTS service to return proper result, not MagicMock
        voice_service.tts_service.synthesize_speech.return_value = mock_tts_result

        # Mock audio playback
        voice_service.audio_processor.play_audio.return_value = True

        # Generate voice output
        result = await voice_service.generate_voice_output(
            session_id,
            "I understand you're feeling anxious. Let's talk about it."
        )

        assert result is not None
        assert result.audio_data == b'synthesized_audio'
        assert result.duration == 2.5
        assert result.provider == 'openai'

    @pytest.mark.asyncio
    async def test_crisis_response_integration(self, voice_service, mock_audio_data):
        """Test crisis response integration."""
        session_id = voice_service.create_session()

        # Create proper STT result mock for crisis detection
        mock_stt_result = MagicMock(spec=STTResult)
        mock_stt_result.text = "I want to kill myself"
        mock_stt_result.confidence = 0.95
        mock_stt_result.is_crisis = True
        mock_stt_result.is_command = False
        mock_stt_result.crisis_keywords_detected = ['kill myself']
        mock_stt_result.alternatives = []
        mock_stt_result.provider = 'openai'

        # Mock STT service to return proper crisis result
        voice_service.stt_service.transcribe_audio.return_value = mock_stt_result

        # Create proper command result mock
        mock_command_result = MagicMock()
        mock_command_result.is_emergency = True
        mock_command_result.is_command = False
        mock_command_result.command.name = 'emergency_help'
        mock_command_result.success = True

        # Mock command processor with proper async methods
        voice_service.command_processor.process_text = AsyncMock(return_value=mock_command_result)
        voice_service.command_processor.execute_command = AsyncMock(return_value={'success': True})

        # Process crisis input
        result = await voice_service.process_voice_input(session_id, mock_audio_data['data'])

        # Verify crisis response triggered
        assert result.is_crisis == True
        assert result.text == "I want to kill myself"
        voice_service.command_processor.process_text.assert_called_once_with("I want to kill myself")
        voice_service.command_processor.execute_command.assert_called_once()

    @pytest.mark.asyncio
    async def test_voice_command_processing(self, voice_service, mock_audio_data):
        """Test voice command processing."""
        session_id = voice_service.create_session()

        # Create proper STT result mock for command processing
        mock_stt_result = MagicMock(spec=STTResult)
        mock_stt_result.text = "start meditation"
        mock_stt_result.confidence = 0.90
        mock_stt_result.is_command = True
        mock_stt_result.is_crisis = False
        mock_stt_result.alternatives = []
        mock_stt_result.provider = 'openai'

        # Mock STT service to return proper command result
        voice_service.stt_service.transcribe_audio.return_value = mock_stt_result

        # Create proper command result mock
        mock_command_result = MagicMock()
        mock_command_result.is_command = True
        mock_command_result.is_emergency = False
        mock_command_result.command.name = 'start_meditation'
        mock_command_result.success = True

        # Mock command processor with proper async methods
        voice_service.command_processor.process_text = AsyncMock(return_value=mock_command_result)
        voice_service.command_processor.execute_command = AsyncMock(return_value={'success': True})

        # Process voice command
        result = await voice_service.process_voice_input(session_id, mock_audio_data['data'])

        # Verify command processing
        assert result.is_command == True
        assert result.text == "start meditation"
        voice_service.command_processor.process_text.assert_called_once_with("start meditation")
        voice_service.command_processor.execute_command.assert_called_once()

    @pytest.mark.asyncio
    async def test_conversation_flow(self, voice_service):
        """Test conversation flow integration."""
        session_id = voice_service.create_session()

        # Create proper STT result mock
        mock_stt_result = MagicMock(spec=STTResult)
        mock_stt_result.text = "I'm feeling stressed"
        mock_stt_result.confidence = 0.95
        mock_stt_result.is_crisis = False
        mock_stt_result.is_command = False
        mock_stt_result.alternatives = []
        mock_stt_result.provider = 'openai'

        # Mock voice input processing
        voice_input = b'user_voice_input'
        voice_service.stt_service.transcribe_audio.return_value = mock_stt_result

        # Mock AI response generation with proper return type
        ai_response = "I understand you're feeling stressed. Let's work on some breathing exercises."
        voice_service.generate_ai_response = MagicMock(return_value=ai_response)

        # Create proper TTS result mock
        mock_tts_result = MagicMock(spec=TTSResult)
        mock_tts_result.audio_data = b'ai_response_audio'
        mock_tts_result.duration = 3.0
        mock_tts_result.provider = 'openai'
        mock_tts_result.voice = 'alloy'
        mock_tts_result.format = 'wav'

        # Mock TTS service to return proper result
        voice_service.tts_service.synthesize_speech.return_value = mock_tts_result

        # Process conversation turn
        result = await voice_service.process_conversation_turn(session_id, voice_input)

        assert result is not None
        assert 'user_input' in result
        assert 'ai_response' in result
        assert 'voice_output' in result
        assert result['ai_response'] == ai_response
        assert result['voice_output'].audio_data == b'ai_response_audio'

    @pytest.mark.asyncio
    async def test_multi_provider_fallback(self, voice_service, mock_audio_data):
        """Test multi-provider fallback integration."""
        session_id = voice_service.create_session()

        # Create proper fallback STT result mock
        mock_fallback_result = MagicMock(spec=STTResult)
        mock_fallback_result.text = "Fallback transcription"
        mock_fallback_result.confidence = 0.85
        mock_fallback_result.is_crisis = False
        mock_fallback_result.is_command = False
        mock_fallback_result.alternatives = []
        mock_fallback_result.provider = 'fallback'

        # Mock primary provider failure
        voice_service.stt_service.transcribe_audio.side_effect = Exception("Primary provider failed")

        # Mock fallback provider success with proper result
        voice_service.fallback_stt_service = MagicMock()
        voice_service.fallback_stt_service.transcribe_audio.return_value = mock_fallback_result

        # Process with fallback
        result = await voice_service.process_voice_input(session_id, mock_audio_data['data'])

        assert result is not None
        assert result.text == "Fallback transcription"
        assert result.confidence == 0.85
        assert result.provider == 'fallback'

    def test_service_statistics(self, voice_service):
        """Test service statistics."""
        # Create some activity
        session_id = voice_service.create_session()
        voice_service.update_session_activity(session_id)

        stats = voice_service.get_service_statistics()

        assert 'sessions_count' in stats
        assert 'active_sessions' in stats
        assert 'total_conversations' in stats
        assert 'service_uptime' in stats
        assert 'error_count' in stats

    @pytest.mark.asyncio
    async def test_error_handling(self, voice_service):
        """Test error handling."""
        session_id = voice_service.create_session()

        # Test invalid session
        result = await voice_service.process_voice_input("invalid_session", b'audio')
        assert result is None

        # Test audio processing error with proper error handling
        voice_service.audio_processor.get_audio_chunk.side_effect = Exception("Audio error")

        result = await voice_service.process_voice_input(session_id, b'audio')
        # Result could be None or could have error attribute depending on implementation
        assert result is None or (hasattr(result, 'error') and result.error is not None)

    def test_conversation_history(self, voice_service):
        """Test conversation history management."""
        session_id = voice_service.create_session()

        # Add conversation entries
        voice_service.add_conversation_entry(
            session_id,
            "user",
            "I'm feeling anxious"
        )
        voice_service.add_conversation_entry(
            session_id,
            "ai",
            "Let's talk about your anxiety"
        )

        # Retrieve history
        history = voice_service.get_conversation_history(session_id)

        assert len(history) == 2
        assert history[0]['speaker'] == 'user'
        assert history[1]['speaker'] == 'ai'

    @pytest.mark.asyncio
    async def test_voice_settings_management(self, voice_service):
        """Test voice settings management."""
        session_id = voice_service.create_session()

        # Update voice settings
        new_settings = {
            'voice_speed': 1.2,
            'voice_pitch': 1.1,
            'volume': 0.8
        }
        voice_service.update_voice_settings(session_id, new_settings)

        # Verify settings
        session = voice_service.get_session(session_id)
        assert session.metadata['voice_settings']['voice_speed'] == 1.2
        assert session.metadata['voice_settings']['voice_pitch'] == 1.1
        assert session.metadata['voice_settings']['volume'] == 0.8

    def test_service_health_check(self, voice_service):
        """Test service health check."""
        health = voice_service.health_check()

        assert 'overall_status' in health
        assert 'audio_processor' in health
        assert 'stt_service' in health
        assert 'tts_service' in health
        assert 'command_processor' in health
        assert 'security' in health

    def test_cleanup(self, voice_service):
        """Test cleanup functionality."""
        # Create some sessions
        for i in range(3):
            voice_service.create_session()

        # Mock component cleanup
        voice_service.audio_processor = MagicMock()
        voice_service.stt_service = MagicMock()
        voice_service.tts_service = MagicMock()
        voice_service.command_processor = MagicMock()

        voice_service.cleanup()

        # Verify all sessions are cleared
        assert len(voice_service.sessions) == 0
        # Verify component cleanup calls
        voice_service.audio_processor.cleanup.assert_called_once()
        voice_service.stt_service.cleanup.assert_called_once()
        voice_service.tts_service.cleanup.assert_called_once()