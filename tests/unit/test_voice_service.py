"""
Comprehensive unit tests for voice/voice_service.py module.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
from datetime import datetime
import json

# Import the module to test with robust error handling
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from voice.voice_service import VoiceService, VoiceServiceError
    from voice.config import VoiceConfig
    from voice.stt_service import STTResult
    from voice.tts_service import TTSResult
    from voice.audio_processor import AudioData
    from voice.commands import VoiceCommandProcessor
except ImportError as e:
    pytest.skip(f"voice.voice_service module not available: {e}", allow_module_level=True)


class TestVoiceService:
    """Test VoiceService class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock voice config."""
        config = Mock(spec=VoiceConfig)
        config.voice_enabled = True
        config.stt_provider = "openai"
        config.tts_provider = "openai"
        config.openai_api_key = "test_key"
        config.elevenlabs_api_key = "test_key"
        config.elevenlabs_voice_id = "test_voice"
        config.voice_commands_enabled = True
        config.voice_command_wake_word = "hey therapist"
        config.voice_command_timeout = 30000
        config.voice_command_min_confidence = 0.6
        return config
    
    @pytest.fixture
    def mock_stt_service(self):
        """Create a mock STT service."""
        stt_service = Mock()
        stt_service.transcribe_audio = AsyncMock(return_value=STTResult(
            text="Hello therapist",
            confidence=0.9,
            provider="openai",
            processing_time=1.0
        ))
        stt_service.is_available = Mock(return_value=True)
        return stt_service
    
    @pytest.fixture
    def mock_tts_service(self):
        """Create a mock TTS service."""
        tts_service = Mock()
        tts_service.synthesize_speech = AsyncMock(return_value=TTSResult(
            audio_data=b"fake_audio_data",
            provider="openai",
            voice_id="test_voice",
            processing_time=1.0
        ))
        tts_service.is_available = Mock(return_value=True)
        return tts_service
    
    @pytest.fixture
    def mock_audio_processor(self):
        """Create a mock audio processor."""
        processor = Mock()
        processor.process_audio = Mock(return_value=AudioData(
            data=b"processed_audio",
            sample_rate=16000,
            channels=1
        ))
        processor.is_available = Mock(return_value=True)
        return processor
    
    @pytest.fixture
    def mock_command_processor(self):
        """Create a mock command processor."""
        processor = Mock(spec=VoiceCommandProcessor)
        processor.process_text = AsyncMock(return_value=None)
        processor.process_audio = AsyncMock(return_value=None)
        return processor
    
    @pytest.fixture
    def voice_service(self, mock_config, mock_stt_service, mock_tts_service, 
                     mock_audio_processor, mock_command_processor):
        """Create a voice service with mocked dependencies."""
        with patch('voice.voice_service.STTService', return_value=mock_stt_service), \
             patch('voice.voice_service.TTSService', return_value=mock_tts_service), \
             patch('voice.voice_service.SimplifiedAudioProcessor', return_value=mock_audio_processor), \
             patch('voice.voice_service.VoiceCommandProcessor', return_value=mock_command_processor):
            
            service = VoiceService(mock_config)
            service.stt_service = mock_stt_service
            service.tts_service = mock_tts_service
            service.audio_processor = mock_audio_processor
            service.command_processor = mock_command_processor
            return service
    
    def test_voice_service_initialization(self, voice_service, mock_config):
        """Test voice service initialization."""
        assert voice_service.config == mock_config
        assert voice_service.stt_service is not None
        assert voice_service.tts_service is not None
        assert voice_service.audio_processor is not None
        assert voice_service.command_processor is not None
        assert voice_service.is_initialized == True
    
    def test_voice_service_initialization_disabled(self, mock_config):
        """Test voice service initialization when disabled."""
        mock_config.voice_enabled = False
        
        with patch('voice.voice_service.STTService'), \
             patch('voice.voice_service.TTSService'), \
             patch('voice.voice_service.SimplifiedAudioProcessor'), \
             patch('voice.voice_service.VoiceCommandProcessor'):
            
            service = VoiceService(mock_config)
            assert service.is_initialized == False
    
    def test_is_available_true(self, voice_service):
        """Test is_available when all services are available."""
        voice_service.stt_service.is_available.return_value = True
        voice_service.tts_service.is_available.return_value = True
        voice_service.audio_processor.is_available.return_value = True
        
        assert voice_service.is_available() == True
    
    def test_is_available_false_stt(self, voice_service):
        """Test is_available when STT service is not available."""
        voice_service.stt_service.is_available.return_value = False
        voice_service.tts_service.is_available.return_value = True
        voice_service.audio_processor.is_available.return_value = True
        
        assert voice_service.is_available() == False
    
    def test_is_available_false_tts(self, voice_service):
        """Test is_available when TTS service is not available."""
        voice_service.stt_service.is_available.return_value = True
        voice_service.tts_service.is_available.return_value = False
        voice_service.audio_processor.is_available.return_value = True
        
        assert voice_service.is_available() == False
    
    def test_is_available_false_processor(self, voice_service):
        """Test is_available when audio processor is not available."""
        voice_service.stt_service.is_available.return_value = True
        voice_service.tts_service.is_available.return_value = True
        voice_service.audio_processor.is_available.return_value = False
        
        assert voice_service.is_available() == False
    
    def test_is_available_false_not_initialized(self, voice_service):
        """Test is_available when service is not initialized."""
        voice_service.is_initialized = False
        
        assert voice_service.is_available() == False
    
    @pytest.mark.asyncio
    async def test_process_audio_input_success(self, voice_service):
        """Test successful audio input processing."""
        audio_data = AudioData(data=b"test_audio", sample_rate=16000, channels=1)
        
        # Mock the command processor to return None (no command)
        voice_service.command_processor.process_audio.return_value = None
        
        result = await voice_service.process_audio_input(audio_data)
        
        assert result is not None
        assert result.text == "Hello therapist"
        assert result.confidence == 0.9
        assert result.provider == "openai"
        assert result.processing_time > 0
        
        # Verify STT service was called
        voice_service.stt_service.transcribe_audio.assert_called_once()
        
        # Verify command processor was called
        voice_service.command_processor.process_audio.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_audio_input_with_command(self, voice_service):
        """Test audio input processing with voice command."""
        audio_data = AudioData(data=b"test_audio", sample_rate=16000, channels=1)
        
        # Mock command match
        mock_command_match = Mock()
        mock_command_match.command.name = "test_command"
        mock_command_match.command.action = "test_action"
        mock_command_match.confidence = 0.9
        
        voice_service.command_processor.process_audio.return_value = mock_command_match
        
        result = await voice_service.process_audio_input(audio_data)
        
        assert result is not None
        assert result.text == "Hello therapist"
        assert result.command_match == mock_command_match
    
    @pytest.mark.asyncio
    async def test_process_audio_input_not_initialized(self, voice_service):
        """Test audio input processing when not initialized."""
        voice_service.is_initialized = False
        audio_data = AudioData(data=b"test_audio", sample_rate=16000, channels=1)
        
        with pytest.raises(VoiceServiceError) as exc_info:
            await voice_service.process_audio_input(audio_data)
        
        assert "Voice service is not initialized" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_process_audio_input_stt_error(self, voice_service):
        """Test audio input processing with STT error."""
        audio_data = AudioData(data=b"test_audio", sample_rate=16000, channels=1)
        
        # Mock STT service to raise an exception
        voice_service.stt_service.transcribe_audio.side_effect = Exception("STT error")
        
        with pytest.raises(VoiceServiceError) as exc_info:
            await voice_service.process_audio_input(audio_data)
        
        assert "Failed to process audio input" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_synthesize_speech_success(self, voice_service):
        """Test successful speech synthesis."""
        text = "Hello, this is a test"
        
        result = await voice_service.synthesize_speech(text)
        
        assert result is not None
        assert result.audio_data == b"fake_audio_data"
        assert result.provider == "openai"
        assert result.voice_id == "test_voice"
        assert result.processing_time > 0
        
        # Verify TTS service was called
        voice_service.tts_service.synthesize_speech.assert_called_once_with(text)
    
    @pytest.mark.asyncio
    async def test_synthesize_speech_not_initialized(self, voice_service):
        """Test speech synthesis when not initialized."""
        voice_service.is_initialized = False
        text = "Hello, this is a test"
        
        with pytest.raises(VoiceServiceError) as exc_info:
            await voice_service.synthesize_speech(text)
        
        assert "Voice service is not initialized" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_synthesize_speech_tts_error(self, voice_service):
        """Test speech synthesis with TTS error."""
        text = "Hello, this is a test"
        
        # Mock TTS service to raise an exception
        voice_service.tts_service.synthesize_speech.side_effect = Exception("TTS error")
        
        with pytest.raises(VoiceServiceError) as exc_info:
            await voice_service.synthesize_speech(text)
        
        assert "Failed to synthesize speech" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_process_text_command_success(self, voice_service):
        """Test successful text command processing."""
        text = "start a new session"
        
        # Mock command match
        mock_command_match = Mock()
        mock_command_match.command.name = "start_session"
        mock_command_match.command.action = "start_session"
        mock_command_match.confidence = 0.9
        
        voice_service.command_processor.process_text.return_value = mock_command_match
        
        result = await voice_service.process_text_command(text)
        
        assert result == mock_command_match
        voice_service.command_processor.process_text.assert_called_once_with(text)
    
    @pytest.mark.asyncio
    async def test_process_text_command_no_match(self, voice_service):
        """Test text command processing with no match."""
        text = "random text"
        
        voice_service.command_processor.process_text.return_value = None
        
        result = await voice_service.process_text_command(text)
        
        assert result is None
        voice_service.command_processor.process_text.assert_called_once_with(text)
    
    @pytest.mark.asyncio
    async def test_process_text_command_not_initialized(self, voice_service):
        """Test text command processing when not initialized."""
        voice_service.is_initialized = False
        text = "start a new session"
        
        with pytest.raises(VoiceServiceError) as exc_info:
            await voice_service.process_text_command(text)
        
        assert "Voice service is not initialized" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_process_text_command_error(self, voice_service):
        """Test text command processing with error."""
        text = "start a new session"
        
        # Mock command processor to raise an exception
        voice_service.command_processor.process_text.side_effect = Exception("Command error")
        
        with pytest.raises(VoiceServiceError) as exc_info:
            await voice_service.process_text_command(text)
        
        assert "Failed to process text command" in str(exc_info.value)
    
    def test_get_service_status(self, voice_service):
        """Test getting service status."""
        # Mock service availability
        voice_service.stt_service.is_available.return_value = True
        voice_service.tts_service.is_available.return_value = True
        voice_service.audio_processor.is_available.return_value = True
        
        status = voice_service.get_service_status()
        
        assert isinstance(status, dict)
        assert 'initialized' in status
        assert 'available' in status
        assert 'stt_service' in status
        assert 'tts_service' in status
        assert 'audio_processor' in status
        assert 'command_processor' in status
        assert 'config' in status
        
        assert status['initialized'] == True
        assert status['available'] == True
        assert status['stt_service']['available'] == True
        assert status['tts_service']['available'] == True
        assert status['audio_processor']['available'] == True
    
    def test_get_service_status_not_initialized(self, voice_service):
        """Test getting service status when not initialized."""
        voice_service.is_initialized = False
        
        status = voice_service.get_service_status()
        
        assert status['initialized'] == False
        assert status['available'] == False
    
    def test_get_supported_voices(self, voice_service):
        """Test getting supported voices."""
        # Mock TTS service voices
        voice_service.tts_service.get_supported_voices.return_value = [
            {"id": "voice1", "name": "Voice 1", "language": "en-US"},
            {"id": "voice2", "name": "Voice 2", "language": "en-GB"}
        ]
        
        voices = voice_service.get_supported_voices()
        
        assert isinstance(voices, list)
        assert len(voices) == 2
        assert voices[0]["id"] == "voice1"
        assert voices[1]["id"] == "voice2"
        
        voice_service.tts_service.get_supported_voices.assert_called_once()
    
    def test_get_supported_voices_not_initialized(self, voice_service):
        """Test getting supported voices when not initialized."""
        voice_service.is_initialized = False
        
        with pytest.raises(VoiceServiceError) as exc_info:
            voice_service.get_supported_voices()
        
        assert "Voice service is not initialized" in str(exc_info.value)
    
    def test_set_voice_profile(self, voice_service):
        """Test setting voice profile."""
        voice_profile = {
            "provider": "openai",
            "voice_id": "new_voice",
            "speed": 1.2,
            "pitch": 1.0
        }
        
        voice_service.set_voice_profile(voice_profile)
        
        # Verify the profile was set
        assert voice_service.voice_profile == voice_profile
    
    def test_set_voice_profile_not_initialized(self, voice_service):
        """Test setting voice profile when not initialized."""
        voice_service.is_initialized = False
        voice_profile = {"provider": "openai", "voice_id": "new_voice"}
        
        with pytest.raises(VoiceServiceError) as exc_info:
            voice_service.set_voice_profile(voice_profile)
        
        assert "Voice service is not initialized" in str(exc_info.value)
    
    def test_get_voice_profile(self, voice_service):
        """Test getting voice profile."""
        # Set a voice profile
        voice_profile = {"provider": "openai", "voice_id": "test_voice"}
        voice_service.voice_profile = voice_profile
        
        profile = voice_service.get_voice_profile()
        
        assert profile == voice_profile
    
    def test_reset_voice_profile(self, voice_service):
        """Test resetting voice profile."""
        # Set a voice profile
        voice_service.voice_profile = {"provider": "openai", "voice_id": "test_voice"}
        
        voice_service.reset_voice_profile()
        
        # Verify the profile was reset to default
        assert voice_service.voice_profile == {}
    
    def test_cleanup(self, voice_service):
        """Test service cleanup."""
        # Mock cleanup methods
        voice_service.stt_service.cleanup = Mock()
        voice_service.tts_service.cleanup = Mock()
        voice_service.command_processor.cleanup = Mock()
        
        voice_service.cleanup()
        
        # Verify cleanup methods were called
        voice_service.stt_service.cleanup.assert_called_once()
        voice_service.tts_service.cleanup.assert_called_once()
        voice_service.command_processor.cleanup.assert_called_once()
        
        # Verify service is marked as not initialized
        assert voice_service.is_initialized == False
    
    def test_cleanup_not_initialized(self, voice_service):
        """Test cleanup when not initialized."""
        voice_service.is_initialized = False
        
        # Should not raise an error
        voice_service.cleanup()
    
    def test_context_manager(self, voice_service):
        """Test using voice service as context manager."""
        # Mock cleanup method
        voice_service.cleanup = Mock()
        
        with voice_service as service:
            assert service == voice_service
        
        # Verify cleanup was called
        voice_service.cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_full_conversation_flow(self, voice_service):
        """Test a full conversation flow."""
        # User speaks
        audio_input = AudioData(data=b"user_speech", sample_rate=16000, channels=1)
        
        # Mock STT result
        stt_result = STTResult(
            text="How are you today?",
            confidence=0.9,
            provider="openai",
            processing_time=1.0
        )
        voice_service.stt_service.transcribe_audio.return_value = stt_result
        
        # Mock command processor (no command)
        voice_service.command_processor.process_audio.return_value = None
        
        # Process user input
        result = await voice_service.process_audio_input(audio_input)
        assert result.text == "How are you today?"
        
        # Generate response
        response_text = "I'm doing well, thank you for asking!"
        
        # Mock TTS result
        tts_result = TTSResult(
            audio_data=b"response_audio",
            provider="openai",
            voice_id="test_voice",
            processing_time=1.5
        )
        voice_service.tts_service.synthesize_speech.return_value = tts_result
        
        # Synthesize response
        audio_output = await voice_service.synthesize_speech(response_text)
        assert audio_output.audio_data == b"response_audio"
    
    @pytest.mark.asyncio
    async def test_error_handling_in_conversation(self, voice_service):
        """Test error handling in conversation flow."""
        # User speaks
        audio_input = AudioData(data=b"user_speech", sample_rate=16000, channels=1)
        
        # Mock STT service to raise an exception
        voice_service.stt_service.transcribe_audio.side_effect = Exception("STT error")
        
        # Process user input should raise VoiceServiceError
        with pytest.raises(VoiceServiceError):
            await voice_service.process_audio_input(audio_input)
    
    def test_str_representation(self, voice_service):
        """Test string representation of voice service."""
        str_repr = str(voice_service)
        
        assert "VoiceService" in str_repr
        assert "initialized" in str_repr
    
    def test_repr_representation(self, voice_service):
        """Test repr representation of voice service."""
        repr_str = repr(voice_service)
        
        assert "VoiceService" in repr_str
        assert "initialized" in repr_str


class TestVoiceServiceError:
    """Test VoiceServiceError exception."""
    
    def test_voice_service_error_creation(self):
        """Test creating VoiceServiceError."""
        error = VoiceServiceError("Test error message")
        
        assert str(error) == "Test error message"
    
    def test_voice_service_error_inheritance(self):
        """Test VoiceServiceError inheritance."""
        error = VoiceServiceError("Test error")
        
        assert isinstance(error, Exception)
        assert isinstance(error, ValueError)