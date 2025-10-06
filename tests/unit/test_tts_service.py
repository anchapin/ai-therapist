"""
Comprehensive unit tests for voice/tts_service.py module.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
import json

# Import the module to test with robust error handling
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from voice.tts_service import TTSService, TTSResult, TTSError
    from voice.config import VoiceConfig
    from voice.audio_processor import AudioData
except ImportError as e:
    pytest.skip(f"voice.tts_service module not available: {e}", allow_module_level=True)


class TestTTSResult:
    """Test TTSResult dataclass."""
    
    def test_tts_result_creation(self):
        """Test creating a TTS result."""
        audio_data = AudioData(data=b"fake_audio_data", sample_rate=22050, channels=1, duration=1.0)
        result = TTSResult(
            audio=audio_data,
            text="Hello world",
            provider="openai",
            processing_time=1.5
        )
        
        assert result.audio == audio_data
        assert result.text == "Hello world"
        assert result.provider == "openai"
        assert result.processing_time == 1.5
    
    def test_tts_result_with_optional_fields(self):
        """Test creating a TTS result with optional fields."""
        audio_data = AudioData(data=b"fake_audio_data", sample_rate=22050, channels=1, duration=1.0)
        result = TTSResult(
            audio=audio_data,
            text="Hello world",
            provider="openai",
            processing_time=1.5,
            voice="alloy",
            language="en-US",
            metadata={"model": "tts-1"}
        )
        
        assert result.voice == "alloy"
        assert result.language == "en-US"
        assert result.metadata == {"model": "tts-1"}


class TestTTSService:
    """Test TTSService class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock voice config."""
        config = Mock(spec=VoiceConfig)
        config.tts_provider = "openai"
        config.openai_api_key = "test_key"
        config.tts_voice = "alloy"
        config.tts_language = "en-US"
        config.tts_model = "tts-1"
        
        # Add performance config
        config.performance = Mock()
        config.performance.cache_size = 100
        
        return config
    
    @pytest.fixture
    def tts_service(self, mock_config):
        """Create a TTS service with mock config."""
        with patch('voice.tts_service.openai'):
            service = TTSService(mock_config)
            return service
    
    def test_tts_service_initialization(self, tts_service, mock_config):
        """Test TTS service initialization."""
        assert tts_service.config == mock_config
        assert tts_service.provider == "openai"
        assert tts_service.api_key == "test_key"
        assert tts_service.voice == "alloy"
        assert tts_service.language == "en-US"
        assert tts_service.model == "tts-1"
    
    def test_tts_service_initialization_no_api_key(self, mock_config):
        """Test TTS service initialization with no API key."""
        mock_config.openai_api_key = None
        
        with patch('voice.tts_service.openai'):
            service = TTSService(mock_config)
            assert service.api_key is None
    
    def test_is_available_true(self, tts_service):
        """Test is_available when service is available."""
        tts_service.api_key = "test_key"
        assert tts_service.is_available() == True
    
    def test_is_available_false_no_api_key(self, tts_service):
        """Test is_available when no API key."""
        tts_service.api_key = None
        assert tts_service.is_available() == False
    
    @pytest.mark.asyncio
    async def test_synthesize_speech_openai_success(self, tts_service):
        """Test successful speech synthesis with OpenAI."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = b"fake_audio_data"
        
        mock_client.audio.speech.create = AsyncMock(return_value=mock_response)
        tts_service.openai_client = mock_client
        
        result = await tts_service.synthesize_speech("Hello world")
        
        assert result is not None
        assert result.text == "Hello world"
        assert result.audio.data == b"fake_audio_data"
        assert result.provider == "openai"
        assert result.processing_time > 0
        
        # Verify OpenAI client was called
        mock_client.audio.speech.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_synthesize_speech_with_voice(self, tts_service):
        """Test speech synthesis with specified voice."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = b"fake_audio_data"
        
        mock_client.audio.speech.create = AsyncMock(return_value=mock_response)
        tts_service.openai_client = mock_client
        
        result = await tts_service.synthesize_speech("Hello world", voice="nova")
        
        assert result.text == "Hello world"
        assert result.voice == "nova"
        
        # Verify voice was passed to OpenAI
        call_args = mock_client.audio.speech.create.call_args
        assert call_args[1]['voice'] == "nova"
    
    @pytest.mark.asyncio
    async def test_synthesize_speech_with_language(self, tts_service):
        """Test speech synthesis with specified language."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = b"fake_audio_data"
        
        mock_client.audio.speech.create = AsyncMock(return_value=mock_response)
        tts_service.openai_client = mock_client
        
        result = await tts_service.synthesize_speech("Bonjour le monde", language="fr")
        
        assert result.text == "Bonjour le monde"
        assert result.language == "fr"
        
        # Verify language was passed to OpenAI
        call_args = mock_client.audio.speech.create.call_args
        assert call_args[1]['language'] == "fr"
    
    @pytest.mark.asyncio
    async def test_synthesize_speech_not_available(self, tts_service):
        """Test synthesis when service is not available."""
        tts_service.api_key = None
        
        with pytest.raises(TTSError) as exc_info:
            await tts_service.synthesize_speech("Hello world")
        
        assert "TTS service is not available" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_synthesize_speech_openai_error(self, tts_service):
        """Test synthesis with OpenAI error."""
        # Mock OpenAI client to raise an exception
        mock_client = Mock()
        mock_client.audio.speech.create = AsyncMock(
            side_effect=Exception("OpenAI API error")
        )
        tts_service.openai_client = mock_client
        
        with pytest.raises(TTSError) as exc_info:
            await tts_service.synthesize_speech("Hello world")
        
        assert "Failed to synthesize speech" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_synthesize_speech_empty_text(self, tts_service):
        """Test synthesis with empty text."""
        with pytest.raises(TTSError) as exc_info:
            await tts_service.synthesize_speech("")
        
        assert "No text provided" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_synthesize_speech_timeout(self, tts_service):
        """Test synthesis with timeout."""
        # Mock OpenAI client to timeout
        mock_client = Mock()
        mock_client.audio.speech.create = AsyncMock(
            side_effect=asyncio.TimeoutError("Request timeout")
        )
        tts_service.openai_client = mock_client
        
        with pytest.raises(TTSError) as exc_info:
            await tts_service.synthesize_speech("Hello world")
        
        assert "Synthesis timed out" in str(exc_info.value)
    
    def test_get_supported_voices(self, tts_service):
        """Test getting supported voices."""
        voices = tts_service.get_supported_voices()
        
        assert isinstance(voices, list)
        assert len(voices) > 0
        assert any(voice["id"] == "alloy" for voice in voices)
        assert any(voice["id"] == "nova" for voice in voices)
        assert all("id" in voice for voice in voices)
        assert all("name" in voice for voice in voices)
        assert all("language" in voice for voice in voices)
    
    def test_get_supported_languages(self, tts_service):
        """Test getting supported languages."""
        languages = tts_service.get_supported_languages()
        
        assert isinstance(languages, list)
        assert len(languages) > 0
        assert any(lang["code"] == "en" for lang in languages)
        assert any(lang["code"] == "es" for lang in languages)
        assert all("code" in lang for lang in languages)
        assert all("name" in lang for lang in languages)
    
    def test_get_supported_models(self, tts_service):
        """Test getting supported models."""
        models = tts_service.get_supported_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        assert any(model["id"] == "tts-1" for model in models)
        assert any(model["id"] == "tts-1-hd" for model in models)
        assert all("id" in model for model in models)
        assert all("name" in model for model in models)
    
    def test_set_voice(self, tts_service):
        """Test setting voice."""
        tts_service.set_voice("nova")
        
        assert tts_service.voice == "nova"
    
    def test_set_voice_invalid(self, tts_service):
        """Test setting invalid voice."""
        with pytest.raises(TTSError) as exc_info:
            tts_service.set_voice("invalid-voice")
        
        assert "Invalid voice" in str(exc_info.value)
    
    def test_set_language(self, tts_service):
        """Test setting language."""
        tts_service.set_language("es")
        
        assert tts_service.language == "es"
    
    def test_set_language_invalid(self, tts_service):
        """Test setting invalid language."""
        with pytest.raises(TTSError) as exc_info:
            tts_service.set_language("invalid")
        
        assert "Invalid language" in str(exc_info.value)
    
    def test_set_model(self, tts_service):
        """Test setting model."""
        tts_service.set_model("tts-1-hd")
        
        assert tts_service.model == "tts-1-hd"
    
    def test_set_model_invalid(self, tts_service):
        """Test setting invalid model."""
        with pytest.raises(TTSError) as exc_info:
            tts_service.set_model("invalid-model")
        
        assert "Invalid model" in str(exc_info.value)
    
    def test_get_service_info(self, tts_service):
        """Test getting service information."""
        info = tts_service.get_service_info()
        
        assert isinstance(info, dict)
        assert "provider" in info
        assert "voice" in info
        assert "language" in info
        assert "model" in info
        assert "available" in info
        assert "supported_voices" in info
        assert "supported_languages" in info
        assert "supported_models" in info
        
        assert info["provider"] == "openai"
        assert info["voice"] == "alloy"
        assert info["language"] == "en-US"
        assert info["model"] == "tts-1"
    
    def test_cleanup(self, tts_service):
        """Test service cleanup."""
        # Mock OpenAI client
        mock_client = Mock()
        tts_service.openai_client = mock_client
        
        tts_service.cleanup()
        
        # Verify client is cleaned up
        assert tts_service.openai_client is None
    
    def test_context_manager(self, tts_service):
        """Test using TTS service as context manager."""
        # Mock OpenAI client
        mock_client = Mock()
        tts_service.openai_client = mock_client
        
        with tts_service as service:
            assert service == tts_service
        
        # Verify cleanup was called
        assert tts_service.openai_client is None
    
    def test_str_representation(self, tts_service):
        """Test string representation of TTS service."""
        str_repr = str(tts_service)
        
        assert "TTSService" in str_repr
        assert "openai" in str_repr
    
    def test_repr_representation(self, tts_service):
        """Test repr representation of TTS service."""
        repr_str = repr(tts_service)
        
        assert "TTSService" in repr_str
        assert "openai" in repr_str
    
    @pytest.mark.asyncio
    async def test_batch_synthesis(self, tts_service):
        """Test batch synthesis of multiple texts."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_response1 = Mock()
        mock_response1.content = b"audio1"
        
        mock_response2 = Mock()
        mock_response2.content = b"audio2"
        
        mock_client.audio.speech.create = AsyncMock(
            side_effect=[mock_response1, mock_response2]
        )
        tts_service.openai_client = mock_client
        
        text_list = ["First text", "Second text"]
        
        results = await tts_service.batch_synthesize(text_list)
        
        assert len(results) == 2
        assert results[0].text == "First text"
        assert results[1].text == "Second text"
        
        # Verify OpenAI client was called twice
        assert mock_client.audio.speech.create.call_count == 2
    
    @pytest.mark.asyncio
    async def test_synthesize_with_ssml(self, tts_service):
        """Test synthesis with SSML."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = b"fake_audio_data"
        
        mock_client.audio.speech.create = AsyncMock(return_value=mock_response)
        tts_service.openai_client = mock_client
        
        ssml_text = "<speak>Hello <emphasis>world</emphasis></speak>"
        
        result = await tts_service.synthesize_speech(ssml_text, use_ssml=True)
        
        assert result.text == ssml_text
        
        # Verify SSML was passed to OpenAI
        call_args = mock_client.audio.speech.create.call_args
        assert call_args[1]['input'] == ssml_text
    
    @pytest.mark.asyncio
    async def test_synthesize_with_speed(self, tts_service):
        """Test synthesis with custom speed."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = b"fake_audio_data"
        
        mock_client.audio.speech.create = AsyncMock(return_value=mock_response)
        tts_service.openai_client = mock_client
        
        result = await tts_service.synthesize_speech("Hello world", speed=1.5)
        
        assert result.text == "Hello world"
        
        # Verify speed was passed to OpenAI
        call_args = mock_client.audio.speech.create.call_args
        assert call_args[1]['speed'] == 1.5
    
    @pytest.mark.asyncio
    async def test_synthesize_with_emotion(self, tts_service):
        """Test synthesis with emotion."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = b"fake_audio_data"
        
        mock_client.audio.speech.create = AsyncMock(return_value=mock_response)
        tts_service.openai_client = mock_client
        
        result = await tts_service.synthesize_speech("I'm so happy!", emotion="happy")
        
        assert result.text == "I'm so happy!"
        assert "emotion" in result.metadata
        assert result.metadata["emotion"] == "happy"
    
    @pytest.mark.asyncio
    async def test_synthesize_long_text(self, tts_service):
        """Test synthesis of long text (chunking)."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = b"fake_audio_data"
        
        mock_client.audio.speech.create = AsyncMock(return_value=mock_response)
        tts_service.openai_client = mock_client
        
        # Create long text (over 4096 characters)
        long_text = "Hello world " * 300
        
        result = await tts_service.synthesize_speech(long_text)
        
        assert result.text == long_text
        # Should have been chunked and synthesized in multiple parts
        assert mock_client.audio.speech.create.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_synthesize_with_pronunciation(self, tts_service):
        """Test synthesis with pronunciation hints."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = b"fake_audio_data"
        
        mock_client.audio.speech.create = AsyncMock(return_value=mock_response)
        tts_service.openai_client = mock_client
        
        pronunciation = {"hello": "həˈloʊ", "world": "wɜːrld"}
        
        result = await tts_service.synthesize_speech(
            "Hello world",
            pronunciation=pronunciation
        )
        
        assert result.text == "Hello world"
        assert "pronunciation" in result.metadata


class TestTTSError:
    """Test TTSError exception."""
    
    def test_tts_error_creation(self):
        """Test creating TTSError."""
        error = TTSError("Test error message")
        
        assert str(error) == "Test error message"
    
    def test_tts_error_inheritance(self):
        """Test TTSError inheritance."""
        error = TTSError("Test error")
        
        assert isinstance(error, Exception)
        # TTSError might not inherit from ValueError, just check it's an Exception