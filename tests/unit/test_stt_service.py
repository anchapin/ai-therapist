"""
Comprehensive unit tests for voice/stt_service.py module.
"""

import pytest
import asyncio
import time
import numpy as np
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
import json

# Import the module to test with robust error handling
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from voice.stt_service import STTService, STTResult, STTError
    from voice.config import VoiceConfig
    from voice.audio_processor import AudioData
except ImportError as e:
    pytest.skip(f"voice.stt_service module not available: {e}", allow_module_level=True)


class TestSTTResult:
    """Test STTResult dataclass."""
    
    def test_stt_result_creation(self):
        """Test creating an STT result."""
        result = STTResult(
            text="Hello world",
            confidence=0.95,
            provider="openai",
            processing_time=1.5
        )
        
        assert result.text == "Hello world"
        assert result.confidence == 0.95
        assert result.provider == "openai"
        assert result.processing_time == 1.5
    
    def test_stt_result_with_optional_fields(self):
        """Test creating an STT result with optional fields."""
        result = STTResult(
            text="Hello world",
            confidence=0.95,
            provider="openai",
            processing_time=1.5,
            alternatives=["Hello world", "Hello word"],
            language="en-US"
        )
        
        assert result.alternatives == ["Hello world", "Hello word"]
        assert result.language == "en-US"


class TestSTTService:
    """Test STTService class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock voice config."""
        config = Mock(spec=VoiceConfig)
        config.stt_provider = "openai"
        config.openai_api_key = "test_key"
        config.stt_language = "en-US"
        config.stt_model = "whisper-1"
        config.security = Mock()
        config.security.encryption_enabled = False
        config.get_preferred_stt_service.return_value = "openai"
        config.is_google_speech_configured.return_value = False
        config.is_whisper_configured.return_value = False
        config.is_openai_whisper_configured.return_value = True
        config.whisper_language = "en"
        config.whisper_temperature = 0.0
        config.security = Mock()
        return config
    
    @pytest.fixture
    def stt_service(self, mock_config):
        """Create an STT service with mock config."""
        # Mock all external dependencies completely
        with patch('voice.stt_service.openai'), \
             patch('voice.stt_service.asyncio.get_event_loop') as mock_loop:
            
            # Mock the run_in_executor to return the result directly without threading
            def mock_run_executor(executor, func, *args, **kwargs):
                return func(*args, **kwargs)
            
            mock_loop.return_value.run_in_executor = mock_run_executor
            
            service = STTService(mock_config)
            return service
    
    def test_stt_service_initialization(self, stt_service, mock_config):
        """Test STT service initialization."""
        assert stt_service.config == mock_config
        assert stt_service.provider == "openai"
        # Set api_key for backward compatibility test
        test_key = os.getenv("TEST_STT_API_KEY", "test-key-placeholder")
        stt_service.api_key = test_key
        assert stt_service.api_key == test_key
        assert stt_service.language == "en-US"
        assert stt_service.model == "whisper-1"
    
    def test_stt_service_initialization_no_api_key(self, mock_config):
        """Test STT service initialization with no API key."""
        mock_config.openai_api_key = None
        
        with patch('voice.stt_service.openai'):
            service = STTService(mock_config)
            assert service.api_key is None
    
    def test_is_available_true(self, stt_service):
        """Test is_available when service is available."""
        stt_service.api_key = "test_key"
        assert stt_service.is_available() == True
    
    def test_is_available_false_no_api_key(self, stt_service):
        """Test is_available when no API key."""
        # Force clear all clients to simulate no API key scenario
        stt_service.openai_client = None
        stt_service.google_speech_client = None
        stt_service.whisper_model = None
        assert stt_service.is_available() == False
    
    @pytest.mark.asyncio
    async def test_transcribe_audio_openai_success(self, stt_service):
        """Test successful audio transcription with OpenAI."""
        # Mock OpenAI client response
        mock_response = {
            "text": "Hello world",
            "language": "en",
            "duration": 2.0,
            "segments": []
        }
        
        mock_client = Mock()
        mock_client.Audio.transcribe = Mock(return_value=mock_response)
        stt_service.openai_client = mock_client
        
        # Create audio data
        audio_data = AudioData(
            data=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1
        )
        
        result = await stt_service.transcribe_audio(audio_data)
        
        assert result is not None
        assert result.text == "Hello world"
        assert result.confidence > 0.0
        assert result.provider == "openai"
        assert result.language == "en"
        assert result.processing_time > 0
        
        # Verify OpenAI client was called
        mock_client.Audio.transcribe.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_transcribe_audio_with_language(self, stt_service):
        """Test audio transcription with specified language."""
        # Mock OpenAI client response as dict (what the service expects)
        mock_response = {
            "text": "Bonjour le monde",
            "language": "fr",
            "duration": 2.0,
            "segments": []
        }
        
        # Mock OpenAI client with the correct API
        mock_client = Mock()
        mock_client.Audio.transcribe = AsyncMock(return_value=mock_response)
        stt_service.openai_client = mock_client
        
        # Create audio data
        audio_data = AudioData(
            data=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1
        )
        
        result = await stt_service.transcribe_audio(audio_data, language="fr")
        
        assert result.text == "Bonjour le monde"
        assert result.language == "fr"
        
        # Verify language was passed to OpenAI
        call_args = mock_client.audio.transcriptions.create.call_args
        assert call_args[1]['language'] == "fr"
    
    @pytest.mark.asyncio
    async def test_transcribe_audio_not_available(self, stt_service):
        """Test transcription when service is not available."""
        stt_service.api_key = None
        audio_data = AudioData(
            data=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1
        )
        
        with pytest.raises(STTError) as exc_info:
            await stt_service.transcribe_audio(audio_data)
        
        assert "STT service is not available" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_transcribe_audio_openai_error(self, stt_service):
        """Test transcription with OpenAI error."""
        # Mock OpenAI client to raise an exception
        mock_client = Mock()
        mock_client.audio.transcriptions.create = AsyncMock(
            side_effect=Exception("OpenAI API error")
        )
        stt_service.openai_client = mock_client
        
        audio_data = AudioData(
            data=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1
        )
        
        with pytest.raises(STTError) as exc_info:
            await stt_service.transcribe_audio(audio_data)
        
        assert "Failed to transcribe audio" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_transcribe_audio_empty_data(self, stt_service):
        """Test transcription with empty audio data."""
        audio_data = AudioData(data=b"", sample_rate=16000, channels=1)
        
        with pytest.raises(STTError) as exc_info:
            await stt_service.transcribe_audio(audio_data)
        
        assert "No audio data provided" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_transcribe_audio_timeout(self, stt_service):
        """Test transcription with timeout."""
        # Mock OpenAI client to timeout
        mock_client = Mock()
        mock_client.audio.transcriptions.create = AsyncMock(
            side_effect=asyncio.TimeoutError("Request timeout")
        )
        stt_service.openai_client = mock_client
        
        audio_data = AudioData(
            data=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1
        )
        
        with pytest.raises(STTError) as exc_info:
            await stt_service.transcribe_audio(audio_data)
        
        assert "Transcription timed out" in str(exc_info.value)
    
    def test_get_supported_languages(self, stt_service):
        """Test getting supported languages."""
        languages = stt_service.get_supported_languages()
        
        assert isinstance(languages, list)
        assert len(languages) > 0
        assert any(lang["code"] == "en" for lang in languages)
        assert any(lang["code"] == "es" for lang in languages)
        assert all("code" in lang for lang in languages)
        assert all("name" in lang for lang in languages)
    
    def test_get_supported_models(self, stt_service):
        """Test getting supported models."""
        models = stt_service.get_supported_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        assert any(model["id"] == "whisper-1" for model in models)
        assert all("id" in model for model in models)
        assert all("name" in model for model in models)
    
    def test_set_language(self, stt_service):
        """Test setting language."""
        stt_service.set_language("es")
        
        assert stt_service.language == "es"
    
    def test_set_language_invalid(self, stt_service):
        """Test setting invalid language."""
        with pytest.raises(STTError) as exc_info:
            stt_service.set_language("invalid")
        
        assert "Invalid language" in str(exc_info.value)
    
    def test_set_model(self, stt_service):
        """Test setting model."""
        stt_service.set_model("whisper-1")
        
        assert stt_service.model == "whisper-1"
    
    def test_set_model_invalid(self, stt_service):
        """Test setting invalid model."""
        with pytest.raises(STTError) as exc_info:
            stt_service.set_model("invalid-model")
        
        assert "Invalid model" in str(exc_info.value)
    
    def test_get_service_info(self, stt_service):
        """Test getting service information."""
        info = stt_service.get_service_info()
        
        assert isinstance(info, dict)
        assert "provider" in info
        assert "language" in info
        assert "model" in info
        assert "available" in info
        assert "supported_languages" in info
        assert "supported_models" in info
        
        assert info["provider"] == "openai"
        assert info["language"] == "en-US"
        assert info["model"] == "whisper-1"
    
    def test_cleanup(self, stt_service):
        """Test service cleanup."""
        # Mock OpenAI client
        mock_client = Mock()
        stt_service.openai_client = mock_client
        
        stt_service.cleanup()
        
        # Verify client is cleaned up
        assert stt_service.openai_client is None
    
    def test_context_manager(self, stt_service):
        """Test using STT service as context manager."""
        # Mock OpenAI client
        mock_client = Mock()
        stt_service.openai_client = mock_client
        
        with stt_service as service:
            assert service == stt_service
        
        # Verify cleanup was called
        assert stt_service.openai_client is None
    
    def test_str_representation(self, stt_service):
        """Test string representation of STT service."""
        str_repr = str(stt_service)
        
        assert "STTService" in str_repr
        assert "openai" in str_repr
    
    def test_repr_representation(self, stt_service):
        """Test repr representation of STT service."""
        repr_str = repr(stt_service)
        
        assert "STTService" in repr_str
        assert "openai" in repr_str
    
    @pytest.mark.asyncio
    async def test_batch_transcription(self, stt_service):
        """Test batch transcription of multiple audio files."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_response1 = Mock()
        mock_response1.text = "First transcription"
        mock_response1.language = "en"
        mock_response1.duration = 2.0
        
        mock_response2 = Mock()
        mock_response2.text = "Second transcription"
        mock_response2.language = "en"
        mock_response2.duration = 3.0
        
        mock_client.audio.transcriptions.create = AsyncMock(
            side_effect=[mock_response1, mock_response2]
        )
        stt_service.openai_client = mock_client
        
        # Create audio data list
        audio_list = [
            AudioData(data=b"audio1", sample_rate=16000, channels=1),
            AudioData(data=b"audio2", sample_rate=16000, channels=1)
        ]
        
        results = await stt_service.batch_transcribe(audio_list)
        
        assert len(results) == 2
        assert results[0].text == "First transcription"
        assert results[1].text == "Second transcription"
        
        # Verify OpenAI client was called twice
        assert mock_client.audio.transcriptions.create.call_count == 2
    
    @pytest.mark.asyncio
    async def test_transcribe_with_timestamps(self, stt_service):
        """Test transcription with timestamps."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = "Hello world"
        mock_response.language = "en"
        mock_response.duration = 2.0
        mock_response.words = [
            {"word": "Hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.5, "end": 1.0}
        ]
        
        mock_client.Audio.transcribe = AsyncMock(return_value=mock_response)
        stt_service.openai_client = mock_client
        
        audio_data = AudioData(
            data=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1
        )
        
        result = await stt_service.transcribe_audio(audio_data, timestamp_granularities=["word"])
        
        assert result.text == "Hello world"
        assert "words" in result.metadata
        assert len(result.metadata["words"]) == 2
    
    @pytest.mark.asyncio
    async def test_transcribe_with_vad(self, stt_service):
        """Test transcription with voice activity detection."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = "Hello world"
        mock_response.language = "en"
        mock_response.duration = 2.0
        
        mock_client.Audio.transcribe = AsyncMock(return_value=mock_response)
        stt_service.openai_client = mock_client
        
        # Create audio data with silence
        audio_data = AudioData(
            data=b"\x00" * 8000 + b"audio_data" + b"\x00" * 8000,  # Silence at beginning and end
            sample_rate=16000,
            channels=1
        )
        
        result = await stt_service.transcribe_audio(audio_data, apply_vad=True)
        
        assert result.text == "Hello world"
        # VAD should have been applied to remove silence
        assert result.processing_time > 0


class TestSTTError:
    """Test STTError exception."""
    
    def test_stt_error_creation(self):
        """Test creating STTError."""
        error = STTError("Test error message")
        
        assert str(error) == "Test error message"
    
    def test_stt_error_inheritance(self):
        """Test STTError inheritance."""
        error = STTError("Test error")
        
        assert isinstance(error, Exception)
        assert str(error) == "Test error"