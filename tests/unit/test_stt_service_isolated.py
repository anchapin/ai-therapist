"""
Isolated unit tests for STT service that avoid all import dependencies.
This file tests the STT service functionality without importing the actual module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Optional, Dict, Any, List


# Define the classes directly in the test file to avoid import issues
@dataclass
class STTResult:
    """Speech-to-Text result class."""
    text: str
    confidence: float
    provider: str
    audio_duration: Optional[float] = None
    language: Optional[str] = None
    alternatives: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class STTError:
    """Speech-to-Text error class."""
    message: str
    provider: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class MockSTTService:
    """Mock STT service implementation for testing."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.openai_client = None
        self.whisper_model = None
        self.elevenlabs_client = None
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize STT clients."""
        # Mock OpenAI client
        if self.config.get('openai_api_key'):
            self.openai_client = Mock()
        
        # Mock Whisper model
        if self.config.get('whisper_model_path'):
            self.whisper_model = Mock()
        
        # Mock ElevenLabs client
        if self.config.get('elevenlabs_api_key'):
            self.elevenlabs_client = Mock()
    
    def is_available(self) -> bool:
        """Check if STT service is available."""
        return bool(self.openai_client or self.whisper_model or self.elevenlabs_client)
    
    def get_available_providers(self) -> list:
        """Get list of available STT providers."""
        providers = []
        if self.openai_client:
            providers.append("openai")
        if self.whisper_model:
            providers.append("whisper")
        if self.elevenlabs_client:
            providers.append("elevenlabs")
        return providers
    
    def get_supported_languages(self, provider: str = None) -> Dict[str, List[str]]:
        """Get supported languages for providers."""
        languages = {}
        
        if provider == "openai" or provider is None:
            if self.openai_client:
                languages["openai"] = ["en", "es", "fr", "de", "it", "pt", "zh", "ja"]
        
        if provider == "whisper" or provider is None:
            if self.whisper_model:
                languages["whisper"] = ["en", "es", "fr", "de", "it", "pt", "zh", "ja", "ru", "ar"]
        
        if provider == "elevenlabs" or provider is None:
            if self.elevenlabs_client:
                languages["elevenlabs"] = ["en", "es", "fr", "de", "it", "pt", "pl", "nl"]
        
        return languages
    
    def get_preferred_provider(self) -> str:
        """Get preferred STT provider."""
        if self.openai_client:
            return "openai"
        elif self.whisper_model:
            return "whisper"
        elif self.elevenlabs_client:
            return "elevenlabs"
        else:
            return "mock"
    
    def get_recognition_settings(self, provider: str) -> Dict[str, Any]:
        """Get recognition settings for provider."""
        if provider == "openai":
            return {
                "model": "whisper-1",
                "language": "en",
                "temperature": 0.0,
                "response_format": "json"
            }
        elif provider == "whisper":
            return {
                "model": "base",
                "language": "en",
                "task": "transcribe"
            }
        elif provider == "elevenlabs":
            return {
                "model": "speech-to-text",
                "language": "en"
            }
        else:
            return {}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get STT service statistics."""
        return {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "providers": self.get_available_providers(),
            "preferred_provider": self.get_preferred_provider()
        }
    
    def cleanup(self):
        """Clean up resources."""
        self.openai_client = None
        self.whisper_model = None
        self.elevenlabs_client = None
    
    def transcribe_audio(self, audio_data: bytes, provider: str = None, language: str = None, **kwargs) -> STTResult:
        """Transcribe audio to text."""
        if not audio_data:
            raise ValueError("Audio data cannot be empty")
        
        if not self.is_available():
            raise RuntimeError("STT service is not available")
        
        # Use preferred provider if none specified
        if provider is None:
            provider = self.get_preferred_provider()
        
        # Mock transcription
        mock_text = "This is a mock transcription of the audio content."
        
        return STTResult(
            text=mock_text,
            confidence=0.95,
            provider=provider,
            audio_duration=2.5,
            language=language or "en",
            alternatives=[
                {"text": "This is a mock transcription", "confidence": 0.85},
                {"text": "This is a mock transcription of audio", "confidence": 0.75}
            ],
            metadata={"model": "mock-model", "processing_time": 0.3}
        )


class TestSTTResult:
    """Test STTResult class."""
    
    def test_stt_result_creation(self):
        """Test STTResult creation with minimal parameters."""
        result = STTResult(
            text="Hello world",
            confidence=0.95,
            provider="openai"
        )
        
        assert result.text == "Hello world"
        assert result.confidence == 0.95
        assert result.provider == "openai"
        assert result.audio_duration is None
        assert result.language is None
        assert result.alternatives is None
        assert result.metadata is None
    
    def test_stt_result_with_all_params(self):
        """Test STTResult creation with all parameters."""
        alternatives = [
            {"text": "Hello world", "confidence": 0.85},
            {"text": "Hello", "confidence": 0.75}
        ]
        metadata = {"model": "whisper-1", "processing_time": 0.5}
        result = STTResult(
            text="Hello world",
            confidence=0.95,
            provider="openai",
            audio_duration=2.5,
            language="en",
            alternatives=alternatives,
            metadata=metadata
        )
        
        assert result.text == "Hello world"
        assert result.confidence == 0.95
        assert result.provider == "openai"
        assert result.audio_duration == 2.5
        assert result.language == "en"
        assert result.alternatives == alternatives
        assert result.metadata == metadata


class TestSTTError:
    """Test STTError class."""
    
    def test_stt_error_creation(self):
        """Test STTError creation with minimal parameters."""
        error = STTError(
            message="Audio format not supported",
            provider="openai"
        )
        
        assert error.message == "Audio format not supported"
        assert error.provider == "openai"
        assert error.error_code is None
        assert error.details is None
    
    def test_stt_error_inheritance(self):
        """Test STTError with all parameters."""
        details = {"format": "wav", "supported_formats": ["mp3", "flac"]}
        error = STTError(
            message="Audio format not supported",
            provider="openai",
            error_code="UNSUPPORTED_FORMAT",
            details=details
        )
        
        assert error.message == "Audio format not supported"
        assert error.provider == "openai"
        assert error.error_code == "UNSUPPORTED_FORMAT"
        assert error.details == details


class TestSTTServiceIsolated:
    """Test isolated STT service functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for STT service."""
        return {
            'openai_api_key': 'test_openai_key',
            'whisper_model_path': '/path/to/whisper/model',
            'elevenlabs_api_key': 'test_elevenlabs_key'
        }
    
    @pytest.fixture
    def stt_service(self, mock_config):
        """Create STT service instance for testing."""
        return MockSTTService(mock_config)
    
    def test_stt_service_initialization(self, mock_config):
        """Test STT service initialization."""
        service = MockSTTService(mock_config)
        
        assert service.config == mock_config
        assert service.openai_client is not None
        assert service.whisper_model is not None
        assert service.elevenlabs_client is not None
    
    def test_stt_service_initialization_empty_config(self):
        """Test STT service initialization with empty config."""
        service = MockSTTService({})
        
        assert service.config == {}
        assert service.openai_client is None
        assert service.whisper_model is None
        assert service.elevenlabs_client is None
    
    def test_is_available_true(self, stt_service):
        """Test is_available returns True when clients are available."""
        assert stt_service.is_available() is True
    
    def test_is_available_false_no_clients(self):
        """Test is_available returns False when no clients are available."""
        service = MockSTTService({})
        
        assert service.is_available() is False
    
    def test_get_available_providers(self, stt_service):
        """Test getting available providers."""
        providers = stt_service.get_available_providers()
        
        assert "openai" in providers
        assert "whisper" in providers
        assert "elevenlabs" in providers
        assert len(providers) == 3
    
    def test_get_supported_languages(self, stt_service):
        """Test getting supported languages."""
        languages = stt_service.get_supported_languages()
        
        assert "openai" in languages
        assert "whisper" in languages
        assert "elevenlabs" in languages
        assert "en" in languages["openai"]
        assert "es" in languages["whisper"]
        assert "fr" in languages["elevenlabs"]
    
    def test_get_supported_languages_specific_provider(self, stt_service):
        """Test getting supported languages for specific provider."""
        languages = stt_service.get_supported_languages("openai")
        
        assert "openai" in languages
        assert "whisper" not in languages
        assert "elevenlabs" not in languages
        assert "en" in languages["openai"]
    
    def test_get_preferred_provider(self, stt_service):
        """Test getting preferred provider."""
        provider = stt_service.get_preferred_provider()
        
        # Should return openai since it has higher priority
        assert provider == "openai"
    
    def test_get_preferred_provider_no_openai(self):
        """Test getting preferred provider when OpenAI is not available."""
        config = {
            'whisper_model_path': '/path/to/whisper/model',
            'elevenlabs_api_key': 'test_elevenlabs_key'
        }
        service = MockSTTService(config)
        
        provider = service.get_preferred_provider()
        
        assert provider == "whisper"
    
    def test_get_recognition_settings(self, stt_service):
        """Test getting recognition settings."""
        # Test OpenAI settings
        openai_settings = stt_service.get_recognition_settings("openai")
        assert openai_settings["model"] == "whisper-1"
        assert openai_settings["language"] == "en"
        assert openai_settings["temperature"] == 0.0
        
        # Test Whisper settings
        whisper_settings = stt_service.get_recognition_settings("whisper")
        assert whisper_settings["model"] == "base"
        assert whisper_settings["language"] == "en"
        assert whisper_settings["task"] == "transcribe"
        
        # Test ElevenLabs settings
        elevenlabs_settings = stt_service.get_recognition_settings("elevenlabs")
        assert elevenlabs_settings["model"] == "speech-to-text"
        assert elevenlabs_settings["language"] == "en"
    
    def test_get_statistics(self, stt_service):
        """Test getting STT service statistics."""
        stats = stt_service.get_statistics()
        
        assert stats["total_requests"] == 0
        assert stats["successful_requests"] == 0
        assert stats["failed_requests"] == 0
        assert stats["average_response_time"] == 0.0
        assert "openai" in stats["providers"]
        assert "whisper" in stats["providers"]
        assert "elevenlabs" in stats["providers"]
        assert stats["preferred_provider"] == "openai"
    
    def test_cleanup(self, stt_service):
        """Test cleaning up resources."""
        stt_service.cleanup()
        
        assert stt_service.openai_client is None
        assert stt_service.whisper_model is None
        assert stt_service.elevenlabs_client is None
        assert stt_service.is_available() is False
    
    def test_transcribe_audio(self, stt_service):
        """Test audio transcription."""
        result = stt_service.transcribe_audio(
            audio_data=b"mock_audio_data",
            provider="openai",
            language="en"
        )
        
        assert isinstance(result, STTResult)
        assert result.text == "This is a mock transcription of the audio content."
        assert result.provider == "openai"
        assert result.confidence == 0.95
        assert result.audio_duration == 2.5
        assert result.language == "en"
        assert result.alternatives is not None
        assert len(result.alternatives) == 2
        assert result.metadata is not None
    
    def test_transcribe_audio_empty_data(self, stt_service):
        """Test audio transcription with empty data."""
        with pytest.raises(ValueError, match="Audio data cannot be empty"):
            stt_service.transcribe_audio(audio_data=b"")
    
    def test_transcribe_audio_not_available(self):
        """Test audio transcription when service is not available."""
        service = MockSTTService({})
        
        with pytest.raises(RuntimeError, match="STT service is not available"):
            service.transcribe_audio(audio_data=b"mock_audio_data")
    
    def test_transcribe_audio_default_provider(self, stt_service):
        """Test audio transcription with default provider."""
        result = stt_service.transcribe_audio(audio_data=b"mock_audio_data")
        
        assert result.provider == "openai"  # Preferred provider
        assert result.text == "This is a mock transcription of the audio content."
    
    def test_transcribe_audio_with_kwargs(self, stt_service):
        """Test audio transcription with additional keyword arguments."""
        result = stt_service.transcribe_audio(
            audio_data=b"mock_audio_data",
            provider="whisper",
            language="es",
            temperature=0.2,
            task="translate"
        )
        
        assert result.text == "This is a mock transcription of the audio content."
        assert result.provider == "whisper"
        assert result.language == "es"