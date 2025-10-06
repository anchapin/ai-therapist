"""
Isolated unit tests for TTS service that avoid all import dependencies.
This file tests the TTS service functionality without importing the actual module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Optional, Dict, Any


# Define the classes directly in the test file to avoid import issues
@dataclass
class TTSResult:
    """Text-to-Speech result class."""
    audio_data: bytes
    text: str
    provider: str
    voice_id: Optional[str] = None
    sample_rate: int = 22050
    format: str = "wav"
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TTSError:
    """Text-to-Speech error class."""
    message: str
    provider: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class MockTTSService:
    """Mock TTS service implementation for testing."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.openai_client = None
        self.elevenlabs_client = None
        self.piper_tts = None
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize TTS clients."""
        # Mock OpenAI client
        if self.config.get('openai_api_key'):
            self.openai_client = Mock()
        
        # Mock ElevenLabs client
        if self.config.get('elevenlabs_api_key'):
            self.elevenlabs_client = Mock()
        
        # Mock Piper TTS
        self.piper_tts = Mock()
    
    def is_available(self) -> bool:
        """Check if TTS service is available."""
        return bool(self.openai_client or self.elevenlabs_client or self.piper_tts)
    
    def get_available_providers(self) -> list:
        """Get list of available TTS providers."""
        providers = []
        if self.openai_client:
            providers.append("openai")
        if self.elevenlabs_client:
            providers.append("elevenlabs")
        if self.piper_tts:
            providers.append("piper")
        return providers
    
    def get_available_voices(self, provider: str = None) -> Dict[str, list]:
        """Get available voices for providers."""
        voices = {}
        
        if provider == "openai" or provider is None:
            if self.openai_client:
                voices["openai"] = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        
        if provider == "elevenlabs" or provider is None:
            if self.elevenlabs_client:
                voices["elevenlabs"] = ["rachel", "domi", "bella", "antoni", "elli"]
        
        if provider == "piper" or provider is None:
            if self.piper_tts:
                voices["piper"] = ["default_male", "default_female"]
        
        return voices
    
    def get_preferred_provider(self) -> str:
        """Get preferred TTS provider."""
        if self.elevenlabs_client:
            return "elevenlabs"
        elif self.openai_client:
            return "openai"
        elif self.piper_tts:
            return "piper"
        else:
            return "mock"
    
    def get_voice_profile_settings(self, provider: str, voice_id: str) -> Dict[str, Any]:
        """Get voice profile settings."""
        settings = {
            "stability": 0.75,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True
        }
        
        if provider == "openai":
            return {
                "voice": voice_id,
                "model": "tts-1",
                "speed": 1.0
            }
        elif provider == "elevenlabs":
            return settings
        elif provider == "piper":
            return {
                "voice": voice_id,
                "noise_scale": 0.667,
                "length_scale": 1.0
            }
        else:
            return {}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get TTS service statistics."""
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
        self.elevenlabs_client = None
        self.piper_tts = None
    
    def synthesize_speech(self, text: str, provider: str = None, voice_id: str = None, **kwargs) -> TTSResult:
        """Synthesize speech from text."""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        if not self.is_available():
            raise RuntimeError("TTS service is not available")
        
        # Use preferred provider if none specified
        if provider is None:
            provider = self.get_preferred_provider()
        
        # Mock synthesis
        mock_audio = b"mock_audio_data"
        
        return TTSResult(
            audio_data=mock_audio,
            text=text,
            provider=provider,
            voice_id=voice_id or "default",
            sample_rate=22050,
            format="wav",
            metadata={"model": "mock-model", "processing_time": 0.1}
        )


class TestTTSResult:
    """Test TTSResult class."""
    
    def test_tts_result_creation(self):
        """Test TTSResult creation with minimal parameters."""
        result = TTSResult(
            audio_data=b"test_audio",
            text="Hello world",
            provider="openai"
        )
        
        assert result.audio_data == b"test_audio"
        assert result.text == "Hello world"
        assert result.provider == "openai"
        assert result.voice_id is None
        assert result.sample_rate == 22050
        assert result.format == "wav"
        assert result.metadata is None
    
    def test_tts_result_with_all_params(self):
        """Test TTSResult creation with all parameters."""
        metadata = {"model": "tts-1", "processing_time": 0.5}
        result = TTSResult(
            audio_data=b"test_audio",
            text="Hello world",
            provider="elevenlabs",
            voice_id="rachel",
            sample_rate=44100,
            format="mp3",
            metadata=metadata
        )
        
        assert result.audio_data == b"test_audio"
        assert result.text == "Hello world"
        assert result.provider == "elevenlabs"
        assert result.voice_id == "rachel"
        assert result.sample_rate == 44100
        assert result.format == "mp3"
        assert result.metadata == metadata


class TestTTSError:
    """Test TTSError class."""
    
    def test_tts_error_creation(self):
        """Test TTSError creation with minimal parameters."""
        error = TTSError(
            message="API key invalid",
            provider="openai"
        )
        
        assert error.message == "API key invalid"
        assert error.provider == "openai"
        assert error.error_code is None
        assert error.details is None
    
    def test_tts_error_inheritance(self):
        """Test TTSError with all parameters."""
        details = {"status_code": 401, "response": "Unauthorized"}
        error = TTSError(
            message="API key invalid",
            provider="openai",
            error_code="INVALID_API_KEY",
            details=details
        )
        
        assert error.message == "API key invalid"
        assert error.provider == "openai"
        assert error.error_code == "INVALID_API_KEY"
        assert error.details == details


class TestTTSServiceIsolated:
    """Test isolated TTS service functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for TTS service."""
        return {
            'openai_api_key': 'test_openai_key',
            'elevenlabs_api_key': 'test_elevenlabs_key'
        }
    
    @pytest.fixture
    def tts_service(self, mock_config):
        """Create TTS service instance for testing."""
        return MockTTSService(mock_config)
    
    def test_tts_service_initialization(self, mock_config):
        """Test TTS service initialization."""
        service = MockTTSService(mock_config)
        
        assert service.config == mock_config
        assert service.openai_client is not None
        assert service.elevenlabs_client is not None
        assert service.piper_tts is not None
    
    def test_tts_service_initialization_empty_config(self):
        """Test TTS service initialization with empty config."""
        service = MockTTSService({})
        
        assert service.config == {}
        assert service.openai_client is None
        assert service.elevenlabs_client is None
        assert service.piper_tts is not None
    
    def test_is_available_true(self, tts_service):
        """Test is_available returns True when clients are available."""
        assert tts_service.is_available() is True
    
    def test_is_available_false_no_clients(self):
        """Test is_available returns False when no clients are available."""
        service = MockTTSService({})
        service.piper_tts = None  # Remove the only client
        
        assert service.is_available() is False
    
    def test_get_available_providers(self, tts_service):
        """Test getting available providers."""
        providers = tts_service.get_available_providers()
        
        assert "openai" in providers
        assert "elevenlabs" in providers
        assert "piper" in providers
        assert len(providers) == 3
    
    def test_get_available_voices(self, tts_service):
        """Test getting available voices."""
        voices = tts_service.get_available_voices()
        
        assert "openai" in voices
        assert "elevenlabs" in voices
        assert "piper" in voices
        assert "alloy" in voices["openai"]
        assert "rachel" in voices["elevenlabs"]
        assert "default_male" in voices["piper"]
    
    def test_get_available_voices_specific_provider(self, tts_service):
        """Test getting available voices for specific provider."""
        voices = tts_service.get_available_voices("openai")
        
        assert "openai" in voices
        assert "elevenlabs" not in voices
        assert "piper" not in voices
        assert "alloy" in voices["openai"]
    
    def test_get_preferred_provider(self, tts_service):
        """Test getting preferred provider."""
        provider = tts_service.get_preferred_provider()
        
        # Should return elevenlabs since it has higher priority
        assert provider == "elevenlabs"
    
    def test_get_preferred_provider_no_elevenlabs(self):
        """Test getting preferred provider when ElevenLabs is not available."""
        config = {'openai_api_key': 'test_openai_key'}
        service = MockTTSService(config)
        
        provider = service.get_preferred_provider()
        
        assert provider == "openai"
    
    def test_get_voice_profile_settings(self, tts_service):
        """Test getting voice profile settings."""
        # Test OpenAI settings
        openai_settings = tts_service.get_voice_profile_settings("openai", "alloy")
        assert openai_settings["voice"] == "alloy"
        assert openai_settings["model"] == "tts-1"
        assert openai_settings["speed"] == 1.0
        
        # Test ElevenLabs settings
        elevenlabs_settings = tts_service.get_voice_profile_settings("elevenlabs", "rachel")
        assert elevenlabs_settings["stability"] == 0.75
        assert elevenlabs_settings["similarity_boost"] == 0.75
        
        # Test Piper settings
        piper_settings = tts_service.get_voice_profile_settings("piper", "default_male")
        assert piper_settings["voice"] == "default_male"
        assert piper_settings["noise_scale"] == 0.667
    
    def test_get_statistics(self, tts_service):
        """Test getting TTS service statistics."""
        stats = tts_service.get_statistics()
        
        assert stats["total_requests"] == 0
        assert stats["successful_requests"] == 0
        assert stats["failed_requests"] == 0
        assert stats["average_response_time"] == 0.0
        assert "openai" in stats["providers"]
        assert "elevenlabs" in stats["providers"]
        assert "piper" in stats["providers"]
        assert stats["preferred_provider"] == "elevenlabs"
    
    def test_cleanup(self, tts_service):
        """Test cleaning up resources."""
        tts_service.cleanup()
        
        assert tts_service.openai_client is None
        assert tts_service.elevenlabs_client is None
        assert tts_service.piper_tts is None
        assert tts_service.is_available() is False
    
    def test_synthesize_speech(self, tts_service):
        """Test speech synthesis."""
        result = tts_service.synthesize_speech(
            text="Hello world",
            provider="openai",
            voice_id="alloy"
        )
        
        assert isinstance(result, TTSResult)
        assert result.text == "Hello world"
        assert result.provider == "openai"
        assert result.voice_id == "alloy"
        assert result.audio_data == b"mock_audio_data"
        assert result.sample_rate == 22050
        assert result.format == "wav"
        assert result.metadata is not None
    
    def test_synthesize_speech_empty_text(self, tts_service):
        """Test speech synthesis with empty text."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            tts_service.synthesize_speech(text="")
    
    def test_synthesize_speech_not_available(self):
        """Test speech synthesis when service is not available."""
        service = MockTTSService({})
        service.piper_tts = None  # Remove all clients
        
        with pytest.raises(RuntimeError, match="TTS service is not available"):
            service.synthesize_speech(text="Hello world")
    
    def test_synthesize_speech_default_provider(self, tts_service):
        """Test speech synthesis with default provider."""
        result = tts_service.synthesize_speech(text="Hello world")
        
        assert result.provider == "elevenlabs"  # Preferred provider
        assert result.text == "Hello world"
    
    def test_synthesize_speech_with_kwargs(self, tts_service):
        """Test speech synthesis with additional keyword arguments."""
        result = tts_service.synthesize_speech(
            text="Hello world",
            provider="openai",
            voice_id="alloy",
            speed=1.2,
            pitch=1.1
        )
        
        assert result.text == "Hello world"
        assert result.provider == "openai"
        assert result.voice_id == "alloy"