"""
Comprehensive tests for STTService missing branch coverage.

This test file covers:
1. _get_provider_fallback_chain - behavior for missing/preferred providers
2. _calculate_audio_quality_score - branches: empty array, clipping, invalid values (nan/inf)
3. set_language - invalid language -> STTError
4. set_model - invalid model -> STTError
5. batch_transcribe - mixed success/error cases returns error STTResult entries
6. get_service_info/__str__/__repr__ - sanity assertions
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from voice.stt_service import STTService, STTResult, STTError
    from voice.config import VoiceConfig
    from voice.audio_processor import AudioData
except ImportError as e:
    pytest.skip(f"voice.stt_service module not available: {e}", allow_module_level=True)


class TestProviderFallbackChain:
    """Test _get_provider_fallback_chain method."""
    
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
        return config
    
    def test_fallback_chain_with_preferred_provider(self, mock_config):
        """Test fallback chain when preferred provider is available."""
        with patch('voice.stt_service.openai'):
            service = STTService(mock_config)
            service.openai_client = Mock()  # Simulate OpenAI available
            
            chain = service._get_provider_fallback_chain("openai")
            
            assert chain[0] == "openai"
            assert len(chain) == 1  # Only openai is available
    
    def test_fallback_chain_with_missing_preferred_provider(self, mock_config):
        """Test fallback chain when preferred provider is not available."""
        with patch('voice.stt_service.openai'):
            service = STTService(mock_config)
            service.openai_client = Mock()
            
            # Request a provider that doesn't exist
            chain = service._get_provider_fallback_chain("nonexistent")
            
            # Should fall back to default priority
            assert "openai" in chain
            assert "nonexistent" not in chain
    
    def test_fallback_chain_with_multiple_providers(self, mock_config):
        """Test fallback chain with multiple providers available."""
        mock_config.is_google_speech_configured.return_value = True
        mock_config.is_whisper_configured.return_value = True
        
        with patch('voice.stt_service.openai'), \
             patch('voice.stt_service.whisper'):
            service = STTService(mock_config)
            service.openai_client = Mock()
            service.whisper_model = Mock()
            service.google_speech_client = Mock()
            
            # Request google, should get google first, then others
            chain = service._get_provider_fallback_chain("google")
            
            assert chain[0] == "google"
            assert "openai" in chain
            assert "whisper" in chain
    
    def test_fallback_chain_with_no_preferred_provider(self, mock_config):
        """Test fallback chain when no preferred provider is specified."""
        with patch('voice.stt_service.openai'):
            service = STTService(mock_config)
            service.openai_client = Mock()
            
            chain = service._get_provider_fallback_chain(None)
            
            # Should use default priority order
            assert len(chain) >= 1
            assert "openai" in chain


class TestAudioQualityScore:
    """Test _calculate_audio_quality_score method."""
    
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
        return config
    
    def test_quality_score_empty_array(self, mock_config):
        """Test quality score calculation with empty audio array."""
        with patch('voice.stt_service.openai'):
            service = STTService(mock_config)
            
            # Create empty audio data
            audio_data = AudioData(
                data=np.array([], dtype=np.float32),
                sample_rate=16000,
                channels=1,
                format="float32",
                duration=0.0
            )
            
            score = service._calculate_audio_quality_score(audio_data)
            
            assert score == 0.0
    
    def test_quality_score_with_clipping(self, mock_config):
        """Test quality score calculation with clipping (high clipping ratio)."""
        with patch('voice.stt_service.openai'):
            service = STTService(mock_config)
            
            # Create audio data with clipping (values > 0.95)
            clipped_data = np.ones(1000, dtype=np.float32)  # All values at 1.0
            audio_data = AudioData(
                data=clipped_data,
                sample_rate=16000,
                channels=1,
                format="float32",
                duration=1.0
            )
            
            score = service._calculate_audio_quality_score(audio_data)
            
            # Score should be lower due to heavy clipping
            assert 0.0 <= score <= 1.0
            assert score < 0.7  # Should be penalized
    
    def test_quality_score_with_nan_values(self, mock_config):
        """Test quality score calculation with NaN values."""
        with patch('voice.stt_service.openai'):
            service = STTService(mock_config)
            
            # Create audio data with NaN values
            nan_data = np.array([0.1, 0.2, np.nan, 0.4, np.nan], dtype=np.float32)
            audio_data = AudioData(
                data=nan_data,
                sample_rate=16000,
                channels=1,
                format="float32",
                duration=1.0
            )
            
            # Should not raise an exception, should handle gracefully
            score = service._calculate_audio_quality_score(audio_data)
            
            assert 0.0 <= score <= 1.0
    
    def test_quality_score_with_inf_values(self, mock_config):
        """Test quality score calculation with inf values."""
        with patch('voice.stt_service.openai'):
            service = STTService(mock_config)
            
            # Create audio data with inf values
            inf_data = np.array([0.1, 0.2, np.inf, 0.4, -np.inf], dtype=np.float32)
            audio_data = AudioData(
                data=inf_data,
                sample_rate=16000,
                channels=1,
                format="float32",
                duration=1.0
            )
            
            # Should not raise an exception, should handle gracefully
            score = service._calculate_audio_quality_score(audio_data)
            
            assert 0.0 <= score <= 1.0
    
    def test_quality_score_normal_audio(self, mock_config):
        """Test quality score calculation with normal audio."""
        with patch('voice.stt_service.openai'):
            service = STTService(mock_config)
            
            # Create normal audio data
            normal_data = np.random.uniform(-0.5, 0.5, 1000).astype(np.float32)
            audio_data = AudioData(
                data=normal_data,
                sample_rate=16000,
                channels=1,
                format="float32",
                duration=1.0
            )
            
            score = service._calculate_audio_quality_score(audio_data)
            
            assert 0.0 <= score <= 1.0
            assert score > 0.3  # Should have reasonable quality


class TestSetLanguage:
    """Test set_language method."""
    
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
        return config
    
    def test_set_language_valid(self, mock_config):
        """Test setting a valid language."""
        with patch('voice.stt_service.openai'):
            service = STTService(mock_config)
            
            # Set to a supported language
            service.set_language("es")
            
            assert service.language == "es"
    
    def test_set_language_invalid(self, mock_config):
        """Test setting an invalid language raises STTError."""
        with patch('voice.stt_service.openai'):
            service = STTService(mock_config)
            
            # Should raise STTError for invalid language
            with pytest.raises(STTError) as exc_info:
                service.set_language("invalid_lang")
            
            assert "Invalid language" in str(exc_info.value)
    
    def test_set_language_empty_string(self, mock_config):
        """Test setting empty string as language raises STTError."""
        with patch('voice.stt_service.openai'):
            service = STTService(mock_config)
            
            with pytest.raises(STTError) as exc_info:
                service.set_language("")
            
            assert "Invalid language" in str(exc_info.value)


class TestSetModel:
    """Test set_model method."""
    
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
        return config
    
    def test_set_model_valid(self, mock_config):
        """Test setting a valid model."""
        with patch('voice.stt_service.openai'):
            service = STTService(mock_config)
            service.openai_client = Mock()  # Make sure openai is available
            
            # Set to a supported model
            service.set_model("whisper-1")
            
            assert service.model == "whisper-1"
    
    def test_set_model_invalid(self, mock_config):
        """Test setting an invalid model raises STTError."""
        with patch('voice.stt_service.openai'):
            service = STTService(mock_config)
            
            # Should raise STTError for invalid model
            with pytest.raises(STTError) as exc_info:
                service.set_model("invalid_model")
            
            assert "Invalid model" in str(exc_info.value)
    
    def test_set_model_empty_string(self, mock_config):
        """Test setting empty string as model raises STTError."""
        with patch('voice.stt_service.openai'):
            service = STTService(mock_config)
            
            with pytest.raises(STTError) as exc_info:
                service.set_model("")
            
            assert "Invalid model" in str(exc_info.value)


class TestBatchTranscribe:
    """Test batch_transcribe method."""
    
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
        return config
    
    @pytest.mark.asyncio
    async def test_batch_transcribe_all_success(self, mock_config):
        """Test batch transcribe with all successes."""
        with patch('voice.stt_service.openai'):
            service = STTService(mock_config)
            
            # Mock successful transcription
            async def mock_transcribe(audio_data, **kwargs):
                return STTResult(
                    text="Transcribed text",
                    confidence=0.95,
                    provider="openai",
                    language="en",
                    duration=1.0
                )
            
            service.transcribe_audio = mock_transcribe
            
            # Create audio data list
            audio_list = [
                AudioData(
                    data=np.array([0.1, 0.2], dtype=np.float32),
                    sample_rate=16000,
                    channels=1,
                    format="float32",
                    duration=1.0
                ),
                AudioData(
                    data=np.array([0.3, 0.4], dtype=np.float32),
                    sample_rate=16000,
                    channels=1,
                    format="float32",
                    duration=1.0
                )
            ]
            
            results = await service.batch_transcribe(audio_list)
            
            assert len(results) == 2
            assert all(r.text == "Transcribed text" for r in results)
            assert all(r.confidence == 0.95 for r in results)
    
    @pytest.mark.asyncio
    async def test_batch_transcribe_mixed_success_error(self, mock_config):
        """Test batch transcribe with mixed success and error cases."""
        with patch('voice.stt_service.openai'):
            service = STTService(mock_config)
            
            # Mock transcription that alternates success and failure
            call_count = [0]
            
            async def mock_transcribe(audio_data, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    return STTResult(
                        text="Success",
                        confidence=0.95,
                        provider="openai",
                        language="en",
                        duration=1.0
                    )
                else:
                    raise Exception("Transcription failed")
            
            service.transcribe_audio = mock_transcribe
            
            # Create audio data list
            audio_list = [
                AudioData(
                    data=np.array([0.1, 0.2], dtype=np.float32),
                    sample_rate=16000,
                    channels=1,
                    format="float32",
                    duration=1.0
                ),
                AudioData(
                    data=np.array([0.3, 0.4], dtype=np.float32),
                    sample_rate=16000,
                    channels=1,
                    format="float32",
                    duration=1.0
                )
            ]
            
            results = await service.batch_transcribe(audio_list)
            
            assert len(results) == 2
            # First should succeed
            assert results[0].text == "Success"
            assert results[0].confidence == 0.95
            # Second should be an error result
            assert results[1].text == ""
            assert results[1].confidence == 0.0
            assert results[1].provider == "error"
            assert results[1].error is not None
            assert "Transcription failed" in results[1].error
    
    @pytest.mark.asyncio
    async def test_batch_transcribe_all_errors(self, mock_config):
        """Test batch transcribe with all errors."""
        with patch('voice.stt_service.openai'):
            service = STTService(mock_config)
            
            # Mock failing transcription
            async def mock_transcribe(audio_data, **kwargs):
                raise Exception("All failed")
            
            service.transcribe_audio = mock_transcribe
            
            # Create audio data list
            audio_list = [
                AudioData(
                    data=np.array([0.1, 0.2], dtype=np.float32),
                    sample_rate=16000,
                    channels=1,
                    format="float32",
                    duration=1.0
                )
            ]
            
            results = await service.batch_transcribe(audio_list)
            
            assert len(results) == 1
            assert results[0].text == ""
            assert results[0].confidence == 0.0
            assert results[0].provider == "error"
            assert results[0].error is not None


class TestServiceInfoAndStringMethods:
    """Test get_service_info, __str__, and __repr__ methods."""
    
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
        return config
    
    def test_get_service_info_complete(self, mock_config):
        """Test get_service_info returns complete information."""
        with patch('voice.stt_service.openai'):
            service = STTService(mock_config)
            service.openai_client = Mock()
            
            info = service.get_service_info()
            
            # Verify all expected keys are present
            assert "provider" in info
            assert "model" in info
            assert "language" in info
            assert "available_providers" in info
            assert "is_available" in info
            assert "available" in info  # Backward compatibility
            assert "request_count" in info
            assert "error_count" in info
            assert "average_processing_time" in info
            assert "supported_languages" in info
            assert "supported_models" in info
            
            # Verify values
            assert info["provider"] == "openai"
            assert info["model"] == "whisper-1"
            assert info["language"] == "en-US"
            assert isinstance(info["available_providers"], list)
            assert isinstance(info["is_available"], bool)
            assert isinstance(info["supported_languages"], list)
            assert isinstance(info["supported_models"], list)
    
    def test_str_representation(self, mock_config):
        """Test __str__ method."""
        with patch('voice.stt_service.openai'):
            service = STTService(mock_config)
            
            str_repr = str(service)
            
            assert isinstance(str_repr, str)
            assert "STTService" in str_repr
            assert "openai" in str_repr
            assert "whisper-1" in str_repr
    
    def test_repr_representation(self, mock_config):
        """Test __repr__ method."""
        with patch('voice.stt_service.openai'):
            service = STTService(mock_config)
            
            repr_str = repr(service)
            
            assert isinstance(repr_str, str)
            assert "STTService" in repr_str
            assert "openai" in repr_str
            assert "whisper-1" in repr_str
            assert "en-US" in repr_str
            assert "available=" in repr_str
    
    def test_str_and_repr_different_providers(self, mock_config):
        """Test __str__ and __repr__ with different configurations."""
        mock_config.get_preferred_stt_service.return_value = "google"
        mock_config.is_google_speech_configured.return_value = True
        
        with patch('voice.stt_service.openai'):
            service = STTService(mock_config)
            service.primary_provider = "google"
            
            str_repr = str(service)
            repr_str = repr(service)
            
            # Both should reflect the actual provider
            assert "google" in str_repr or "openai" in str_repr  # Depends on initialization
            assert "STTService" in repr_str
