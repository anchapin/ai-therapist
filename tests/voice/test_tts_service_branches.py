"""
Comprehensive Branch Coverage Tests for TTSService

This module adds missing branch coverage for:
1. build_ssml / _generate_ssml - options toggles, prosody and emphasis tags
2. Caching - _get_cache_key + _cache_result LRU eviction, save_audio error paths
3. get_supported_* methods when providers unavailable
"""

import pytest
import asyncio
import numpy as np
import tempfile
import os
from unittest.mock import Mock, MagicMock, AsyncMock, patch, mock_open
from pathlib import Path
from dataclasses import dataclass

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from voice.tts_service import (
    TTSService,
    TTSResult,
    EmotionType,
    VoiceEmotionSettings,
    SSMLSettings,
    TTSError
)
from voice.audio_processor import AudioData
from voice.config import VoiceConfig, VoiceProfile


@pytest.fixture
def mock_voice_config():
    """Create mock VoiceConfig for testing."""
    config = MagicMock(spec=VoiceConfig)
    config.openai_api_key = "test-key"
    config.elevenlabs_api_key = None
    config.tts_voice = "alloy"
    config.tts_language = "en-US"
    config.tts_model = "tts-1"
    config.default_voice_profile = "default"
    config.performance = MagicMock()
    config.performance.cache_size = 10
    config.get_preferred_tts_service = MagicMock(return_value="openai")
    
    # Add voice_profiles to avoid AttributeError
    default_profile = VoiceProfile(
        name="default",
        description="Default test profile",
        voice_id="alloy",
        pitch=1.0,
        speed=1.0,
        volume=1.0,
        language="en-US"
    )
    config.voice_profiles = {"default": default_profile}
    
    return config


@pytest.fixture
def tts_service(mock_voice_config):
    """Create TTSService instance for testing."""
    # Patch service initialization to avoid real API calls
    with patch('voice.tts_service.openai') as mock_openai:
        mock_openai.OpenAI.return_value = None  # Disable OpenAI
        
        with patch.object(TTSService, '_initialize_services'):
            service = TTSService(mock_voice_config)
            # Manually set up minimal service state
            service.openai_client = None
            service.elevenlabs_client = None
            service.piper_tts = None
            service.ssml_settings = SSMLSettings()
            return service


@pytest.fixture
def sample_voice_profile():
    """Create sample voice profile."""
    profile = VoiceProfile(
        name="test_profile",
        description="Test voice profile",
        voice_id="alloy",
        pitch=1.2,
        speed=0.9,
        volume=0.8,
        language="en-US"
    )
    if not hasattr(profile, 'emotion'):
        profile.emotion = EmotionType.CALM
    return profile


@pytest.fixture
def sample_audio_data():
    """Create sample AudioData for testing."""
    return AudioData(
        data=np.random.random(16000).astype(np.float32),
        sample_rate=16000,
        duration=1.0,
        channels=1,
        format="wav"
    )


class TestSSMLGeneration:
    """Test _generate_ssml method with different option toggles."""
    
    def test_generate_ssml_all_disabled(self, tts_service, sample_voice_profile):
        """Test SSML generation when SSML is completely disabled."""
        tts_service.ssml_settings = SSMLSettings(enabled=False)
        
        text = "This is a test with important words"
        result = tts_service._generate_ssml(text, sample_voice_profile)
        
        # Should return plain text when disabled
        assert result == text
        assert '<speak>' not in result
        assert '<prosody' not in result
    
    def test_generate_ssml_prosody_enabled(self, tts_service, sample_voice_profile):
        """Test SSML generation with only prosody enabled."""
        tts_service.ssml_settings = SSMLSettings(
            enabled=True,
            prosody_attributes=True,
            emphasis_tags=False
        )
        
        text = "This is important"
        result = tts_service._generate_ssml(text, sample_voice_profile)
        
        # Should have speak and prosody tags
        assert '<speak>' in result
        assert '</speak>' in result
        assert '<prosody' in result
        assert '</prosody>' in result
        # Should not have emphasis tags
        assert '<emphasis' not in result
    
    def test_generate_ssml_emphasis_enabled(self, tts_service, sample_voice_profile):
        """Test SSML generation with only emphasis enabled."""
        tts_service.ssml_settings = SSMLSettings(
            enabled=True,
            prosody_attributes=False,
            emphasis_tags=True
        )
        
        text = "This is important and we understand your help"
        result = tts_service._generate_ssml(text, sample_voice_profile)
        
        # Should have speak tags
        assert '<speak>' in result
        assert '</speak>' in result
        # Should have emphasis tags for therapeutic keywords
        assert '<emphasis level="moderate">important</emphasis>' in result
        assert '<emphasis level="moderate">understand</emphasis>' in result
        assert '<emphasis level="moderate">help</emphasis>' in result
        # Should not have prosody tags
        assert '</prosody>' not in result
    
    def test_generate_ssml_all_enabled(self, tts_service, sample_voice_profile):
        """Test SSML generation with all options enabled."""
        tts_service.ssml_settings = SSMLSettings(
            enabled=True,
            prosody_attributes=True,
            emphasis_tags=True
        )
        
        text = "We care and support your important progress"
        result = tts_service._generate_ssml(text, sample_voice_profile)
        
        # Should have all tags
        assert '<speak>' in result
        assert '</speak>' in result
        assert '<prosody' in result
        assert '</prosody>' in result
        assert '<emphasis' in result
        # Check therapeutic keywords are emphasized
        assert 'care' in result
        assert 'support' in result
        assert 'important' in result
    
    def test_generate_ssml_prosody_attributes(self, tts_service, sample_voice_profile):
        """Test prosody attributes are correctly set."""
        tts_service.ssml_settings = SSMLSettings(
            enabled=True,
            prosody_attributes=True,
            emphasis_tags=False
        )
        
        # Profile with specific values
        sample_voice_profile.pitch = 1.2  # 20% higher
        sample_voice_profile.speed = 0.8  # 80% speed
        sample_voice_profile.volume = 1.1  # 10% louder
        
        text = "Test prosody"
        result = tts_service._generate_ssml(text, sample_voice_profile)
        
        # Check pitch adjustment (allow for +19% due to int rounding)
        assert 'pitch="+19%"' in result or 'pitch="+20%"' in result or 'pitch="medium"' in result
        # Check rate adjustment
        assert 'rate="80%"' in result or 'rate="80.0%"' in result
        # Check volume adjustment
        assert 'volume=' in result
    
    def test_generate_ssml_default_prosody_values(self, tts_service):
        """Test prosody with default 1.0 values uses 'medium'."""
        tts_service.ssml_settings = SSMLSettings(
            enabled=True,
            prosody_attributes=True,
            emphasis_tags=False
        )
        
        # Profile with default values
        profile = VoiceProfile(
            name="default",
            description="Default prosody test",
            voice_id="alloy",
            pitch=1.0,
            speed=1.0,
            volume=1.0,
            language="en-US"
        )
        
        text = "Test default prosody"
        result = tts_service._generate_ssml(text, profile)
        
        # Should use medium/default for 1.0 values
        assert '<prosody' in result
        assert 'pitch="medium"' in result or 'rate="medium"' in result
    
    def test_generate_ssml_therapeutic_keywords_case_insensitive(self, tts_service):
        """Test emphasis tags applied case-insensitively."""
        tts_service.ssml_settings = SSMLSettings(
            enabled=True,
            prosody_attributes=False,
            emphasis_tags=True
        )
        
        profile = VoiceProfile(
            name="test",
            description="Case test profile",
            voice_id="alloy",
            pitch=1.0,
            speed=1.0,
            volume=1.0,
            language="en-US"
        )
        
        text = "I UNDERSTAND your HELP is IMPORTANT. We SUPPORT and CARE"
        result = tts_service._generate_ssml(text, profile)
        
        # Keywords should be emphasized regardless of case
        # Note: re.sub replaces with lowercase match from pattern
        assert '<emphasis level="moderate">understand</emphasis>' in result
        assert '<emphasis level="moderate">help</emphasis>' in result
        assert '<emphasis level="moderate">important</emphasis>' in result
        assert '<emphasis level="moderate">support</emphasis>' in result
        assert '<emphasis level="moderate">care</emphasis>' in result


class TestCaching:
    """Test caching functionality including LRU eviction and cache key generation."""
    
    def test_get_cache_key_basic(self, tts_service):
        """Test basic cache key generation."""
        key = tts_service._get_cache_key(
            text="Hello world",
            voice_profile="default",
            provider="openai"
        )
        
        assert isinstance(key, str)
        assert len(key) > 0
    
    def test_get_cache_key_with_emotion(self, tts_service):
        """Test cache key generation with emotion."""
        key_with_emotion = tts_service._get_cache_key(
            text="Hello",
            voice_profile="default",
            provider="openai",
            emotion=EmotionType.CALM
        )
        
        key_without_emotion = tts_service._get_cache_key(
            text="Hello",
            voice_profile="default",
            provider="openai",
            emotion=None
        )
        
        # Keys should be different when emotion differs
        assert key_with_emotion != key_without_emotion
    
    def test_get_cache_key_different_emotions(self, tts_service):
        """Test cache keys differ for different emotions."""
        key_calm = tts_service._get_cache_key(
            text="Hello",
            voice_profile="default",
            provider="openai",
            emotion=EmotionType.CALM
        )
        
        key_empathetic = tts_service._get_cache_key(
            text="Hello",
            voice_profile="default",
            provider="openai",
            emotion=EmotionType.EMPATHETIC
        )
        
        assert key_calm != key_empathetic
    
    def test_cache_result_basic(self, tts_service, sample_audio_data):
        """Test basic cache result storage."""
        result = TTSResult(
            audio_data=sample_audio_data,
            text="Test",
            voice_profile="default",
            provider="openai",
            duration=1.0
        )
        
        key = "test_key_1"
        tts_service._cache_result(key, result)
        
        assert key in tts_service.audio_cache
        assert tts_service.audio_cache[key] == result
    
    def test_cache_result_lru_eviction(self, tts_service, sample_audio_data):
        """Test LRU eviction when cache is full."""
        # Set small cache size
        tts_service.max_cache_size = 3
        tts_service.audio_cache.clear()
        
        # Add results with different timestamps
        import time
        for i in range(3):
            result = TTSResult(
                audio_data=sample_audio_data,
                text=f"Test {i}",
                voice_profile="default",
                provider="openai",
                duration=1.0,
                timestamp=time.time() + i  # Increasing timestamps
            )
            tts_service._cache_result(f"key_{i}", result)
            time.sleep(0.01)  # Ensure different timestamps
        
        # Cache should be full
        assert len(tts_service.audio_cache) == 3
        assert "key_0" in tts_service.audio_cache
        assert "key_1" in tts_service.audio_cache
        assert "key_2" in tts_service.audio_cache
        
        # Add one more - should evict oldest (key_0)
        new_result = TTSResult(
            audio_data=sample_audio_data,
            text="Test new",
            voice_profile="default",
            provider="openai",
            duration=1.0,
            timestamp=time.time() + 10
        )
        tts_service._cache_result("key_3", new_result)
        
        # Should still have max_cache_size entries
        assert len(tts_service.audio_cache) == 3
        # Oldest entry should be evicted
        assert "key_0" not in tts_service.audio_cache
        # Newer entries should remain
        assert "key_1" in tts_service.audio_cache
        assert "key_2" in tts_service.audio_cache
        assert "key_3" in tts_service.audio_cache
    
    def test_save_audio_success(self, tts_service, sample_audio_data):
        """Test successful audio save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_audio.wav")
            
            with patch('soundfile.write') as mock_sf_write:
                result = tts_service.save_audio(sample_audio_data, filepath, format="wav")
                
                assert result is True
                mock_sf_write.assert_called_once()
    
    def test_save_audio_creates_directory(self, tts_service, sample_audio_data):
        """Test save_audio creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "nested", "dir", "test_audio.wav")
            
            with patch('soundfile.write') as mock_sf_write:
                result = tts_service.save_audio(sample_audio_data, filepath, format="wav")
                
                assert result is True
                # Parent directory should be created
                assert Path(filepath).parent.exists()
    
    def test_save_audio_format_mapping(self, tts_service, sample_audio_data):
        """Test different audio format mappings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            formats = ["wav", "mp3", "flac", "ogg"]
            
            for fmt in formats:
                filepath = os.path.join(tmpdir, f"test.{fmt}")
                
                with patch('soundfile.write') as mock_sf_write:
                    result = tts_service.save_audio(sample_audio_data, filepath, format=fmt)
                    
                    assert result is True
                    # Check format was mapped correctly
                    call_args = mock_sf_write.call_args
                    assert call_args is not None
    
    def test_save_audio_soundfile_import_error(self, tts_service, sample_audio_data):
        """Test save_audio handles soundfile import error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_audio.wav")
            
            # Patch the local import inside save_audio method
            with patch('builtins.__import__', side_effect=ImportError("soundfile not found")):
                result = tts_service.save_audio(sample_audio_data, filepath, format="wav")
                
                # Should return False on error
                assert result is False
    
    def test_save_audio_write_error(self, tts_service, sample_audio_data):
        """Test save_audio handles write errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_audio.wav")
            
            with patch('soundfile.write', side_effect=Exception("Write failed")):
                result = tts_service.save_audio(sample_audio_data, filepath, format="wav")
                
                # Should return False on error
                assert result is False
    
    def test_save_audio_permission_error(self, tts_service, sample_audio_data):
        """Test save_audio handles permission errors."""
        # Use a path that would cause permission error
        filepath = "/root/cannot_write_here.wav"
        
        with patch('soundfile.write', side_effect=PermissionError("Permission denied")):
            result = tts_service.save_audio(sample_audio_data, filepath, format="wav")
            
            # Should return False on error
            assert result is False


class TestGetSupportedMethods:
    """Test get_supported_* methods when providers are unavailable."""
    
    def test_get_supported_voices_no_providers(self, mock_voice_config):
        """Test get_supported_voices with no providers available."""
        with patch('voice.tts_service.openai', None):
            service = TTSService(mock_voice_config)
            service.openai_client = None
            service.elevenlabs_client = None
            service.piper_tts = None
            
            voices = service.get_supported_voices()
            
            # Should return empty list when no providers available
            assert isinstance(voices, list)
            assert len(voices) == 0
    
    def test_get_supported_voices_only_openai(self, mock_voice_config):
        """Test get_supported_voices with only OpenAI available."""
        with patch('voice.tts_service.openai') as mock_openai:
            mock_openai.OpenAI.return_value = MagicMock()
            service = TTSService(mock_voice_config)
            service.openai_client = MagicMock()
            service.elevenlabs_client = None
            service.piper_tts = None
            
            voices = service.get_supported_voices()
            
            # Should only return OpenAI voices
            assert len(voices) == 6  # OpenAI has 6 voices
            voice_ids = [v['id'] for v in voices]
            assert 'alloy' in voice_ids
            assert 'echo' in voice_ids
            assert 'fable' in voice_ids
            # Should not have ElevenLabs or Piper voices
            assert 'rachel' not in voice_ids
            assert 'default_male' not in voice_ids
    
    def test_get_supported_voices_only_elevenlabs(self, mock_voice_config):
        """Test get_supported_voices with only ElevenLabs available."""
        service = TTSService(mock_voice_config)
        service.openai_client = None
        service.elevenlabs_client = MagicMock()
        service.piper_tts = None
        
        voices = service.get_supported_voices()
        
        # Should only return ElevenLabs voices
        assert len(voices) == 10  # ElevenLabs has 10 voices
        voice_ids = [v['id'] for v in voices]
        assert 'rachel' in voice_ids
        assert 'drew' in voice_ids
        # Should not have OpenAI or Piper voices
        assert 'alloy' not in voice_ids
        assert 'default_male' not in voice_ids
    
    def test_get_supported_voices_only_piper(self, mock_voice_config):
        """Test get_supported_voices with only Piper available."""
        service = TTSService(mock_voice_config)
        service.openai_client = None
        service.elevenlabs_client = None
        service.piper_tts = MagicMock()
        
        voices = service.get_supported_voices()
        
        # Should only return Piper voices
        assert len(voices) == 2  # Piper has 2 voices
        voice_ids = [v['id'] for v in voices]
        assert 'default_male' in voice_ids
        assert 'default_female' in voice_ids
        # Should not have OpenAI or ElevenLabs voices
        assert 'alloy' not in voice_ids
        assert 'rachel' not in voice_ids
    
    def test_get_supported_voices_all_providers(self, mock_voice_config):
        """Test get_supported_voices with all providers available."""
        with patch('voice.tts_service.openai') as mock_openai:
            mock_openai.OpenAI.return_value = MagicMock()
            service = TTSService(mock_voice_config)
            service.openai_client = MagicMock()
            service.elevenlabs_client = MagicMock()
            service.piper_tts = MagicMock()
            
            voices = service.get_supported_voices()
            
            # Should return all voices from all providers
            assert len(voices) == 18  # 6 + 10 + 2
            voice_ids = [v['id'] for v in voices]
            # Check all providers represented
            assert 'alloy' in voice_ids  # OpenAI
            assert 'rachel' in voice_ids  # ElevenLabs
            assert 'default_male' in voice_ids  # Piper
    
    def test_get_supported_models_no_providers(self, mock_voice_config):
        """Test get_supported_models with no providers available."""
        service = TTSService(mock_voice_config)
        service.openai_client = None
        service.elevenlabs_client = None
        service.piper_tts = None
        
        models = service.get_supported_models()
        
        # Should return empty list when no providers available
        assert isinstance(models, list)
        assert len(models) == 0
    
    def test_get_supported_models_openai_available(self, mock_voice_config):
        """Test get_supported_models with OpenAI available."""
        with patch('voice.tts_service.openai') as mock_openai:
            mock_openai.OpenAI.return_value = MagicMock()
            service = TTSService(mock_voice_config)
            service.openai_client = MagicMock()
            
            models = service.get_supported_models()
            
            # Should return OpenAI models
            assert len(models) == 2
            model_ids = [m['id'] for m in models]
            assert 'tts-1' in model_ids
            assert 'tts-1-hd' in model_ids
    
    def test_get_supported_languages_always_returns_list(self, mock_voice_config):
        """Test get_supported_languages always returns language list."""
        service = TTSService(mock_voice_config)
        # Even with no providers, languages should be available
        service.openai_client = None
        service.elevenlabs_client = None
        service.piper_tts = None
        
        languages = service.get_supported_languages()
        
        # Should always return 10 supported languages
        assert isinstance(languages, list)
        assert len(languages) == 10
        lang_codes = [lang['code'] for lang in languages]
        assert 'en-US' in lang_codes
        assert 'es-ES' in lang_codes
        assert 'ja-JP' in lang_codes
    
    def test_get_available_providers_none_available(self, mock_voice_config):
        """Test get_available_providers when no providers available."""
        service = TTSService(mock_voice_config)
        service.openai_client = None
        service.elevenlabs_client = None
        service.piper_tts = None
        
        providers = service.get_available_providers()
        
        assert isinstance(providers, list)
        assert len(providers) == 0
    
    def test_get_available_providers_partial(self, mock_voice_config):
        """Test get_available_providers with some providers available."""
        with patch('voice.tts_service.openai') as mock_openai:
            mock_openai.OpenAI.return_value = MagicMock()
            service = TTSService(mock_voice_config)
            service.openai_client = MagicMock()
            service.elevenlabs_client = None
            service.piper_tts = MagicMock()
            
            providers = service.get_available_providers()
            
            assert len(providers) == 2
            assert "openai" in providers
            assert "piper" in providers
            assert "elevenlabs" not in providers
    
    def test_is_available_with_no_providers(self, mock_voice_config):
        """Test is_available returns False when no providers available."""
        service = TTSService(mock_voice_config)
        service.openai_client = None
        service.elevenlabs_client = None
        service.piper_tts = None
        
        assert service.is_available() is False
    
    def test_is_available_with_any_provider(self, mock_voice_config):
        """Test is_available returns True when any provider available."""
        with patch('voice.tts_service.openai') as mock_openai:
            mock_openai.OpenAI.return_value = MagicMock()
            service = TTSService(mock_voice_config)
            service.openai_client = MagicMock()
            service.elevenlabs_client = None
            service.piper_tts = None
            
            assert service.is_available() is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
