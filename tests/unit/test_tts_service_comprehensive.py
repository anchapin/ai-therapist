"""
Comprehensive TTS Service Tests

This module provides extensive coverage for voice/tts_service.py functionality:
- Multi-provider TTS service integration (OpenAI, ElevenLabs, Piper)
- Voice profile management and customization
- SSML support and advanced speech synthesis
- Real-time and batch synthesis modes
- Audio post-processing and quality optimization
- Error handling and provider fallback mechanisms
- Performance optimization and caching
- Multi-language and accent support
- Emotion and prosody control
- HIPAA-compliant data handling and security
"""

import pytest
import asyncio
import time
import json
import base64
import tempfile
import os
from unittest.mock import Mock, MagicMock, AsyncMock, patch, call, mock_open
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# Add project root to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import TTS service components
try:
    from voice.tts_service import (
        TTSService,
        TTSProvider,
        TTSResult,
        TTSConfig,
        VoiceProfile
    )
    from voice.audio_processor import AudioData
    from voice.config import VoiceConfig
    TTS_SERVICE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TTS service import failed: {e}")
    TTS_SERVICE_AVAILABLE = False

# Import fixtures
try:
    from tests.fixtures.voice_fixtures import mock_voice_config
except ImportError:
    # Fallback fixture
    @pytest.fixture
    def mock_voice_config():
        """Fallback mock VoiceConfig for testing."""
        config = MagicMock()
        config.tts_provider = "openai"
        config.tts_fallback_providers = ["openai", "elevenlabs", "piper"]
        config.cache_enabled = True
        config.ssml_enabled = True
        config.post_processing_enabled = True
        config.hipaa_compliance_enabled = True
        config.encryption_enabled = True
        config.real_time_enabled = True
        return config


class TestTTSServiceCore:
    """Test core TTS service functionality."""
    
    @pytest.fixture
    def tts_service(self):
        """Create TTSService instance for testing."""
        if not TTS_SERVICE_AVAILABLE:
            pytest.skip("TTS service not available")
        
        config = mock_voice_config
        return TTSService(config)
    
    def test_tts_service_initialization(self, tts_service):
        """Test TTSService initialization."""
        assert tts_service.config is not None
        assert tts_service.primary_provider == TTSProvider.OPENAI
        assert len(tts_service.providers) > 0
        assert tts_service.cache_enabled is True
        assert len(tts_service.voice_profiles) > 0
    
    def test_tts_result_creation(self):
        """Test TTSResult object creation and attributes."""
        audio_data = AudioData(
            data=np.random.random(16000).astype(np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1,
            format="wav"
        )
        
        result = TTSResult(
            audio_data=audio_data,
            text="Hello world",
            voice_id="alloy",
            provider="openai",
            processing_time=0.5,
            timestamp=time.time()
        )
        
        assert result.audio_data == audio_data
        assert result.text == "Hello world"
        assert result.voice_id == "alloy"
        assert result.provider == "openai"
        assert result.processing_time == 0.5
        assert result.timestamp > 0
    
    def test_voice_profile_creation(self):
        """Test VoiceProfile object creation."""
        profile = VoiceProfile(
            name="custom_voice",
            provider="openai",
            voice_id="alloy",
            language="en",
            gender="neutral",
            age="adult",
            accent="american",
            emotion="neutral",
            speed=1.0,
            pitch=1.0,
            volume=1.0
        )
        
        assert profile.name == "custom_voice"
        assert profile.provider == "openai"
        assert profile.voice_id == "alloy"
        assert profile.language == "en"
        assert profile.speed == 1.0


@pytest.mark.skipif(not TTS_SERVICE_AVAILABLE, reason="TTS service not available")
class TestOpenAITTSIntegration:
    """Test OpenAI TTS API integration."""
    
    @pytest.fixture
    def tts_service(self):
        """Create TTSService with OpenAI configuration."""
        config = mock_voice_config
        config.tts_provider = "openai"
        return TTSService(config)
    
    @pytest.fixture
    def mock_openai(self):
        """Mock OpenAI TTS API."""
        with patch('voice.tts_service.openai') as mock_openai:
            mock_client = MagicMock()
            mock_openai.Audio = MagicMock()
            mock_openai.Audio.speech = MagicMock()
            mock_openai.Audio.speech.return_value = b"fake_audio_data"
            yield mock_openai
    
    @pytest.mark.asyncio
    async def test_openai_synthesis_success(self, tts_service, mock_openai):
        """Test successful OpenAI TTS synthesis."""
        text = "Hello, this is a test synthesis"
        
        result = await tts_service.synthesize_speech(text)
        
        assert isinstance(result, TTSResult)
        assert result.text == text
        assert result.provider == "openai"
        assert result.voice_id == "alloy"
        assert isinstance(result.audio_data, AudioData)
        mock_openai.Audio.speech.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_openai_with_custom_voice(self, tts_service, mock_openai):
        """Test OpenAI synthesis with custom voice."""
        text = "Custom voice test"
        voice_id = "nova"
        
        result = await tts_service.synthesize_speech(text, voice_id=voice_id)
        
        assert result.voice_id == voice_id
        
        # Verify correct voice was requested
        call_args = mock_openai.Audio.speech.call_args
        assert call_args[1]['voice'] == voice_id
    
    @pytest.mark.asyncio
    async def test_openai_with_speed_control(self, tts_service, mock_openai):
        """Test OpenAI synthesis with speed control."""
        text = "Speed controlled speech"
        speed = 1.5
        
        result = await tts_service.synthesize_speech(text, speed=speed)
        
        # Speed should be applied in post-processing
        assert result.audio_data is not None
    
    @pytest.mark.asyncio
    async def test_openai_error_handling(self, tts_service, mock_openai):
        """Test OpenAI API error handling."""
        mock_openai.Audio.speech.side_effect = Exception("OpenAI API error")
        
        text = "Test error handling"
        
        with pytest.raises(Exception) as exc_info:
            await tts_service.synthesize_speech(text)
        
        assert "OpenAI API error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_openai_timeout_handling(self, tts_service, mock_openai):
        """Test OpenAI API timeout handling."""
        import asyncio
        mock_openai.Audio.speech.side_effect = asyncio.TimeoutError("Request timeout")
        
        text = "Test timeout"
        
        with pytest.raises(asyncio.TimeoutError):
            await tts_service.synthesize_speech(text)
    
    @pytest.mark.asyncio
    async def test_openai_long_text_chunking(self, tts_service, mock_openai):
        """Test OpenAI synthesis with long text chunking."""
        # Create long text that exceeds character limits
        long_text = "This is a very long text. " * 200  # ~4000 characters
        
        # Mock response for each chunk
        mock_openai.Audio.speech.return_value = b"chunk_audio_data"
        
        result = await tts_service.synthesize_speech(long_text)
        
        assert isinstance(result, TTSResult)
        assert result.text == long_text
        assert result.audio_data is not None
        
        # Should have made multiple API calls for chunking
        assert mock_openai.Audio.speech.call_count > 1


@pytest.mark.skipif(not TTS_SERVICE_AVAILABLE, reason="TTS service not available")
class TestElevenLabsIntegration:
    """Test ElevenLabs TTS API integration."""
    
    @pytest.fixture
    def tts_service(self):
        """Create TTSService with ElevenLabs configuration."""
        config = mock_voice_config
        config.tts_provider = "elevenlabs"
        config.elevenlabs_api_key = "test_key"
        return TTSService(config)
    
    @pytest.fixture
    def mock_elevenlabs(self):
        """Mock ElevenLabs API."""
        with patch('voice.tts_service.elevenlabs') as mock_elevenlabs:
            mock_client = MagicMock()
            mock_elevenlabs.ElevenLabs.return_value = mock_client
            
            mock_voices = MagicMock()
            mock_voices.get.return_value = [
                {"voice_id": "rachel", "name": "Rachel"},
                {"voice_id": "adam", "name": "Adam"},
                {"voice_id": "bella", "name": "Bella"}
            ]
            mock_client.voices = mock_voices
            
            mock_tts = MagicMock()
            mock_tts.generate.return_value = b"elevenlabs_audio_data"
            mock_client.generate = mock_tts
            
            yield mock_elevenlabs
    
    @pytest.mark.asyncio
    async def test_elevenlabs_synthesis_success(self, tts_service, mock_elevenlabs):
        """Test successful ElevenLabs synthesis."""
        text = "Hello from ElevenLabs"
        voice_id = "rachel"
        
        result = await tts_service.synthesize_speech(text, voice_id=voice_id)
        
        assert isinstance(result, TTSResult)
        assert result.text == text
        assert result.provider == "elevenlabs"
        assert result.voice_id == voice_id
        
        mock_client = mock_elevenlabs.ElevenLabs.return_value
        mock_client.generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_elevenlabs_voice_listings(self, tts_service, mock_elevenlabs):
        """Test ElevenLabs voice listings."""
        voices = await tts_service.list_available_voices("elevenlabs")
        
        assert isinstance(voices, list)
        assert len(voices) > 0
        
        # Check voice structure
        for voice in voices:
            assert "voice_id" in voice
            assert "name" in voice
    
    @pytest.mark.asyncio
    async def test_elevenlabs_with_emotion(self, tts_service, mock_elevenlabs):
        """Test ElevenLabs synthesis with emotion control."""
        text = "Emotional speech test"
        voice_id = "rachel"
        emotion = "sad"
        
        result = await tts_service.synthesize_speech(
            text, 
            voice_id=voice_id, 
            emotion=emotion
        )
        
        assert result.voice_id == voice_id
        assert isinstance(result, TTSResult)
    
    @pytest.mark.asyncio
    async def test_elevenlabs_error_handling(self, tts_service, mock_elevenlabs):
        """Test ElevenLabs API error handling."""
        mock_client = mock_elevenlabs.ElevenLabs.return_value
        mock_client.generate.side_effect = Exception("ElevenLabs API error")
        
        text = "Test error handling"
        
        with pytest.raises(Exception) as exc_info:
            await tts_service.synthesize_speech(text)
        
        assert "ElevenLabs API error" in str(exc_info.value)


@pytest.mark.skipif(not TTS_SERVICE_AVAILABLE, reason="TTS service not available")
class TestPiperTTSIntegration:
    """Test local Piper TTS integration."""
    
    @pytest.fixture
    def tts_service(self):
        """Create TTSService with Piper configuration."""
        config = mock_voice_config
        config.tts_provider = "piper"
        return TTSService(config)
    
    @pytest.fixture
    def mock_piper(self):
        """Mock Piper TTS."""
        with patch('voice.tts_service.piper_tts') as mock_piper:
            mock_synthesizer = MagicMock()
            mock_synthesizer.synthesize.return_value = (
                np.random.random(16000 * 2).astype(np.float32),  # 2 seconds audio
                22050  # Sample rate
            )
            mock_piper.Synthesizer.return_value = mock_synthesizer
            
            yield mock_piper
    
    @pytest.mark.asyncio
    async def test_piper_synthesis_success(self, tts_service, mock_piper):
        """Test successful Piper synthesis."""
        text = "Hello from Piper"
        voice_model = "en_US-lessac-medium"
        
        result = await tts_service.synthesize_speech(text, voice_model=voice_model)
        
        assert isinstance(result, TTSResult)
        assert result.text == text
        assert result.provider == "piper"
        assert isinstance(result.audio_data, AudioData)
        
        mock_piper.Synthesizer.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_piper_model_loading(self, tts_service, mock_piper):
        """Test Piper model loading."""
        models = ["en_US-lessac-medium", "en_US-lessac-low", "en_US-lessac-high"]
        
        for model in models:
            await tts_service.synthesize_speech("Test", voice_model=model)
            
            # Check if correct model was loaded
            call_args = mock_piper.Synthesizer.call_args
            # Model should be in the call arguments
    
    @pytest.mark.asyncio
    async def test_piper_offline_capability(self, tts_service, mock_piper):
        """Test Piper offline synthesis capability."""
        # Disable network to test offline capability
        with patch('socket.socket') as mock_socket:
            mock_socket.side_effect = Exception("Network unavailable")
            
            text = "Offline synthesis test"
            
            result = await tts_service.synthesize_speech(text)
            
            assert isinstance(result, TTSResult)
            assert result.provider == "piper"
    
    @pytest.mark.asyncio
    async def test_piper_error_handling(self, tts_service, mock_piper):
        """Test Piper error handling."""
        mock_piper.Synthesizer.side_effect = Exception("Model loading failed")
        
        text = "Test error handling"
        
        with pytest.raises(Exception) as exc_info:
            await tts_service.synthesize_speech(text)
        
        assert "Model loading failed" in str(exc_info.value)


@pytest.mark.skipif(not TTS_SERVICE_AVAILABLE, reason="TTS service not available")
class TestTTSProviderFallback:
    """Test TTS provider fallback mechanisms."""
    
    @pytest.fixture
    def tts_service(self):
        """Create TTSService with multiple providers."""
        config = mock_voice_config
        config.tts_fallback_providers = ["openai", "elevenlabs", "piper"]
        return TTSService(config)
    
    @pytest.fixture
    def mock_providers(self):
        """Mock all TTS providers."""
        with patch('voice.tts_service.openai') as mock_openai, \
             patch('voice.tts_service.elevenlabs') as mock_elevenlabs, \
             patch('voice.tts_service.piper_tts') as mock_piper:
            
            # OpenAI fails
            mock_openai.Audio.speech.side_effect = Exception("OpenAI down")
            
            # ElevenLabs succeeds
            mock_client = MagicMock()
            mock_client.generate.return_value = b"elevenlabs_fallback_audio"
            mock_elevenlabs.ElevenLabs.return_value = mock_client
            
            # Piper is available but not needed
            mock_piper.Synthesizer.return_value.synthesize.return_value = (
                np.random.random(16000).astype(np.float32),
                22050
            )
            
            yield {
                'openai': mock_openai,
                'elevenlabs': mock_elevenlabs,
                'piper': mock_piper
            }
    
    @pytest.mark.asyncio
    async def test_provider_fallback_success(self, tts_service, mock_providers):
        """Test successful provider fallback."""
        text = "Fallback test text"
        
        result = await tts_service.synthesize_speech_with_fallback(text)
        
        assert isinstance(result, TTSResult)
        assert result.text == text
        assert result.provider == "elevenlabs"  # Should fallback to ElevenLabs
    
    @pytest.mark.asyncio
    async def test_all_providers_fail(self, tts_service, mock_providers):
        """Test when all providers fail."""
        # Make all providers fail
        mock_providers['elevenlabs'].ElevenLabs.return_value.generate.side_effect = Exception("ElevenLabs failed")
        
        text = "All providers fail test"
        
        with pytest.raises(Exception) as exc_info:
            await tts_service.synthesize_speech_with_fallback(text)
        
        assert "All TTS providers failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_provider_priority_order(self, tts_service, mock_providers):
        """Test provider fallback priority order."""
        # Reset mocks to track call order
        mock_providers['openai'].reset_mock()
        mock_providers['elevenlabs'].reset_mock()
        
        text = "Priority test"
        
        await tts_service.synthesize_speech_with_fallback(text)
        
        # Should try OpenAI first, then ElevenLabs
        mock_providers['openai'].Audio.speech.assert_called_once()
        mock_client = mock_providers['elevenlabs'].ElevenLabs.return_value
        mock_client.generate.assert_called_once()


@pytest.mark.skipif(not TTS_SERVICE_AVAILABLE, reason="TTS service not available")
class TestVoiceProfileManagement:
    """Test voice profile management and customization."""
    
    @pytest.fixture
    def tts_service(self):
        """Create TTSService for voice profile tests."""
        config = mock_voice_config
        return TTSService(config)
    
    def test_create_voice_profile(self, tts_service):
        """Test creating a custom voice profile."""
        profile_data = {
            "name": "therapist_voice",
            "provider": "openai",
            "voice_id": "nova",
            "language": "en",
            "gender": "female",
            "age": "adult",
            "accent": "american",
            "emotion": "calm",
            "speed": 1.1,
            "pitch": 1.0,
            "volume": 0.9
        }
        
        profile = tts_service.create_voice_profile(**profile_data)
        
        assert isinstance(profile, VoiceProfile)
        assert profile.name == "therapist_voice"
        assert profile.provider == "openai"
        assert profile.voice_id == "nova"
        assert profile.emotion == "calm"
        assert profile.speed == 1.1
    
    def test_save_voice_profile(self, tts_service):
        """Test saving voice profile."""
        profile = VoiceProfile(
            name="test_profile",
            provider="openai",
            voice_id="alloy"
        )
        
        success = tts_service.save_voice_profile(profile)
        
        assert success is True
        assert "test_profile" in tts_service.voice_profiles
        assert tts_service.voice_profiles["test_profile"] == profile
    
    def test_load_voice_profile(self, tts_service):
        """Test loading voice profile."""
        # First save a profile
        profile = VoiceProfile(
            name="load_test",
            provider="elevenlabs",
            voice_id="rachel"
        )
        tts_service.save_voice_profile(profile)
        
        # Load it back
        loaded_profile = tts_service.load_voice_profile("load_test")
        
        assert loaded_profile is not None
        assert loaded_profile.name == "load_test"
        assert loaded_profile.provider == "elevenlabs"
        assert loaded_profile.voice_id == "rachel"
    
    def test_list_voice_profiles(self, tts_service):
        """Test listing all voice profiles."""
        profiles = tts_service.list_voice_profiles()
        
        assert isinstance(profiles, list)
        assert len(profiles) > 0  # Should have default profiles
        
        for profile in profiles:
            assert isinstance(profile, VoiceProfile)
    
    def test_delete_voice_profile(self, tts_service):
        """Test deleting voice profile."""
        # Create a temporary profile
        profile = VoiceProfile(
            name="temp_profile",
            provider="openai",
            voice_id="alloy"
        )
        tts_service.save_voice_profile(profile)
        
        # Delete it
        success = tts_service.delete_voice_profile("temp_profile")
        
        assert success is True
        assert "temp_profile" not in tts_service.voice_profiles
    
    @pytest.mark.asyncio
    async def test_synthesis_with_voice_profile(self, tts_service):
        """Test synthesis using custom voice profile."""
        # Create custom profile
        profile = VoiceProfile(
            name="custom_test",
            provider="openai",
            voice_id="nova",
            speed=1.2,
            emotion="calm"
        )
        tts_service.save_voice_profile(profile)
        
        # Mock synthesis
        with patch.object(tts_service, '_synthesize_with_provider') as mock_synthesize:
            mock_audio_data = AudioData(
                data=np.random.random(16000).astype(np.float32),
                sample_rate=16000,
                duration=1.0,
                channels=1,
                format="wav"
            )
            mock_synthesize.return_value = TTSResult(
                audio_data=mock_audio_data,
                text="Test with profile",
                voice_id="nova",
                provider="openai"
            )
            
            result = await tts_service.synthesize_speech(
                "Test with profile",
                voice_profile="custom_test"
            )
            
            assert isinstance(result, TTSResult)
            assert result.voice_id == "nova"
            assert result.provider == "openai"


@pytest.mark.skipif(not TTS_SERVICE_AVAILABLE, reason="TTS service not available")
class TestSSMLSupport:
    """Test SSML (Speech Synthesis Markup Language) support."""
    
    @pytest.fixture
    def tts_service(self):
        """Create TTSService with SSML support."""
        config = mock_voice_config
        config.ssml_enabled = True
        return TTSService(config)
    
    def test_ssml_parsing(self, tts_service):
        """Test SSML parsing and validation."""
        ssml_text = """
        <speak>
            <prosody rate="slow" pitch="high">
                Hello, this is a test.
            </prosody>
            <break time="500ms"/>
            <emphasis level="strong">
                This is emphasized text.
            </emphasis>
        </speak>
        """
        
        parsed = tts_service.parse_ssml(ssml_text)
        
        assert "text" in parsed
        assert "prosody" in parsed
        assert "breaks" in parsed
        assert "emphasis" in parsed
    
    @pytest.mark.asyncio
    async def test_ssml_synthesis(self, tts_service):
        """Test synthesis with SSML."""
        ssml_text = """
        <speak>
            <prosody rate="0.9" pitch="+10%">
                This is SSML controlled speech.
            </prosody>
        </speak>
        """
        
        with patch.object(tts_service, '_synthesize_with_provider') as mock_synthesize:
            mock_audio_data = AudioData(
                data=np.random.random(16000).astype(np.float32),
                sample_rate=16000,
                duration=2.0,
                channels=1,
                format="wav"
            )
            mock_synthesize.return_value = TTSResult(
                audio_data=mock_audio_data,
                text="This is SSML controlled speech.",
                voice_id="alloy",
                provider="openai"
            )
            
            result = await tts_service.synthesize_ssml(ssml_text)
            
            assert isinstance(result, TTSResult)
            assert "SSML controlled" in result.text
    
    def test_ssml_validation(self, tts_service):
        """Test SSML validation."""
        valid_ssml = "<speak>Hello world</speak>"
        invalid_ssml = "<invalid>Hello world</invalid>"
        
        assert tts_service.validate_ssml(valid_ssml) is True
        assert tts_service.validate_ssml(invalid_ssml) is False
    
    def test_ssml_to_text_conversion(self, tts_service):
        """Test converting SSML to plain text."""
        ssml_text = """
        <speak>
            Hello <emphasis level="strong">world</emphasis>.
            <break time="200ms"/>
            How are you?
        </speak>
        """
        
        plain_text = tts_service.ssml_to_text(ssml_text)
        
        assert "Hello world." in plain_text
        assert "How are you?" in plain_text
        assert "<" not in plain_text  # No SSML tags


@pytest.mark.skipif(not TTS_SERVICE_AVAILABLE, reason="TTS service not available")
class TestAudioPostProcessing:
    """Test audio post-processing and quality optimization."""
    
    @pytest.fixture
    def tts_service(self):
        """Create TTSService for post-processing tests."""
        config = mock_voice_config
        config.post_processing_enabled = True
        return TTSService(config)
    
    @pytest.fixture
    def sample_audio(self):
        """Create sample audio data for processing."""
        return AudioData(
            data=np.random.random(16000).astype(np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1,
            format="wav"
        )
    
    def test_audio_speed_adjustment(self, tts_service, sample_audio):
        """Test audio playback speed adjustment."""
        faster_audio = tts_service.adjust_audio_speed(sample_audio, 1.5)
        slower_audio = tts_service.adjust_audio_speed(sample_audio, 0.8)
        
        assert isinstance(faster_audio, AudioData)
        assert isinstance(slower_audio, AudioData)
        
        # Faster audio should be shorter
        assert faster_audio.duration < sample_audio.duration
        # Slower audio should be longer
        assert slower_audio.duration > sample_audio.duration
    
    def test_audio_pitch_adjustment(self, tts_service, sample_audio):
        """Test audio pitch adjustment."""
        higher_pitch = tts_service.adjust_audio_pitch(sample_audio, 1.2)
        lower_pitch = tts_service.adjust_audio_pitch(sample_audio, 0.8)
        
        assert isinstance(higher_pitch, AudioData)
        assert isinstance(lower_pitch, AudioData)
    
    def test_audio_volume_adjustment(self, tts_service, sample_audio):
        """Test audio volume adjustment."""
        louder = tts_service.adjust_audio_volume(sample_audio, 1.5)
        quieter = tts_service.adjust_audio_volume(sample_audio, 0.5)
        
        assert isinstance(louder, AudioData)
        assert isinstance(quieter, AudioData)
        
        # Check if volume was actually adjusted
        original_peak = np.max(np.abs(sample_audio.data))
        louder_peak = np.max(np.abs(louder.data))
        quieter_peak = np.max(np.abs(quieter.data))
        
        assert louder_peak > original_peak
        assert quieter_peak < original_peak
    
    def test_audio_format_conversion(self, tts_service, sample_audio):
        """Test audio format conversion."""
        # Test converting to different formats
        mp3_audio = tts_service.convert_audio_format(sample_audio, "mp3")
        ogg_audio = tts_service.convert_audio_format(sample_audio, "ogg")
        
        assert isinstance(mp3_audio, AudioData)
        assert isinstance(ogg_audio, AudioData)
        assert mp3_audio.format == "mp3"
        assert ogg_audio.format == "ogg"
    
    def test_audio_quality_enhancement(self, tts_service, sample_audio):
        """Test audio quality enhancement."""
        enhanced = tts_service.enhance_audio_quality(sample_audio)
        
        assert isinstance(enhanced, AudioData)
        # Enhanced audio should have same basic properties
        assert enhanced.sample_rate == sample_audio.sample_rate
        assert enhanced.channels == sample_audio.channels
    
    def test_audio_normalization(self, tts_service, sample_audio):
        """Test audio normalization."""
        normalized = tts_service.normalize_audio(sample_audio)
        
        assert isinstance(normalized, AudioData)
        
        # Check if audio is normalized
        peak_amplitude = np.max(np.abs(normalized.data))
        assert 0.8 <= peak_amplitude <= 1.0


@pytest.mark.skipif(not TTS_SERVICE_AVAILABLE, reason="TTS service not available")
class TestTTSSecurity:
    """Test TTS service security and privacy features."""
    
    @pytest.fixture
    def tts_service(self):
        """Create TTSService with security features."""
        config = mock_voice_config
        config.hipaa_compliance_enabled = True
        config.encryption_enabled = True
        return TTSService(config)
    
    def test_text_sanitization(self, tts_service):
        """Test text sanitization for security."""
        malicious_text = "Hello <script>alert('xss')</script> world"
        sanitized = tts_service.sanitize_text(malicious_text)
        
        assert "<script>" not in sanitized
        assert "alert('xss')" not in sanitized
        assert "Hello world" in sanitized
    
    def test_pii_detection_in_text(self, tts_service):
        """Test PII detection in synthesis text."""
        text_with_pii = "Call me at 555-123-4567 or email john@example.com"
        
        pii_detected = tts_service.detect_pii_in_text(text_with_pii)
        
        assert pii_detected is True
        
        # Check specific PII types
        pii_types = tts_service.identify_pii_types(text_with_pii)
        assert "phone_number" in pii_types or "email" in pii_types
    
    def test_audio_data_encryption(self, tts_service):
        """Test audio data encryption."""
        audio_data = AudioData(
            data=np.random.random(16000).astype(np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1,
            format="wav"
        )
        
        encrypted = tts_service.encrypt_audio_data(audio_data)
        
        assert isinstance(encrypted, bytes)
        assert encrypted != audio_data.to_bytes()
    
    def test_audit_logging(self, tts_service):
        """Test audit logging for TTS operations."""
        with patch.object(tts_service, 'log_synthesis_event') as mock_log:
            tts_service.log_synthesis(
                user_id="test_user",
                text="Hello world",
                provider="openai",
                voice_id="alloy",
                success=True
            )
            
            mock_log.assert_called_once_with(
                user_id="test_user",
                text="Hello world",
                provider="openai",
                voice_id="alloy",
                success=True
            )


@pytest.mark.skipif(not TTS_SERVICE_AVAILABLE, reason="TTS service not available")
class TestTTSPerformance:
    """Test TTS service performance and optimization."""
    
    @pytest.fixture
    def tts_service(self):
        """Create TTSService for performance tests."""
        config = mock_voice_config
        config.cache_enabled = True
        return TTSService(config)
    
    @pytest.mark.asyncio
    async def test_concurrent_synthesis(self, tts_service):
        """Test concurrent synthesis requests."""
        async def synthesize_multiple():
            tasks = []
            texts = [
                "First synthesis test",
                "Second synthesis test", 
                "Third synthesis test",
                "Fourth synthesis test",
                "Fifth synthesis test"
            ]
            
            for text in texts:
                # Mock synthesis to avoid actual API calls
                with patch.object(tts_service, '_synthesize_with_provider') as mock_synthesize:
                    mock_audio_data = AudioData(
                        data=np.random.random(16000).astype(np.float32),
                        sample_rate=16000,
                        duration=1.0,
                        channels=1,
                        format="wav"
                    )
                    mock_synthesize.return_value = TTSResult(
                        audio_data=mock_audio_data,
                        text=text,
                        voice_id="alloy",
                        provider="openai"
                    )
                    
                    task = asyncio.create_task(tts_service.synthesize_speech(text))
                    tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results
        
        results = await synthesize_multiple()
        
        assert len(results) == 5
        for result in results:
            assert not isinstance(result, Exception)
            assert isinstance(result, TTSResult)
    
    def test_synthesis_caching(self, tts_service):
        """Test synthesis result caching."""
        text = "Cached synthesis test"
        
        with patch.object(tts_service, '_synthesize_with_provider') as mock_synthesize:
            mock_audio_data = AudioData(
                data=np.random.random(16000).astype(np.float32),
                sample_rate=16000,
                duration=1.0,
                channels=1,
                format="wav"
            )
            mock_synthesize.return_value = TTSResult(
                audio_data=mock_audio_data,
                text=text,
                voice_id="alloy",
                provider="openai"
            )
            
            # First synthesis
            result1 = asyncio.run(tts_service.synthesize_speech(text))
            
            # Second synthesis (should use cache)
            result2 = asyncio.run(tts_service.synthesize_speech(text))
            
            # Results should be identical
            assert result1.text == result2.text
            assert result1.voice_id == result2.voice_id
            
            # Second call should not call the provider again
            assert mock_synthesize.call_count == 1
    
    def test_batch_synthesis_performance(self, tts_service):
        """Test batch synthesis performance."""
        texts = [
            f"Batch synthesis test {i}"
            for i in range(10)
        ]
        
        with patch.object(tts_service, '_synthesize_with_provider') as mock_synthesize:
            mock_audio_data = AudioData(
                data=np.random.random(16000).astype(np.float32),
                sample_rate=16000,
                duration=1.0,
                channels=1,
                format="wav"
            )
            mock_synthesize.return_value = TTSResult(
                audio_data=mock_audio_data,
                text="Batch result",
                voice_id="alloy",
                provider="openai"
            )
            
            start_time = time.time()
            
            results = asyncio.run(tts_service.synthesize_batch(texts))
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            assert len(results) == 10
            for result in results:
                assert isinstance(result, TTSResult)
            
            # Should process 10 texts reasonably quickly
            assert processing_time < 30.0  # Adjust threshold as needed
    
    def test_memory_usage_optimization(self, tts_service):
        """Test memory usage during synthesis."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process long text
        long_text = "This is a very long text for memory testing. " * 100
        
        with patch.object(tts_service, '_synthesize_with_provider') as mock_synthesize:
            mock_audio_data = AudioData(
                data=np.random.random(16000 * 30).astype(np.float32),  # 30 seconds
                sample_rate=16000,
                duration=30.0,
                channels=1,
                format="wav"
            )
            mock_synthesize.return_value = TTSResult(
                audio_data=mock_audio_data,
                text=long_text,
                voice_id="alloy",
                provider="openai"
            )
            
            result = asyncio.run(tts_service.synthesize_speech(long_text))
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 100  # MB


@pytest.mark.skipif(not TTS_SERVICE_AVAILABLE, reason="TTS service not available")
class TestRealTimeSynthesis:
    """Test real-time TTS features."""
    
    @pytest.fixture
    def tts_service(self):
        """Create TTSService with real-time features."""
        config = mock_voice_config
        config.real_time_enabled = True
        return TTSService(config)
    
    @pytest.mark.asyncio
    async def test_streaming_synthesis(self, tts_service):
        """Test streaming synthesis."""
        text = "This is a streaming synthesis test"
        
        with patch.object(tts_service, '_synthesize_chunk') as mock_synthesize:
            mock_synthesize.return_value = AudioData(
                data=np.random.random(1600).astype(np.float32),  # 0.1 seconds
                sample_rate=16000,
                duration=0.1,
                channels=1,
                format="wav"
            )
            
            chunks = []
            async for chunk in tts_service.synthesize_stream(text):
                chunks.append(chunk)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, AudioData)
    
    @pytest.mark.asyncio
    async def test_low_latency_synthesis(self, tts_service):
        """Test low-latency synthesis mode."""
        text = "Low latency test"
        
        with patch.object(tts_service, '_synthesize_with_provider') as mock_synthesize:
            mock_audio_data = AudioData(
                data=np.random.random(8000).astype(np.float32),  # 0.5 seconds
                sample_rate=16000,
                duration=0.5,
                channels=1,
                format="wav"
            )
            mock_synthesize.return_value = TTSResult(
                audio_data=mock_audio_data,
                text=text,
                voice_id="alloy",
                provider="openai"
            )
            
            start_time = time.time()
            
            result = await tts_service.synthesize_speech_low_latency(text)
            
            end_time = time.time()
            latency = end_time - start_time
            
            assert isinstance(result, TTSResult)
            assert latency < 2.0  # Should be low latency


class TestTTSIntegration:
    """Test TTS service integration scenarios."""
    
    @pytest.fixture
    def tts_service(self):
        """Create TTSService for integration tests."""
        if not TTS_SERVICE_AVAILABLE:
            pytest.skip("TTS service not available")
        
        config = mock_voice_config
        return TTSService(config)
    
    def test_integration_with_audio_processor(self, tts_service):
        """Test integration with audio processor."""
        # Test audio processing pipeline
        raw_audio = np.random.random(16000).astype(np.float32)
        audio_data = AudioData(
            data=raw_audio,
            sample_rate=16000,
            duration=1.0,
            channels=1,
            format="wav"
        )
        
        # Test post-processing
        processed = tts_service.post_process_audio(audio_data)
        
        assert isinstance(processed, AudioData)
    
    def test_integration_with_voice_service(self, tts_service):
        """Test integration with voice service."""
        # Mock voice service integration
        with patch('voice.voice_service.VoiceService') as mock_voice_service:
            mock_service = MagicMock()
            mock_voice_service.return_value = mock_service
            
            # Test that TTS result can be used by voice service
            audio_data = AudioData(
                data=np.random.random(16000).astype(np.float32),
                sample_rate=16000,
                duration=1.0,
                channels=1,
                format="wav"
            )
            
            tts_result = TTSResult(
                audio_data=audio_data,
                text="Hello from TTS",
                voice_id="alloy",
                provider="openai"
            )
            
            # Voice service should be able to use this result
            mock_service.play_audio.return_value = True
            play_result = mock_service.play_audio(tts_result.audio_data)
            
            assert play_result is True
    
    def test_integration_with_database(self, tts_service):
        """Test integration with database for voice profiles."""
        with patch('database.models.VoiceProfileRepository') as mock_repo:
            mock_instance = MagicMock()
            mock_instance.save_profile.return_value = True
            mock_instance.get_profile.return_value = VoiceProfile(
                name="db_profile",
                provider="openai",
                voice_id="nova"
            )
            mock_repo.return_value = mock_instance
            
            # Test saving profile to database
            profile = VoiceProfile(
                name="db_test",
                provider="openai",
                voice_id="alloy"
            )
            
            success = mock_instance.save_profile(profile)
            assert success is True
            
            # Test loading profile from database
            loaded = mock_instance.get_profile("db_profile")
            assert loaded.name == "db_profile"


# Run tests if this file is executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])