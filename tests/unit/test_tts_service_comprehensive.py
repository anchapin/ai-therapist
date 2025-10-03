#!/usr/bin/env python3
"""
Comprehensive unit tests for TTS service module.
"""

import pytest
import asyncio
import sys
import os
import tempfile
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import dataclass
from typing import Optional, Dict, List, Any

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import test utilities for safe module loading
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from tests.test_utils import (
    setup_voice_module_mocks,
    get_voice_config_module,
    get_tts_service_module,
    get_audio_processor_module
)

# Set up mocks
setup_voice_module_mocks(project_root)

# Import modules safely
config_module = get_voice_config_module(project_root)
VoiceConfig = config_module.VoiceConfig

audio_processor_module = get_audio_processor_module(project_root)
tts_service_module = get_tts_service_module(project_root)

# Extract classes from the module
TTSService = tts_service_module.TTSService
TTSResult = tts_service_module.TTSResult
AudioData = audio_processor_module.AudioData


class TestTTSService:
    """Test TTSService class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = MagicMock()
        config.audio = MagicMock()
        config.audio.sample_rate = 22050
        config.audio.channels = 1
        config.audio.tts_provider = "openai"
        config.audio.tts_voice = "alloy"
        config.audio.tts_model = "tts-1"
        config.audio.tts_speed = 1.0
        config.audio.tts_pitch = 1.0
        config.audio.tts_volume = 1.0
        config.audio.tts_emotion = "neutral"
        config.audio.tts_cache_enabled = True
        config.audio.tts_cache_size = 50
        config.audio.tts_max_text_length = 1000
        return config

    @pytest.fixture
    def tts_service(self, mock_config):
        """Create TTS service for testing."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            with patch('openai', MagicMock()):
                return TTSService(mock_config)

    def test_initialization(self, tts_service, mock_config):
        """Test TTS service initialization."""
        assert tts_service.config == mock_config
        assert tts_service.primary_provider == "openai"
        assert tts_service.is_initialized == True
        assert tts_service.cache == {}
        assert tts_service.request_count == 0
        assert tts_service.error_count == 0

    def test_initialization_without_api_key(self, mock_config):
        """Test initialization without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('openai', MagicMock()):
                service = TTSService(mock_config)
                # Should still initialize but without OpenAI client
                assert service.config == mock_config

    @pytest.mark.asyncio
    async def test_synthesize_speech_basic(self, tts_service):
        """Test basic speech synthesis."""
        # Mock the OpenAI response
        with patch.object(tts_service, '_synthesize_with_openai', new_callable=AsyncMock) as mock_openai:
            mock_result = TTSResult(
                audio_data=b'synthesized_audio',
                duration=1.5,
                sample_rate=22050,
                provider='openai',
                voice='alloy'
            )
            mock_openai.return_value = mock_result

            result = await tts_service.synthesize_speech("Hello world")

            assert isinstance(result, TTSResult)
            assert result.audio_data == b'synthesized_audio'
            assert result.duration == 1.5
            assert result.provider == 'openai'
            assert result.voice == 'alloy'
            mock_openai.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesize_speech_with_voice_profile(self, tts_service):
        """Test speech synthesis with custom voice profile."""
        voice_profile = {
            'voice': 'echo',
            'speed': 1.2,
            'pitch': 0.9,
            'emotion': 'calm'
        }

        with patch.object(tts_service, '_synthesize_with_openai', new_callable=AsyncMock) as mock_openai:
            mock_result = TTSResult(
                audio_data=b'synthesized_audio',
                duration=1.5,
                sample_rate=22050,
                provider='openai',
                voice='echo'
            )
            mock_openai.return_value = mock_result

            result = await tts_service.synthesize_speech("Hello world", voice_profile=voice_profile)

            assert result.voice == 'echo'
            mock_openai.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesize_speech_no_provider_available(self, tts_service):
        """Test synthesis when no provider is available."""
        tts_service.openai_client = None
        tts_service.elevenlabs_client = None
        tts_service.piper_voices = {}

        with pytest.raises(Exception):
            await tts_service.synthesize_speech("Hello world")

    @pytest.mark.asyncio
    async def test_synthesize_speech_cache_hit(self, tts_service):
        """Test synthesis with cache hit."""
        text = "Hello world"
        voice_profile = {'voice': 'alloy'}
        cache_key = tts_service._generate_cache_key(text, voice_profile)

        # Pre-populate cache
        cached_result = TTSResult(
            audio_data=b'cached_audio',
            duration=1.0,
            sample_rate=22050,
            provider='openai',
            voice='alloy',
            cached=True
        )
        tts_service.cache[cache_key] = cached_result

        result = await tts_service.synthesize_speech(text, voice_profile=voice_profile)

        assert result.audio_data == b'cached_audio'
        assert result.cached == True

    @pytest.mark.asyncio
    async def test_synthesize_speech_empty_text(self, tts_service):
        """Test synthesis with empty text."""
        result = await tts_service.synthesize_speech("")
        assert isinstance(result, TTSResult)
        assert len(result.audio_data) == 0

    @pytest.mark.asyncio
    async def test_synthesize_speech_text_too_long(self, tts_service):
        """Test synthesis with text that's too long."""
        long_text = "Hello " * 500  # Very long text
        tts_service.config.audio.tts_max_text_length = 100

        with pytest.raises(ValueError):
            await tts_service.synthesize_speech(long_text)

    @pytest.mark.asyncio
    async def test_synthesize_with_openai(self, tts_service):
        """Test OpenAI synthesis."""
        audio_data = AudioData(
            data=np.array([0.1, 0.2, 0.3]),
            sample_rate=22050,
            duration=0.136,
            channels=1
        )

        with patch.object(tts_service, '_call_openai_tts', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = b'openai_audio'

            result = await tts_service._synthesize_with_openai("Hello world")

            assert isinstance(result, TTSResult)
            assert result.audio_data == b'openai_audio'
            assert result.provider == 'openai'
            mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesize_with_elevenlabs(self, tts_service):
        """Test ElevenLabs synthesis."""
        tts_service.elevenlabs_client = MagicMock()

        with patch.object(tts_service, '_call_elevenlabs_tts', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = b'elevenlabs_audio'

            result = await tts_service._synthesize_with_elevenlabs("Hello world")

            assert isinstance(result, TTSResult)
            assert result.audio_data == b'elevenlabs_audio'
            assert result.provider == 'elevenlabs'
            mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesize_with_piper(self, tts_service):
        """Test Piper synthesis."""
        tts_service.piper_voices = {'test_voice': MagicMock()}

        with patch.object(tts_service, '_call_piper_tts', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = b'piper_audio'

            result = await tts_service._synthesize_with_piper("Hello world")

            assert isinstance(result, TTSResult)
            assert result.audio_data == b'piper_audio'
            assert result.provider == 'piper'
            mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_provider_fallback(self, tts_service):
        """Test provider fallback mechanism."""
        # Set up multiple providers
        tts_service.elevenlabs_client = MagicMock()
        tts_service.piper_voices = {'test_voice': MagicMock()}

        # Mock primary provider to fail
        with patch.object(tts_service, '_synthesize_with_openai', new_callable=AsyncMock) as mock_openai:
            mock_openai.side_effect = Exception("OpenAI failed")

            # Mock fallback to succeed
            with patch.object(tts_service, '_synthesize_with_elevenlabs', new_callable=AsyncMock) as mock_elevenlabs:
                mock_result = TTSResult(
                    audio_data=b'fallback_audio',
                    duration=1.0,
                    sample_rate=22050,
                    provider='elevenlabs',
                    voice='alloy'
                )
                mock_elevenlabs.return_value = mock_result

                result = await tts_service.synthesize_speech("Hello world")

                assert result.provider == 'elevenlabs'
                assert result.audio_data == b'fallback_audio'
                mock_elevenlabs.assert_called_once()

    def test_get_available_voices(self, tts_service):
        """Test getting available voices."""
        tts_service.openai_voices = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
        tts_service.elevenlabs_voices = ['rachel', 'domi', 'bella', 'antoni', 'elli']
        tts_service.piper_voices = {'lessac': 'path', 'danny': 'path'}

        voices = tts_service.get_available_voices()

        assert 'openai' in voices
        assert 'elevenlabs' in voices
        assert 'piper' in voices
        assert len(voices['openai']) == 6
        assert len(voices['elevenlabs']) == 5
        assert len(voices['piper']) == 2

    def test_get_available_providers(self, tts_service):
        """Test getting available providers."""
        tts_service.openai_client = MagicMock()
        tts_service.elevenlabs_client = None
        tts_service.piper_voices = {}

        providers = tts_service.get_available_providers()

        assert 'openai' in providers
        assert 'elevenlabs' not in providers
        assert 'piper' not in providers

    def test_is_available(self, tts_service):
        """Test availability check."""
        tts_service.openai_client = MagicMock()
        assert tts_service.is_available() == True

        tts_service.openai_client = None
        tts_service.elevenlabs_client = None
        tts_service.piper_voices = {}
        assert tts_service.is_available() == False

    def test_get_statistics(self, tts_service):
        """Test getting service statistics."""
        tts_service.request_count = 10
        tts_service.error_count = 2

        stats = tts_service.get_statistics()

        assert stats['request_count'] == 10
        assert stats['error_count'] == 2
        assert stats['success_rate'] == 0.8
        assert 'cache_size' in stats
        assert 'available_providers' in stats

    def test_test_service(self, tts_service):
        """Test service functionality test."""
        tts_service.openai_client = MagicMock()

        with patch.object(tts_service, '_synthesize_with_openai', new_callable=AsyncMock) as mock_openai:
            mock_result = TTSResult(
                audio_data=b'test_audio',
                duration=0.5,
                sample_rate=22050,
                provider='openai',
                voice='alloy'
            )
            mock_openai.return_value = mock_result

            result = asyncio.run(tts_service.test_service())

            assert result == True
            mock_openai.assert_called_once()

    def test_test_service_failure(self, tts_service):
        """Test service functionality test failure."""
        tts_service.openai_client = None
        tts_service.elevenlabs_client = None
        tts_service.piper_voices = {}

        result = tts_service.test_service()
        assert result == False

    def test_cache_key_generation(self, tts_service):
        """Test cache key generation."""
        text = "Hello world"
        voice_profile = {'voice': 'alloy', 'speed': 1.2}

        key1 = tts_service._generate_cache_key(text, voice_profile)
        key2 = tts_service._generate_cache_key(text, voice_profile)

        assert key1 == key2
        assert isinstance(key1, str)
        assert len(key1) > 0

        # Different text should generate different key
        key3 = tts_service._generate_cache_key("Different text", voice_profile)
        assert key1 != key3

    def test_cache_operations(self, tts_service):
        """Test cache add/get operations."""
        cache_key = "test_key"
        test_result = TTSResult(
            audio_data=b'test_audio',
            duration=1.0,
            sample_rate=22050,
            provider='openai',
            voice='alloy'
        )

        # Add to cache
        tts_service._add_to_cache(cache_key, test_result)
        assert cache_key in tts_service.cache

        # Get from cache
        result = tts_service._get_from_cache(cache_key)
        assert result == test_result

        # Test cache expiry
        old_timestamp = time.time() - 25 * 3600  # 25 hours ago
        tts_service.cache[cache_key]['timestamp'] = old_timestamp
        result = tts_service._get_from_cache(cache_key)
        assert result is None

    def test_cache_max_size_enforcement(self, tts_service):
        """Test cache max size enforcement."""
        tts_service.max_cache_size = 2

        # Add items up to max size
        for i in range(3):
            result = TTSResult(
                audio_data=f'audio_{i}'.encode(),
                duration=1.0,
                sample_rate=22050,
                provider='openai',
                voice='alloy'
            )
            tts_service._add_to_cache(f'key_{i}', result)

        # Should only have 2 items (oldest removed)
        assert len(tts_service.cache) == 2
        assert 'key_0' not in tts_service.cache
        assert 'key_1' in tts_service.cache
        assert 'key_2' in tts_service.cache

    def test_cleanup(self, tts_service):
        """Test cleanup."""
        tts_service.openai_client = MagicMock()
        tts_service.elevenlabs_client = MagicMock()
        tts_service.cache = {'test': 'value'}

        tts_service.cleanup()

        assert tts_service.openai_client is None
        assert tts_service.elevenlabs_client is None
        assert len(tts_service.cache) == 0

    def test_set_voice_profile(self, tts_service):
        """Test setting voice profile."""
        voice_profile = {
            'voice': 'echo',
            'speed': 1.2,
            'pitch': 0.9,
            'emotion': 'calm'
        }

        tts_service.set_voice_profile(voice_profile)

        assert tts_service.current_voice_profile == voice_profile

    def test_get_preferred_provider(self, tts_service):
        """Test getting preferred provider."""
        provider = tts_service.get_preferred_provider()
        assert provider == "openai"

    def test_error_handling_in_synthesis(self, tts_service):
        """Test error handling during synthesis."""
        with patch.object(tts_service, '_synthesize_with_openai', new_callable=AsyncMock) as mock_openai:
            mock_openai.side_effect = Exception("Synthesis failed")

            with pytest.raises(Exception):
                asyncio.run(tts_service.synthesize_speech("Hello world"))

    def test_voice_validation(self, tts_service):
        """Test voice validation."""
        # Test valid voice
        assert tts_service._validate_voice('alloy') == True

        # Test invalid voice
        assert tts_service._validate_voice('invalid_voice') == False

    def test_text_preprocessing(self, tts_service):
        """Test text preprocessing."""
        # Test normal text
        processed = tts_service._preprocess_text("Hello world!")
        assert processed == "Hello world!"

        # Test text with special characters
        processed = tts_service._preprocess_text("Hello @#$% world")
        assert "@" not in processed and "#" not in processed

        # Test very long text truncation
        long_text = "Hello " * 100
        processed = tts_service._preprocess_text(long_text)
        assert len(processed) <= tts_service.max_text_length

    def test_audio_format_conversion(self, tts_service):
        """Test audio format conversion."""
        # Test numpy array conversion
        audio_array = np.array([0.1, 0.2, 0.3])
        converted = tts_service._convert_audio_format(audio_array, "bytes")
        assert isinstance(converted, bytes)

        # Test bytes conversion
        audio_bytes = b'raw_audio_data'
        converted = tts_service._convert_audio_format(audio_bytes, "numpy")
        assert isinstance(converted, np.ndarray)

    @pytest.mark.asyncio
    async def test_streaming_synthesis(self, tts_service):
        """Test streaming synthesis."""
        with patch.object(tts_service, '_stream_with_openai') as mock_stream:
            async def mock_generator():
                yield TTSResult(
                    audio_data=b'chunk1',
                    duration=0.5,
                    sample_rate=22050,
                    provider='openai',
                    voice='alloy'
                )
                yield TTSResult(
                    audio_data=b'chunk2',
                    duration=0.5,
                    sample_rate=22050,
                    provider='openai',
                    voice='alloy'
                )

            mock_stream.return_value = mock_generator()

            chunks = []
            async for chunk in tts_service.synthesize_stream("Hello world"):
                chunks.append(chunk)

            assert len(chunks) == 2
            assert chunks[0].audio_data == b'chunk1'
            assert chunks[1].audio_data == b'chunk2'


class TestTTSResult:
    """Test TTSResult class."""

    def test_tts_result_creation(self):
        """Test TTS result creation."""
        result = TTSResult(
            audio_data=b'test_audio',
            duration=1.5,
            sample_rate=22050,
            provider='openai',
            voice='alloy'
        )

        assert result.audio_data == b'test_audio'
        assert result.duration == 1.5
        assert result.sample_rate == 22050
        assert result.provider == 'openai'
        assert result.voice == 'alloy'

    def test_tts_result_with_optional_fields(self):
        """Test TTS result with optional fields."""
        result = TTSResult(
            audio_data=b'test_audio',
            duration=1.5,
            sample_rate=22050,
            provider='openai',
            voice='alloy',
            alternatives=[{'text': 'alternative', 'confidence': 0.8}],
            processing_time=0.5,
            emotion='happy'
        )

        assert len(result.alternatives) == 1
        assert result.alternatives[0]['text'] == 'alternative'
        assert result.processing_time == 0.5
        assert result.emotion == 'happy'

    def test_tts_result_to_dict(self):
        """Test TTS result to dictionary conversion."""
        result = TTSResult(
            audio_data=b'test_audio',
            duration=1.5,
            sample_rate=22050,
            provider='openai',
            voice='alloy'
        )

        result_dict = result.__dict__ if hasattr(result, '__dict__') else {}
        assert 'audio_data' in result_dict
        assert 'duration' in result_dict
        assert 'provider' in result_dict