#!/usr/bin/env python3
"""
Additional unit tests for STT service to reach 90%+ coverage.
"""

import pytest
import asyncio
import sys
import os
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import test utilities for safe module loading
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from tests.test_utils import (
    setup_voice_module_mocks,
    get_voice_config_module,
    get_stt_service_module,
    get_audio_processor_module
)

# Set up mocks
setup_voice_module_mocks(project_root)

# Import modules safely
config_module = get_voice_config_module(project_root)
VoiceConfig = config_module.VoiceConfig

audio_processor_module = get_audio_processor_module(project_root)
stt_service_module = get_stt_service_module(project_root)

# Extract classes from the module
STTService = stt_service_module.STTService
STTResult = stt_service_module.STTResult
AudioData = audio_processor_module.AudioData


class TestSTTServiceAdditional:
    """Additional tests for STT service to improve coverage."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = MagicMock()
        config.audio = MagicMock()
        config.audio.sample_rate = 16000
        config.audio.channels = 1
        config.audio.max_recording_duration = 30.0
        config.audio.stt_provider = "openai"
        config.audio.stt_confidence_threshold = 0.7
        config.audio.stt_language = "en"
        config.audio.stt_model = "whisper-1"
        config.audio.stt_fallback_enabled = True
        config.audio.stt_cache_enabled = True
        config.audio.stt_cache_size = 100
        return config

    @pytest.fixture
    def stt_service(self, mock_config):
        """Create STT service for testing."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            with patch('openai', MagicMock()):
                return STTService(mock_config)

    def test_audio_quality_score_comprehensive_scenarios(self, stt_service):
        """Test audio quality score calculation with various scenarios."""

        # Test optimal energy levels
        optimal_audio = AudioData(
            data=np.array([0.2, -0.3, 0.4, -0.5, 0.6]),  # RMS around 0.4
            sample_rate=16000,
            duration=0.3125,
            channels=1
        )
        score = stt_service._calculate_audio_quality_score(optimal_audio)
        assert 0.5 <= score <= 1.0  # Should get good score

        # Test suboptimal but acceptable energy levels
        suboptimal_audio = AudioData(
            data=np.array([0.06, -0.08, 0.09]),  # RMS around 0.07
            sample_rate=16000,
            duration=0.1875,
            channels=1
        )
        score = stt_service._calculate_audio_quality_score(suboptimal_audio)
        assert 0.2 <= score <= 0.7  # Should get moderate score

        # Test very low energy levels
        low_energy_audio = AudioData(
            data=np.array([0.01, -0.02, 0.01]),  # RMS around 0.01
            sample_rate=16000,
            duration=0.1875,
            channels=1
        )
        score = stt_service._calculate_audio_quality_score(low_energy_audio)
        assert 0.1 <= score <= 0.4  # Should get low score

        # Test high energy but non-clipping audio
        high_energy_audio = AudioData(
            data=np.array([0.85, -0.88, 0.92]),  # RMS around 0.88
            sample_rate=16000,
            duration=0.1875,
            channels=1
        )
        score = stt_service._calculate_audio_quality_score(high_energy_audio)
        assert 0.2 <= score <= 0.6  # Should get moderate-low score

    def test_sentiment_score_calculation_various_texts(self, stt_service):
        """Test sentiment score calculation with various text inputs."""

        # Test positive sentiment
        positive_text = "I feel much better today and I'm hopeful about the future"
        score = stt_service._calculate_sentiment_score(positive_text)
        assert score > 0.0

        # Test negative sentiment
        negative_text = "I feel very sad and worried about everything"
        score = stt_service._calculate_sentiment_score(negative_text)
        assert score < 0.0

        # Test neutral/mixed sentiment
        neutral_text = "Today was okay but tomorrow might be better"
        score = stt_service._calculate_sentiment_score(neutral_text)
        assert -0.5 <= score <= 0.5

        # Test empty text
        score = stt_service._calculate_sentiment_score("")
        assert score == 0.0

        # Test text with no sentiment words
        score = stt_service._calculate_sentiment_score("The weather is cloudy today")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_enhance_stt_result_with_therapy_keywords(self, stt_service):
        """Test STT result enhancement with therapy keyword detection."""

        # Create base result
        base_result = STTResult(
            text="I feel anxious about my therapy session",
            confidence=0.8,
            language="en",
            duration=2.5,
            provider="openai",
            alternatives=[],
            therapy_keywords=[],
            crisis_keywords=[]
        )

        audio_data = AudioData(
            data=np.array([0.1, 0.2, 0.3]),
            sample_rate=16000,
            duration=0.1875,
            channels=1
        )

        enhanced_result = await stt_service._enhance_stt_result(base_result, audio_data)

        # Check that therapy keywords were detected
        assert len(enhanced_result.therapy_keywords) > 0
        assert "anxious" in enhanced_result.therapy_keywords or "therapy" in enhanced_result.therapy_keywords
        assert enhanced_result.therapy_keywords_detected == enhanced_result.therapy_keywords

    @pytest.mark.asyncio
    async def test_enhance_stt_result_with_crisis_keywords(self, stt_service):
        """Test STT result enhancement with crisis keyword detection."""

        # Create base result with crisis indicators
        base_result = STTResult(
            text="I'm having suicidal thoughts and want to harm myself",
            confidence=0.9,
            language="en",
            duration=3.0,
            provider="openai",
            alternatives=[],
            therapy_keywords=[],
            crisis_keywords=[]
        )

        audio_data = AudioData(
            data=np.array([0.1, 0.2, 0.3]),
            sample_rate=16000,
            duration=0.1875,
            channels=1
        )

        enhanced_result = await stt_service._enhance_stt_result(base_result, audio_data)

        # Check that crisis keywords were detected
        assert len(enhanced_result.crisis_keywords) > 0
        assert enhanced_result.is_crisis == True
        assert enhanced_result.crisis_keywords_detected == enhanced_result.crisis_keywords

    @pytest.mark.asyncio
    async def test_enhance_stt_result_with_sentiment_analysis(self, stt_service):
        """Test STT result enhancement with sentiment analysis."""

        # Create base result
        base_result = STTResult(
            text="I feel happy and positive about my progress",
            confidence=0.85,
            language="en",
            duration=2.0,
            provider="openai",
            alternatives=[],
            therapy_keywords=[],
            crisis_keywords=[]
        )

        audio_data = AudioData(
            data=np.array([0.1, 0.2, 0.3]),
            sample_rate=16000,
            duration=0.1875,
            channels=1
        )

        enhanced_result = await stt_service._enhance_stt_result(base_result, audio_data)

        # Check that sentiment was calculated
        assert enhanced_result.sentiment_score is not None
        assert enhanced_result.sentiment_score > 0.0  # Should be positive

    def test_audio_conversion_openai_with_resampling(self, stt_service):
        """Test OpenAI audio conversion with resampling required."""

        # Create audio with wrong sample rate
        audio_data = AudioData(
            data=np.array([0.1, 0.2, 0.3, 0.4]),
            sample_rate=44100,  # Not 16kHz
            duration=0.0907,
            channels=1,
            format="float32"
        )

        with patch('librosa.resample') as mock_resample:
            mock_resample.return_value = np.array([0.15, 0.25, 0.35])

            result = stt_service._convert_audio_for_openai(audio_data)

            mock_resample.assert_called_once()
            assert isinstance(result, bytes)

    def test_cache_operations_with_expiry(self, stt_service):
        """Test cache operations with expiry functionality."""

        # Add item to cache
        cache_key = "test_key"
        test_result = STTResult(
            text="test text",
            confidence=0.9,
            language="en",
            duration=1.0,
            provider="openai",
            alternatives=[]
        )

        # Mock timestamp to be older than 24 hours
        with patch('time.time', return_value=1000000):
            stt_service._add_to_cache(cache_key, test_result)

        # Mock current time to be more than 24 hours later
        with patch('time.time', return_value=1000000 + 25 * 3600):
            # Try to get from cache - should return None due to expiry
            result = stt_service._get_from_cache(cache_key)
            assert result is None

    def test_cache_operations_max_size_enforcement(self, stt_service):
        """Test cache operations with max size enforcement."""

        # Set a small cache size for testing
        stt_service.max_cache_size = 2

        # Add first item
        result1 = STTResult(
            text="first text",
            confidence=0.9,
            language="en",
            duration=1.0,
            provider="openai",
            alternatives=[]
        )
        stt_service._add_to_cache("key1", result1)

        # Add second item
        result2 = STTResult(
            text="second text",
            confidence=0.8,
            language="en",
            duration=1.0,
            provider="openai",
            alternatives=[]
        )
        stt_service._add_to_cache("key2", result2)

        # Add third item - should remove oldest
        result3 = STTResult(
            text="third text",
            confidence=0.85,
            language="en",
            duration=1.0,
            provider="openai",
            alternatives=[]
        )
        stt_service._add_to_cache("key3", result3)

        # Check that oldest item was removed
        assert stt_service._get_from_cache("key1") is None
        assert stt_service._get_from_cache("key2") is not None
        assert stt_service._get_from_cache("key3") is not None

    def test_provider_fallback_edge_cases(self, stt_service):
        """Test provider fallback chain with edge cases."""

        # Test with no available providers
        stt_service.openai_client = None
        stt_service.google_speech_client = None
        stt_service.whisper_model = None

        chain = stt_service._get_provider_fallback_chain("openai")
        assert chain == []

        # Test with non-existent preferred provider
        stt_service.openai_client = MagicMock()
        chain = stt_service._get_provider_fallback_chain("nonexistent")
        assert "openai" in chain

        # Test with multiple available providers
        stt_service.google_speech_client = MagicMock()
        stt_service.whisper_model = MagicMock()

        chain = stt_service._get_provider_fallback_chain("google")
        assert chain[0] == "google"
        assert "openai" in chain[1:]
        assert "whisper" in chain[1:]

    def test_error_handling_in_quality_calculation(self, stt_service):
        """Test error handling in audio quality calculation."""

        # Test with invalid audio data that causes exception
        with patch.object(np, 'std', side_effect=Exception("Calculation error")):
            audio_data = AudioData(
                data=np.array([0.1, 0.2, 0.3]),
                sample_rate=16000,
                duration=0.1875,
                channels=1
            )

            score = stt_service._calculate_audio_quality_score(audio_data)
            assert score == 0.5  # Default quality score on error

    def test_error_handling_in_sentiment_calculation(self, stt_service):
        """Test error handling in sentiment calculation."""

        # Test with invalid input that causes exception
        with patch.object(str, 'lower', side_effect=Exception("Processing error")):
            score = stt_service._calculate_sentiment_score("test text")
            assert score == 0.0  # Default sentiment on error

    def test_enhance_stt_result_error_handling(self, stt_service):
        """Test error handling in STT result enhancement."""

        base_result = STTResult(
            text="test text",
            confidence=0.8,
            language="en",
            duration=1.0,
            provider="openai",
            alternatives=[]
        )

        # Mock audio data that causes issues
        audio_data = MagicMock()
        audio_data.data = MagicMock()

        with patch.object(stt_service, '_calculate_sentiment_score', side_effect=Exception("Sentiment error")):
            # Should not crash, should return original result with minimal enhancement
            result = asyncio.run(stt_service._enhance_stt_result(base_result, audio_data))
            assert result.text == base_result.text
            assert result.confidence == base_result.confidence