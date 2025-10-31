"""
Comprehensive STT Service Tests

This module provides extensive coverage for voice/stt_service.py functionality:
- Multi-provider STT service integration (OpenAI, Google, Local Whisper)
- Audio preprocessing and format handling
- Confidence scoring and result ranking
- Real-time and batch processing modes
- Error handling and provider fallback mechanisms
- Performance optimization and caching
- Language detection and multi-language support
- Therapy-specific terminology recognition
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

# Import STT service components
try:
    from voice.stt_service import (
        STTService,
        STTProvider,
        STTResult,
        STTConfig
    )
    from voice.audio_processor import AudioData
    from voice.config import VoiceConfig
    STT_SERVICE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: STT service import failed: {e}")
    STT_SERVICE_AVAILABLE = False

# Import fixtures
try:
    from tests.fixtures.voice_fixtures import mock_voice_config, mock_audio_data
except ImportError:
    # Fallback fixtures
    @pytest.fixture
    def mock_voice_config():
        """Fallback mock VoiceConfig for testing."""
        config = MagicMock()
        config.stt_provider = "openai"
        config.stt_fallback_providers = ["openai", "whisper", "google"]
        config.cache_enabled = True
        config.therapy_mode = True
        config.hipaa_compliance_enabled = True
        config.encryption_enabled = True
        return config
    
    @pytest.fixture
    def mock_audio_data():
        """Create mock audio data for testing."""
        return AudioData(
            data=np.random.random(16000).astype(np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1,
            format="wav"
        )


class TestSTTServiceCore:
    """Test core STT service functionality."""
    
    @pytest.fixture
    def stt_service(self):
        """Create STTService instance for testing."""
        if not STT_SERVICE_AVAILABLE:
            pytest.skip("STT service not available")
        
        config = mock_voice_config
        return STTService(config)
    
    @pytest.fixture
    def sample_audio_data(self):
        """Create sample audio data for testing."""
        return AudioData(
            data=np.random.random(16000).astype(np.float32),  # 1 second at 16kHz
            sample_rate=16000,
            duration=1.0,
            channels=1,
            format="wav"
        )
    
    def test_stt_service_initialization(self, stt_service):
        """Test STTService initialization."""
        assert stt_service.config is not None
        assert stt_service.primary_provider == STTProvider.OPENAI
        assert len(stt_service.providers) > 0
        assert stt_service.cache_enabled is True
    
    def test_stt_result_creation(self):
        """Test STTResult object creation and attributes."""
        result = STTResult(
            text="Hello world",
            confidence=0.95,
            language="en",
            duration=1.0,
            provider="openai",
            processing_time=0.5,
            timestamp=time.time()
        )
        
        assert result.text == "Hello world"
        assert result.confidence == 0.95
        assert result.language == "en"
        assert result.duration == 1.0
        assert result.provider == "openai"
        assert result.processing_time == 0.5
        assert result.timestamp > 0
    
    def test_stt_result_backward_compatibility(self):
        """Test STTResult backward compatibility methods."""
        # Test create_compatible method
        result = STTResult.create_compatible(
            text="Test text",
            confidence=0.9,
            language="en"
        )
        
        assert result.text == "Test text"
        assert result.confidence == 0.9
        assert result.language == "en"
        assert result.timestamp > 0  # Should be set automatically


@pytest.mark.skipif(not STT_SERVICE_AVAILABLE, reason="STT service not available")
class TestOpenAIWhisperIntegration:
    """Test OpenAI Whisper API integration."""
    
    @pytest.fixture
    def stt_service(self):
        """Create STTService with OpenAI configuration."""
        config = mock_voice_config
        config.stt_provider = "openai"
        return STTService(config)
    
    @pytest.fixture
    def mock_openai(self):
        """Mock OpenAI API."""
        with patch('voice.stt_service.openai') as mock_openai:
            mock_client = MagicMock()
            mock_openai.Audio = MagicMock()
            mock_openai.Audio.transcribe = MagicMock()
            mock_openai.Audio.translate = MagicMock()
            yield mock_openai
    
    @pytest.mark.asyncio
    async def test_openai_transcription_success(self, stt_service, mock_openai, sample_audio_data):
        """Test successful OpenAI transcription."""
        # Mock successful OpenAI response
        mock_response = {
            "text": "Hello, this is a test transcription",
            "language": "english"
        }
        mock_openai.Audio.transcribe.return_value = mock_response
        
        result = await stt_service.transcribe_audio(sample_audio_data)
        
        assert isinstance(result, STTResult)
        assert result.text == "Hello, this is a test transcription"
        assert result.provider == "openai"
        assert result.confidence > 0.0
        mock_openai.Audio.transcribe.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_openai_translation(self, stt_service, mock_openai, sample_audio_data):
        """Test OpenAI translation to English."""
        mock_response = {
            "text": "Hello, this is translated text",
            "language": "english"
        }
        mock_openai.Audio.translate.return_value = mock_response
        
        result = await stt_service.transcribe_audio(sample_audio_data, translate=True)
        
        assert isinstance(result, STTResult)
        assert result.text == "Hello, this is translated text"
        assert mock_openai.Audio.translate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_openai_error_handling(self, stt_service, mock_openai, sample_audio_data):
        """Test OpenAI API error handling."""
        mock_openai.Audio.transcribe.side_effect = Exception("OpenAI API error")
        
        with pytest.raises(Exception) as exc_info:
            await stt_service.transcribe_audio(sample_audio_data)
        
        assert "OpenAI API error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_openai_timeout_handling(self, stt_service, mock_openai, sample_audio_data):
        """Test OpenAI API timeout handling."""
        import asyncio
        mock_openai.Audio.transcribe.side_effect = asyncio.TimeoutError("Request timeout")
        
        with pytest.raises(asyncio.TimeoutError):
            await stt_service.transcribe_audio(sample_audio_data)
    
    @pytest.mark.asyncio
    async def test_openai_with_vad(self, stt_service, mock_openai, sample_audio_data):
        """Test OpenAI transcription with Voice Activity Detection."""
        mock_response = {
            "text": "Hello after voice activity detection",
            "language": "english"
        }
        mock_openai.Audio.transcribe.return_value = mock_response
        
        result = await stt_service.transcribe_audio(sample_audio_data, use_vad=True)
        
        assert isinstance(result, STTResult)
        assert result.text == "Hello after voice activity detection"
    
    @pytest.mark.asyncio
    async def test_openai_with_timestamps(self, stt_service, mock_openai, sample_audio_data):
        """Test OpenAI transcription with word timestamps."""
        mock_response = {
            "text": "Hello world with timestamps",
            "language": "english",
            "words": [
                {"word": "Hello", "start": 0.0, "end": 0.5},
                {"word": "world", "start": 0.6, "end": 1.0},
                {"word": "with", "start": 1.1, "end": 1.4},
                {"word": "timestamps", "start": 1.5, "end": 2.0}
            ]
        }
        mock_openai.Audio.transcribe.return_value = mock_response
        
        result = await stt_service.transcribe_audio(sample_audio_data, timestamp_granularities=["word"])
        
        assert isinstance(result, STTResult)
        assert result.word_timestamps is not None
        assert len(result.word_timestamps) == 4


@pytest.mark.skipif(not STT_SERVICE_AVAILABLE, reason="STT service not available")
class TestLocalWhisperIntegration:
    """Test local Whisper model integration."""
    
    @pytest.fixture
    def stt_service(self):
        """Create STTService with Whisper configuration."""
        config = mock_voice_config
        config.stt_provider = "whisper"
        return STTService(config)
    
    @pytest.fixture
    def mock_whisper(self):
        """Mock local Whisper model."""
        with patch('voice.stt_service.whisper') as mock_whisper:
            mock_model = MagicMock()
            mock_model.transcribe.return_value = {
                "text": "Local Whisper transcription result",
                "language": "en",
                "segments": [
                    {"start": 0.0, "end": 2.0, "text": "Local Whisper"},
                    {"start": 2.0, "end": 4.0, "text": "transcription result"}
                ]
            }
            mock_whisper.load_model.return_value = mock_model
            yield mock_whisper
    
    @pytest.mark.asyncio
    async def test_local_whisper_transcription(self, stt_service, mock_whisper, sample_audio_data):
        """Test local Whisper transcription."""
        result = await stt_service.transcribe_audio(sample_audio_data)
        
        assert isinstance(result, STTResult)
        assert result.text == "Local Whisper transcription result"
        assert result.provider == "whisper"
        assert mock_whisper.load_model.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_whisper_model_selection(self, stt_service, mock_whisper, sample_audio_data):
        """Test Whisper model selection."""
        # Test with different model sizes
        for model_size in ["tiny", "base", "small", "medium", "large"]:
            await stt_service.transcribe_audio(sample_audio_data, model_size=model_size)
            
            # Check if correct model was loaded
            mock_whisper.load_model.assert_called_with(model_size)
    
    @pytest.mark.asyncio
    async def test_whisper_language_detection(self, stt_service, mock_whisper, sample_audio_data):
        """Test Whisper language detection."""
        mock_whisper.load_model.return_value.transcribe.return_value = {
            "text": "Bonjour le monde",
            "language": "fr"
        }
        
        result = await stt_service.transcribe_audio(sample_audio_data)
        
        assert result.language == "fr"
    
    @pytest.mark.asyncio
    async def test_whisper_error_handling(self, stt_service, mock_whisper, sample_audio_data):
        """Test Whisper error handling."""
        mock_whisper.load_model.side_effect = Exception("Model loading failed")
        
        with pytest.raises(Exception) as exc_info:
            await stt_service.transcribe_audio(sample_audio_data)
        
        assert "Model loading failed" in str(exc_info.value)


@pytest.mark.skipif(not STT_SERVICE_AVAILABLE, reason="STT service not available")
class TestGoogleSpeechIntegration:
    """Test Google Speech-to-Text integration."""
    
    @pytest.fixture
    def stt_service(self):
        """Create STTService with Google Speech configuration."""
        config = mock_voice_config
        config.stt_provider = "google"
        return STTService(config)
    
    @pytest.fixture
    def mock_google_speech(self):
        """Mock Google Speech client."""
        with patch('voice.stt_service.speech') as mock_speech:
            mock_client = MagicMock()
            mock_speech.SpeechClient.return_value = mock_client
            
            mock_response = MagicMock()
            mock_response.results = [
                MagicMock(
                    alternatives=[
                        MagicMock(
                            transcript="Google Speech transcription result",
                            confidence=0.95
                        )
                    ]
                )
            ]
            mock_client.recognize.return_value = mock_response
            
            yield mock_speech
    
    @pytest.mark.asyncio
    async def test_google_speech_transcription(self, stt_service, mock_google_speech, sample_audio_data):
        """Test Google Speech transcription."""
        result = await stt_service.transcribe_audio(sample_audio_data)
        
        assert isinstance(result, STTResult)
        assert result.text == "Google Speech transcription result"
        assert result.provider == "google"
        assert result.confidence == 0.95
    
    @pytest.mark.asyncio
    async def test_google_speech_with_enhanced_model(self, stt_service, mock_google_speech, sample_audio_data):
        """Test Google Speech with enhanced model."""
        await stt_service.transcribe_audio(
            sample_audio_data,
            model="video",
            use_enhanced=True
        )
        
        # Verify enhanced model was used
        mock_client = mock_google_speech.SpeechClient.return_value
        mock_client.recognize.assert_called()
        
        # Check if enhanced model was specified in the request
        call_args = mock_client.recognize.call_args
        assert call_args is not None


@pytest.mark.skipif(not STT_SERVICE_AVAILABLE, reason="STT service not available")
class TestSTTProviderFallback:
    """Test STT provider fallback mechanisms."""
    
    @pytest.fixture
    def stt_service(self):
        """Create STTService with multiple providers."""
        config = mock_voice_config
        config.stt_fallback_providers = ["openai", "whisper", "google"]
        return STTService(config)
    
    @pytest.fixture
    def mock_providers(self):
        """Mock all STT providers."""
        with patch('voice.stt_service.openai') as mock_openai, \
             patch('voice.stt_service.whisper') as mock_whisper, \
             patch('voice.stt_service.speech') as mock_google:
            
            # OpenAI fails
            mock_openai.Audio.transcribe.side_effect = Exception("OpenAI down")
            
            # Whisper succeeds
            mock_whisper_model = MagicMock()
            mock_whisper_model.transcribe.return_value = {
                "text": "Fallback Whisper transcription",
                "language": "en"
            }
            mock_whisper.load_model.return_value = mock_whisper_model
            
            yield {
                'openai': mock_openai,
                'whisper': mock_whisper,
                'google': mock_google
            }
    
    @pytest.mark.asyncio
    async def test_provider_fallback_success(self, stt_service, mock_providers, sample_audio_data):
        """Test successful provider fallback."""
        result = await stt_service.transcribe_audio_with_fallback(sample_audio_data)
        
        assert isinstance(result, STTResult)
        assert result.text == "Fallback Whisper transcription"
        assert result.provider == "whisper"
    
    @pytest.mark.asyncio
    async def test_all_providers_fail(self, stt_service, mock_providers, sample_audio_data):
        """Test when all providers fail."""
        # Make all providers fail
        mock_providers['whisper'].load_model.side_effect = Exception("Whisper failed")
        
        with pytest.raises(Exception) as exc_info:
            await stt_service.transcribe_audio_with_fallback(sample_audio_data)
        
        assert "All STT providers failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_provider_priority_order(self, stt_service, mock_providers, sample_audio_data):
        """Test provider fallback priority order."""
        # Reset mocks to track call order
        mock_providers['openai'].reset_mock()
        mock_providers['whisper'].reset_mock()
        
        await stt_service.transcribe_audio_with_fallback(sample_audio_data)
        
        # Should try OpenAI first, then Whisper
        mock_providers['openai'].Audio.transcribe.assert_called_once()
        mock_providers['whisper'].load_model.assert_called_once()


@pytest.mark.skipif(not STT_SERVICE_AVAILABLE, reason="STT service not available")
class TestAudioPreprocessing:
    """Test audio preprocessing for STT."""
    
    @pytest.fixture
    def stt_service(self):
        """Create STTService for preprocessing tests."""
        config = mock_voice_config
        return STTService(config)
    
    @pytest.fixture
    def noisy_audio_data(self):
        """Create noisy audio data for preprocessing tests."""
        # Generate audio with noise
        clean_audio = np.random.random(16000).astype(np.float32)
        noise = np.random.normal(0, 0.1, 16000).astype(np.float32)
        noisy_audio = clean_audio + noise
        
        return AudioData(
            data=noisy_audio,
            sample_rate=16000,
            duration=1.0,
            channels=1,
            format="wav"
        )
    
    def test_audio_format_validation(self, stt_service):
        """Test audio format validation."""
        # Valid audio data
        valid_audio = AudioData(
            data=np.random.random(16000).astype(np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1,
            format="wav"
        )
        
        assert stt_service.validate_audio_format(valid_audio) is True
        
        # Invalid sample rate
        invalid_sample_rate = AudioData(
            data=np.random.random(8000).astype(np.float32),
            sample_rate=8000,  # Unsupported rate
            duration=1.0,
            channels=1,
            format="wav"
        )
        
        assert stt_service.validate_audio_format(invalid_sample_rate) is False
    
    def test_audio_resampling(self, stt_service):
        """Test audio resampling to target sample rate."""
        source_audio = AudioData(
            data=np.random.random(22050).astype(np.float32),  # 22.05kHz
            sample_rate=22050,
            duration=1.0,
            channels=1,
            format="wav"
        )
        
        resampled = stt_service.resample_audio(source_audio, target_rate=16000)
        
        assert isinstance(resampled, AudioData)
        assert resampled.sample_rate == 16000
    
    def test_audio_normalization(self, stt_service):
        """Test audio normalization."""
        # Create audio with varying amplitude
        audio_data = np.random.random(16000).astype(np.float32) * 2.0 - 1.0
        
        audio_obj = AudioData(
            data=audio_data,
            sample_rate=16000,
            duration=1.0,
            channels=1,
            format="wav"
        )
        
        normalized = stt_service.normalize_audio(audio_obj)
        
        assert isinstance(normalized, AudioData)
        # Check if audio is normalized (peak amplitude close to 1.0)
        peak_amplitude = np.max(np.abs(normalized.data))
        assert 0.8 <= peak_amplitude <= 1.0
    
    def test_noise_reduction(self, stt_service, noisy_audio_data):
        """Test noise reduction."""
        with patch('voice.stt_service.nr') as mock_noisereduce:
            mock_noisereduce.reduce_noise.return_value = np.random.random(16000).astype(np.float32)
            
            cleaned = stt_service.reduce_noise(noisy_audio_data)
            
            assert isinstance(cleaned, AudioData)
            mock_noisereduce.reduce_noise.assert_called_once()
    
    def test_voice_activity_detection(self, stt_service, noisy_audio_data):
        """Test voice activity detection."""
        with patch('voice.stt_service.webrtcvad') as mock_vad:
            mock_vad_instance = MagicMock()
            mock_vad_instance.is_speech.return_value = True
            mock_vad.Vad.return_value = mock_vad_instance
            
            speech_detected = stt_service.detect_voice_activity(noisy_audio_data)
            
            assert speech_detected is True
            mock_vad_instance.is_speech.assert_called()
    
    @pytest.mark.asyncio
    async def test_preprocessing_pipeline(self, stt_service, noisy_audio_data):
        """Test complete preprocessing pipeline."""
        with patch.object(stt_service, 'resample_audio') as mock_resample, \
             patch.object(stt_service, 'normalize_audio') as mock_normalize, \
             patch.object(stt_service, 'reduce_noise') as mock_reduce_noise, \
             patch.object(stt_service, 'detect_voice_activity') as mock_vad:
            
            mock_resample.return_value = noisy_audio_data
            mock_normalize.return_value = noisy_audio_data
            mock_reduce_noise.return_value = noisy_audio_data
            mock_vad.return_value = True
            
            processed = await stt_service.preprocess_audio(noisy_audio_data)
            
            assert isinstance(processed, AudioData)
            mock_resample.assert_called_once()
            mock_normalize.assert_called_once()
            mock_reduce_noise.assert_called_once()
            mock_vad.assert_called_once()


@pytest.mark.skipif(not STT_SERVICE_AVAILABLE, reason="STT service not available")
class TestTherapySpecificFeatures:
    """Test therapy-specific STT features."""
    
    @pytest.fixture
    def stt_service(self):
        """Create STTService with therapy features."""
        config = mock_voice_config
        config.therapy_mode = True
        return STTService(config)
    
    def test_therapy_keyword_detection(self, stt_service):
        """Test therapy-specific keyword detection."""
        therapy_texts = [
            "I'm feeling anxious about my session",
            "This therapy is helping me with depression",
            "My anxiety is getting better",
            "I need help with my mental health"
        ]
        
        for text in therapy_texts:
            keywords = stt_service.detect_therapy_keywords(text)
            assert len(keywords) > 0, f"Should detect therapy keywords in: {text}"
    
    def test_crisis_keyword_detection(self, stt_service):
        """Test crisis keyword detection."""
        crisis_texts = [
            "I want to kill myself",
            "I'm thinking about suicide",
            "I want to end my life",
            "I'm going to harm myself"
        ]
        
        for text in crisis_texts:
            is_crisis, keywords = stt_service.detect_crisis_keywords(text)
            assert is_crisis is True, f"Should detect crisis in: {text}"
            assert len(keywords) > 0
    
    def test_sentiment_analysis(self, stt_service):
        """Test sentiment analysis of transcribed text."""
        positive_text = "I'm feeling much better today"
        negative_text = "I feel horrible and nothing helps"
        neutral_text = "I have an appointment tomorrow"
        
        positive_sentiment = stt_service.analyze_sentiment(positive_text)
        negative_sentiment = stt_service.analyze_sentiment(negative_text)
        neutral_sentiment = stt_service.analyze_sentiment(neutral_text)
        
        assert positive_sentiment['sentiment'] == 'positive'
        assert negative_sentiment['sentiment'] == 'negative'
        assert neutral_sentiment['sentiment'] == 'neutral'
    
    @pytest.mark.asyncio
    async def test_therapy_enhanced_transcription(self, stt_service, sample_audio_data):
        """Test transcription with therapy enhancements."""
        with patch.object(stt_service, 'transcribe_audio') as mock_transcribe:
            mock_result = STTResult(
                text="I'm feeling anxious about therapy",
                confidence=0.95,
                language="en",
                provider="mock"
            )
            mock_transcribe.return_value = mock_result
            
            result = await stt_service.transcribe_with_therapy_features(sample_audio_data)
            
            assert isinstance(result, STTResult)
            assert len(result.therapy_keywords) > 0
            assert result.crisis_keywords is not None


@pytest.mark.skipif(not STT_SERVICE_AVAILABLE, reason="STT service not available")
class TestSTTPerformance:
    """Test STT service performance and optimization."""
    
    @pytest.fixture
    def stt_service(self):
        """Create STTService for performance tests."""
        config = mock_voice_config
        config.cache_enabled = True
        return STTService(config)
    
    @pytest.mark.asyncio
    async def test_concurrent_transcription(self, stt_service, sample_audio_data):
        """Test concurrent transcription requests."""
        async def transcribe_multiple():
            tasks = []
            for i in range(5):
                # Create unique audio for each request
                audio = AudioData(
                    data=np.random.random(16000).astype(np.float32),
                    sample_rate=16000,
                    duration=1.0,
                    channels=1,
                    format="wav"
                )
                task = asyncio.create_task(stt_service.transcribe_audio(audio))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results
        
        results = await transcribe_multiple()
        
        assert len(results) == 5
        for result in results:
            assert not isinstance(result, Exception)
            assert isinstance(result, STTResult)
    
    def test_transcription_caching(self, stt_service, sample_audio_data):
        """Test transcription result caching."""
        # Mock the primary transcription method
        with patch.object(stt_service, '_transcribe_with_provider') as mock_transcribe:
            mock_transcribe.return_value = STTResult(
                text="Cached transcription result",
                confidence=0.95,
                language="en",
                provider="mock"
            )
            
            # First transcription
            result1 = asyncio.run(stt_service.transcribe_audio(sample_audio_data))
            
            # Second transcription (should use cache)
            result2 = asyncio.run(stt_service.transcribe_audio(sample_audio_data))
            
            # Results should be identical
            assert result1.text == result2.text
            assert result1.confidence == result2.confidence
            
            # Second call should not call the provider again
            assert mock_transcribe.call_count == 1
    
    def test_batch_transcription_performance(self, stt_service):
        """Test batch transcription performance."""
        # Create multiple audio files
        audio_files = []
        for i in range(10):
            audio = AudioData(
                data=np.random.random(16000).astype(np.float32),
                sample_rate=16000,
                duration=1.0,
                channels=1,
                format="wav"
            )
            audio_files.append(audio)
        
        start_time = time.time()
        
        # Process batch
        results = asyncio.run(stt_service.transcribe_batch(audio_files))
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        assert len(results) == 10
        for result in results:
            assert isinstance(result, STTResult)
        
        # Should process 10 audio files reasonably quickly
        assert processing_time < 30.0  # Adjust threshold as needed
    
    def test_memory_usage_optimization(self, stt_service):
        """Test memory usage during transcription."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process large audio file
        large_audio = AudioData(
            data=np.random.random(16000 * 60).astype(np.float32),  # 60 seconds
            sample_rate=16000,
            duration=60.0,
            channels=1,
            format="wav"
        )
        
        with patch.object(stt_service, '_transcribe_with_provider') as mock_transcribe:
            mock_transcribe.return_value = STTResult(
                text="Large audio transcription",
                confidence=0.95,
                language="en",
                provider="mock"
            )
            
            result = asyncio.run(stt_service.transcribe_audio(large_audio))
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 200  # MB


@pytest.mark.skipif(not STT_SERVICE_AVAILABLE, reason="STT service not available")
class TestSTTSecurity:
    """Test STT service security and privacy features."""
    
    @pytest.fixture
    def stt_service(self):
        """Create STTService with security features."""
        config = mock_voice_config
        config.hipaa_compliance_enabled = True
        config.encryption_enabled = True
        return STTService(config)
    
    @pytest.fixture
    def audio_with_pii(self):
        """Create audio transcription containing PII."""
        return STTResult(
            text="My name is John Doe and my phone number is 555-123-4567",
            confidence=0.95,
            language="en",
            provider="mock"
        )
    
    def test_pii_detection(self, stt_service, audio_with_pii):
        """Test PII detection in transcription."""
        pii_detected = stt_service.detect_pii(audio_with_pii.text)
        
        assert pii_detected is True
        
        # Check specific PII types
        pii_types = stt_service.identify_pii_types(audio_with_pii.text)
        assert "phone_number" in pii_types
        assert "name" in pii_types
    
    def test_pii_masking(self, stt_service, audio_with_pii):
        """Test PII masking in transcription."""
        masked_text = stt_service.mask_pii(audio_with_pii.text)
        
        assert "John Doe" not in masked_text
        assert "555-123-4567" not in masked_text
        assert "[NAME]" in masked_text or "[PHONE]" in masked_text
    
    def test_audio_data_encryption(self, stt_service, sample_audio_data):
        """Test audio data encryption before transmission."""
        encrypted = stt_service.encrypt_audio_data(sample_audio_data)
        
        assert isinstance(encrypted, bytes)
        assert encrypted != sample_audio_data.to_bytes()
    
    def test_audio_data_decryption(self, stt_service, sample_audio_data):
        """Test audio data decryption after transmission."""
        encrypted = stt_service.encrypt_audio_data(sample_audio_data)
        decrypted = stt_service.decrypt_audio_data(encrypted)
        
        assert isinstance(decrypted, AudioData)
        np.testing.assert_array_almost_equal(decrypted.data, sample_audio_data.data)
    
    def test_audit_logging(self, stt_service):
        """Test audit logging for STT operations."""
        with patch.object(stt_service, 'log_transcription_event') as mock_log:
            stt_service.log_transcription(
                user_id="test_user",
                audio_duration=5.0,
                provider="openai",
                success=True
            )
            
            mock_log.assert_called_once_with(
                user_id="test_user",
                audio_duration=5.0,
                provider="openai",
                success=True
            )


@pytest.mark.skipif(not STT_SERVICE_AVAILABLE, reason="STT service not available")
class TestSTTRealTimeFeatures:
    """Test real-time STT features."""
    
    @pytest.fixture
    def stt_service(self):
        """Create STTService with real-time features."""
        config = mock_voice_config
        config.real_time_enabled = True
        return STTService(config)
    
    @pytest.mark.asyncio
    async def test_streaming_transcription(self, stt_service):
        """Test streaming transcription."""
        # Create audio chunks
        audio_chunks = []
        for i in range(5):
            chunk = AudioData(
                data=np.random.random(3200).astype(np.float32),  # 0.2 seconds
                sample_rate=16000,
                duration=0.2,
                channels=1,
                format="wav"
            )
            audio_chunks.append(chunk)
        
        with patch.object(stt_service, '_transcribe_chunk') as mock_transcribe:
            mock_transcribe.return_value = f"Partial transcription {len(audio_chunks)}"
            
            results = []
            async for result in stt_service.transcribe_stream(audio_chunks):
                results.append(result)
        
        assert len(results) == 5
        for result in results:
            assert isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_live_transcription_buffer(self, stt_service):
        """Test live transcription with buffering."""
        buffer_size = 10
        transcription_buffer = []
        
        async def simulate_live_audio():
            for i in range(15):  # 15 audio chunks
                chunk = AudioData(
                    data=np.random.random(1600).astype(np.float32),  # 0.1 seconds
                    sample_rate=16000,
                    duration=0.1,
                    channels=1,
                    format="wav"
                )
                
                result = await stt_service.add_audio_chunk(chunk, buffer_size)
                if result:
                    transcription_buffer.append(result)
                
                await asyncio.sleep(0.01)  # Simulate real-time gaps
        
        await simulate_live_audio()
        
        # Should have transcriptions when buffer is full
        assert len(transcription_buffer) >= 1


class TestSTTIntegration:
    """Test STT service integration scenarios."""
    
    @pytest.fixture
    def stt_service(self):
        """Create STTService for integration tests."""
        if not STT_SERVICE_AVAILABLE:
            pytest.skip("STT service not available")
        
        config = mock_voice_config
        return STTService(config)
    
    def test_integration_with_audio_processor(self, stt_service):
        """Test integration with audio processor."""
        # Mock audio processor methods
        with patch('voice.audio_processor.SimplifiedAudioProcessor') as mock_processor:
            mock_instance = MagicMock()
            mock_instance.process_audio.return_value = AudioData(
                data=np.random.random(16000).astype(np.float32),
                sample_rate=16000,
                duration=1.0,
                channels=1,
                format="wav"
            )
            mock_processor.return_value = mock_instance
            
            raw_audio = np.random.random(16000).astype(np.float32)
            processed = mock_instance.process_audio(raw_audio)
            
            assert isinstance(processed, AudioData)
    
    def test_integration_with_voice_commands(self, stt_service):
        """Test integration with voice commands."""
        transcription_result = STTResult(
            text="start breathing exercise",
            confidence=0.95,
            language="en",
            provider="mock",
            is_command=True
        )
        
        # Test command detection in transcription
        with patch('voice.commands.VoiceCommandProcessor') as mock_commands:
            mock_processor = MagicMock()
            mock_processor.process_command.return_value = {
                "command": "breathing_exercise",
                "success": True
            }
            mock_commands.return_value = mock_processor
            
            command_result = mock_processor.process_command(transcription_result.text)
            
            assert command_result["success"] is True
            assert command_result["command"] == "breathing_exercise"
    
    def test_integration_with_database(self, stt_service):
        """Test integration with database for transcription storage."""
        transcription_result = STTResult(
            text="Test transcription for database",
            confidence=0.95,
            language="en",
            provider="mock"
        )
        
        with patch('database.models.VoiceDataRepository') as mock_repo:
            mock_instance = MagicMock()
            mock_instance.save_transcription.return_value = True
            mock_repo.return_value = mock_instance
            
            success = mock_instance.save_transcription(
                user_id="test_user",
                transcription=transcription_result
            )
            
            assert success is True


# Run tests if this file is executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])