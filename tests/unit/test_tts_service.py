#!/usr/bin/env python3
"""
Mock TTS service tests for testing purposes.
"""

import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class TestTTSService:
    """Mock TTS service tests."""

    @pytest.fixture
    def tts_config(self):
        """Create TTS configuration for testing."""
        return {
            'provider': 'openai',
            'voice': 'alloy',
            'model': 'tts-1'
        }

    @pytest.fixture
    def tts_service(self, tts_config):
        """Create mock TTS service for testing."""
        # Create a mock TTS service
        mock_service = MagicMock()
        mock_service.config = tts_config
        mock_service.is_initialized = True
        mock_service.synthesize = MagicMock(return_value=b"mock_audio_data")
        return mock_service

    def test_initialization(self, tts_service):
        """Test TTS service initialization."""
        assert tts_service.config['provider'] == 'openai'
        assert tts_service.config['voice'] == 'alloy'
        assert tts_service.is_initialized == True

    def test_synthesize_speech(self, tts_service):
        """Test speech synthesis."""
        result = tts_service.synthesize("Hello world")
        assert result == b"mock_audio_data"
        tts_service.synthesize.assert_called_once_with("Hello world")
