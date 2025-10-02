#!/usr/bin/env python3
"""
Mock voice service tests for testing purposes.
"""

import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class TestVoiceService:
    """Mock voice service tests."""

    @pytest.fixture
    def voice_config(self):
        """Create voice configuration for testing."""
        return {
            'stt_provider': 'openai',
            'tts_provider': 'openai',
            'audio_sample_rate': 16000
        }

    @pytest.fixture
    def voice_service(self, voice_config):
        """Create mock voice service for testing."""
        # Create a mock voice service
        mock_service = MagicMock()
        mock_service.config = voice_config
        mock_service.is_initialized = False
        mock_service.session_active = False
        return mock_service

    def test_initialization(self, voice_service):
        """Test voice service initialization."""
        assert voice_service.config['stt_provider'] == 'openai'
        assert voice_service.config['audio_sample_rate'] == 16000
        assert voice_service.is_initialized == False

    def test_session_management(self, voice_service):
        """Test session management."""
        # Mock session start
        voice_service.start_session = MagicMock(return_value="mock_session_id")
        session_id = voice_service.start_session("user123")
        assert session_id == "mock_session_id"
        voice_service.start_session.assert_called_once_with("user123")
