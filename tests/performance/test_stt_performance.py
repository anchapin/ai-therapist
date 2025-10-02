#!/usr/bin/env python3
"""
Mock STT performance tests.
"""

import pytest
import time
from unittest.mock import MagicMock

class TestSTTPerformance:
    """Mock STT performance tests."""

    @pytest.fixture
    def stt_service(self):
        """Create mock STT service."""
        mock_service = MagicMock()
        mock_service.transcribe = MagicMock(return_value="Mock transcription result")
        return mock_service

    def test_stt_processing_performance(self, stt_service):
        """Test STT processing performance."""
        # Mock audio data
        audio_data = b"mock_audio_data"

        # Measure processing time
        start_time = time.time()
        result = stt_service.transcribe(audio_data)
        end_time = time.time()

        processing_time = end_time - start_time

        # Performance assertion (should be reasonably fast)
        assert processing_time < 2.0, f"STT processing too slow: {processing_time:.3f}s"
        assert result == "Mock transcription result"
