"""
Comprehensive unit tests for voice/optimized_audio_processor.py
"""

import os
import sys
import tempfile
import shutil
import unittest
from unittest.mock import Mock, patch, MagicMock, call
import pytest
import time
import threading
import numpy as np
from pathlib import Path
from queue import Queue, Empty

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from voice.optimized_audio_processor import (
    OptimizedAudioData,
    OptimizedAudioProcessorState,
    AudioProcessingMetrics,
    OptimizedAudioProcessor,
    create_optimized_audio_processor,
    AudioProcessingMode
)

class TestOptimizedAudioData(unittest.TestCase):
    """Test OptimizedAudioData class."""

    def test_initialization(self):
        """Test audio data initialization."""
        data = np.array([1, 2, 3, 4, 5])
        audio_data = OptimizedAudioData(
            data=data,
            sample_rate=16000,
            channels=1,
            metadata={'test': True}
        )

        self.assertEqual(audio_data.sample_rate, 16000)
        self.assertEqual(audio_data.channels, 1)
        self.assertTrue(audio_data.metadata['test'])
        self.assertTrue(audio_data.quality_score > 0)

    def test_empty_data(self):
        """Test handling of empty audio data."""
        audio_data = OptimizedAudioData(
            data=np.array([]),
            sample_rate=16000,
            channels=1,
            metadata={}
        )

        self.assertEqual(audio_data.quality_score, 0.0)
        self.assertEqual(len(audio_data.data), 0)

class TestOptimizedAudioProcessorState(unittest.TestCase):
    """Test OptimizedAudioProcessorState class."""

    def test_initialization(self):
        """Test state initialization."""
        state = OptimizedAudioProcessorState()

        self.assertFalse(state.is_processing)
        self.assertEqual(state.mode, AudioProcessingMode.REALTIME)
        self.assertEqual(len(state.metrics_history), 0)

    def test_metrics_update(self):
        """Test metrics updating."""
        state = OptimizedAudioProcessorState()
        metrics = AudioProcessingMetrics(
            processing_time=0.1,
            samples_processed=1000,
            memory_usage=50.0,
            cpu_usage=25.0,
            quality_score=0.8,
            timestamp=time.time()
        )

        state.update_metrics(metrics)
        self.assertEqual(len(state.metrics_history), 1)
        self.assertEqual(state.metrics_history[0], metrics)

class TestOptimizedAudioProcessor(unittest.TestCase):
    """Test OptimizedAudioProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'sample_rate': 16000,
            'channels': 1,
            'buffer_size': 1024,
            'noise_reduction': True,
            'vad': True
        }
        self.processor = OptimizedAudioProcessor(self.config)

    def test_initialization(self):
        """Test processor initialization."""
        self.assertEqual(self.processor.sample_rate, 16000)
        self.assertEqual(self.processor.channels, 1)
        self.assertTrue(self.processor.noise_reduction_enabled)
        self.assertTrue(self.processor.voice_activity_detection)

    def test_process_audio_empty(self):
        """Test processing empty audio data."""
        empty_data = np.array([])
        result = self.processor.process_audio(empty_data)

        self.assertEqual(len(result.data), 0)
        self.assertEqual(result.quality_score, 0.0)

    def test_process_audio_valid(self):
        """Test processing valid audio data."""
        audio_data = np.array([1, 2, 3, 4, 5] * 100)  # 500 samples
        result = self.processor.process_audio(audio_data)

        self.assertEqual(len(result.data), len(audio_data))
        self.assertTrue(result.quality_score > 0)
        self.assertIn('processing_time', result.metadata)

    def test_validate_audio(self):
        """Test audio validation."""
        # Valid audio
        valid_audio = np.array([1, 2, 3, 4, 5])
        self.assertTrue(self.processor.validate_audio(valid_audio))

        # Invalid audio (empty)
        empty_audio = np.array([])
        self.assertFalse(self.processor.validate_audio(empty_audio))

        # Invalid audio (contains NaN)
        nan_audio = np.array([1, 2, np.nan, 4])
        self.assertFalse(self.processor.validate_audio(nan_audio))

    def test_performance_metrics(self):
        """Test performance metrics calculation."""
        metrics = self.processor.get_performance_metrics()

        self.assertIn('avg_processing_time', metrics)
        self.assertIn('samples_per_second', metrics)
        self.assertIn('avg_quality_score', metrics)

    def test_batch_processing(self):
        """Test batch processing."""
        batch = [
            np.array([1, 2, 3, 4, 5]),
            np.array([6, 7, 8, 9, 10]),
            np.array([11, 12, 13, 14, 15])
        ]

        results = self.processor.process_batch(batch)

        self.assertEqual(len(results), len(batch))
        for result in results:
            self.assertIsInstance(result, OptimizedAudioData)

class TestFactoryFunction(unittest.TestCase):
    """Test factory function."""

    def test_create_optimized_audio_processor(self):
        """Test factory function."""
        config = {'sample_rate': 22050}
        processor = create_optimized_audio_processor(config)

        self.assertIsInstance(processor, OptimizedAudioProcessor)
        self.assertEqual(processor.sample_rate, 22050)

if __name__ == '__main__':
    unittest.main()
