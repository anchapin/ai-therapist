#!/usr/bin/env python3
"""
Simple Performance Tests

This module contains simplified performance tests that avoid threading issues
and focus on core functionality verification.
"""

import pytest
import time
import gc
from unittest.mock import Mock, patch
import numpy as np

# Import voice services
from voice.audio_processor import SimplifiedAudioProcessor, AudioData


class TestSimplePerformance:
    """Simple performance tests that avoid complex threading."""

    def setup_method(self):
        """Set up test environment."""
        # Mock config for audio processor
        self.mock_config = Mock()
        self.mock_config.audio.max_buffer_size = 300
        self.mock_config.audio.max_memory_mb = 100
        self.mock_config.audio.sample_rate = 16000
        self.mock_config.audio.channels = 1
        self.mock_config.audio.chunk_size = 1024
        self.mock_config.audio.format = 'wav'
        self.mock_config.audio.stream_buffer_size = 10
        self.mock_config.audio.stream_chunk_duration = 0.1
        self.mock_config.audio.compression_enabled = True
        self.mock_config.audio.compression_level = 6

    def test_audio_processor_basic_performance(self):
        """Test audio processor basic performance."""
        processor = SimplifiedAudioProcessor(self.mock_config)
        
        # Test buffer operations performance
        start_time = time.time()
        
        for i in range(10):
            audio_data = np.random.random(1600).astype(np.float32)  # Small chunks
            processor.add_to_buffer(audio_data)
        
        buffer_time = time.time() - start_time
        
        # Should be fast
        assert buffer_time < 0.1, f"Buffer operations too slow: {buffer_time:.6f}s"
        assert len(processor.audio_buffer) > 0
        
        # Test cleanup performance
        start_time = time.time()
        cleaned_count = processor.force_cleanup_buffers()
        cleanup_time = time.time() - start_time
        
        assert cleanup_time < 0.01, f"Buffer cleanup too slow: {cleanup_time:.6f}s"
        assert cleaned_count > 0
        assert len(processor.audio_buffer) == 0

    def test_memory_usage_tracking(self):
        """Test basic memory usage tracking."""
        # Get baseline memory
        gc.collect()  # Force garbage collection
        baseline_objects = len(gc.get_objects())
        
        # Create some objects
        test_objects = []
        for i in range(10):
            test_objects.append([f"test_data_{j}" * 10 for j in range(5)])
        
        # Check memory increased
        current_objects = len(gc.get_objects())
        assert current_objects > baseline_objects
        
        # Clean up
        del test_objects
        gc.collect()
        
        # Memory should decrease
        final_objects = len(gc.get_objects())
        assert final_objects < current_objects

    def test_simple_cache_operations(self):
        """Test simple cache operations without background threads."""
        # Simple dict-based cache for testing
        cache = {}
        cache_stats = {'hits': 0, 'misses': 0, 'sets': 0}
        
        # Test cache operations
        start_time = time.time()
        
        # Set operations
        for i in range(50):
            key = f"key_{i}"
            value = f"value_{i}" * 10
            cache[key] = value
            cache_stats['sets'] += 1
        
        # Get operations (hits)
        for i in range(50):
            key = f"key_{i}"
            if key in cache:
                cache_stats['hits'] += 1
            else:
                cache_stats['misses'] += 1
        
        # Get operations (misses)
        for i in range(50, 100):
            key = f"key_{i}"
            if key in cache:
                cache_stats['hits'] += 1
            else:
                cache_stats['misses'] += 1
        
        cache_time = time.time() - start_time
        
        # Performance assertions
        assert cache_time < 0.1, f"Cache operations too slow: {cache_time:.6f}s"
        assert cache_stats['sets'] == 50
        assert cache_stats['hits'] == 50
        assert cache_stats['misses'] == 50

    def test_voice_service_session_performance(self):
        """Test voice service session performance."""
        # Mock config
        mock_config = Mock()
        mock_config.voice_enabled = True
        mock_config.default_voice_profile = 'default'
        mock_config.voice_commands_enabled = True
        mock_config.audio = self.mock_config.audio
        mock_config.voice_profiles = {}  # Add voice profiles to avoid len() error
        mock_config.voice_command_timeout = 30000  # Add timeout to avoid Mock/int division error
        
        security = Mock()
        
        # Mock external dependencies
        with patch('voice.stt_service.STTService'), \
             patch('voice.tts_service.TTSService'), \
             patch('voice.commands.VoiceCommandProcessor'):
            
            from voice.voice_service import VoiceService
            service = VoiceService(mock_config, security)
            
            # Test session creation performance
            start_time = time.time()
            session_ids = []
            
            for i in range(5):
                session_id = service.create_session(f"test_user_{i}")
                session_ids.append(session_id)
            
            creation_time = time.time() - start_time
            
            # Should be fast
            avg_creation_time = creation_time / 5
            assert avg_creation_time < 0.01, f"Session creation too slow: {avg_creation_time:.6f}s"
            assert len(session_ids) == 5
            
            # Test session statistics
            stats = service.get_service_statistics()
            assert isinstance(stats, dict)
            assert 'sessions_count' in stats
            
            # Test session cleanup performance
            start_time = time.time()
            for session_id in session_ids:
                service.end_session(session_id)
            
            cleanup_time = time.time() - start_time
            avg_cleanup_time = cleanup_time / 5
            assert avg_cleanup_time < 0.01, f"Session cleanup too slow: {avg_cleanup_time:.6f}s"

    def test_concurrent_operations_simple(self):
        """Test simple concurrent operations."""
        import threading
        
        results = []
        errors = []
        
        def simple_worker(worker_id):
            try:
                # Simple computation
                start_time = time.time()
                result = sum(i * i for i in range(100))
                end_time = time.time()
                
                results.append({
                    'worker_id': worker_id,
                    'result': result,
                    'time': end_time - start_time
                })
                
            except Exception as e:
                errors.append(f"Worker {worker_id}: {e}")
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=simple_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 3
        assert all(r['result'] == sum(i * i for i in range(100)) for r in results)
        
        # Check performance
        avg_time = sum(r['time'] for r in results) / len(results)
        assert avg_time < 0.1, f"Concurrent operations too slow: {avg_time:.6f}s"

    def test_audio_data_processing_performance(self):
        """Test audio data processing performance."""
        # Create test audio data
        audio_data = np.random.random(8000).astype(np.float32)  # 0.5 second at 16kHz
        audio_obj = AudioData(
            data=audio_data,
            sample_rate=16000,
            duration=0.5
        )
        
        # Test processing performance
        start_time = time.time()
        
        # Simple processing (volume adjustment)
        processed_data = audio_obj.data * 0.5
        
        processing_time = time.time() - start_time
        
        # Should be fast
        assert processing_time < 0.01, f"Audio processing too slow: {processing_time:.6f}s"
        assert len(processed_data) == len(audio_data)

    def test_performance_metrics_collection(self):
        """Test performance metrics collection."""
        metrics = {
            'operation_times': [],
            'memory_usage': [],
            'error_count': 0
        }
        
        # Collect metrics
        for i in range(10):
            start_time = time.time()
            
            # Simulate operation
            result = sum(j * j for j in range(50))
            
            end_time = time.time()
            metrics['operation_times'].append(end_time - start_time)
            
            # Simulate memory check
            if i % 3 == 0:
                metrics['memory_usage'].append(len(gc.get_objects()))
        
        # Analyze metrics
        avg_time = sum(metrics['operation_times']) / len(metrics['operation_times'])
        max_time = max(metrics['operation_times'])
        
        # Performance assertions
        assert avg_time < 0.01, f"Average operation time too high: {avg_time:.6f}s"
        assert max_time < 0.05, f"Maximum operation time too high: {max_time:.6f}s"
        assert len(metrics['memory_usage']) > 0
        assert metrics['error_count'] == 0

    def test_resource_cleanup_performance(self):
        """Test resource cleanup performance."""
        # Create resources
        resources = []
        for i in range(10):
            resources.append([f"resource_{j}" * 5 for j in range(10)])
        
        # Test cleanup performance
        start_time = time.time()
        
        # Clean up resources by clearing the list
        resources.clear()
        
        # Force garbage collection
        gc.collect()
        
        cleanup_time = time.time() - start_time
        
        # Should be fast (relaxed threshold for test environment)
        assert cleanup_time < 0.2, f"Resource cleanup too slow: {cleanup_time:.6f}s"
        
        # Verify cleanup (list should be empty)
        assert len(resources) == 0
        # Resources should be garbage collected

    def test_scalability_basic(self):
        """Test basic scalability."""
        operation_counts = [10, 50, 100]
        performance_results = []
        
        for count in operation_counts:
            start_time = time.time()
            
            # Perform operations
            results = []
            for i in range(count):
                results.append(i * i)
            
            end_time = time.time()
            total_time = end_time - start_time
            avg_time = total_time / count
            
            performance_results.append({
                'count': count,
                'total_time': total_time,
                'avg_time': avg_time
            })
        
        # Check scalability
        base_avg_time = performance_results[0]['avg_time']
        for result in performance_results[1:]:
            time_ratio = result['avg_time'] / base_avg_time
            assert time_ratio < 2.0, f"Performance degraded too much at {result['count']} operations: {time_ratio:.2f}x"

    def test_error_handling_performance(self):
        """Test error handling performance."""
        start_time = time.time()
        
        # Test error handling
        errors_caught = 0
        for i in range(10):
            try:
                # Intentionally cause an error
                result = 1 / 0
            except ZeroDivisionError:
                errors_caught += 1
        
        end_time = time.time()
        error_handling_time = end_time - start_time
        
        # Should be fast
        assert error_handling_time < 0.01, f"Error handling too slow: {error_handling_time:.6f}s"
        assert errors_caught == 10