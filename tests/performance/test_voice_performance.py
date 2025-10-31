"""
Voice Performance Tests

Comprehensive performance testing specifically for voice features including:
- STT/TTS service performance benchmarks
- Real-time audio processing latency
- Voice quality metrics
- Concurrent voice session handling
- Voice command processing performance
"""

import pytest
import time
import threading
import asyncio
import statistics
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any, Tuple
import queue
import sys
import os

# Add parent directories for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from voice.audio_processor import SimplifiedAudioProcessor, AudioData
from voice.voice_service import VoiceService
from voice.stt_service import STTService
from voice.tts_service import TTSService
from voice.commands import VoiceCommandProcessor
from voice.security import VoiceSecurity


class TestVoicePerformance:
    """Test performance of voice-specific features."""

    def setup_method(self):
        """Set up test environment with realistic voice configuration."""
        # Create realistic voice config
        class MockVoiceConfig:
            voice_enabled = True
            voice_commands_enabled = True
            stt_provider = "openai"
            tts_provider = "openai"
            default_voice_profile = "alloy"
            
            # Security configuration attributes
            session_timeout_minutes = 60
            encryption_key_rotation_days = 90
            audit_log_retention_days = 365
            consent_retention_days = 2555  # 7 years
            pii_detection_enabled = True
            encryption_enabled = True
            anonymization_enabled = True
            
            # Voice configuration attributes
            voice_profiles = {}
            elevenlabs_api_key = None
            elevenlabs_voice_id = None
            elevenlabs_model = "eleven_multilingual_v2"
            openai_whisper_model = "whisper-1"
            openai_whisper_language = "en"
            openai_whisper_temperature = 0.0
            whisper_model = "base"
            whisper_language = "en"
            whisper_temperature = 0.0
            whisper_beam_size = 5
            whisper_best_of = 5
            
            class audio:
                max_buffer_size = 300
                max_memory_mb = 100
                sample_rate = 16000
                channels = 1
                chunk_size = 1024
                format = 'wav'
                stream_buffer_size = 10
                stream_chunk_duration = 0.1
                compression_enabled = True
                compression_level = 6
            
            class performance:
                cache_size = 100
                max_concurrent_sessions = 50
                session_timeout = 300
            
            def get_preferred_stt_service(self):
                return self.stt_provider
            
            def get_preferred_tts_service(self):
                return self.tts_provider
            
            def is_google_speech_configured(self):
                return False
            
            def is_elevenlabs_configured(self):
                return False
            
            def is_whisper_configured(self):
                return True
            
            def is_piper_configured(self):
                return False
        
        self.config = MockVoiceConfig()
        self.security = VoiceSecurity(self.config)
        
        # Mock external services for controlled performance testing
        with patch('voice.stt_service.STTService') as mock_stt, \
             patch('voice.tts_service.TTSService') as mock_tts, \
             patch('voice.commands.VoiceCommandProcessor') as mock_commands:
            
            # Configure realistic performance mocks
            mock_stt_instance = mock_stt.return_value
            mock_stt_instance.transcribe_audio.return_value = asyncio.Future()
            mock_stt_instance.transcribe_audio.return_value.set_result({
                'text': 'Hello therapist',
                'confidence': 0.95,
                'processing_time': 0.5
            })
            
            mock_tts_instance = mock_tts.return_value
            mock_tts_instance.synthesize_speech.return_value = asyncio.Future()
            mock_tts_instance.synthesize_speech.return_value.set_result({
                'audio_data': np.random.random(16000).astype(np.float32),
                'duration': 1.0,
                'processing_time': 0.3
            })
            
            mock_commands_instance = mock_commands.return_value
            mock_commands_instance.process_command.return_value = asyncio.Future()
            mock_commands_instance.process_command.return_value.set_result({
                'command': None,
                'is_crisis': False,
                'processing_time': 0.1
            })
            
            self.voice_service = VoiceService(self.config, self.security)
            self.mock_stt = mock_stt_instance
            self.mock_tts = mock_tts_instance
            self.mock_commands = mock_commands_instance

    def test_stt_service_performance_benchmark(self):
        """Benchmark STT service performance with various audio sizes."""
        audio_sizes = [8000, 16000, 32000, 64000]  # Different audio lengths
        performance_results = []

        for size in audio_sizes:
            # Create test audio data
            audio_data = np.random.random(size).astype(np.float32)
            audio_obj = AudioData(
                data=audio_data,
                sample_rate=16000,
                duration=size/16000
            )

            # Measure STT performance
            start_time = time.time()
            result = asyncio.run(self.voice_service.stt_service.transcribe_audio(audio_obj))
            end_time = time.time()

            processing_time = end_time - start_time
            realtime_factor = processing_time / (size/16000)  # Processing time vs audio duration

            performance_results.append({
                'audio_size': size,
                'duration': size/16000,
                'processing_time': processing_time,
                'realtime_factor': realtime_factor,
                'result': result
            })

        # Analyze performance benchmarks
        processing_times = [r['processing_time'] for r in performance_results]
        realtime_factors = [r['realtime_factor'] for r in performance_results]

        # Performance assertions for STT
        assert max(processing_times) < 2.0, f"STT processing too slow: {max(processing_times):.2f}s"
        assert max(realtime_factors) < 0.5, f"Real-time factor too high: {max(realtime_factors):.2f}"

        # Verify results are consistent across different sizes
        confidences = [r['result']['confidence'] for r in performance_results]
        assert min(confidences) > 0.8, "STT confidence too low for some audio sizes"

    def test_tts_service_performance_benchmark(self):
        """Benchmark TTS service performance with various text lengths."""
        text_lengths = [10, 50, 100, 200, 500]  # Different text lengths
        performance_results = []

        for length in text_lengths:
            text = "Hello " * length  # Generate test text
            
            start_time = time.time()
            result = asyncio.run(self.voice_service.tts_service.synthesize_speech(text))
            end_time = time.time()

            processing_time = end_time - start_time
            synthesis_rate = len(text) / processing_time  # Characters per second

            performance_results.append({
                'text_length': len(text),
                'processing_time': processing_time,
                'synthesis_rate': synthesis_rate,
                'audio_duration': result['duration'],
                'result': result
            })

        # Analyze performance benchmarks
        processing_times = [r['processing_time'] for r in performance_results]
        synthesis_rates = [r['synthesis_rate'] for r in performance_results]

        # Performance assertions for TTS
        assert max(processing_times) < 3.0, f"TTS processing too slow: {max(processing_times):.2f}s"
        assert min(synthesis_rates) > 50, f"TTS synthesis rate too low: {min(synthesis_rates):.0f} chars/s"

        # Verify audio quality metrics
        audio_durations = [r['audio_duration'] for r in performance_results]
        expected_duration_ratio = 0.08  # Approximate chars to seconds ratio
        actual_ratios = [r['text_length'] / r['audio_duration'] for r in performance_results]
        
        # Should be within reasonable range of expected ratio
        for ratio in actual_ratios:
            assert expected_duration_ratio * 0.5 < ratio < expected_duration_ratio * 2.0, \
                f"Audio duration inconsistent: {ratio:.3f}"

    def test_voice_session_throughput(self):
        """Test voice session processing throughput under concurrent load."""
        num_concurrent_sessions = 20
        session_results = []
        results_lock = threading.Lock()

        def session_worker(session_id: int):
            """Worker function for processing a voice session."""
            try:
                session_start = time.time()
                
                # Create session
                voice_session_id = self.voice_service.create_session(f"user_{session_id}")
                
                # Process multiple voice interactions
                interaction_times = []
                for i in range(3):  # 3 interactions per session
                    # Create audio data
                    audio_data = np.random.random(16000).astype(np.float32)  # 1 second
                    audio_obj = AudioData(
                        data=audio_data,
                        sample_rate=16000,
                        duration=1.0
                    )

                    # Process voice input
                    interaction_start = time.time()
                    result = asyncio.run(self.voice_service.process_voice_input(
                        audio_obj, voice_session_id
                    ))
                    interaction_end = time.time()
                    
                    interaction_times.append(interaction_end - interaction_start)
                    
                    # Small delay between interactions
                    time.sleep(0.1)

                # End session
                self.voice_service.end_session(voice_session_id)
                session_end = time.time()

                with results_lock:
                    session_results.append({
                        'session_id': session_id,
                        'total_time': session_end - session_start,
                        'interaction_times': interaction_times,
                        'avg_interaction_time': statistics.mean(interaction_times),
                        'interactions_per_second': len(interaction_times) / (session_end - session_start)
                    })

            except Exception as e:
                with results_lock:
                    session_results.append({
                        'session_id': session_id,
                        'error': str(e)
                    })

        # Run concurrent sessions
        threads = []
        for i in range(num_concurrent_sessions):
            thread = threading.Thread(target=session_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=30.0)

        # Analyze throughput results
        successful_sessions = [r for r in session_results if 'error' not in r]
        
        assert len(successful_sessions) > num_concurrent_sessions * 0.8, \
            f"Too many failed sessions: {len(successful_sessions)}/{num_concurrent_sessions}"

        # Calculate overall throughput metrics
        total_interactions = sum(len(r['interaction_times']) for r in successful_sessions)
        total_time = max(r['total_time'] for r in successful_sessions)
        overall_throughput = total_interactions / total_time if total_time > 0 else 0

        avg_interaction_times = [r['avg_interaction_time'] for r in successful_sessions]
        avg_interaction_time = statistics.mean(avg_interaction_times)

        # Performance assertions
        assert overall_throughput > 5.0, f"Overall throughput too low: {overall_throughput:.1f} interactions/s"
        assert avg_interaction_time < 2.0, f"Average interaction time too high: {avg_interaction_time:.2f}s"

    def test_realtime_audio_processing_latency(self):
        """Test real-time audio processing latency for voice interactions."""
        chunk_sizes = [1024, 2048, 4096, 8192]  # Different chunk sizes
        latency_results = []

        for chunk_size in chunk_sizes:
            # Create audio processor
            processor = SimplifiedAudioProcessor(self.config)
            
            # Process audio chunks and measure latency
            latencies = []
            for i in range(50):  # 50 chunks per test
                audio_chunk = np.random.random(chunk_size).astype(np.float32)
                
                # Measure buffer addition latency
                start_time = time.time()
                processor.add_to_buffer(audio_chunk)
                end_time = time.time()
                
                latencies.append(end_time - start_time)
            
            # Calculate statistics for this chunk size
            avg_latency = statistics.mean(latencies)
            max_latency = max(latencies)
            p95_latency = np.percentile(latencies, 95)
            
            latency_results.append({
                'chunk_size': chunk_size,
                'avg_latency': avg_latency,
                'max_latency': max_latency,
                'p95_latency': p95_latency,
                'samples_per_second': chunk_size * 16000 / chunk_size  # Simplified calculation
            })

        # Analyze latency results
        avg_latencies = [r['avg_latency'] for r in latency_results]
        max_latencies = [r['max_latency'] for r in latency_results]
        p95_latencies = [r['p95_latency'] for r in latency_results]

        # Real-time processing assertions (should be under 10ms for most chunks)
        assert max(avg_latencies) < 0.01, f"Average latency too high: {max(avg_latencies):.6f}s"
        assert max(p95_latencies) < 0.02, f"95th percentile latency too high: {max(p95_latencies):.6f}s"

    def test_voice_command_processing_performance(self):
        """Test voice command processing performance under various conditions."""
        command_types = [
            "breathing exercise",
            "emergency help",
            "start session",
            "pause therapy",
            "repeat that"
        ]
        
        num_commands = 100
        command_results = []

        for i in range(num_commands):
            command = command_types[i % len(command_types)]
            
            # Create audio with command
            audio_data = np.random.random(8000).astype(np.float32)
            audio_obj = AudioData(
                data=audio_data,
                sample_rate=16000,
                duration=0.5
            )

            # Measure command processing
            start_time = time.time()
            result = asyncio.run(self.voice_service.commands.process_command(
                audio_obj, "test_session"
            ))
            end_time = time.time()

            processing_time = end_time - start_time
            command_results.append({
                'command': command,
                'processing_time': processing_time,
                'detected_command': result.get('command'),
                'is_crisis': result.get('is_crisis', False)
            })

        # Analyze command processing performance
        processing_times = [r['processing_time'] for r in command_results]
        crisis_commands = [r for r in command_results if r['is_crisis']]

        # Performance assertions
        avg_processing_time = statistics.mean(processing_times)
        max_processing_time = max(processing_times)

        assert avg_processing_time < 0.5, f"Command processing too slow: {avg_processing_time:.3f}s"
        assert max_processing_time < 1.0, f"Max command processing too slow: {max_processing_time:.3f}s"

        # Crisis commands should be processed faster
        if crisis_commands:
            crisis_times = [r['processing_time'] for r in crisis_commands]
            avg_crisis_time = statistics.mean(crisis_times)
            assert avg_crisis_time < avg_processing_time * 0.8, \
                f"Crisis commands not prioritized: {avg_crisis_time:.3f}s vs {avg_processing_time:.3f}s"

    def test_voice_quality_metrics(self):
        """Test voice quality metrics for audio processing."""
        # Test with different audio qualities
        audio_qualities = [
            {'noise_level': 0.0, 'volume': 1.0, 'name': 'clean'},
            {'noise_level': 0.1, 'volume': 0.8, 'name': 'noisy'},
            {'noise_level': 0.2, 'volume': 0.6, 'name': 'very_noisy'},
            {'noise_level': 0.05, 'volume': 1.2, 'name': 'loud'}
        ]
        
        quality_results = []

        for quality in audio_qualities:
            # Generate audio with specified quality
            duration = 2.0  # 2 seconds
            sample_rate = 16000
            num_samples = int(duration * sample_rate)
            
            # Clean audio signal
            t = np.linspace(0, duration, num_samples)
            clean_audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
            
            # Add noise and adjust volume
            noise = np.random.normal(0, quality['noise_level'], num_samples)
            audio_data = (clean_audio + noise) * quality['volume']
            audio_data = audio_data.astype(np.float32)

            audio_obj = AudioData(
                data=audio_data,
                sample_rate=sample_rate,
                duration=duration
            )

            # Process audio and measure quality metrics
            start_time = time.time()
            result = asyncio.run(self.voice_service.process_voice_input(audio_obj))
            end_time = time.time()

            # Calculate quality metrics
            snr = 10 * np.log10(np.var(clean_audio) / np.var(noise)) if np.var(noise) > 0 else float('inf')
            rms = np.sqrt(np.mean(audio_data**2))
            
            quality_results.append({
                'quality_name': quality['name'],
                'snr_db': snr,
                'rms_level': rms,
                'processing_time': end_time - start_time,
                'success': result is not None
            })

        # Analyze quality results
        successful_processing = [r for r in quality_results if r['success']]
        processing_times = [r['processing_time'] for r in successful_processing]
        
        # Clean audio should process best
        clean_result = next((r for r in quality_results if r['quality_name'] == 'clean'), None)
        assert clean_result and clean_result['success'], "Clean audio processing failed"
        
        # Processing time should not vary dramatically with quality
        if len(processing_times) > 1:
            time_variance = statistics.variance(processing_times)
            assert time_variance < 1.0, f"Processing time varies too much with quality: {time_variance:.3f}"

    def test_concurrent_voice_session_memory_usage(self):
        """Test memory usage with concurrent voice sessions."""
        import psutil
        process = psutil.Process()
        
        # Get baseline memory
        baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        num_concurrent_sessions = 50
        session_data = {}
        memory_measurements = []

        def session_memory_worker(session_id: int):
            """Worker that creates and maintains a voice session with memory monitoring."""
            try:
                # Create session
                voice_session_id = self.voice_service.create_session(f"mem_test_user_{session_id}")
                
                # Store session data
                session_data[session_id] = {
                    'session_id': voice_session_id,
                    'start_time': time.time()
                }
                
                # Process some audio to populate session
                for i in range(5):
                    audio_data = np.random.random(16000).astype(np.float32)
                    audio_obj = AudioData(
                        data=audio_data,
                        sample_rate=16000,
                        duration=1.0
                    )
                    
                    # Process without waiting for result (focus on memory)
                    asyncio.run(self.voice_service.process_voice_input(audio_obj, voice_session_id))
                    
                    # Measure memory after each interaction
                    current_memory = process.memory_info().rss / (1024 * 1024)
                    memory_measurements.append({
                        'session_id': session_id,
                        'interaction': i,
                        'memory_mb': current_memory,
                        'timestamp': time.time()
                    })
                    
                    time.sleep(0.1)  # Small delay between interactions

                # Clean up session
                self.voice_service.end_session(voice_session_id)
                session_data[session_id]['end_time'] = time.time()

            except Exception as e:
                session_data[session_id] = {'error': str(e)}

        # Run concurrent sessions
        threads = []
        for i in range(num_concurrent_sessions):
            thread = threading.Thread(target=session_memory_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=60.0)

        # Get final memory
        final_memory = process.memory_info().rss / (1024 * 1024)
        
        # Analyze memory usage
        memory_increase = final_memory - baseline_memory
        max_memory = max(m['memory_mb'] for m in memory_measurements) if memory_measurements else final_memory
        avg_memory_per_session = memory_increase / num_concurrent_sessions

        # Memory assertions
        assert memory_increase < 200, f"Memory increase too high: {memory_increase:.1f} MB"
        assert avg_memory_per_session < 5, f"Average memory per session too high: {avg_memory_per_session:.1f} MB"
        assert max_memory < baseline_memory + 500, f"Peak memory usage too high: {max_memory:.1f} MB"

        # Verify cleanup - memory should return close to baseline
        successful_sessions = len([s for s in session_data.values() if 'error' not in s])
        if successful_sessions > 0:
            # Most sessions should be cleaned up
            assert len(self.voice_service.sessions) < num_concurrent_sessions * 0.1, \
                "Sessions not properly cleaned up"

    def test_voice_service_scalability(self):
        """Test voice service scalability with increasing load."""
        concurrency_levels = [1, 5, 10, 25, 50]
        scalability_results = []

        for concurrency in concurrency_levels:
            response_times = []
            errors = 0
            throughput_start = time.time()

            def scalability_worker():
                nonlocal errors
                try:
                    # Create session
                    session_id = self.voice_service.create_session(f"scale_test_user_{threading.current_thread().ident}")
                    
                    # Process voice interaction
                    audio_data = np.random.random(8000).astype(np.float32)
                    audio_obj = AudioData(
                        data=audio_data,
                        sample_rate=16000,
                        duration=0.5
                    )

                    start_time = time.time()
                    result = asyncio.run(self.voice_service.process_voice_input(audio_obj, session_id))
                    end_time = time.time()

                    if result is not None:
                        response_times.append(end_time - start_time)
                    else:
                        errors += 1

                    # Clean up
                    self.voice_service.end_session(session_id)

                except Exception:
                    errors += 1

            # Run concurrent workers
            threads = []
            for i in range(concurrency):
                thread = threading.Thread(target=scalability_worker)
                threads.append(thread)
                thread.start()

            # Wait for completion
            for thread in threads:
                thread.join(timeout=30.0)

            throughput_end = time.time()
            total_time = throughput_end - throughput_start

            # Calculate metrics
            successful_requests = len(response_times)
            avg_response_time = statistics.mean(response_times) if response_times else 0
            throughput = successful_requests / total_time if total_time > 0 else 0
            error_rate = errors / concurrency

            scalability_results.append({
                'concurrency': concurrency,
                'avg_response_time': avg_response_time,
                'throughput': throughput,
                'error_rate': error_rate,
                'successful_requests': successful_requests
            })

        # Analyze scalability
        if len(scalability_results) >= 2:
            # Calculate throughput degradation
            baseline_throughput = scalability_results[0]['throughput']
            max_throughput = max(r['throughput'] for r in scalability_results)
            
            # Throughput should increase with concurrency up to a point
            assert max_throughput >= baseline_throughput * 0.8, \
                f"Throughput degradation too severe: {max_throughput:.1f} vs {baseline_throughput:.1f}"

            # Response time should not increase dramatically
            response_times = [r['avg_response_time'] for r in scalability_results if r['avg_response_time'] > 0]
            if response_times:
                max_response_time = max(response_times)
                min_response_time = min(response_times)
                response_ratio = max_response_time / min_response_time
                assert response_ratio < 5.0, \
                    f"Response time degradation too high: {response_ratio:.1f}x"

            # Error rate should remain acceptable
            max_error_rate = max(r['error_rate'] for r in scalability_results)
            assert max_error_rate < 0.3, \
                f"Error rate too high under load: {max_error_rate:.2%}"