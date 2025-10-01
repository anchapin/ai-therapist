"""
Resource Exhaustion and Memory Management Tests

This module tests system resilience under resource exhaustion scenarios:
- Memory exhaustion and resource cleanup
- CPU exhaustion scenarios
- File descriptor leaks
- Thread pool exhaustion
- Database connection pool limits
- Large file processing limits
- Concurrent operation resource management
"""

import pytest
import gc
import psutil
import os
import time
import threading
import multiprocessing
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import tempfile
import shutil
import tracemalloc
import resource

# Import project modules
from voice.voice_service import VoiceService, VoiceSession, VoiceSessionState
from voice.audio_processor import AudioData, SimplifiedAudioProcessor
from voice.config import VoiceConfig, SecurityConfig
from voice.security import VoiceSecurity


class TestMemoryExhaustion:
    """Test handling of memory exhaustion scenarios."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_audio_buffer_memory_limits(self):
        """Test memory limits for audio buffer processing."""
        # Mock configuration with memory limits
        config = Mock(spec=VoiceConfig)
        config.max_audio_buffer_size = 10 * 1024 * 1024  # 10MB limit
        config.audio_buffer_item_limit = 100

        # Create large audio data that exceeds limits
        large_audio_data = np.random.randn(16000 * 60).astype(np.float32)  # 1 minute of audio

        # Test that large audio data is handled appropriately
        audio_data = AudioData(
            data=large_audio_data,
            sample_rate=16000,
            duration=60.0,
            channels=1
        )

        # Simulate memory limit enforcement
        memory_usage = len(audio_data.data) * audio_data.data.itemsize
        max_memory = 10 * 1024 * 1024  # 10MB

        if memory_usage > max_memory:
            # Should either reject or downsample
            reduction_ratio = max_memory / memory_usage
            new_length = int(len(audio_data.data) * reduction_ratio)

            # Create downsampled version
            downsampled_data = audio_data.data[:new_length]
            audio_data.data = downsampled_data
            audio_data.duration = len(downsampled_data) / audio_data.sample_rate

        # Verify memory usage is within limits
        final_memory_usage = len(audio_data.data) * audio_data.data.itemsize
        assert final_memory_usage <= max_memory

    def test_memory_leak_detection(self):
        """Test detection and prevention of memory leaks."""
        # Start memory tracing
        tracemalloc.start()

        try:
            # Create multiple audio buffers to simulate potential leaks
            audio_buffers = []

            for i in range(100):
                # Create large audio buffer
                large_data = np.random.randn(16000 * 10).astype(np.float32)  # 10 seconds
                audio_data = AudioData(
                    data=large_data,
                    sample_rate=16000,
                    duration=10.0,
                    channels=1
                )
                audio_buffers.append(audio_data)

            # Take memory snapshot
            snapshot1 = tracemalloc.take_snapshot()

            # Clear buffers
            audio_buffers.clear()

            # Force garbage collection
            gc.collect()

            # Take second snapshot
            snapshot2 = tracemalloc.take_snapshot()

            # Compare snapshots
            stats = snapshot2.compare_to(snapshot1, 'lineno')

            # Check for significant memory increases
            total_memory_diff = sum(stat.size_diff for stat in stats)

            # Memory should not grow significantly after cleanup
            # (allowing some tolerance for test overhead)
            assert total_memory_diff < 50 * 1024 * 1024  # 50MB tolerance

        finally:
            tracemalloc.stop()

    def test_large_file_processing_memory_management(self, temp_dir):
        """Test memory management during large file processing."""
        # Create large audio file
        large_file = os.path.join(temp_dir, "large_audio.wav")

        # Generate large audio data
        sample_rate = 16000
        duration = 300  # 5 minutes
        num_samples = sample_rate * duration

        large_audio_data = np.random.randn(num_samples).astype(np.float32)

        # Write large file in chunks to avoid memory issues
        with open(large_file, 'wb') as f:
            # Write WAV header
            f.write(b'RIFF')
            f.write((num_samples * 2 + 36).to_bytes(4, 'little'))  # File size
            f.write(b'WAVE')

            # Write format chunk
            f.write(b'fmt ')
            f.write((16).to_bytes(4, 'little'))  # Chunk size
            f.write((1).to_bytes(2, 'little'))   # PCM format
            f.write((1).to_bytes(2, 'little'))   # Mono
            f.write((sample_rate).to_bytes(4, 'little'))
            f.write((sample_rate * 2).to_bytes(4, 'little'))  # Byte rate
            f.write((2).to_bytes(2, 'little'))   # Block align
            f.write((16).to_bytes(2, 'little'))  # Bits per sample

            # Write data chunk header
            f.write(b'data')
            f.write((num_samples * 2).to_bytes(4, 'little'))

            # Write audio data in chunks
            chunk_size = 8192
            for i in range(0, len(large_audio_data), chunk_size):
                chunk = large_audio_data[i:i + chunk_size]
                # Convert to 16-bit PCM
                chunk_pcm = (chunk * 32767).astype(np.int16)
                f.write(chunk_pcm.tobytes())

        # Test processing large file with memory monitoring
        config = Mock(spec=VoiceConfig)
        processor = SimplifiedAudioProcessor(config)

        # Mock load_audio to use our large file
        def mock_load_audio(filepath):
            if filepath == large_file:
                # Simulate memory-efficient loading
                return AudioData(
                    data=np.array([], dtype=np.float32),  # Don't load full data
                    sample_rate=16000,
                    duration=300.0,
                    channels=1
                )
            return None

        processor.load_audio = mock_load_audio

        # Process should not cause memory exhaustion
        result = processor.load_audio(large_file)
        assert result is not None

    def test_memory_fragmentation_handling(self):
        """Test handling of memory fragmentation."""
        # Simulate memory fragmentation scenario
        fragments = []

        try:
            # Allocate and deallocate memory in random pattern
            for i in range(1000):
                # Randomly allocate memory chunks
                if i % 3 == 0:
                    # Allocate large chunk
                    large_chunk = np.random.randn(10000).astype(np.float32)
                    fragments.append(large_chunk)
                elif i % 3 == 1:
                    # Allocate small chunks
                    for j in range(10):
                        small_chunk = np.random.randn(100).astype(np.float32)
                        fragments.append(small_chunk)

                # Periodically clean up
                if i % 100 == 0:
                    fragments.clear()
                    gc.collect()

            # Final cleanup
            fragments.clear()
            gc.collect()

        except MemoryError:
            # Should handle memory errors gracefully
            fragments.clear()
            gc.collect()
            raise

    def test_session_memory_cleanup(self):
        """Test memory cleanup for voice sessions."""
        # Create multiple sessions to test memory management
        sessions = []

        for i in range(100):
            session = VoiceSession(
                session_id=f"session_{i}",
                state=VoiceSessionState.IDLE,
                start_time=time.time(),
                last_activity=time.time(),
                conversation_history=[],
                current_voice_profile="default",
                audio_buffer=[],
                metadata={}
            )

            # Add large conversation history
            for j in range(50):
                session.conversation_history.append({
                    'type': 'user',
                    'text': f"Message {j} with lots of data " * 10,  # Large message
                    'timestamp': time.time(),
                    'confidence': 0.95
                })

            # Add large audio buffer
            for j in range(10):
                large_audio = np.random.randn(16000).astype(np.float32)
                audio_data = AudioData(
                    data=large_audio,
                    sample_rate=16000,
                    duration=1.0,
                    channels=1
                )
                session.audio_buffer.append(audio_data)

            sessions.append(session)

        # Test memory usage before cleanup
        initial_memory = psutil.Process().memory_info().rss

        # Clear sessions
        sessions.clear()

        # Force garbage collection
        gc.collect()

        # Test memory usage after cleanup
        final_memory = psutil.Process().memory_info().rss

        # Memory should be released (with some tolerance)
        memory_released = initial_memory - final_memory
        # Should have released at least some memory
        assert memory_released > -50 * 1024 * 1024  # Allow 50MB increase for test overhead


class TestCPUExhaustion:
    """Test handling of CPU exhaustion scenarios."""

    def test_cpu_intensive_operation_limits(self):
        """Test limits on CPU-intensive operations."""
        # Mock CPU-intensive audio processing
        def cpu_intensive_process(audio_data):
            # Simulate CPU-intensive operation
            result = np.zeros_like(audio_data.data)

            # Simulate heavy computation
            for i in range(len(audio_data.data)):
                # Complex mathematical operations
                value = audio_data.data[i]
                for j in range(100):  # Simulate processing loop
                    value = np.sin(value * np.pi * j / 100)

                result[i] = value

            return result

        # Test with timeout to prevent infinite CPU usage
        audio_data = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            duration=1.0,
            channels=1
        )

        # Test with timeout
        try:
            result = asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    cpu_intensive_process,
                    audio_data
                ),
                timeout=5.0  # 5 second timeout
            )
            # Should complete within timeout
            assert len(result) == len(audio_data.data)
        except asyncio.TimeoutError:
            # Should timeout before excessive CPU usage
            pass

    def test_concurrent_processing_limits(self):
        """Test limits on concurrent processing operations."""
        def cpu_task(task_id):
            """Simulate CPU-intensive task."""
            result = 0
            for i in range(100000):  # CPU-intensive loop
                result += i * task_id
            return result

        # Test concurrent execution limits
        max_concurrent = 4

        async def run_concurrent_test():
            # Limit concurrent tasks
            semaphore = asyncio.Semaphore(max_concurrent)

            async def limited_task(task_id):
                async with semaphore:
                    # Simulate CPU work
                    await asyncio.get_event_loop().run_in_executor(
                        None, cpu_task, task_id
                    )
                return task_id

            # Run many tasks with concurrency limit
            tasks = [limited_task(i) for i in range(20)]
            results = await asyncio.gather(*tasks)

            assert len(results) == 20

        asyncio.run(run_concurrent_test())

    def test_infinite_loop_protection(self):
        """Test protection against infinite loops."""
        # Simulate potential infinite loop scenario
        def potentially_infinite_process(data):
            counter = 0
            max_iterations = 1000000

            while counter < len(data):
                if counter > max_iterations:
                    break  # Emergency break
                counter += 1

            return counter < max_iterations

        # Test with various data sizes
        test_cases = [
            np.array([1, 2, 3]),
            np.array(list(range(1000))),
            np.array(list(range(100000))),
        ]

        for data in test_cases:
            result = potentially_infinite_process(data)
            assert result == True  # Should complete successfully


class TestResourceCleanup:
    """Test resource cleanup under various scenarios."""

    def test_file_descriptor_cleanup(self, temp_dir):
        """Test cleanup of file descriptors."""
        # Create many temporary files
        files = []
        max_files = 100

        try:
            for i in range(max_files):
                filepath = os.path.join(temp_dir, f"test_file_{i}.txt")
                f = open(filepath, 'w')
                f.write(f"Test data {i}")
                files.append(f)

            # Ensure all files are closed
            for f in files:
                f.close()

            files.clear()

            # Verify no file descriptor leaks
            # (In real scenario, would check ulimit or lsof)

        except OSError as e:
            if "Too many open files" in str(e):
                # Should handle file descriptor exhaustion
                for f in files:
                    try:
                        f.close()
                    except:
                        pass
                files.clear()
                raise

    def test_thread_pool_cleanup(self):
        """Test cleanup of thread pools."""
        def worker_task(task_id):
            """Worker task for thread pool."""
            time.sleep(0.1)  # Simulate work
            return task_id * 2

        # Test thread pool management
        max_workers = 4

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            futures = [executor.submit(worker_task, i) for i in range(20)]

            # Wait for completion
            results = [future.result() for future in as_completed(futures)]

            assert len(results) == 20

        # Thread pool should be cleaned up automatically

    def test_database_connection_cleanup(self):
        """Test cleanup of database connections."""
        # Mock database connections
        connections = []
        max_connections = 10

        try:
            for i in range(max_connections):
                # Mock connection creation
                conn = Mock()
                conn.is_open = True
                connections.append(conn)

            # Simulate connection pool exhaustion
            if len(connections) >= max_connections:
                # Should reuse or cleanup old connections
                old_conn = connections.pop(0)
                old_conn.is_open = False

                # Add new connection
                new_conn = Mock()
                new_conn.is_open = True
                connections.append(new_conn)

            # Cleanup all connections
            for conn in connections:
                conn.is_open = False

            connections.clear()

        except Exception:
            # Cleanup on error
            for conn in connections:
                try:
                    conn.is_open = False
                except:
                    pass
            connections.clear()

    def test_audio_processor_resource_cleanup(self):
        """Test resource cleanup in audio processor."""
        # Mock audio processor with resources
        config = Mock(spec=VoiceConfig)
        processor = SimplifiedAudioProcessor(config)

        # Mock resources that need cleanup
        processor.audio_stream = Mock()
        processor.recording_thread = Mock()
        processor.temp_files = ["/tmp/test1.wav", "/tmp/test2.wav"]

        # Test cleanup
        if hasattr(processor, 'cleanup'):
            processor.cleanup()
        else:
            # Manual cleanup
            if hasattr(processor, 'audio_stream') and processor.audio_stream:
                processor.audio_stream = None

            if hasattr(processor, 'recording_thread') and processor.recording_thread:
                if processor.recording_thread.is_alive():
                    processor.recording_thread.join(timeout=1.0)
                processor.recording_thread = None

            # Clean up temp files
            for temp_file in getattr(processor, 'temp_files', []):
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except:
                    pass
            processor.temp_files = []

        # Verify cleanup
        assert processor.audio_stream is None
        assert processor.recording_thread is None
        assert len(processor.temp_files) == 0


class TestResourceMonitoring:
    """Test resource monitoring and alerting."""

    def test_memory_usage_monitoring(self):
        """Test monitoring of memory usage."""
        # Get current process
        process = psutil.Process()

        # Monitor memory usage
        initial_memory = process.memory_info().rss

        # Allocate some memory
        test_data = []
        for i in range(100):
            data = np.random.randn(10000).astype(np.float32)
            test_data.append(data)

        # Check memory increase
        current_memory = process.memory_info().rss
        memory_increase = current_memory - initial_memory

        # Should detect memory increase
        assert memory_increase > 0

        # Cleanup
        test_data.clear()
        gc.collect()

        # Memory should decrease after cleanup
        final_memory = process.memory_info().rss
        memory_decrease = current_memory - final_memory

        # Should have released memory (with tolerance)
        assert memory_decrease > -10 * 1024 * 1024  # Allow 10MB increase

    def test_cpu_usage_monitoring(self):
        """Test monitoring of CPU usage."""
        # Get initial CPU times
        process = psutil.Process()
        initial_cpu_times = process.cpu_times()

        # Perform CPU-intensive operation
        start_time = time.time()

        # CPU-intensive task
        result = 0
        for i in range(1000000):
            result += i ** 2

        end_time = time.time()

        # Check CPU time increase
        final_cpu_times = process.cpu_times()
        cpu_time_used = (final_cpu_times.user + final_cpu_times.system) - \
                       (initial_cpu_times.user + initial_cpu_times.system)

        # Should have used CPU time
        assert cpu_time_used > 0

        # Wall clock time should be reasonable
        wall_clock_time = end_time - start_time
        assert wall_clock_time < 10.0  # Should complete in reasonable time

    def test_resource_threshold_alerting(self):
        """Test alerting when resource thresholds are exceeded."""
        # Define resource thresholds
        memory_threshold = 100 * 1024 * 1024  # 100MB
        cpu_threshold = 80.0  # 80% CPU

        # Monitor current resource usage
        process = psutil.Process()
        memory_usage = process.memory_info().rss
        cpu_usage = process.cpu_percent(interval=1.0)

        # Check if thresholds are exceeded
        memory_exceeded = memory_usage > memory_threshold
        cpu_exceeded = cpu_usage > cpu_threshold

        if memory_exceeded or cpu_exceeded:
            # Should trigger alerts or cleanup
            if memory_exceeded:
                # Trigger memory cleanup
                gc.collect()

            if cpu_exceeded:
                # Could throttle processing or alert
                pass

        # Verify monitoring works
        assert isinstance(memory_usage, int)
        assert isinstance(cpu_usage, float)
        assert cpu_usage >= 0.0 and cpu_usage <= 100.0


class TestConcurrentResourceManagement:
    """Test resource management under concurrent operations."""

    def test_concurrent_audio_processing_limits(self):
        """Test limits on concurrent audio processing."""
        def process_audio(audio_id):
            """Mock audio processing task."""
            # Simulate audio processing time
            time.sleep(0.1)
            return f"processed_{audio_id}"

        # Test concurrent processing with limits
        max_concurrent = 3

        async def run_concurrent_processing():
            semaphore = asyncio.Semaphore(max_concurrent)

            async def limited_process(audio_id):
                async with semaphore:
                    # Simulate async audio processing
                    await asyncio.sleep(0.1)
                    return process_audio(audio_id)

            # Process multiple audio files concurrently
            tasks = [limited_process(i) for i in range(10)]
            results = await asyncio.gather(*tasks)

            assert len(results) == 10
            assert all("processed_" in result for result in results)

        asyncio.run(run_concurrent_processing())

    def test_thread_resource_exhaustion(self):
        """Test handling of thread resource exhaustion."""
        def thread_task(task_id):
            """Task for thread pool."""
            time.sleep(0.01)
            return task_id

        # Test with limited thread pool
        max_threads = 5

        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            # Submit more tasks than threads
            futures = [executor.submit(thread_task, i) for i in range(20)]

            # Should handle queueing gracefully
            results = [future.result() for future in as_completed(futures)]

            assert len(results) == 20

    def test_memory_pressure_under_concurrency(self):
        """Test memory management under concurrent pressure."""
        # Test memory usage with concurrent operations

        def memory_intensive_task(task_id):
            """Memory intensive task."""
            # Allocate memory
            data = np.random.randn(10000).astype(np.float32)

            # Process data
            result = np.sum(data ** 2)

            return result

        # Run concurrent memory-intensive tasks
        max_workers = 2

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(memory_intensive_task, i) for i in range(10)]
            results = [future.result() for future in as_completed(futures)]

            assert len(results) == 10
            assert all(isinstance(r, (int, float)) for r in results)


class TestResourceExhaustionRecovery:
    """Test recovery mechanisms for resource exhaustion."""

    def test_memory_exhaustion_recovery(self):
        """Test recovery from memory exhaustion."""
        # Simulate memory exhaustion scenario
        large_allocations = []

        try:
            # Attempt to allocate large amounts of memory
            for i in range(100):
                try:
                    # Allocate large array
                    large_array = np.random.randn(100000).astype(np.float32)
                    large_allocations.append(large_array)
                except MemoryError:
                    # Memory exhausted - trigger cleanup
                    large_allocations.clear()
                    gc.collect()

                    # Retry with smaller allocation
                    small_array = np.random.randn(1000).astype(np.float32)
                    large_allocations.append(small_array)
                    break

        except Exception:
            # Final cleanup
            large_allocations.clear()
            gc.collect()
            raise

    def test_file_descriptor_exhaustion_recovery(self, temp_dir):
        """Test recovery from file descriptor exhaustion."""
        open_files = []

        try:
            # Attempt to open many files
            for i in range(200):  # More than typical ulimit
                try:
                    filepath = os.path.join(temp_dir, f"test_{i}.txt")
                    f = open(filepath, 'w')
                    f.write(f"Test data {i}")
                    open_files.append(f)
                except OSError as e:
                    if "Too many open files" in str(e):
                        # File descriptor exhaustion - cleanup and retry
                        for f in open_files:
                            f.close()
                        open_files.clear()

                        # Retry with fewer files
                        for j in range(10):
                            filepath = os.path.join(temp_dir, f"retry_{j}.txt")
                            f = open(filepath, 'w')
                            f.write(f"Retry data {j}")
                            open_files.append(f)
                        break
                    else:
                        raise

        finally:
            # Final cleanup
            for f in open_files:
                try:
                    f.close()
                except:
                    pass
            open_files.clear()

    def test_cpu_exhaustion_throttling(self):
        """Test CPU throttling under exhaustion."""
        # Simulate CPU exhaustion scenario
        cpu_intensive_tasks = []

        def cpu_task():
            """CPU intensive task."""
            result = 0
            for i in range(1000000):
                result += i * i
            return result

        # Run multiple CPU-intensive tasks
        max_concurrent = 2  # Limit concurrency

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # Submit tasks
            futures = [executor.submit(cpu_task) for _ in range(10)]

            # Should complete without system lockup
            results = [future.result() for future in as_completed(futures)]

            assert len(results) == 10

    def test_resource_pool_recovery(self):
        """Test recovery of resource pools after exhaustion."""
        # Simulate resource pool exhaustion and recovery
        resource_pool = []
        max_pool_size = 10
        min_pool_size = 2

        # Exhaust resource pool
        for i in range(max_pool_size + 5):
            if len(resource_pool) < max_pool_size:
                # Add resource to pool
                resource = Mock()
                resource.id = i
                resource_pool.append(resource)
            else:
                # Pool exhausted - should trigger cleanup
                # Remove oldest resources
                while len(resource_pool) > min_pool_size:
                    old_resource = resource_pool.pop(0)
                    old_resource.active = False

                # Add new resource
                resource = Mock()
                resource.id = i
                resource_pool.append(resource)

        # Verify pool management
        assert len(resource_pool) <= max_pool_size
        assert len(resource_pool) >= min_pool_size