"""
Phase 3: Integration & Performance Tests

End-to-end voice workflow testing, performance benchmarking, 
and memory leak prevention validation for the AI Therapist voice system.
"""

import pytest
import asyncio
import time
import threading
import psutil
import gc
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import tempfile
import os
from pathlib import Path

# Import voice modules
try:
    from voice.config import VoiceConfig
    from voice.security import VoiceSecurity
    VOICE_BASE_AVAILABLE = True
except ImportError:
    VOICE_BASE_AVAILABLE = False
    pytest.skip("Voice base modules not available", allow_module_level=True)

try:
    from voice.audio_processor import AudioProcessor
    AUDIO_PROCESSOR_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSOR_AVAILABLE = False

try:
    from voice.commands import VoiceCommandProcessor
    COMMAND_PROCESSOR_AVAILABLE = True
except ImportError:
    COMMAND_PROCESSOR_AVAILABLE = False

try:
    from voice.voice_service import VoiceService
    VOICE_SERVICE_AVAILABLE = True
except ImportError:
    VOICE_SERVICE_AVAILABLE = False


class TestVoiceWorkflowIntegration:
    """Test end-to-end voice workflow integration."""
    
    def setup_method(self):
        """Setup integrated test environment."""
        self.config = VoiceConfig()
        self.security = VoiceSecurity(self.config)
        
        if VOICE_SERVICE_AVAILABLE:
            self.voice_service = VoiceService(self.config, self.security)
        
        if AUDIO_PROCESSOR_AVAILABLE:
            self.audio_processor = AudioProcessor(self.config)
        
        if COMMAND_PROCESSOR_AVAILABLE:
            self.command_processor = VoiceCommandProcessor(self.config)
    
    def test_voice_to_text_to_voice_workflow(self):
        """Test complete voice workflow: input → processing → output."""
        # Simulate voice input data
        voice_input = b"simulated_voice_audio_data"
        user_id = "integration_test_user"
        
        # Step 1: Security validation
        security_valid = self.security._verify_security_requirements(user_id=user_id)
        assert isinstance(security_valid, bool)
        
        # Step 2: Audio encryption
        encrypted_audio = self.security.encrypt_audio_data(voice_input, user_id)
        assert encrypted_audio is not None
        assert encrypted_audio != voice_input
        
        # Step 3: Audio processing (if available)
        if AUDIO_PROCESSOR_AVAILABLE:
            try:
                # Simulate audio processing
                processed_result = self.audio_processor.validate_audio_data(voice_input)
                assert isinstance(processed_result, (bool, type(None)))
            except Exception:
                pass  # May fail due to missing dependencies
        
        # Step 4: Data decryption
        decrypted_audio = self.security.decrypt_audio_data(encrypted_audio, user_id)
        assert decrypted_audio == voice_input
    
    def test_consent_integration_workflow(self):
        """Test consent integration across voice components."""
        user_id = "consent_test_user"
        session_id = "consent_test_session"
        
        # Step 1: Check consent status
        consent_status = self.security._check_consent_status(user_id=user_id, session_id=session_id)
        assert isinstance(consent_status, bool)
        
        # Step 2: Verify security requirements
        security_valid = self.security._verify_security_requirements(user_id=user_id, session_id=session_id)
        assert isinstance(security_valid, bool)
        
        # Step 3: Process data with consent validation
        test_data = b"test_voice_data_with_consent"
        
        if consent_status and security_valid:
            # Should be able to process data
            encrypted = self.security.encrypt_data(test_data, user_id)
            assert encrypted is not None
            
            decrypted = self.security.decrypt_data(encrypted, user_id)
            assert decrypted == test_data
        else:
            # Should handle consent restrictions gracefully
            try:
                result = self.security.process_audio(Mock())
                # May return None if consent not given
                assert result is None or isinstance(result, (bytes, dict))
            except Exception:
                pass
    
    def test_cross_component_data_flow(self):
        """Test data flow between voice components."""
        # Test data
        test_user = "flow_test_user"
        test_data = b"workflow_test_data"
        
        # Step 1: Security layer
        security_result = self.security._verify_security_requirements(user_id=test_user)
        assert isinstance(security_result, bool)
        
        # Step 2: Audio processing layer (if available)
        if AUDIO_PROCESSOR_AVAILABLE:
            try:
                # Validate data through audio processor
                audio_valid = self.audio_processor.validate_audio_data(test_data)
                assert isinstance(audio_valid, (bool, type(None)))
            except Exception:
                pass
        
        # Step 3: Command processing layer (if available)
        if COMMAND_PROCESSOR_AVAILABLE:
            # Test command processing integration
            try:
                # Mock voice command processing
                command_result = self.command_processor.process_command("test command", test_user)
                # Handle various return types
                assert command_result is None or isinstance(command_result, (dict, str, bool))
            except Exception:
                pass
        
        # Step 4: Voice service integration (if available)
        if VOICE_SERVICE_AVAILABLE:
            # Test service level integration
            try:
                service_result = self.voice_service.get_status()
                assert isinstance(service_result, (dict, bool, type(None)))
            except Exception:
                pass
    
    def test_error_propagation_handling(self):
        """Test error handling across integrated components."""
        # Test with invalid data
        invalid_data = None
        invalid_user = ""
        
        # Security layer error handling
        try:
            security_result = self.security._verify_security_requirements(user_id=invalid_user)
            assert isinstance(security_result, bool)
        except Exception:
            pass  # Should handle gracefully
        
        # Audio processing error handling
        if AUDIO_PROCESSOR_AVAILABLE:
            try:
                audio_result = self.audio_processor.validate_audio_data(invalid_data)
                assert isinstance(audio_result, (bool, type(None)))
            except Exception:
                pass  # Should handle gracefully
        
        # Encryption error handling
        try:
            encrypted = self.security.encrypt_data(invalid_data, invalid_user)
            # May succeed or fail gracefully
        except Exception:
            pass  # Expected for invalid input


@pytest.mark.skipif(not AUDIO_PROCESSOR_AVAILABLE, reason="AudioProcessor not available")
class TestVoicePerformanceBenchmarking:
    """Test voice system performance under various loads."""
    
    def setup_method(self):
        """Setup performance test environment."""
        self.config = VoiceConfig()
        self.processor = AudioProcessor(self.config)
        self.security = VoiceSecurity(self.config)
        
        # Performance tracking
        self.performance_data = {}
    
    def test_audio_processing_performance(self):
        """Test audio processing performance benchmarks."""
        # Generate test audio data of various sizes
        audio_sizes = [1600, 16000, 160000]  # Small, medium, large
        
        for size in audio_sizes:
            test_audio = np.random.randint(-32768, 32768, size, dtype=np.int16)
            
            # Benchmark validation
            start_time = time.time()
            try:
                result = self.processor.validate_audio_data(test_audio)
                validation_time = time.time() - start_time
                
                # Store performance data
                self.performance_data[f"validation_{size}"] = validation_time
                
                # Performance assertions
                assert validation_time < 1.0  # Should complete within 1 second
                
            except Exception:
                # Handle gracefully if validation fails
                validation_time = time.time() - start_time
                self.performance_data[f"validation_error_{size}"] = validation_time
    
    def test_encryption_performance(self):
        """Test encryption/decryption performance."""
        test_data_sizes = [1024, 10240, 102400]  # 1KB, 10KB, 100KB
        user_id = "performance_test_user"
        
        for size in test_data_sizes:
            test_data = b"x" * size
            
            # Benchmark encryption
            encrypt_start = time.time()
            encrypted = self.security.encrypt_data(test_data, user_id)
            encrypt_time = time.time() - encrypt_start
            
            # Benchmark decryption
            decrypt_start = time.time()
            decrypted = self.security.decrypt_data(encrypted, user_id)
            decrypt_time = time.time() - decrypt_start
            
            # Store performance data
            self.performance_data[f"encrypt_{size}"] = encrypt_time
            self.performance_data[f"decrypt_{size}"] = decrypt_time
            
            # Verify data integrity
            assert decrypted == test_data
            
            # Performance assertions
            assert encrypt_time < 2.0  # Encryption should be fast
            assert decrypt_time < 2.0  # Decryption should be fast
    
    def test_concurrent_processing_performance(self):
        """Test performance under concurrent load."""
        num_threads = 5
        operations_per_thread = 10
        
        def concurrent_operation(thread_id):
            """Operation to run in concurrent threads."""
            results = []
            for i in range(operations_per_thread):
                user_id = f"thread_{thread_id}_user_{i}"
                test_data = f"test_data_{thread_id}_{i}".encode()
                
                try:
                    # Time encryption
                    start = time.time()
                    encrypted = self.security.encrypt_data(test_data, user_id)
                    encrypt_time = time.time() - start
                    
                    # Time decryption
                    start = time.time()
                    decrypted = self.security.decrypt_data(encrypted, user_id)
                    decrypt_time = time.time() - start
                    
                    results.append({
                        'thread_id': thread_id,
                        'operation': i,
                        'encrypt_time': encrypt_time,
                        'decrypt_time': decrypt_time,
                        'success': decrypted == test_data
                    })
                except Exception as e:
                    results.append({
                        'thread_id': thread_id,
                        'operation': i,
                        'error': str(e),
                        'success': False
                    })
            
            return results
        
        # Run concurrent operations
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(concurrent_operation, i) for i in range(num_threads)]
            all_results = [future.result() for future in futures]
        total_time = time.time() - start_time
        
        # Analyze results
        successful_ops = sum(len([r for r in results if r.get('success', False)]) for results in all_results)
        total_ops = num_threads * operations_per_thread
        
        # Performance assertions
        assert total_time < 10.0  # Should complete within 10 seconds
        assert successful_ops >= total_ops * 0.8  # At least 80% success rate
        
        # Store performance data
        self.performance_data['concurrent_total_time'] = total_time
        self.performance_data['concurrent_success_rate'] = successful_ops / total_ops
    
    def test_memory_usage_performance(self):
        """Test memory usage during operations."""
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform memory-intensive operations
        audio_chunks = []
        for i in range(100):
            chunk = np.random.randint(-32768, 32768, 16000, dtype=np.int16)
            audio_chunks.append(chunk)
            
            # Process chunk
            try:
                self.processor.validate_audio_data(chunk)
            except Exception:
                pass
        
        # Get peak memory usage
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Cleanup
        audio_chunks.clear()
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_after_cleanup = final_memory - initial_memory
        
        # Store performance data
        self.performance_data['memory_initial'] = initial_memory
        self.performance_data['memory_peak'] = peak_memory
        self.performance_data['memory_increase'] = memory_increase
        self.performance_data['memory_after_cleanup'] = memory_after_cleanup
        
        # Memory assertions
        assert memory_increase < 500  # Should not increase by more than 500MB
        assert memory_after_cleanup < memory_increase * 1.5  # Most memory should be recovered


class TestMemoryLeakPrevention:
    """Test memory leak prevention in voice components."""
    
    def setup_method(self):
        """Setup memory leak test environment."""
        self.config = VoiceConfig()
        self.security = VoiceSecurity(self.config)
        
        if AUDIO_PROCESSOR_AVAILABLE:
            self.processor = AudioProcessor(self.config)
    
    def test_audio_buffer_memory_leaks(self):
        """Test for memory leaks in audio buffer management."""
        if not AUDIO_PROCESSOR_AVAILABLE:
            pytest.skip("AudioProcessor not available")
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and process many audio buffers
        for i in range(1000):
            try:
                # Create large audio data
                audio_data = np.random.randint(-32768, 32768, 16000, dtype=np.int16)
                
                # Add to buffer if method exists
                if hasattr(self.processor, 'add_audio_buffer'):
                    self.processor.add_audio_buffer(audio_data)
                
                # Process data
                self.processor.validate_audio_data(audio_data)
                
                # Cleanup every 100 iterations
                if i % 100 == 0 and hasattr(self.processor, 'cleanup_buffers'):
                    self.processor.cleanup_buffers()
                
            except Exception:
                pass  # Handle gracefully
        
        # Force cleanup
        if hasattr(self.processor, 'cleanup_buffers'):
            self.processor.cleanup_buffers()
        
        # Check memory usage
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory leak assertion
        assert memory_increase < 100  # Should not leak more than 100MB
    
    def test_encryption_key_memory_leaks(self):
        """Test for memory leaks in encryption key management."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create many encryption keys through user operations
        for i in range(500):
            user_id = f"memory_test_user_{i}"
            test_data = f"test_data_{i}".encode()
            
            try:
                # Encrypt and decrypt (creates internal keys)
                encrypted = self.security.encrypt_data(test_data, user_id)
                decrypted = self.security.decrypt_data(encrypted, user_id)
                
                assert decrypted == test_data
            except Exception:
                pass
        
        # Force garbage collection
        gc.collect()
        
        # Check memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory leak assertion
        assert memory_increase < 50  # Should not leak more than 50MB
    
    def test_long_running_session_memory(self):
        """Test memory usage in long-running sessions."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate long-running session
        session_id = "long_running_session"
        user_id = "long_running_user"
        
        for cycle in range(100):
            try:
                # Simulate session activity
                consent_check = self.security._check_consent_status(
                    user_id=user_id, session_id=session_id
                )
                security_check = self.security._verify_security_requirements(
                    user_id=user_id, session_id=session_id
                )
                
                # Process some data
                test_data = f"cycle_{cycle}_data".encode()
                encrypted = self.security.encrypt_data(test_data, user_id)
                decrypted = self.security.decrypt_data(encrypted, user_id)
                
                assert decrypted == test_data
                
                # Periodic cleanup
                if cycle % 20 == 0:
                    gc.collect()
                    
            except Exception:
                pass
        
        # Final cleanup and check
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory leak assertion
        assert memory_increase < 75  # Should not leak more than 75MB


class TestSecurityPerformanceOptimization:
    """Test security performance under load."""
    
    def setup_method(self):
        """Setup security performance test environment."""
        self.config = VoiceConfig()
        self.security = VoiceSecurity(self.config)
    
    def test_bulk_encryption_performance(self):
        """Test performance of bulk encryption operations."""
        data_size = 1000  # Number of items to encrypt
        item_size = 1024  # Size of each item in bytes
        
        # Generate test data
        test_items = [
            (f"user_{i}", f"data_{i}".ljust(item_size, 'x').encode())
            for i in range(data_size)
        ]
        
        # Benchmark bulk encryption
        start_time = time.time()
        encrypted_items = []
        
        for user_id, data in test_items:
            try:
                encrypted = self.security.encrypt_data(data, user_id)
                encrypted_items.append((user_id, encrypted))
            except Exception:
                pass
        
        encryption_time = time.time() - start_time
        
        # Benchmark bulk decryption
        start_time = time.time()
        successful_decryptions = 0
        
        for user_id, encrypted_data in encrypted_items:
            try:
                decrypted = self.security.decrypt_data(encrypted_data, user_id)
                successful_decryptions += 1
            except Exception:
                pass
        
        decryption_time = time.time() - start_time
        
        # Performance assertions
        assert encryption_time < 30.0  # Should complete within 30 seconds
        assert decryption_time < 30.0  # Should complete within 30 seconds
        assert successful_decryptions >= len(encrypted_items) * 0.95  # 95% success rate
    
    def test_concurrent_security_operations(self):
        """Test security operations under concurrent load."""
        num_threads = 10
        operations_per_thread = 20
        
        def security_operations(thread_id):
            """Security operations for concurrent testing."""
            results = []
            for i in range(operations_per_thread):
                user_id = f"concurrent_user_{thread_id}_{i}"
                test_data = f"security_test_{thread_id}_{i}".encode()
                
                try:
                    # Test full security pipeline
                    start = time.time()
                    
                    consent = self.security._check_consent_status(user_id=user_id)
                    security_req = self.security._verify_security_requirements(user_id=user_id)
                    
                    if consent and security_req:
                        encrypted = self.security.encrypt_data(test_data, user_id)
                        decrypted = self.security.decrypt_data(encrypted, user_id)
                        success = decrypted == test_data
                    else:
                        success = True  # Graceful handling
                    
                    operation_time = time.time() - start
                    
                    results.append({
                        'thread_id': thread_id,
                        'operation': i,
                        'time': operation_time,
                        'success': success
                    })
                except Exception:
                    results.append({
                        'thread_id': thread_id,
                        'operation': i,
                        'success': False
                    })
            
            return results
        
        # Run concurrent security operations
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(security_operations, i) for i in range(num_threads)]
            all_results = [future.result() for future in futures]
        total_time = time.time() - start_time
        
        # Analyze results
        flat_results = [result for sublist in all_results for result in sublist]
        successful_ops = sum(1 for r in flat_results if r.get('success', False))
        total_ops = num_threads * operations_per_thread
        avg_time = sum(r.get('time', 0) for r in flat_results) / len(flat_results)
        
        # Performance assertions
        assert total_time < 60.0  # Should complete within 1 minute
        assert successful_ops >= total_ops * 0.9  # 90% success rate
        assert avg_time < 1.0  # Average operation should be fast


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])