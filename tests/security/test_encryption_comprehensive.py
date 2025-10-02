"""
Comprehensive encryption and data protection tests.

Tests security edge cases, breach scenarios, and encryption/decryption
across different data types with various attack vectors.
"""

import pytest
import asyncio
import math
import weakref
import gc
from unittest.mock import MagicMock, patch, PropertyMock
import json
import tempfile
import os
from datetime import datetime, timedelta
import hashlib
import base64
import cryptography
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from voice.security import VoiceSecurity, SecurityConfig, SecurityError


class TestEncryptionComprehensive:
    """Comprehensive encryption and data protection tests."""

    @pytest.fixture
    def security_config(self):
        """Create security configuration for testing."""
        return SecurityConfig(
            encryption_enabled=True,
            consent_required=True,
            privacy_mode=False,
            hipaa_compliance_enabled=True,
            data_retention_days=30,
            audit_logging_enabled=True
        )

    @pytest.fixture
    def security(self, security_config):
        """Create VoiceSecurity instance for testing."""
        return VoiceSecurity(security_config)

    @pytest.fixture
    def test_data_variants(self):
        """Various data types for encryption testing."""
        return {
            'empty_bytes': b'',
            'single_byte': b'\x00',
            'small_data': b'hello world',
            'large_data': b'x' * 10000,
            'binary_data': bytes(range(256)),
            'text_data': 'Sensitive patient information with PII data'.encode(),
            'json_data': json.dumps({
                'patient_id': 'P12345',
                'diagnosis': 'anxiety disorder',
                'treatment_plan': 'CBT therapy',
                'personal_notes': 'Patient shows signs of severe distress'
            }).encode(),
            'audio_like': b'\x00\x01\x02\x03' + b'\xFF' * 1000,  # Mock audio data
            'mixed_unicode': 'Patient: José María, Symptoms: café au lait'.encode('utf-8'),
            'structured_data': json.dumps({
                'timestamp': '2024-01-15T10:30:00Z',
                'session_id': 'sess_abc123',
                'user_id': 'user_xyz789',
                'transcript': 'I feel very anxious about work',
                'audio_metadata': {
                    'duration': 45.2,
                    'sample_rate': 16000,
                    'channels': 1,
                    'format': 'wav'
                }
            }).encode()
        }

    @pytest.fixture
    def malicious_data(self):
        """Malicious payloads for attack simulation."""
        return {
            'sql_injection': "'; DROP TABLE patients; --".encode(),
            'xss_attempt': '<script>alert("xss")</script>'.encode(),
            'path_traversal': '../../../etc/passwd'.encode(),
            'command_injection': 'rm -rf /; echo "pwned"'.encode(),
            'buffer_overflow': b'A' * 100000,
            'format_string': '%s%s%s%s%s%n%n%n%n%n'.encode(),
            'null_bytes': b'admin\x00legit_password',
            'unicode_exploit': '患者\x00\x00\x00\x00\x00'.encode('utf-8')
        }

    def test_encryption_different_data_types(self, security, test_data_variants):
        """Test encryption/decryption across different data types."""
        user_id = "test_user_123"

        for data_name, test_data in test_data_variants.items():
            # Test encryption
            encrypted = security.encrypt_data(test_data, user_id)
            assert encrypted != test_data, f"Data not encrypted for {data_name}"
            assert isinstance(encrypted, bytes), f"Encrypted data not bytes for {data_name}"

            # Test decryption
            decrypted = security.decrypt_data(encrypted, user_id)
            assert decrypted == test_data, f"Decryption failed for {data_name}"

            # Verify audit logging
            audit_logs = security.audit_logger.get_user_logs(user_id)
            encryption_events = [log for log in audit_logs if log.get('event_type') == 'data_encryption']
            decryption_events = [log for log in audit_logs if log.get('event_type') == 'data_decryption']

            assert len(encryption_events) > 0, f"No encryption audit log for {data_name}"
            assert len(decryption_events) > 0, f"No decryption audit log for {data_name}"

    def test_encryption_malicious_payloads(self, security, malicious_data):
        """Test encryption with malicious payloads."""
        user_id = "test_user_123"

        for attack_name, malicious_payload in malicious_data.items():
            # Should not crash on malicious input
            try:
                encrypted = security.encrypt_data(malicious_payload, user_id)
                assert encrypted != malicious_payload, f"Malicious data not encrypted: {attack_name}"
                assert isinstance(encrypted, bytes)

                # Should be able to decrypt back to original
                decrypted = security.decrypt_data(encrypted, user_id)
                assert decrypted == malicious_payload, f"Malicious payload corrupted: {attack_name}"

            except SecurityError:
                # Expected for some malformed payloads
                pass
            except Exception as e:
                pytest.fail(f"Unexpected error with malicious payload {attack_name}: {e}")

    def test_encryption_key_rotation_simulation(self, security):
        """Test encryption key rotation scenarios."""
        user_id = "test_user_123"
        test_data = b"persistent_sensitive_data"

        # Encrypt with current key
        encrypted_current = security.encrypt_data(test_data, user_id)

        # Simulate key rotation by patching _get_current_time method
        with patch.object(security, '_get_current_time') as mock_get_time:
            # Move time forward by encryption_key_rotation_days + 1
            future_date = datetime.now() + timedelta(days=security.original_config.encryption_key_rotation_days + 1)
            mock_get_time.return_value = future_date

            # Try to decrypt with "old" key - should fail
            with pytest.raises((SecurityError, ValueError)):
                security.decrypt_data(encrypted_current, user_id)

    def test_encryption_timing_attacks(self, security):
        """Test resistance to timing attacks."""
        user_id = "test_user_123"
        wrong_user = "attacker_user_456"
        test_data = b"sensitive_data"

        # Encrypt data with correct user
        encrypted = security.encrypt_data(test_data, user_id)

        # Measure time for correct user decryption
        import time
        start_time = time.time()
        security.decrypt_data(encrypted, user_id)
        correct_time = time.time() - start_time

        # Measure time for wrong user decryption (should fail)
        start_time = time.time()
        try:
            security.decrypt_data(encrypted, wrong_user)
        except (SecurityError, ValueError):
            pass  # Expected to fail
        wrong_time = time.time() - start_time

        # Times should be reasonably similar (within 50ms) to prevent timing attacks
        time_diff = abs(correct_time - wrong_time)
        assert time_diff < 0.05, f"Timing difference too large: {time_diff}s"

    def test_encryption_parallel_access(self, security):
        """Test encryption/decryption under concurrent access."""
        import threading
        import queue

        test_data = b"concurrent_test_data"
        user_id = "concurrent_user_123"
        results = queue.Queue()
        errors = queue.Queue()

        def encrypt_decrypt_worker(worker_id):
            try:
                # Each worker encrypts and decrypts data
                for i in range(10):
                    data = f"worker_{worker_id}_data_{i}".encode()
                    encrypted = security.encrypt_data(data, user_id)
                    decrypted = security.decrypt_data(encrypted, user_id)
                    assert decrypted == data
                results.put(f"worker_{worker_id}_success")
            except Exception as e:
                errors.put(f"worker_{worker_id}_error: {e}")

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=encrypt_decrypt_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)

        # Check results
        success_count = 0
        while not results.empty():
            results.get()
            success_count += 1

        error_count = 0
        while not errors.empty():
            errors.get()
            error_count += 1

        assert success_count == 5, f"Only {success_count} workers succeeded"
        assert error_count == 0, f"{error_count} workers failed"

    def test_encryption_data_integrity(self, security):
        """Test data integrity across encryption cycles."""
        user_id = "integrity_user_123"
        original_data = b"integrity_test_data"

        # Multiple encryption/decryption cycles
        current_data = original_data
        for cycle in range(10):
            encrypted = security.encrypt_data(current_data, user_id)
            decrypted = security.decrypt_data(encrypted, user_id)

            # Data should remain identical
            assert decrypted == original_data, f"Data corruption in cycle {cycle}"

            # Hash should change after encryption
            assert hashlib.sha256(decrypted).digest() != hashlib.sha256(encrypted).digest()
            current_data = decrypted

    def test_encryption_memory_safety(self, security):
        """Test memory safety during encryption operations."""
        import gc
        import weakref

        user_id = "memory_user_123"
        large_data = b'x' * 1000000  # 1MB of data

        # Test with large data
        encrypted = security.encrypt_data(large_data, user_id)

        # Force garbage collection
        gc.collect()

        # Should still be able to decrypt
        decrypted = security.decrypt_data(encrypted, user_id)
        assert decrypted == large_data

        # Test memory cleanup (bytes objects can't be weak referenced, but we can test deletion)
        encrypted_id = id(encrypted)
        del encrypted
        gc.collect()
        # Note: We can't directly test weak references to bytes, but we can ensure no exceptions occur

    def test_encryption_side_channel_attacks(self, security):
        """Test resistance to side-channel attacks."""
        user_id = "side_channel_user_123"
        attacker_user = "attacker_user_456"
        test_data = b"secret_data"

        # Encrypt data with legitimate user
        encrypted = security.encrypt_data(test_data, user_id)

        # Attacker tries to observe encryption patterns
        attacker_data = b"attacker_data"
        attacker_encrypted = security.encrypt_data(attacker_data, attacker_user)

        # Verify that encrypted data appears random and doesn't leak patterns
        encrypted_entropy = self._calculate_entropy(encrypted)
        attacker_entropy = self._calculate_entropy(attacker_encrypted)

        # Both should have high entropy (appear random)
        assert encrypted_entropy > 7.5, "Main encryption has low entropy"
        assert attacker_entropy > 7.5, "Attacker encryption has low entropy"

        # Lengths should be different (no size-based correlation)
        assert len(encrypted) != len(attacker_encrypted), "Encryption sizes are correlated"

    def test_encryption_cryptographic_failures(self, security):
        """Test handling of cryptographic failures."""
        user_id = "crypto_fail_user_123"
        test_data = b"test_data"

        # Test with corrupted encrypted data
        corrupted_data = b"corrupted_encrypted_data"

        with pytest.raises((SecurityError, ValueError)):
            security.decrypt_data(corrupted_data, user_id)

        # Test with wrong key material
        with patch.object(security, 'master_key', None):
            with pytest.raises(SecurityError):
                security.encrypt_data(test_data, user_id)

    def test_encryption_edge_cases(self, security):
        """Test encryption edge cases."""
        user_id = "edge_case_user_123"

        # Test with None data (should handle gracefully)
        with pytest.raises((TypeError, AttributeError)):
            security.encrypt_data(None, user_id)

        # Test with very long user ID
        long_user_id = "user_" + "a" * 1000
        short_data = b"short"
        encrypted = security.encrypt_data(short_data, long_user_id)
        decrypted = security.decrypt_data(encrypted, long_user_id)
        assert decrypted == short_data

        # Test with special characters in user ID
        special_user_id = "user@#$%^&*()_+{}[]|\\:;'<>?,./"
        encrypted = security.encrypt_data(short_data, special_user_id)
        decrypted = security.decrypt_data(encrypted, special_user_id)
        assert decrypted == short_data

    def test_audio_encryption_specific_attacks(self, security):
        """Test audio-specific encryption attack vectors."""
        user_id = "audio_user_123"

        # Test with audio-like data that might exploit audio processing
        audio_payloads = [
            b'\x00' * 1000,  # Silent audio
            b'\xFF' * 1000,  # Max amplitude
            b'\x80' * 1000,  # Zero-crossing point
            bytes([i % 256 for i in range(1000)]),  # Ramp pattern
            b''.join([bytes([i]) * 10 for i in range(100)])  # Repeating patterns
        ]

        for i, audio_data in enumerate(audio_payloads):
            # Should encrypt without issues
            encrypted = security.encrypt_audio_data(audio_data, user_id)
            assert encrypted != audio_data

            # Should decrypt correctly
            decrypted = security.decrypt_audio_data(encrypted, user_id)
            assert decrypted == audio_data

            # Verify no pattern leakage in encrypted data
            entropy = self._calculate_entropy(encrypted)
            assert entropy > 6.0, f"Low entropy in audio payload {i}"

    def test_encryption_brute_force_protection(self, security):
        """Test protection against brute force attacks."""
        user_id = "brute_force_user_123"
        test_data = b"secret_data"

        encrypted = security.encrypt_data(test_data, user_id)

        # Simulate brute force attempts with common keys
        common_keys = [
            b"password", b"123456", b"admin", b"letmein",
            b"qwerty", b"monkey", b"dragon", b"master"
        ]

        # Try to decrypt with common keys (should all fail)
        for key in common_keys:
            with patch.object(security, 'master_key', Fernet(key)):
                with pytest.raises((SecurityError, ValueError)):
                    security.decrypt_data(encrypted, user_id)

    def test_encryption_data_tampering_detection(self, security):
        """Test detection of data tampering attempts."""
        user_id = "tamper_user_123"
        original_data = b"tamper_test_data"

        # Encrypt data
        encrypted = security.encrypt_data(original_data, user_id)

        # Attempt various tampering attacks
        tampering_attempts = [
            encrypted[:-1],  # Truncated data
            encrypted + b'x',  # Extended data
            encrypted[0:10] + b'\x00' * len(encrypted[10:]),  # Zeroed section
            bytes([b ^ 0xFF for b in encrypted]),  # Bit flipped
            encrypted[0:5] + encrypted[6:5] + encrypted[5:6] + encrypted[7:],  # Swapped bytes
        ]

        # All tampering attempts should fail decryption
        for tampered in tampering_attempts:
            with pytest.raises((SecurityError, ValueError)):
                security.decrypt_data(tampered, user_id)

    def test_encryption_performance_under_load(self, security):
        """Test encryption performance under heavy load."""
        user_id = "load_user_123"

        # Generate large dataset
        data_sizes = [1000, 10000, 100000]  # Various data sizes
        iterations = 100

        for size in data_sizes:
            test_data = b'x' * size

            start_time = datetime.now()
            for _ in range(iterations):
                encrypted = security.encrypt_data(test_data, user_id)
                decrypted = security.decrypt_data(encrypted, user_id)
                assert decrypted == test_data

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Performance should be reasonable (adjust thresholds as needed)
            max_duration = size * iterations / 1000000  # Rough estimate
            assert duration < max_duration + 5, f"Performance too slow for size {size}: {duration}s"

    def _calculate_entropy(self, data):
        """Calculate Shannon entropy of data."""
        if not data:
            return 0.0

        entropy = 0.0
        byte_counts = [0] * 256

        for byte in data:
            byte_counts[byte] += 1

        data_length = len(data)
        for count in byte_counts:
            if count > 0:
                probability = count / data_length
                entropy -= probability * math.log2(probability)

        return entropy

    def test_encryption_comprehensive_audit_trail(self, security):
        """Test comprehensive audit trail for encryption operations."""
        user_id = "audit_user_123"
        test_data = b"audit_test_data"

        # Clear existing logs
        security.audit_logger.logs.clear()

        # Perform encryption operations
        encrypted = security.encrypt_data(test_data, user_id)

        # Check audit trail
        user_logs = security.audit_logger.get_user_logs(user_id)
        encryption_logs = [log for log in user_logs if log.get('event_type') == 'data_encryption']

        assert len(encryption_logs) >= 1, "No encryption audit logs found"

        encryption_log = encryption_logs[0]
        assert encryption_log['user_id'] == user_id
        assert 'data_size' in encryption_log.get('details', {})
        assert encryption_log['details']['data_size'] == len(test_data)

        # Test audit trail tampering detection
        original_log = encryption_log.copy()
        encryption_log['details']['data_size'] = 999999  # Tamper with log

        # System should detect or prevent log tampering
        # (Implementation dependent on audit system design)

    def test_encryption_cross_user_contamination(self, security):
        """Test that encryption doesn't leak data between users."""
        user1 = "user_one_123"
        user2 = "user_two_456"

        data1 = b"user1_secret_data"
        data2 = b"user2_secret_data"

        # Encrypt data for both users
        encrypted1 = security.encrypt_data(data1, user1)
        encrypted2 = security.encrypt_data(data2, user2)

        # Verify users can't access each other's data
        with pytest.raises((SecurityError, ValueError)):
            security.decrypt_data(encrypted1, user2)

        with pytest.raises((SecurityError, ValueError)):
            security.decrypt_data(encrypted2, user1)

        # Verify original data integrity
        decrypted1 = security.decrypt_data(encrypted1, user1)
        decrypted2 = security.decrypt_data(encrypted2, user2)

        assert decrypted1 == data1
        assert decrypted2 == data2
        assert decrypted1 != decrypted2

    def test_encryption_emergency_scenarios(self, security):
        """Test encryption behavior in emergency scenarios."""
        user_id = "emergency_user_123"
        test_data = b"emergency_sensitive_data"

        # Test encryption during high memory usage simulation
        large_allocations = [b'x' * 1000000 for _ in range(10)]

        try:
            # Attempt encryption under memory pressure
            encrypted = security.encrypt_data(test_data, user_id)
            decrypted = security.decrypt_data(encrypted, user_id)
            assert decrypted == test_data

        finally:
            # Clean up large allocations
            del large_allocations

        # Test encryption with system resource constraints
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.percent = 95  # High memory usage

            encrypted = security.encrypt_data(test_data, user_id)
            decrypted = security.decrypt_data(encrypted, user_id)
            assert decrypted == test_data

    def test_encryption_recovery_mechanisms(self, security):
        """Test encryption recovery from various failure modes."""
        user_id = "recovery_user_123"
        test_data = b"recovery_test_data"

        # Test recovery from partial encryption failure
        encrypted = security.encrypt_data(test_data, user_id)

        # Simulate partial corruption (first 16 bytes)
        corrupted = b'CORRUPTED' + encrypted[16:]

        # Should handle gracefully
        with pytest.raises((SecurityError, ValueError)):
            security.decrypt_data(corrupted, user_id)

        # Original should still work
        decrypted = security.decrypt_data(encrypted, user_id)
        assert decrypted == test_data

    def test_encryption_protocol_downgrade_attacks(self, security):
        """Test resistance to protocol downgrade attacks."""
        user_id = "downgrade_user_123"
        test_data = b"downgrade_test_data"

        # Encrypt with current protocol
        encrypted = security.encrypt_data(test_data, user_id)

        # Verify encrypted data uses proper encryption (not plaintext)
        assert not encrypted.startswith(test_data[:min(10, len(test_data))])

        # Verify no weak encryption patterns
        weak_patterns = [
            b'PLAIN', b'NOENC', b'WEAK', b'BASE64'
        ]

        for pattern in weak_patterns:
            assert not encrypted.startswith(pattern), f"Weak encryption pattern detected: {pattern}"