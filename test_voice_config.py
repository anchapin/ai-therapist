"""
Test configuration for voice module security tests.

This module provides mock configurations and utilities for testing
voice security features without requiring actual audio hardware
or external services.
"""

import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


class MockSecurityConfig:
    """Mock security configuration for testing."""
    def __init__(self):
        self.encryption_enabled = True
        self.consent_required = True
        self.hipaa_compliance_enabled = True
        self.gdpr_compliance_enabled = True
        self.data_localization = True
        self.data_retention_hours = 24
        self.emergency_protocols_enabled = True
        self.privacy_mode = False
        self.anonymization_enabled = False


class MockAudioConfig:
    """Mock audio configuration for testing."""
    def __init__(self):
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        self.format = "wav"
        self.max_buffer_size = 100  # Small for testing
        self.max_memory_mb = 10     # Small for testing


class MockVoiceConfig:
    """Mock voice configuration for testing."""
    def __init__(self):
        self.voice_enabled = True
        self.default_voice_profile = "default"
        self.security = MockSecurityConfig()
        self.audio = MockAudioConfig()


class MockAudioDevices:
    """Mock audio device information."""
    def __init__(self):
        self.input_devices = [
            {
                'index': 0,
                'name': 'Mock Microphone',
                'channels': 1,
                'sample_rate': 16000
            }
        ]
        self.output_devices = [
            {
                'index': 0,
                'name': 'Mock Speaker',
                'channels': 1,
                'sample_rate': 16000
            }
        ]


def create_test_audio_data(duration=1.0, sample_rate=16000):
    """Create test audio data for testing."""
    num_samples = int(duration * sample_rate)
    # Generate sine wave test signal
    t = np.linspace(0, duration, num_samples)
    frequency = 440  # A4 note
    audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
    return audio_data.astype(np.float32)


def create_mock_voice_service():
    """Create a mock voice service for testing."""
    config = MockVoiceConfig()
    security_config = MockSecurityConfig()

    with patch('voice.security.Path'), \
         patch('voice.voice_service.SimplifiedAudioProcessor'), \
         patch('voice.voice_service.STTService'), \
         patch('voice.voice_service.TTSService'), \
         patch('voice.voice_service.VoiceCommandProcessor'):

        from voice.security import VoiceSecurity
        from voice.voice_service import VoiceService

        security = VoiceSecurity(config)
        service = VoiceService(config, security)

        # Mock device availability
        service.audio_processor.input_devices = MockAudioDevices().input_devices
        service.audio_processor.output_devices = MockAudioDevices().output_devices
        service.audio_processor.features = {
            'audio_capture': True,
            'audio_playback': True,
            'noise_reduction': False,
            'vad': False,
            'quality_analysis': False,
            'format_conversion': True
        }

        return service, security


def create_test_environment():
    """Create a test environment with temporary directories."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix='voice_test_')
    temp_path = Path(temp_dir)

    # Create required subdirectories
    (temp_path / "encrypted").mkdir(exist_ok=True)
    (temp_path / "consents").mkdir(exist_ok=True)
    (temp_path / "audit").mkdir(exist_ok=True)
    (temp_path / "emergency").mkdir(exist_ok=True)

    return temp_path


def cleanup_test_environment(temp_path):
    """Clean up test environment."""
    shutil.rmtree(temp_path, ignore_errors=True)


def patch_voice_directories(temp_path):
    """Patch voice module to use test directories."""
    def mock_path_side_effect(path_str):
        if path_str == "./voice_data":
            return temp_path
        return Path(path_str)

    return patch('voice.security.Path', side_effect=mock_path_side_effect)


class MockAudioData:
    """Mock AudioData class for testing."""
    def __init__(self, data=None, sample_rate=16000, channels=1, format="float32", duration=1.0):
        self.data = data if data is not None else create_test_audio_data(duration, sample_rate)
        self.sample_rate = sample_rate
        self.channels = channels
        self.format = format
        self.duration = duration
        self.timestamp = None

    def to_bytes(self):
        """Convert to bytes (mock implementation)."""
        return self.data.tobytes()

    @property
    def nbytes(self):
        """Get size in bytes."""
        return self.data.nbytes

    @property
    def size(self):
        """Get array size."""
        return self.data.size


def create_security_test_suite():
    """Create a complete security test suite configuration."""
    config = {
        'test_user_ids': [
            'valid_user',
            'test_user_123',
            'user-with-dash',
            'user_with_underscore',
            'User123Mixed',
        ],
        'invalid_user_ids': [
            '', None, 'user@domain.com', 'user#123', 'user space',
            'user\ttab', 'user\nnewline', 'a' * 51, 123, [],
            'user<test>', 'user&test', "user'test", 'user"test'
        ],
        'valid_ips': [
            '', '192.168.1.1', '10.0.0.1', '127.0.0.1', '0.0.0.0',
            '255.255.255.255', '1.2.3.4'
        ],
        'invalid_ips': [
            '999.999.999.999', '192.168.1', '192.168.1.1.1',
            '192.168.-1.1', '192.168.1.256', '192.168.1.a',
            '192,168,1,1', 'abc.def.ghi.jkl'
        ],
        'valid_consent_types': [
            'voice_processing', 'data_storage', 'transcription',
            'analysis', 'all_consent', 'emergency_protocol'
        ],
        'invalid_consent_types': [
            'invalid_consent', 'voice_processing_extra', 'custom_consent',
            '', None, 'VOICE_PROCESSING', 'all', 'none'
        ],
        'sql_injection_attempts': [
            "'; DROP TABLE users; --",
            "user' OR '1'='1",
            "admin'; INSERT INTO users VALUES('hacker', 'password'); --",
            "user' UNION SELECT * FROM sensitive_data --",
        ],
        'xss_attempts': [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "javascript:alert('xss')",
            "<iframe src='javascript:alert(\"xss\")'></iframe>",
            "<svg onload=alert('xss')>",
        ],
        'malicious_paths': [
            '/etc/passwd',
            'C:\\Windows\\System32\\config\\SAM',
            '${HOME}/.ssh/id_rsa',
            'file:///etc/passwd',
            'jdbc:mysql://localhost:3306/mysql',
            '../../etc/shadow',
            '..\\..\\windows\\system.ini'
        ],
        'large_data_sizes': [
            1_000_000,      # 1MB
            10_000_000,     # 10MB
            100_000_000,    # 100MB
            1_000_000_000,  # 1GB
        ]
    }

    return config


# Test utilities
def assert_security_validation_passes(result, test_name):
    """Assert that a security validation passes."""
    if not result:
        raise AssertionError(f"Security validation failed for {test_name}")


def assert_security_validation_fails(result, test_name):
    """Assert that a security validation fails."""
    if result:
        raise AssertionError(f"Security validation should have failed for {test_name}")


def assert_no_exceptions_raised(func, *args, **kwargs):
    """Assert that a function doesn't raise exceptions."""
    try:
        func(*args, **kwargs)
    except Exception as e:
        raise AssertionError(f"Unexpected exception raised: {e}")


def assert_exception_raised(func, expected_exception, *args, **kwargs):
    """Assert that a function raises an expected exception."""
    try:
        func(*args, **kwargs)
        raise AssertionError(f"Expected {expected_exception.__name__} was not raised")
    except expected_exception:
        pass  # Expected exception
    except Exception as e:
        raise AssertionError(f"Expected {expected_exception.__name__} but got {type(e).__name__}: {e}")


def generate_test_report(results, test_name="Security Tests"):
    """Generate a test report from results."""
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    percentage = (passed / total) * 100 if total > 0 else 0

    report = f"\n{'='*60}\n"
    report += f"{test_name} Report\n"
    report += f"{'='*60}\n"
    report += f"Total Tests: {total}\n"
    report += f"Passed: {passed}\n"
    report += f"Failed: {total - passed}\n"
    report += f"Success Rate: {percentage:.1f}%\n"
    report += f"{'='*60}\n"

    if not all(results.values()):
        report += "\nFailed Tests:\n"
        for test_name, result in results.items():
            if not result:
                report += f"  ✗ {test_name}\n"

    return report