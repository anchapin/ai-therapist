"""
Module Compatibility Layer
Provides mock implementations for missing modules to enable testing
"""

import sys
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, List, Optional

# Mock modules that don't exist yet
def setup_mock_modules():
    """Setup mock modules for testing compatibility."""
    
    # Mock audit logging
    if 'security.audit_logging' not in sys.modules:
        audit_logging = Mock()
        audit_logging.AuditLogger = Mock
        audit_logging.log_access = Mock(return_value=True)
        audit_logging.get_audit_trail = Mock(return_value=[])
        audit_logging.verify_integrity = Mock(return_value=True)
        sys.modules['security.audit_logging'] = audit_logging
    
    # Mock encryption service
    if 'security.encryption_service' not in sys.modules:
        encryption_service = Mock()
        encryption_service.EncryptionService = Mock
        encryption_service.encrypt = Mock(return_value=b'encrypted_data')
        encryption_service.decrypt = Mock(return_value=b'decrypted_data')
        encryption_service.generate_key = Mock(return_value=b'mock_key')
        sys.modules['security.encryption_service'] = encryption_service
    
    # Mock voice UI components
    if 'voice.voice_ui' not in sys.modules:
        voice_ui = Mock()
        voice_ui.render_voice_controls = Mock()
        voice_ui.update_waveform_display = Mock()
        voice_ui.handle_microphone_error = Mock()
        voice_ui.display_crisis_alert = Mock()
        sys.modules['voice.voice_ui'] = voice_ui
    
    # Mock additional voice components
    if 'voice.audio_processor' not in sys.modules:
        audio_processor = Mock()
        audio_processor.get_audio_devices = Mock(return_value=[{'name': 'Default', 'index': 0}])
        audio_processor.get_default_input_device = Mock(return_value={'name': 'Default', 'index': 0})
        sys.modules['voice.audio_processor'] = audio_processor

# Setup mocks when imported
setup_mock_modules()