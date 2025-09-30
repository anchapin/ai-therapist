#!/usr/bin/env python3
"""
Test script for voice feature setup verification.

This script tests:
1. Voice configuration loading
2. Voice service initialization
3. Audio device detection
4. STT/TTS service availability
5. Security module initialization
"""

import sys
import os
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_voice_configuration():
    """Test voice configuration loading."""
    print("🔧 Testing voice configuration...")
    try:
        from voice.config import VoiceConfig
        config = VoiceConfig()
        print(f"✅ Voice configuration loaded successfully")
        print(f"   - Voice enabled: {config.voice_enabled}")
        print(f"   - Voice input enabled: {config.voice_input_enabled}")
        print(f"   - Voice output enabled: {config.voice_output_enabled}")
        print(f"   - Preferred STT: {config.get_preferred_stt_service()}")
        print(f"   - Preferred TTS: {config.get_preferred_tts_service()}")
        return config
    except Exception as e:
        print(f"❌ Failed to load voice configuration: {e}")
        return None

def test_voice_security(config):
    """Test voice security initialization."""
    print("\n🔒 Testing voice security...")
    try:
        from voice.security import VoiceSecurity
        security = VoiceSecurity(config)
        print("✅ Voice security initialized successfully")
        print(f"   - Encryption enabled: {security.security_config.encryption_enabled}")
        print(f"   - Consent required: {security.security_config.consent_required}")
        print(f"   - Privacy mode: {security.security_config.privacy_mode}")
        return security
    except Exception as e:
        print(f"❌ Failed to initialize voice security: {e}")
        return None

def test_audio_processor():
    """Test audio processor initialization."""
    print("\n🎙️ Testing audio processor...")
    try:
        from voice.audio_processor import AudioProcessor
        from voice.config import VoiceConfig

        config = VoiceConfig()
        processor = AudioProcessor(config)

        print("✅ Audio processor initialized successfully")
        print(f"   - Input devices: {len(processor.input_devices)}")
        print(f"   - Output devices: {len(processor.output_devices)}")

        if processor.input_devices:
            print("   Available input devices:")
            for device in processor.input_devices[:3]:  # Show first 3
                print(f"     - {device['name']}")

        if processor.output_devices:
            print("   Available output devices:")
            for device in processor.output_devices[:3]:  # Show first 3
                print(f"     - {device['name']}")

        return processor
    except Exception as e:
        print(f"❌ Failed to initialize audio processor: {e}")
        return None

def test_stt_service():
    """Test STT service initialization."""
    print("\n🎤 Testing STT service...")
    try:
        from voice.stt_service import STTService
        from voice.config import VoiceConfig

        config = VoiceConfig()
        stt_service = STTService(config)

        print("✅ STT service initialized successfully")
        print(f"   - Available providers: {stt_service.get_available_providers()}")
        print(f"   - Preferred provider: {config.get_preferred_stt_service()}")

        # Test service statistics
        stats = stt_service.get_statistics()
        print(f"   - Request count: {stats['request_count']}")
        print(f"   - Error count: {stats['error_count']}")

        return stt_service
    except Exception as e:
        print(f"❌ Failed to initialize STT service: {e}")
        return None

def test_tts_service():
    """Test TTS service initialization."""
    print("\n🔊 Testing TTS service...")
    try:
        from voice.tts_service import TTSService
        from voice.config import VoiceConfig

        config = VoiceConfig()
        tts_service = TTSService(config)

        print("✅ TTS service initialized successfully")
        print(f"   - Available providers: {tts_service.get_available_providers()}")
        print(f"   - Preferred provider: {config.get_preferred_tts_service()}")
        print(f"   - Voice profiles: {len(tts_service.voice_profiles)}")

        # Test service statistics
        stats = tts_service.get_statistics()
        print(f"   - Request count: {stats['request_count']}")
        print(f"   - Cache size: {stats['cache_size']}")

        return tts_service
    except Exception as e:
        print(f"❌ Failed to initialize TTS service: {e}")
        return None

def test_voice_commands():
    """Test voice command processor."""
    print("\n🎯 Testing voice commands...")
    try:
        from voice.commands import VoiceCommandProcessor
        from voice.config import VoiceConfig

        config = VoiceConfig()
        command_processor = VoiceCommandProcessor(config)

        print("✅ Voice command processor initialized successfully")
        print(f"   - Available commands: {len(command_processor.get_available_commands())}")
        print(f"   - Wake word enabled: {command_processor.wake_word_enabled}")
        print(f"   - Wake word: {command_processor.wake_word}")

        # Show some available commands
        commands = command_processor.get_available_commands()[:5]
        print("   Sample commands:")
        for cmd in commands:
            print(f"     - {cmd['name']}: {cmd['description']}")

        return command_processor
    except Exception as e:
        print(f"❌ Failed to initialize voice command processor: {e}")
        return None

def test_voice_service():
    """Test complete voice service initialization."""
    print("\n🚀 Testing complete voice service...")
    try:
        from voice.voice_service import VoiceService
        from voice.config import VoiceConfig
        from voice.security import VoiceSecurity

        config = VoiceConfig()
        security = VoiceSecurity(config)
        voice_service = VoiceService(config, security)

        # Initialize the service
        if voice_service.initialize():
            print("✅ Voice service initialized successfully")

            # Create a test session
            session_id = voice_service.create_session()
            print(f"   - Test session created: {session_id}")

            # Get service statistics
            stats = voice_service.get_service_statistics()
            print(f"   - Sessions: {stats['sessions_count']}")
            print(f"   - Is running: {stats['is_running']}")

            return voice_service
        else:
            print("❌ Voice service initialization failed")
            return None
    except Exception as e:
        print(f"❌ Failed to initialize voice service: {e}")
        return None

def test_module_imports():
    """Test that all voice modules can be imported."""
    print("\n📦 Testing module imports...")

    modules_to_test = [
        'voice.config',
        'voice.security',
        'voice.audio_processor',
        'voice.stt_service',
        'voice.tts_service',
        'voice.voice_service',
        'voice.voice_ui',
        'voice.commands'
    ]

    failed_imports = []

    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✅ {module_name}")
        except Exception as e:
            print(f"❌ {module_name}: {e}")
            failed_imports.append(module_name)

    if failed_imports:
        print(f"\n❌ Failed to import {len(failed_imports)} modules")
        return False
    else:
        print(f"\n✅ All {len(modules_to_test)} modules imported successfully")
        return True

def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n📋 Checking dependencies...")

    required_packages = [
        'streamlit',
        'langchain',
        'langchain_community',
        'langchain_ollama',
        'numpy',
        'pyaudio',
        'librosa',
        'soundfile',
        'noisereduce',
        'webrtcvad',
        'cryptography',
        'pydantic'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n❌ Missing {len(missing_packages)} required packages")
        print("   Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    else:
        print(f"\n✅ All required packages are installed")
        return True

def main():
    """Main test function."""
    print("🎙️ AI Therapist Voice Feature Setup Test")
    print("=" * 50)

    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install missing packages.")
        return

    # Test module imports
    if not test_module_imports():
        print("\n❌ Module import test failed. Please check for syntax errors.")
        return

    # Test individual components
    config = test_voice_configuration()
    if not config:
        print("\n❌ Configuration test failed.")
        return

    security = test_voice_security(config)
    if not security:
        print("\n❌ Security test failed.")
        return

    processor = test_audio_processor()
    if not processor:
        print("\n❌ Audio processor test failed.")
        return

    stt_service = test_stt_service()
    if not stt_service:
        print("\n❌ STT service test failed.")
        return

    tts_service = test_tts_service()
    if not tts_service:
        print("\n❌ TTS service test failed.")
        return

    command_processor = test_voice_commands()
    if not command_processor:
        print("\n❌ Voice command test failed.")
        return

    voice_service = test_voice_service()
    if not voice_service:
        print("\n❌ Voice service test failed.")
        return

    # Cleanup
    print("\n🧹 Cleaning up...")
    try:
        if processor:
            processor.cleanup()
        if stt_service:
            stt_service.cleanup()
        if tts_service:
            tts_service.cleanup()
        if voice_service:
            voice_service.cleanup()
        print("✅ Cleanup completed")
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")

    print("\n🎉 All voice feature tests passed!")
    print("\n📝 Next Steps:")
    print("1. Configure your voice service API keys in .env")
    print("2. Run 'streamlit run app.py' to start the application")
    print("3. Use the voice setup wizard in the sidebar")
    print("4. Test voice recording and playback")

    print("\n🔧 Configuration Tips:")
    print("- Set ELEVENLABS_API_KEY for ElevenLabs TTS")
    print("- Set GOOGLE_CLOUD_CREDENTIALS_PATH for Google Speech-to-Text")
    print("- Ensure Ollama is running for local LLM processing")
    print("- Check audio permissions on your system")

if __name__ == "__main__":
    main()