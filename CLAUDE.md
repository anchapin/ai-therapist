# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv ai-therapist-env
source ai-therapist-env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp template.env .env
# Edit .env with your configuration (see Voice Configuration section)
```

### Application Commands
```bash
# Run the main AI therapist application with voice features
streamlit run app.py

# Test Ollama connection and models
python test_ollama.py

# Manually download knowledge files
python download_knowledge.py

# Build vector store from PDFs (legacy - app.py does this automatically)
python build_vectorstore.py

# Test voice features setup
python test_voice_setup.py

# Test voice commands
python test_enhanced_voice_commands.py
```

### Testing Commands
```bash
# Run all tests with comprehensive reporting
python tests/test_runner.py

# Run specific test categories
python -m pytest tests/unit/ -v --tb=short
python -m pytest tests/integration/ -v --tb=short
python -m pytest tests/security/ -v --tb=short
python -m pytest tests/performance/ -v --tb=short

# Run individual test files
python -m pytest tests/unit/test_audio_processor.py::TestAudioProcessor::test_initialization -v
python -m pytest tests/integration/test_voice_service.py -v

# Test with coverage
python -m pytest tests/ --cov=voice --cov-report=term-missing

# Voice-specific testing
python test_voice_security_comprehensive.py
python test_integration_voice_crisis.py
```

## Architecture Overview

This is a **Streamlit-based AI therapist application** with comprehensive **voice interaction capabilities** that uses **Ollama** for local LLM inference and **LangChain** for RAG (Retrieval-Augmented Generation) capabilities.

### Core Architecture Components

**Main Application (`app.py`)**
- Streamlit web interface with dual text/voice chat functionality
- Integrates voice features with traditional text-based conversation
- Security features: input sanitization, crisis detection, encryption
- Performance optimizations: caching, streaming responses, memory management
- Voice feature initialization and session management

**Voice Features Module (`voice/`)**
- **Voice Service (`voice_service.py`)**: Central orchestrator for voice features
- **Audio Processor (`audio_processor.py`)**: Real-time audio capture and processing
- **STT Service (`stt_service.py`)**: Multi-provider speech-to-text with fallback
- **TTS Service (`tts_service.py`)**: Multi-provider text-to-speech synthesis
- **Voice UI (`voice_ui.py`)**: Streamlit components for voice interaction
- **Voice Commands (`commands.py`)**: Voice command processing and execution
- **Security (`security.py`)**: Voice data encryption and privacy protection
- **Configuration (`config.py`)**: Centralized voice feature configuration

**Knowledge System**
- **Knowledge Downloader (`download_knowledge.py`)**: Downloads therapy resources
- **Vector Store System**: FAISS with Ollama embeddings (nomic-embed-text:latest)
- **Knowledge Files Configuration (`knowledge_files.txt`)**: URL mappings
- **Knowledge Base**: Located in `knowledge/` directory (gitignored)

**Testing Infrastructure (`tests/`)**
- **Unit Tests**: Component-level testing for all voice modules
- **Integration Tests**: End-to-end voice service testing
- **Security Tests**: HIPAA compliance and security validation
- **Performance Tests**: Load and stress testing
- **Test Runner**: Comprehensive reporting and CI integration

### Voice Features Architecture

**Voice Service Orchestration**
- Manages voice session lifecycle and state
- Coordinates STT, AI processing, and TTS pipeline
- Handles voice commands and emergency protocols
- Integrates with main application conversation flow

**Multi-Provider STT System**
- Primary: OpenAI Whisper API (whisper-1)
- Fallback: Local Whisper processing (base model)
- Alternative: Google Cloud Speech-to-Text
- Features: VAD, noise reduction, confidence scoring

**Multi-Provider TTS System**
- Primary: OpenAI TTS API (tts-1, alloy voice)
- Premium: ElevenLabs TTS (multilingual models)
- Local: Piper TTS (offline processing)
- Features: Voice profiles, emotion control, streaming

**Voice Commands System**
- Wake word detection and command parsing
- Emergency protocol handling (crisis detection)
- Therapy-specific commands (breathing exercises, reflections)
- Security levels and authentication

**Security & Privacy**
- End-to-end encryption for voice data
- Configurable data retention policies
- HIPAA compliance features
- Consent management and anonymization

### Key Configuration

**Environment Variables (`.env`)**
Core application settings:
```
KNOWLEDGE_PATH=./knowledge
VECTORSTORE_PATH=./vectorstore
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2:latest
OLLAMA_EMBEDDING_MODEL=nomic-embed-text:latest
```

Voice feature settings:
```
VOICE_ENABLED=true
VOICE_INPUT_ENABLED=true
VOICE_OUTPUT_ENABLED=true
VOICE_COMMANDS_ENABLED=true

# Audio Processing
AUDIO_SAMPLE_RATE=16000
AUDIO_CHANNELS=1
NOISE_REDUCTION_ENABLED=true
VAD_ENABLED=true

# STT Configuration
OPENAI_WHISPER_MODEL=whisper-1
WHISPER_MODEL=base

# TTS Configuration
OPENAI_TTS_MODEL=tts-1
OPENAI_TTS_VOICE=alloy
ELEVENLABS_VOICE_ID=your_preferred_voice_id

# Security
VOICE_ENCRYPTION_ENABLED=true
VOICE_CONSENT_REQUIRED=true
VOICE_HIPAA_COMPLIANCE_ENABLED=true
```

### Data Flow

**Voice Conversation Flow:**
1. **Voice Input**: Audio capture → VAD → STT processing → Text transcription
2. **Command Processing**: Voice command detection → Emergency/crisis checks
3. **AI Processing**: Conversation context → RAG retrieval → LLM generation
4. **Voice Output**: Text response → TTS synthesis → Audio playback
5. **Session Management**: State updates, caching, security logging

**Security Flow:**
1. **Input Validation**: Sanitization, prompt injection protection
2. **Crisis Detection**: Keyword analysis → Emergency response
3. **Data Encryption**: Voice data encryption at rest and in transit
4. **Privacy Controls**: Data retention, consent management, anonymization

### Critical Implementation Details

**Voice Module Dependencies**
- Streamlit (required for UI components)
- NumPy, SciPy (audio processing)
- librosa, soundfile (audio handling)
- webrtcvad (voice activity detection)
- OpenAI APIs (STT/TTS services)
- cryptography (security features)

**Testing Infrastructure**
- Pytest with async support for voice service testing
- Comprehensive mocking for external dependencies
- CI/CD pipeline with multi-Python version testing
- Security compliance testing and performance benchmarking

**Performance Optimizations**
- Response caching for common queries
- Streaming audio processing
- Parallel voice command processing
- Vector store persistence and optimization
- Background task management for voice features

**Security Architecture**
- Voice data encryption using Fernet symmetric encryption
- Input sanitization and prompt injection prevention
- Crisis keyword detection and emergency response protocols
- HIPAA compliance features (data retention, audit logging)
- Voice profile isolation and access controls

### Development Workflow

**Voice Feature Development:**
1. Set up environment with audio dependencies (may require system packages)
2. Configure `.env` with required API keys and voice settings
3. Test audio hardware: `python test_voice_setup.py`
4. Run voice command tests: `python test_enhanced_voice_commands.py`
5. Start application: `streamlit run app.py`
6. Test voice interaction flow and emergency protocols
7. Run security tests: `python test_voice_security_comprehensive.py`

**Testing Workflow:**
1. Run unit tests for individual components: `python -m pytest tests/unit/`
2. Run integration tests: `python -m pytest tests/integration/`
3. Run security compliance tests: `python -m pytest tests/security/`
4. Run performance tests: `python -m pytest tests/performance/`
5. Generate comprehensive report: `python tests/test_runner.py`

**CI/CD Integration:**
- Automated testing across Python 3.9, 3.10, 3.11
- Ollama service integration in CI environment
- Voice feature testing with mocked dependencies
- Security compliance validation
- Performance benchmarking and regression testing

### Voice Module Error Handling

**Graceful Degradation:**
- Optional UI imports when streamlit unavailable
- Fallback STT providers when primary services fail
- Local TTS processing when cloud services unavailable
- Emergency protocols for crisis situations

**Error Recovery:**
- Audio device detection and fallback handling
- Network connectivity resilience for cloud services
- Voice service restart and session recovery
- Comprehensive logging and error reporting

**Security Incident Response:**
- Voice data encryption key rotation
- Emergency protocol activation for crisis detection
- Audit logging and compliance reporting
- Data anonymization and privacy protection