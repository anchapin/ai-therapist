# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Testing

- Run comprehensive test suite with `python tests/test_runner.py` (not standard pytest)
- Maintain 90%+ test coverage requirement enforced by test runner
- Test categories use pytest markers: unit, integration, security, performance
- Voice features require specialized tests in `tests/voice/` directory

## Voice Services

- Multi-provider STT/TTS pattern with automatic fallbacks implemented in voice service
- Voice configuration in `voice/config.py` requires provider-specific API keys
- Mock voice services available in `voice/mock_config.py` for testing without API calls
- Voice commands must implement security checks in `voice/security.py`

## Security Patterns

- HIPAA compliance requires PII detection and masking for all user data
- JWT authentication with session management implemented in `auth/` module
- All database operations must use connection pooling from `database/db_manager.py`
- Security features must include audit logging in `security/pii_protection.py`

## Performance Management

- Memory management uses bounded buffers in `performance/memory_manager.py`
- Cache management implemented in `performance/cache_manager.py`
- Resource monitoring available in `performance/monitor.py`
- Performance tests must verify memory leak prevention

## Vector Store

- Knowledge base requires building with `python build_vectorstore.py` before first run
- FAISS vector store used for therapy material embeddings
- PDF processing requires knowledge files listed in `knowledge_files.txt`

## Configuration

- Required environment variables: OPENAI_API_KEY, ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, JWT_SECRET_KEY
- Use `.env` file for sensitive configuration (see template.env)
- Voice profiles must be updated in `voice/config.py` when adding new providers