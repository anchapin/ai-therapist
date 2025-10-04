# AI Therapist Project - Agent Instructions

## Architecture Overview

The AI Therapist is a HIPAA-compliant voice-enabled therapy system with these core components:

### Core Components

1. **Voice Processing Layer**
   - Speech-to-Text services with multiple provider support
   - Text-to-Speech with emotion-aware voice synthesis
   - Audio processing with quality analysis
   - Fallback mechanisms for reliability

2. **Security Layer**
   - HIPAA-compliant data protection
   - JWT-based authentication
   - PII detection and masking
   - Comprehensive audit logging
   - Role-based access control

3. **Database Layer**
   - SQLite with connection pooling
   - HIPAA-compliant data storage
   - Automatic data retention
   - Transaction management
   - Audit trail

4. **Performance Layer**
   - Real-time metrics collection
   - Resource usage monitoring
   - Automatic alerts
   - Performance optimization

5. **Knowledge Layer**
   - FAISS vector store
   - PDF document processing
   - Therapy material embeddings
   - Resource management

## Development Guidelines

### Environment Setup
```bash
# Create Python virtual environment
python -m venv ai-therapist-env

# Install dependencies
python -m pip install -r requirements.txt

# Build vector store
python build_vectorstore.py
```

### Configuration
- Use `.env` file for sensitive configuration
- Required environment variables:
  ```
  OPENAI_API_KEY=your-key
  ELEVENLABS_API_KEY=your-key
  ELEVENLABS_VOICE_ID=voice-id
  JWT_SECRET_KEY=your-secret
  ```

### Code Style
- Follow Python PEP 8 conventions
- Use snake_case for functions and variables
- Include docstrings for all modules/classes
- Use f-strings for string formatting
- Handle errors with try/except blocks
- Add logging for important operations

### Testing Requirements
- Maintain 90%+ test coverage
- Run full test suite before commits:
  ```bash
  python tests/test_runner.py
  ```
- Test categories:
  - Unit tests: Component testing
  - Integration tests: Service interaction
  - Security tests: HIPAA compliance
  - Performance tests: Load testing

## Security Guidelines

1. **Data Protection**
   - All PII must be encrypted at rest
   - Use PII detection for data masking
   - Implement role-based access
   - Maintain audit trails

2. **Authentication**
   - Use JWT tokens for auth
   - Implement session management
   - Enable consent tracking
   - Follow HIPAA requirements

3. **Monitoring**
   - Track security events
   - Monitor system health
   - Log access patterns
   - Alert on violations

## Development Workflow

1. **Setup Development Environment**
   ```bash
   git clone <repository>
   cd ai-therapist
   python -m venv ai-therapist-env
   source ai-therapist-env/bin/activate  # or activate.bat on Windows
   pip install -r requirements.txt
   ```

2. **Local Development**
   ```bash
   # Run local development server
   python app.py

   # Build vector store
   python build_vectorstore.py
   ```

3. **Testing**
   ```bash
   # Run all tests
   python tests/test_runner.py

   # Run specific test category
   pytest tests/unit/ -v
   pytest tests/integration/ -v
   pytest tests/security/ -v
   ```

4. **Deployment**
   - Update environment variables
   - Run security compliance checks
   - Verify HIPAA requirements
   - Deploy with monitoring enabled

## Common Tasks

### Adding New Voice Features
1. Update voice profile in `voice/config.py`
2. Implement handler in `voice/commands.py`
3. Add security checks in `voice/security.py`
4. Write tests in `tests/voice/`

### Implementing Security Features
1. Define requirements in `security/pii_config.py`
2. Add protection in `security/pii_protection.py`
3. Update audit logging
4. Add security tests

### Database Operations
1. Define schema in `database/migrations/`
2. Update models in `database/models.py`
3. Use connection pool from `db_manager.py`
4. Add database tests

## Performance Guidelines

1. **Resource Usage**
   - Monitor memory usage
   - Track response times
   - Set performance alerts
   - Optimize heavy operations

2. **Optimizations**
   - Use connection pooling
   - Enable caching
   - Implement streaming
   - Batch operations

3. **Monitoring**
   - Track system metrics
   - Monitor API latency
   - Check resource usage
   - Set up alerts

## Troubleshooting

1. **Voice Processing Issues**
   - Check audio device configuration
   - Verify API keys
   - Review error logs
   - Test provider fallbacks

2. **Security Issues**
   - Review audit logs
   - Check PII protection
   - Verify encryption
   - Test access control

3. **Performance Issues**
   - Check monitoring dashboard
   - Review resource usage
   - Analyze response times
   - Verify optimizations

## Contact

For questions or issues:
1. Check documentation
2. Review existing issues
3. Create detailed bug report
4. Follow security protocol for sensitive issues