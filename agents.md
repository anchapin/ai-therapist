# Agents

This document describes the specialized agents available in this AI Therapist repository and how to use them effectively.

## Available Agents

### Custom Droids

#### `generated-droid`
- **Location**: Personal (`.factory/droids/`)
- **Model**: inherit
- **Tools**: All tools available
- **Description**: Debug and fix tests from GitHub workflow runs in pull requests
- **Usage**: 
  ```bash
  task-cli --subagent-type=generated-droid --description="Fix tests" --prompt="Debug and fix failing tests from PR workflow"
  ```

### Repository-Specific Agent Patterns

The AI Therapist project follows several agent-specific patterns:

#### Testing Agents
- Focus on maintaining 90%+ test coverage
- Use custom test runner: `python tests/test_runner.py` (not standard pytest)
- Handle pytest markers: `unit`, `integration`, `security`, `performance`
- Mock external services (OpenAI, ElevenLabs) in test environments

#### Voice Service Agents
- Multi-provider STT/TTS with automatic fallbacks
- Security implementation in `voice/security.py`
- Configuration management in `voice/config.py`
- Voice command processing in `voice/commands.py`

#### Security Agents
- HIPAA compliance enforcement
- PII detection and masking
- JWT authentication management
- Audit logging and monitoring

#### Performance Agents
- Memory management with bounded buffers
- Resource monitoring and alerting
- Cache optimization
- Performance benchmarking

## Agent Usage Guidelines

### When to Use Agents

Use specialized agents for:
1. **Complex multi-step tasks** requiring 3+ distinct actions
2. **Non-trivial work** needing deliberate planning
3. **Multi-file operations** across different modules
4. **Security-sensitive tasks** requiring HIPAA compliance
5. **Performance optimization** across multiple components

### When NOT to Use Agents

Skip agents for:
1. **Single, straightforward tasks**
2. **Trivial operations** with minimal impact
3. **Informational requests** about code
4. **Simple file edits** in one location

### Agent Communication Patterns

#### For Testing Tasks
```bash
# Comprehensive test suite
task-cli --subagent-type=generated-droid --description="Test suite" --prompt="Run comprehensive test suite and fix any failures"

# Security testing
task-cli --subagent-type=generated-droid --description="Security tests" --prompt="Run HIPAA compliance tests and fix violations"

# Voice feature testing
task-cli --subagent-type=generated-droid --description="Voice tests" --prompt="Test voice service integration and STT/TTS fallbacks"
```

#### For Development Tasks
```bash
# Feature implementation
task-cli --subagent-type=generated-droid --description="Feature dev" --prompt="Implement new voice therapy feature with security and testing"

# Bug fixes
task-cli --subagent-type=generated-droid --description="Bug fix" --prompt="Debug and fix voice processing issue with fallback handling"

# Performance optimization
task-cli --subagent-type=generated-droid --description="Performance" --prompt="Optimize memory usage in audio processing pipeline"
```

## Agent-Specific Configuration

### Environment Setup
Agents should verify:
- Virtual environment activation
- Required environment variables (OPENAI_API_KEY, ELEVENLABS_API_KEY, JWT_SECRET_KEY)
- Vector store built with `python build_vectorstore.py`
- Test dependencies installed

### Security Requirements
All agents must:
- Maintain HIPAA compliance
- Use PII protection for user data
- Implement proper audit logging
- Follow authentication patterns

### Performance Standards
Agents should:
- Monitor memory usage
- Track response times
- Use connection pooling
- Implement caching where appropriate

## Agent Development

### Creating New Agents

To create a new specialized agent:
```bash
# Generate new droid configuration
factory generate-droid --description="Your agent description" --location="project|personal"
```

### Agent Best Practices
1. **Specific Focus**: Each agent should have a clear, narrow scope
2. **Security First**: Always consider HIPAA compliance
3. **Test Coverage**: Maintain 90%+ test coverage
4. **Documentation**: Include clear usage examples
5. **Error Handling**: Implement graceful fallbacks

### Agent Integration
- Use `task-cli` for agent invocation
- Provide clear task descriptions
- Include specific prompts for context
- Handle agent responses appropriately

## Common Agent Workflows

### Voice Feature Development
1. **Setup Agent**: Configure environment and dependencies
2. **Implementation Agent**: Write voice service code
3. **Security Agent**: Add HIPAA compliance features
4. **Testing Agent**: Verify functionality and coverage
5. **Performance Agent**: Optimize resource usage

### Security Enhancement
1. **Audit Agent**: Review current security posture
2. **Implementation Agent**: Add security features
3. **Testing Agent**: Verify HIPAA compliance
4. **Documentation Agent**: Update security guidelines

### Performance Optimization
1. **Analysis Agent**: Identify bottlenecks
2. **Optimization Agent**: Implement improvements
3. **Testing Agent**: Verify performance gains
4. **Monitoring Agent**: Set up alerts and tracking

## Agent Troubleshooting

### Common Issues
1. **Environment Problems**: Check virtual environment and dependencies
2. **Configuration Errors**: Verify `.env` file and API keys
3. **Test Failures**: Review test runner output and coverage
4. **Security Violations**: Check PII protection and audit logs
5. **Performance Issues**: Monitor resource usage and response times

### Debugging Agent Failures
1. Check agent logs in `/home/anchapin/.factory/logs/`
2. Review task history in `/home/anchapin/.factory/history.json`
3. Verify agent configuration and permissions
4. Test with simpler tasks to isolate issues

## Agent Resources

### Documentation
- `AGENT.md`: High-level project instructions for agents
- `AGENTS.md`: Specific agent guidance and patterns
- `CLAUDE.md`: Development commands and architecture
- `SECURITY_GUIDELINES.md`: Security requirements and compliance

### Tools and Utilities
- `tests/test_runner.py`: Comprehensive test suite runner
- `voice/mock_config.py`: Mock services for testing
- `security/pii_protection.py`: PII detection and masking
- `performance/monitor.py`: Resource monitoring utilities

### Configuration Files
- `template.env`: Environment variable template
- `knowledge_files.txt`: Therapy resource mappings
- `voice/config.py`: Voice service configuration
- `security/pii_config.py`: Security configuration
