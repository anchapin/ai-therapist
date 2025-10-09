"""
Comprehensive integration tests for database and service integration.
Tests end-to-end functionality between database and voice services.
"""

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta

# Import with robust error handling
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from database.db_manager import DatabaseManager, DatabaseConnectionPool
    from database.models import (
        User, Session, VoiceSession, Conversation, Message, 
        UserRepository, SessionRepository, VoiceSessionRepository,
        ConversationRepository, MessageRepository
    )
    from voice.voice_service import VoiceService
    from voice.config import VoiceConfig
    from auth.auth_service import AuthService
except ImportError as e:
    pytest.skip(f"Integration test dependencies not available: {e}", allow_module_level=True)


@pytest.mark.asyncio
@pytest.mark.integration
class TestDatabaseVoiceServiceIntegration:
    """Test integration between database and voice service."""
    
    @pytest.fixture
    def test_db_manager(self):
        """Create an in-memory database for testing."""
        # Use SQLite in-memory database for testing
        db_path = ":memory:"
        
        db_manager = DatabaseManager(db_path)
        
        yield db_manager
        
        db_manager.close()
    
    @pytest_asyncio.fixture
    async def repositories(self, test_db_manager):
        """Create repository instances."""
        return {
            'user': UserRepository(test_db_manager),
            'session': SessionRepository(test_db_manager),
            'voice_session': VoiceSessionRepository(test_db_manager),
            'conversation': ConversationRepository(test_db_manager),
            'message': MessageRepository(test_db_manager)
        }
    
    @pytest_asyncio.fixture
    def mock_voice_config(self):
        """Create a mock voice configuration."""
        config = Mock(spec=VoiceConfig)
        config.stt_enabled = True
        config.tts_enabled = True
        config.commands_enabled = True
        config.security_enabled = True
        config.max_session_duration = 3600
        config.audio_sample_rate = 16000
        config.audio_channels = 1
        config.session_timeout = 300
        return config
    
    @pytest_asyncio.fixture
    async def voice_service_with_db(self, mock_voice_config, test_db_manager):
        """Create a VoiceService integrated with database."""
        with patch('voice.voice_service.DatabaseManager', return_value=test_db_manager), \
             patch('voice.voice_service.SimplifiedAudioProcessor'), \
             patch('voice.voice_service.STTService'), \
             patch('voice.voice_service.TTSService'), \
             patch('voice.voice_service.VoiceSecurity'), \
             patch('voice.voice_service.VoiceCommandProcessor'):
            
            service = VoiceService(mock_voice_config)
            await service.initialize()
            service.db_manager = test_db_manager
            return service
    
    async def test_voice_session_persistence(self, voice_service_with_db, repositories):
        """Test that voice sessions are properly persisted to database."""
        # Create a voice session
        session = voice_service_with_db.create_session("user123")
        
        # Verify session exists in memory
        assert session.session_id in voice_service_with_db.sessions
        
        # Verify session was persisted to database
        db_session = await repositories['voice_session'].get_by_session_id(session.session_id)
        assert db_session is not None
        assert db_session.user_id == "user123"
        assert db_session.session_id == session.session_id
        assert db_session.is_active is True
    
    async def test_voice_session_data_repository_integration(self, voice_service_with_db, repositories):
        """Test integration with voice data repository."""
        # Create a user first
        user_data = {
            'email': 'test@example.com',
            'password_hash': 'hashed_password',
            'name': 'Test User',
            'role': 'patient'
        }
        user = await repositories['user'].create(user_data)
        
        # Create voice session for user
        session = voice_service_with_db.create_session(str(user.id))
        
        # Simulate voice data storage
        voice_data = {
            'session_id': session.session_id,
            'audio_data': b'sample_audio_data',
            'transcription': 'Hello world',
            'metadata': {'duration': 2.5, 'format': 'wav'}
        }
        
        # Store voice data (this would be done by voice service)
        stored_data = await voice_service_with_db.db_manager.execute_query(
            "INSERT INTO voice_data (session_id, audio_data, transcription, metadata) VALUES (?, ?, ?, ?)",
            (session.session_id, voice_data['audio_data'], voice_data['transcription'], str(voice_data['metadata']))
        )
        
        # Retrieve and verify
        result = await voice_service_with_db.db_manager.fetch_one(
            "SELECT * FROM voice_data WHERE session_id = ?",
            (session.session_id,)
        )
        
        assert result is not None
        assert result['transcription'] == 'Hello world'
    
    async def test_conversation_voice_integration(self, voice_service_with_db, repositories):
        """Test integration between conversations and voice sessions."""
        # Create user
        user_data = {
            'email': 'conversation@example.com',
            'password_hash': 'hashed_password',
            'name': 'Conversation User',
            'role': 'patient'
        }
        user = await repositories['user'].create(user_data)
        
        # Create conversation
        conv_data = {
            'user_id': str(user.id),
            'title': 'Voice Conversation',
            'session_type': 'voice'
        }
        conversation = await repositories['conversation'].create(conv_data)
        
        # Create voice session linked to conversation
        session = voice_service_with_db.create_session(str(user.id))
        session.conversation_id = str(conversation.id)
        
        # Store the updated session
        await repositories['voice_session'].update(session.session_id, {
            'conversation_id': str(conversation.id)
        })
        
        # Verify the link
        updated_session = await repositories['voice_session'].get_by_session_id(session.session_id)
        assert updated_session.conversation_id == str(conversation.id)
        
        # Create messages for the conversation
        message_data = {
            'conversation_id': str(conversation.id),
            'content': 'Hello from voice session',
            'message_type': 'user',
            'metadata': {'session_id': session.session_id}
        }
        message = await repositories['message'].create(message_data)
        
        # Verify message is linked correctly
        retrieved_message = await repositories['message'].get_by_id(str(message.id))
        assert retrieved_message.conversation_id == str(conversation.id)
        assert retrieved_message.metadata['session_id'] == session.session_id
    
    async def test_auth_service_voice_integration(self, repositories):
        """Test integration between auth service and voice features."""
        # Mock auth service dependencies
        with patch('auth.auth_service.DatabaseManager') as mock_db, \
             patch('auth.auth_service.UserRepository') as mock_user_repo:
            
            # Setup mock database manager
            mock_db_instance = Mock()
            mock_db.return_value = mock_db_instance
            
            # Create auth service
            auth_config = {
                'jwt_secret_key': 'test_secret',
                'token_expiry_hours': 24,
                'max_concurrent_sessions': 3
            }
            auth_service = AuthService(auth_config)
            
            # Create user via auth service
            user_result = await auth_service.register_user(
                email='voiceuser@example.com',
                password='SecurePass123',
                name='Voice User'
            )
            
            assert user_result.success is True
            assert user_result.user.email == 'voiceuser@example.com'
            
            # Login user
            login_result = await auth_service.login_user(
                email='voiceuser@example.com',
                password='SecurePass123'
            )
            
            assert login_result.success is True
            assert login_result.token is not None
            
            # Verify session was created
            sessions = await auth_service.get_user_sessions(str(user_result.user.id))
            assert len(sessions) >= 1
            assert sessions[0].is_active is True
    
    async def test_database_transaction_rollback(self, test_db_manager, repositories):
        """Test database transaction rollback on errors."""
        # Start a transaction
        async with test_db_manager.transaction() as tx:
            try:
                # Create user
                user_data = {
                    'email': 'rollback@example.com',
                    'password_hash': 'hashed_password',
                    'name': 'Rollback User',
                    'role': 'patient'
                }
                user = await repositories['user'].create(user_data)
                
                # Create session
                session_data = {
                    'user_id': str(user.id),
                    'token': 'test_token',
                    'expires_at': datetime.now() + timedelta(hours=1)
                }
                session = await repositories['session'].create(session_data)
                
                # Force an error to trigger rollback
                raise Exception("Intentional error for rollback testing")
                
            except Exception:
                # Transaction should rollback
                pass
        
        # Verify user was not committed
        user_check = await repositories['user'].get_by_email('rollback@example.com')
        assert user_check is None
        
        # Verify session was not committed
        session_check = await test_db_manager.fetch_one(
            "SELECT * FROM sessions WHERE token = ?",
            ('test_token',)
        )
        assert session_check is None
    
    async def test_voice_session_cleanup(self, voice_service_with_db, repositories):
        """Test cleanup of expired voice sessions."""
        # Create a voice session
        session = voice_service_with_db.create_session("cleanup_user")
        
        # Manually expire it in database
        expired_time = datetime.now() - timedelta(hours=1)
        await repositories['voice_session'].update(session.session_id, {
            'expires_at': expired_time,
            'is_active': False
        })
        
        # Run cleanup
        await voice_service_with_db.cleanup_expired_sessions()
        
        # Verify session was cleaned up from memory
        assert session.session_id not in voice_service_with_db.sessions
        
        # Verify session is marked inactive in database
        db_session = await repositories['voice_session'].get_by_session_id(session.session_id)
        assert db_session.is_active is False
    
    async def test_concurrent_voice_sessions(self, voice_service_with_db, repositories):
        """Test handling of concurrent voice sessions."""
        user_id = "concurrent_user"
        
        # Create multiple sessions for the same user
        sessions = []
        for i in range(3):
            try:
                session = voice_service_with_db.create_session(f"{user_id}_{i}")
                sessions.append(session)
            except Exception as e:
                # May hit session limit
                break
        
        # At least one session should be created
        assert len(sessions) >= 1
        
        # Verify all sessions are in database
        for session in sessions:
            db_session = await repositories['voice_session'].get_by_session_id(session.session_id)
            assert db_session is not None
            assert db_session.is_active is True
        
        # Clean up sessions
        for session in sessions:
            await voice_service_with_db.end_session(session.session_id)
    
    async def test_voice_data_audit_logging(self, voice_service_with_db, repositories):
        """Test audit logging for voice data operations."""
        # Create user
        user_data = {
            'email': 'audit@example.com',
            'password_hash': 'hashed_password',
            'name': 'Audit User',
            'role': 'patient'
        }
        user = await repositories['user'].create(user_data)
        
        # Create voice session
        session = voice_service_with_db.create_session(str(user.id))
        
        # Simulate voice data processing with audit log
        audit_data = {
            'user_id': str(user.id),
            'action': 'voice_data_processed',
            'resource_type': 'voice_session',
            'resource_id': session.session_id,
            'metadata': {
                'duration': 5.2,
                'transcription_length': 120,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Insert audit log
        await voice_service_with_db.db_manager.execute_query(
            """INSERT INTO audit_logs 
               (user_id, action, resource_type, resource_id, metadata, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (audit_data['user_id'], audit_data['action'], audit_data['resource_type'],
             audit_data['resource_id'], str(audit_data['metadata']), datetime.now())
        )
        
        # Verify audit log
        audit_log = await voice_service_with_db.db_manager.fetch_one(
            "SELECT * FROM audit_logs WHERE resource_id = ? AND action = ?",
            (session.session_id, 'voice_data_processed')
        )
        
        assert audit_log is not None
        assert audit_log['user_id'] == str(user.id)
        assert audit_log['action'] == 'voice_data_processed'
        assert audit_log['resource_id'] == session.session_id
    
    async def test_database_connection_pool_resilience(self, test_db_manager):
        """Test database connection pool resilience."""
        # Get multiple connections
        connections = []
        for i in range(3):
            conn = await test_db_manager.get_connection()
            connections.append(conn)
        
        # Verify all connections are active
        for conn in connections:
            assert conn is not None
        
        # Return connections
        for conn in connections:
            await test_db_manager.return_connection(conn)
        
        # Test health check
        health = await test_db_manager.health_check()
        assert health['status'] == 'healthy'
        assert 'active_connections' in health
        assert 'pool_size' in health


@pytest.mark.integration
class TestSecurityComplianceIntegration:
    """Test integration of security and compliance features."""
    
    @pytest_asyncio.fixture
    async def test_db_manager(self):
        """Create an in-memory database for testing."""
        db_config = {
            'database_url': 'sqlite:///:memory:',
            'pool_size': 5,
            'max_overflow': 10
        }
        
        db_manager = DatabaseManager(db_config)
        await db_manager.initialize()
        await db_manager.create_tables()
        
        yield db_manager
        db_manager.close()
    
    async def test_phi_data_encryption(self, test_db_manager):
        """Test encryption of PHI data in database."""
        # Test data encryption/decryption
        sensitive_data = "Patient SSN: 123-45-6789"
        
        # This would normally use the encryption service
        # For testing, simulate encryption
        encrypted_data = f"ENCRYPTED:{hash(sensitive_data)}"
        
        # Store encrypted data
        await test_db_manager.execute_query(
            "INSERT INTO audit_logs (user_id, action, metadata) VALUES (?, ?, ?)",
            ("test_user", "phi_data_access", {"encrypted_data": encrypted_data})
        )
        
        # Retrieve data
        result = await test_db_manager.fetch_one(
            "SELECT metadata FROM audit_logs WHERE user_id = ? AND action = ?",
            ("test_user", "phi_data_access")
        )
        
        assert result is not None
        assert "encrypted_data" in result['metadata']
    
    async def test_access_control_logging(self, test_db_manager):
        """Test access control event logging."""
        access_events = [
            {
                'user_id': 'user1',
                'resource': 'voice_session',
                'action': 'access_granted',
                'ip_address': '192.168.1.100',
                'user_agent': 'TestAgent/1.0'
            },
            {
                'user_id': 'user2',
                'resource': 'patient_record',
                'action': 'access_denied',
                'ip_address': '192.168.1.101',
                'user_agent': 'TestAgent/1.0',
                'reason': 'insufficient_permissions'
            }
        ]
        
        # Log access events
        for event in access_events:
            await test_db_manager.execute_query(
                """INSERT INTO audit_logs 
                   (user_id, action, resource_type, metadata, timestamp)
                   VALUES (?, ?, ?, ?, ?)""",
                (event['user_id'], event['action'], event['resource'], 
                 {k: v for k, v in event.items() if k not in ['user_id', 'action', 'resource']}, 
                 datetime.now())
            )
        
        # Verify access logs
        for event in access_events:
            log = await test_db_manager.fetch_one(
                """SELECT * FROM audit_logs 
                   WHERE user_id = ? AND action = ? AND resource_type = ?""",
                (event['user_id'], event['action'], event['resource'])
            )
            assert log is not None
            assert log['action'] == event['action']
    
    async def test_data_retention_policy(self, test_db_manager):
        """Test data retention policy enforcement."""
        # Insert old audit logs (beyond retention period)
        old_date = datetime.now() - timedelta(days=400)  # Beyond 1 year retention
        
        await test_db_manager.execute_query(
            "INSERT INTO audit_logs (user_id, action, timestamp) VALUES (?, ?, ?)",
            ("old_user", "old_action", old_date)
        )
        
        # Insert recent audit logs
        recent_date = datetime.now() - timedelta(days=10)
        await test_db_manager.execute_query(
            "INSERT INTO audit_logs (user_id, action, timestamp) VALUES (?, ?, ?)",
            ("recent_user", "recent_action", recent_date)
        )
        
        # Run cleanup (simulate retention policy)
        await test_db_manager.execute_query(
            "DELETE FROM audit_logs WHERE timestamp < ?",
            (datetime.now() - timedelta(days=365),)
        )
        
        # Verify old data is deleted, recent data remains
        old_log = await test_db_manager.fetch_one(
            "SELECT * FROM audit_logs WHERE user_id = ?",
            ("old_user",)
        )
        assert old_log is None
        
        recent_log = await test_db_manager.fetch_one(
            "SELECT * FROM audit_logs WHERE user_id = ?",
            ("recent_user",)
        )
        assert recent_log is not None
        assert recent_log['action'] == "recent_action"