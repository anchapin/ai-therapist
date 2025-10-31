"""
Database Integration Tests for AI Therapist.

Tests for database integration with authentication and voice services,
ensuring proper data persistence, transaction management, and HIPAA compliance.
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta

from database.db_manager import DatabaseManager
from database.models import UserRepository, SessionRepository, VoiceDataRepository, AuditLogRepository, ConsentRepository
from auth.auth_service import AuthService
from auth.user_model import UserModel, UserRole
from voice.security import VoiceSecurity, SecurityConfig


class TestDatabaseIntegration:
    """Test database integration with application services."""

    @pytest.fixture
    def db_manager(self):
        """Create a temporary database manager for testing."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            db_path = tmp.name

        manager = DatabaseManager(db_path)
        yield manager

        # Cleanup
        manager.close()
        os.unlink(db_path)

    def test_auth_service_database_integration(self, db_manager):
        """Test authentication service integration with database."""
        # Create repositories with test database
        user_repo = UserRepository()
        session_repo = SessionRepository()
        user_repo.db = db_manager
        session_repo.db = db_manager

        # Create auth service with repositories
        user_model = UserModel()
        user_model.user_repo = user_repo

        auth_service = AuthService(user_model)
        auth_service.session_repo = session_repo

        # Test user registration
        result = auth_service.register_user(
            email="test@example.com",
            password="TestPass123",
            full_name="Test User"
        )

        assert result.success
        assert result.user is not None
        assert result.user.email == "test@example.com"

        # Verify user was saved to database
        db_user = user_repo.find_by_email("test@example.com")
        assert db_user is not None
        assert db_user.full_name == "Test User"

        # Test user login
        login_result = auth_service.login_user("test@example.com", "TestPass123")
        assert login_result.success
        assert login_result.token is not None
        assert login_result.session is not None

        # Verify session was saved to database
        db_session = session_repo.find_by_id(login_result.session.session_id)
        assert db_session is not None
        assert db_session.user_id == result.user.user_id

        # Test token validation
        validated_user = auth_service.validate_token(login_result.token)
        assert validated_user is not None
        assert validated_user.user_id == result.user.user_id

    def test_voice_security_database_integration(self, db_manager):
        """Test voice security integration with database."""
        # Create repositories with test database
        audit_repo = AuditLogRepository()
        consent_repo = ConsentRepository()
        voice_repo = VoiceDataRepository()
        audit_repo.db = db_manager
        consent_repo.db = db_manager
        voice_repo.db = db_manager

        # Create voice security with repositories
        config = SecurityConfig()
        security = VoiceSecurity(config)
        security.audit_repo = audit_repo
        security.consent_repo = consent_repo
        security.voice_data_repo = voice_repo

        # Test audit logging
        security._log_security_event(
            event_type="VOICE_INPUT",
            user_id="test_user",
            action="record_audio",
            resource="voice_session",
            result="success"
        )

        # Verify audit log was saved
        user_logs = audit_repo.find_by_user_id("test_user")
        assert len(user_logs) == 1
        assert user_logs[0].event_type == "VOICE_INPUT"

        # Test consent recording
        from database.models import ConsentRecord
        consent = ConsentRecord.create(
            user_id="test_user",
            consent_type="voice_recording",
            granted=True
        )
        consent_repo.save(consent)

        # Verify consent was saved
        user_consents = consent_repo.find_by_user_id("test_user")
        assert len(user_consents) == 1
        assert user_consents[0].consent_type == "voice_recording"

        # Test voice data storage
        from database.models import VoiceData
        voice_data = VoiceData.create(
            user_id="test_user",
            data_type="recording",
            encrypted_data=b"encrypted_test_data"
        )
        voice_repo.save(voice_data)

        # Verify voice data was saved
        user_data = voice_repo.find_by_user_id("test_user")
        assert len(user_data) == 1
        assert user_data[0].data_type == "recording"

    def test_transaction_integrity(self, db_manager):
        """Test transaction integrity across multiple operations."""
        user_repo = UserRepository()
        session_repo = SessionRepository()
        audit_repo = AuditLogRepository()
        user_repo.db = db_manager
        session_repo.db = db_manager
        audit_repo.db = db_manager

        # Test successful transaction
        operations = [
            (
                """INSERT INTO users (user_id, email, full_name, role, status, created_at, updated_at, password_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ('user1', 'user1@example.com', 'User One', 'patient', 'active',
                 '2025-01-01T00:00:00', '2025-01-01T00:00:00', 'hash1')
            ),
            (
                """INSERT INTO sessions (session_id, user_id, created_at, expires_at, is_active)
                   VALUES (?, ?, ?, ?, ?)""",
                ('session1', 'user1', '2025-01-01T00:00:00', '2025-01-01T01:00:00', 1)
            ),
            (
                """INSERT INTO audit_logs (log_id, timestamp, event_type, user_id, session_id, details, severity)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                ('log1', '2025-01-01T00:00:00', 'LOGIN', 'user1', 'session1', '{}', 'INFO')
            )
        ]

        success = db_manager.execute_in_transaction(operations)
        assert success

        # Verify all operations succeeded
        assert user_repo.find_by_id('user1') is not None
        assert session_repo.find_by_id('session1') is not None
        user_logs = audit_repo.find_by_user_id('user1')
        assert len(user_logs) == 1

    def test_data_retention_and_cleanup(self, db_manager):
        """Test data retention and cleanup functionality."""
        voice_repo = VoiceDataRepository()
        voice_repo.db = db_manager

        # Create voice data with different retention periods
        now = datetime.now()

        # Data that should be retained
        retained_data = VoiceData.create(
            user_id="test_user",
            data_type="recording",
            encrypted_data=b"retained_data",
            retention_days=30  # Expires in 30 days
        )
        voice_repo.save(retained_data)

        # Data that should be expired (set creation date in past)
        expired_data = VoiceData.create(
            user_id="test_user",
            data_type="recording",
            encrypted_data=b"expired_data",
            retention_days=1  # Should have expired
        )
        # Manually set creation date to past
        expired_data.created_at = now - timedelta(days=2)
        expired_data.retention_until = expired_data.created_at + timedelta(days=1)
        voice_repo.save(expired_data)

        # Run cleanup
        removed_count = db_manager.cleanup_expired_data()
        assert removed_count > 0  # Should have removed expired data

        # Verify retained data still exists
        retained = voice_repo.find_by_id(retained_data.data_id)
        assert retained is not None

        # Verify expired data was marked as deleted
        expired = voice_repo.find_by_id(expired_data.data_id)
        assert expired is None or expired.is_deleted

    def test_concurrent_access_simulation(self, db_manager):
        """Test concurrent database access simulation."""
        import threading
        import time

        user_repo = UserRepository()
        user_repo.db = db_manager

        results = []
        errors = []

        def create_user(user_id: str):
            """Worker function to create a user."""
            try:
                user = User.create(
                    email=f"user{user_id}@example.com",
                    full_name=f"User {user_id}",
                    role="patient",
                    password_hash=f"hash{user_id}"
                )
                success = user_repo.save(user)
                results.append(success)
            except Exception as e:
                errors.append(str(e))

        # Create multiple threads to simulate concurrent access
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_user, args=(str(i),))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify results
        assert len(results) == 5
        assert all(results)  # All operations should succeed
        assert len(errors) == 0  # No errors should occur

        # Verify all users were created
        all_users = user_repo.find_all()
        assert len(all_users) == 5

    def test_hipaa_compliance_features(self, db_manager):
        """Test HIPAA compliance features."""
        audit_repo = AuditLogRepository()
        voice_repo = VoiceDataRepository()
        audit_repo.db = db_manager
        voice_repo.db = db_manager

        # Test audit logging of PHI access
        phi_access_log = AuditLog.create(
            event_type="PHI_ACCESS",
            user_id="therapist_123",
            details={
                "action": "access",
                "resource": "medical_records",
                "purpose": "treatment",
                "patient_id": "patient_456"
            }
        )
        audit_repo.save(phi_access_log)

        # Verify audit log was created
        therapist_logs = audit_repo.find_by_user_id("therapist_123")
        assert len(therapist_logs) == 1
        assert therapist_logs[0].event_type == "PHI_ACCESS"

        # Test encrypted voice data storage
        encrypted_data = VoiceData.create(
            user_id="patient_456",
            data_type="therapy_session",
            encrypted_data=b"encrypted_session_data",
            metadata={
                "hipaa_compliant": True,
                "retention_years": 7,
                "data_classification": "PHI"
            }
        )
        voice_repo.save(encrypted_data)

        # Verify voice data was stored with metadata
        stored_data = voice_repo.find_by_id(encrypted_data.data_id)
        assert stored_data is not None
        assert stored_data.metadata["hipaa_compliant"] is True
        assert stored_data.metadata["data_classification"] == "PHI"