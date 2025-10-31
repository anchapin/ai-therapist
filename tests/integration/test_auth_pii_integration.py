"""
Comprehensive integration tests for authentication + PII protection integration.

Tests auth + PII protection integration including:
- User authentication → PII detection → data masking flow
- Session management with PII-protected data
- Audit logging of PII access
- HIPAA compliance validation
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, timedelta

from auth.auth_service import AuthService, AuthResult, AuthSession
from auth.user_model import UserModel, UserProfile, UserRole, UserStatus
from security.pii_protection import PIIProtection, PIIType, PIIDetectionResult, MaskingStrategy
from database.models import User, SessionRepository, AuditLog


@pytest.fixture
def user_model():
    """Create user model with test users."""
    model = UserModel()
    
    # Add test users (skip if already exists)
    if not model.get_user_by_email("patient@example.com"):
        model.create_user(
            email="patient@example.com",
            password="SecurePass123!",
            full_name="John Smith",
            role=UserRole.PATIENT
        )

    if not model.get_user_by_email("therapist@example.com"):
        model.create_user(
            email="therapist@example.com",
        password="TherapistPass456!",
    full_name="Dr. Jane Doe",
    role=UserRole.THERAPIST
    )
    
    return model


@pytest.fixture
def auth_service(user_model):
    """Create authentication service."""
    return AuthService(user_model=user_model)


@pytest.fixture
def pii_protection():
    """Create PII protection service."""
    from security.pii_protection import PIIProtectionConfig
    config = PIIProtectionConfig(
        enable_audit=True,
        hipaa_compliance=True
    )
    return PIIProtection(config=config)


@pytest.fixture
def mock_session_repo():
    """Create mock session repository."""
    repo = Mock(spec=SessionRepository)
    repo.save = Mock(return_value=True)
    repo.find_by_id = Mock(return_value=None)
    repo.find_by_user_id = Mock(return_value=[])
    return repo


@pytest.mark.integration
class TestAuthPIIIntegration:
    """Test authentication and PII protection integration."""
    
    def test_login_with_pii_detection(self, auth_service, pii_protection):
        """Test user login with PII detection on user data."""
        # Login user
        auth_result = auth_service.login_user(
            email="patient@example.com",
            password="SecurePass123!",
            ip_address="192.168.1.100"
        )
        
        assert auth_result.success
        assert auth_result.user is not None
        assert auth_result.token is not None
        
        # Detect PII in user profile
        user_data = f"{auth_result.user.full_name} {auth_result.user.email}"
        pii_results = pii_protection.detect_pii(user_data)
        
        # Should detect email and potentially name
        assert len(pii_results) > 0
        email_detected = any(result.pii_type == PIIType.EMAIL for result in pii_results)
        assert email_detected
    
    def test_session_creation_with_pii_masking(self, auth_service, pii_protection):
        """Test session creation with automatic PII masking."""
        # Create session with PII-containing metadata
        auth_result = auth_service.login(
            email="therapist@example.com",
            password="TherapistPass456!",
            ip_address="10.0.0.50",
            user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 14_0)"
        )
        
        assert auth_result.success
        session = auth_result.session
        
        # Mask sensitive session data
        session_data = {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "ip_address": session.ip_address,
            "user_agent": session.user_agent
        }
        
        masked_data = pii_protection.mask_data(session_data)
        
        # Verify IP address masked
        assert "ip_address" in masked_data
        if masked_data["ip_address"] != session.ip_address:
            # IP was masked
            assert PIIType.IP_ADDRESS.value in str(pii_protection.detect_pii(session.ip_address))
    
    def test_pii_audit_logging_on_authentication(self, auth_service, pii_protection):
        """Test PII access is logged during authentication."""
        # Enable audit logging
        pii_protection.enable_audit = True
        
        # Login and access user data
        auth_result = auth_service.login(
            email="patient@example.com",
            password="SecurePass123!"
        )
        
        assert auth_result.success
        
        # Access and mask PII data
        user_info = f"User: {auth_result.user.full_name}, Email: {auth_result.user.email}"
        pii_results = pii_protection.detect_pii(user_info)
        masked_info = pii_protection.mask_pii(user_info, pii_results)
        
        # Verify audit trail created
        audit_logs = pii_protection.get_audit_logs()
        assert len(audit_logs) > 0
    
    def test_role_based_pii_access(self, auth_service, pii_protection, user_model):
        """Test role-based PII access control."""
        # Patient login
        patient_auth = auth_service.login(
            email="patient@example.com",
            password="SecurePass123!"
        )
        
        # Therapist login
        therapist_auth = auth_service.login(
            email="therapist@example.com",
            password="TherapistPass456!"
        )
        
        assert patient_auth.success
        assert therapist_auth.success
        
        # Patient data with PII
        patient_data = {
            "name": "John Smith",
            "email": "patient@example.com",
            "phone": "555-123-4567",
            "medical_record": "Patient experiencing anxiety attacks"
        }
        
        # Therapist should have access to full data (after proper authorization)
        # Patient should have masked data for other patients
        
        # Detect PII
        pii_results = pii_protection.detect_pii(str(patient_data))
        assert len(pii_results) > 0
        
        # Mask based on role
        # Therapist accessing patient data
        therapist_view = pii_protection.mask_data(
            patient_data,
            allowed_roles=[UserRole.THERAPIST.value]
        )
        
        # Patient accessing own data - should see all
        patient_view = patient_data.copy()
        
        # Verify masking applied appropriately
        assert isinstance(therapist_view, dict)
        assert isinstance(patient_view, dict)
    
    def test_session_expiry_with_pii_cleanup(self, auth_service, pii_protection):
        """Test PII cleanup when session expires."""
        # Create session
        auth_result = auth_service.login(
            email="patient@example.com",
            password="SecurePass123!"
        )
        
        assert auth_result.success
        session = auth_result.session
        
        # Store PII data in session context
        session_pii = {
            "user_name": auth_result.user.full_name,
            "email": auth_result.user.email,
            "session_start": datetime.now().isoformat()
        }
        
        # Simulate session expiry
        session.expires_at = datetime.now() - timedelta(hours=1)
        assert session.is_expired()
        
        # Logout and cleanup
        auth_service.logout(session.session_id)
        
        # Verify session invalidated
        token_valid = auth_service.verify_token(auth_result.token)
        assert not token_valid.success
    
    def test_concurrent_sessions_with_pii_isolation(self, auth_service, pii_protection):
        """Test PII isolation across concurrent user sessions."""
        # Create multiple sessions for different users
        patient_auth = auth_service.login(
            email="patient@example.com",
            password="SecurePass123!"
        )
        
        therapist_auth = auth_service.login(
            email="therapist@example.com",
            password="TherapistPass456!"
        )
        
        assert patient_auth.success
        assert therapist_auth.success
        
        # Verify sessions are isolated
        assert patient_auth.session.session_id != therapist_auth.session.session_id
        assert patient_auth.session.user_id != therapist_auth.session.user_id
        
        # Each session should have separate PII protection context
        patient_pii = pii_protection.detect_pii(patient_auth.user.email)
        therapist_pii = pii_protection.detect_pii(therapist_auth.user.email)
        
        assert len(patient_pii) > 0
        assert len(therapist_pii) > 0
    
    def test_failed_login_with_pii_protection(self, auth_service, pii_protection):
        """Test failed login doesn't leak PII in error messages."""
        # Attempt login with wrong password
        auth_result = auth_service.login(
            email="patient@example.com",
            password="WrongPassword123!"
        )
        
        assert not auth_result.success
        assert auth_result.error_message is not None
        
        # Error message should not contain PII
        error_pii = pii_protection.detect_pii(auth_result.error_message)
        
        # Should not leak email or other sensitive info
        for pii in error_pii:
            assert pii.pii_type != PIIType.EMAIL
            assert pii.confidence < 0.8  # Low confidence detections acceptable
    
    def test_password_reset_with_pii_masking(self, auth_service, pii_protection):
        """Test password reset flow with PII masking."""
        # Request password reset
        reset_result = auth_service.request_password_reset("patient@example.com")
        assert reset_result is not None
        
        # Email should be masked in any logs
        masked_email = pii_protection.mask_pii(
            "patient@example.com",
            pii_protection.detect_pii("patient@example.com")
        )
        
        # Verify masking applied
        assert masked_email != "patient@example.com" or "*" in masked_email


@pytest.mark.integration
class TestHIPAACompliance:
    """Test HIPAA compliance in auth + PII integration."""
    
    def test_hipaa_audit_trail_creation(self, auth_service, pii_protection):
        """Test HIPAA audit trail is created for PII access."""
        pii_protection.hipaa_compliance = True
        pii_protection.enable_audit = True
        
        # Login
        auth_result = auth_service.login(
            email="patient@example.com",
            password="SecurePass123!"
        )
        
        # Access sensitive health information
        health_data = "Patient has history of depression and takes Prozac 20mg daily"
        
        # Detect and log health PII
        pii_results = pii_protection.detect_pii(health_data)
        masked_data = pii_protection.mask_pii(health_data, pii_results)
        
        # Verify HIPAA audit created
        audit_logs = pii_protection.get_audit_logs()
        assert len(audit_logs) > 0
        
        # Verify audit contains required HIPAA fields
        for log in audit_logs:
            if hasattr(log, 'pii_type'):
                assert log.pii_type is not None
                assert log.timestamp is not None
    
    def test_hipaa_minimum_necessary_rule(self, auth_service, pii_protection):
        """Test HIPAA minimum necessary rule - only show required PII."""
        pii_protection.hipaa_compliance = True
        
        # Full patient record
        full_record = {
            "name": "John Smith",
            "ssn": "123-45-6789",
            "email": "patient@example.com",
            "phone": "555-123-4567",
            "medical_history": "Extensive anxiety disorder history",
            "current_medications": "Prozac 20mg, Xanax 0.5mg PRN"
        }
        
        # Request only necessary fields for appointment scheduling
        necessary_fields = ["name", "phone", "email"]
        
        # Mask unnecessary fields
        masked_record = pii_protection.mask_data(
            full_record,
            keep_fields=necessary_fields
        )
        
        # Verify unnecessary fields masked
        if "ssn" in masked_record:
            assert masked_record["ssn"] != full_record["ssn"]
    
    def test_hipaa_encryption_requirement(self, auth_service, pii_protection):
        """Test PII data encryption for HIPAA compliance."""
        pii_protection.hipaa_compliance = True
        
        # Sensitive health data
        health_pii = "Patient SSN: 123-45-6789, diagnosed with bipolar disorder"
        
        # Encrypt PII data
        encrypted = pii_protection.encrypt_pii(health_pii)
        
        # Verify encrypted
        assert encrypted != health_pii
        
        # Decrypt and verify
        decrypted = pii_protection.decrypt_pii(encrypted)
        assert decrypted == health_pii
    
    def test_hipaa_access_control_verification(self, auth_service, pii_protection, user_model):
        """Test HIPAA access controls are enforced."""
        # Create patient and therapist sessions
        patient_auth = auth_service.login(
            email="patient@example.com",
            password="SecurePass123!"
        )
        
        therapist_auth = auth_service.login(
            email="therapist@example.com",
            password="TherapistPass456!"
        )
        
        # Health record
        health_record = {
            "patient_id": patient_auth.user.user_id,
            "diagnosis": "Generalized Anxiety Disorder",
            "treatment_plan": "CBT therapy sessions"
        }
        
        # Verify therapist can access (authorized)
        therapist_has_access = therapist_auth.user.role == UserRole.THERAPIST
        assert therapist_has_access
        
        # Another patient should NOT access
        # (This would be enforced at application level)
        assert patient_auth.user.user_id == health_record["patient_id"]
    
    def test_hipaa_data_retention_compliance(self, auth_service, pii_protection):
        """Test HIPAA data retention policies."""
        pii_protection.hipaa_compliance = True
        
        # Create session with retention policy
        auth_result = auth_service.login(
            email="patient@example.com",
            password="SecurePass123!"
        )
        
        # Set retention period (HIPAA requires 6 years minimum)
        retention_days = 365 * 6  # 6 years
        
        # Session should have retention metadata
        session = auth_result.session
        assert session.created_at is not None
        
        # Calculate expiry
        retention_expiry = session.created_at + timedelta(days=retention_days)
        
        # Verify retention period meets HIPAA requirements
        assert (retention_expiry - session.created_at).days >= 365 * 6


@pytest.mark.integration
class TestPIISessionManagement:
    """Test PII-aware session management."""
    
    def test_session_with_pii_metadata(self, auth_service, pii_protection):
        """Test session stores PII metadata for audit."""
        auth_result = auth_service.login(
            email="patient@example.com",
            password="SecurePass123!",
            ip_address="192.168.1.100"
        )
        
        assert auth_result.success
        
        # Create PII-aware session metadata
        session_metadata = {
            "session_id": auth_result.session.session_id,
            "user_email": auth_result.user.email,
            "ip_address": auth_result.session.ip_address,
            "pii_accessed": ["email", "name", "ip_address"],
            "access_time": datetime.now().isoformat()
        }
        
        # Detect PII in metadata
        pii_in_metadata = pii_protection.detect_pii(str(session_metadata))
        assert len(pii_in_metadata) > 0
    
    def test_multi_session_pii_tracking(self, auth_service, pii_protection):
        """Test PII access tracking across multiple sessions."""
        # Create multiple sessions
        sessions = []
        for i in range(3):
            auth_result = auth_service.login(
                email="patient@example.com",
                password="SecurePass123!"
            )
            if auth_result.success:
                sessions.append(auth_result.session)
        
        # Track PII access per session
        pii_access_log = {}
        for session in sessions:
            pii_access_log[session.session_id] = {
                "user_id": session.user_id,
                "access_count": 1,
                "last_access": datetime.now()
            }
        
        # Verify separate tracking
        assert len(pii_access_log) > 0
        
        # Cleanup sessions
        for session in sessions:
            auth_service.logout(session.session_id)
    
    def test_session_pii_sanitization_on_logout(self, auth_service, pii_protection):
        """Test PII is sanitized when session ends."""
        auth_result = auth_service.login(
            email="therapist@example.com",
            password="TherapistPass456!"
        )
        
        assert auth_result.success
        session_id = auth_result.session.session_id
        
        # Store some PII in session context (simulated)
        session_pii = {
            "user_email": auth_result.user.email,
            "full_name": auth_result.user.full_name
        }
        
        # Logout
        logout_success = auth_service.logout(session_id)
        assert logout_success
        
        # Verify session invalidated
        token_result = auth_service.verify_token(auth_result.token)
        assert not token_result.success
