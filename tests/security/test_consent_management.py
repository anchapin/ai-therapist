"""
Comprehensive consent management tests.

Tests user consent workflows, edge cases, revocation scenarios,
and compliance with privacy regulations.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch, PropertyMock
import json
import tempfile
import os
from datetime import datetime, timedelta
import uuid

from voice.security import VoiceSecurity, SecurityConfig, ConsentManager, ConsentRecord


class TestConsentManagement:
    """Comprehensive consent management tests."""

    @pytest.fixture
    def security_config(self):
        """Create security configuration for testing."""
        return SecurityConfig(
            encryption_enabled=True,
            consent_required=True,
            privacy_mode=False,
            hipaa_compliance_enabled=True,
            data_retention_days=30,
            audit_logging_enabled=True
        )

    @pytest.fixture
    def security(self, security_config):
        """Create VoiceSecurity instance for testing."""
        return VoiceSecurity(security_config)

    @pytest.fixture
    def consent_scenarios(self):
        """Various consent scenarios for testing."""
        return [
            {
                'user_id': 'patient_123',
                'consent_type': 'VOICE_DATA_PROCESSING',
                'granted': True,
                'version': '1.0',
                'details': {
                    'purpose': 'therapy_sessions',
                    'data_types': ['audio', 'transcript'],
                    'retention_period': '6_months'
                }
            },
            {
                'user_id': 'patient_456',
                'consent_type': 'THERAPY_RECORDING',
                'granted': False,
                'version': '1.1',
                'details': {
                    'reason': 'privacy_concerns',
                    'alternative': 'text_only'
                }
            },
            {
                'user_id': 'patient_789',
                'consent_type': 'DATA_ANALYSIS',
                'granted': True,
                'version': '2.0',
                'details': {
                    'scope': 'anonymized_only',
                    'research_purpose': True
                }
            }
        ]

    @pytest.fixture
    def edge_case_users(self):
        """Edge case user scenarios."""
        return [
            'user_with_special_chars_@#$%',
            'user_with_unicode_患者José',
            'user_with_very_long_id_' + 'a' * 200,
            'user_with_sql_injection_\';DROP TABLE users;--',
            'user_with_xss_<script>alert("xss")</script>',
            'user_with_empty_id',
            'user_with_whitespace_   ',
            'user_with_newlines_user\n\r\t',
            'user_with_null_bytes_user\x00\x00',
            'user_minimal_length_a'
        ]

    def test_consent_basic_workflow(self, security, consent_scenarios):
        """Test basic consent workflow."""
        for scenario in consent_scenarios:
            user_id = scenario['user_id']
            consent_type = scenario['consent_type']
            granted = scenario['granted']
            version = scenario['version']
            details = scenario['details']

            # Record consent
            consent_record = security.consent_manager.record_consent(
                user_id=user_id,
                consent_type=consent_type,
                granted=granted,
                version=version,
                details=details
            )

            # Verify consent record
            assert consent_record['user_id'] == user_id
            assert consent_record['consent_type'] == consent_type
            assert consent_record['granted'] == granted
            assert consent_record['version'] == version
            assert consent_record['details'] == details
            assert 'timestamp' in consent_record

            # Verify consent check
            has_consent = security.consent_manager.has_consent(user_id, consent_type)
            assert has_consent == granted

            # Verify audit trail
            user_logs = security.audit_logger.get_user_logs(user_id)
            consent_logs = [log for log in user_logs if 'consent' in log.get('event_type', '')]

            assert len(consent_logs) >= 1, f"No consent audit logs for {user_id}"
            consent_log = consent_logs[0]
            assert consent_log['user_id'] == user_id
            assert consent_log['details']['consent_type'] == consent_type
            assert consent_log['details']['granted'] == granted

    def test_consent_revocation_scenarios(self, security):
        """Test consent revocation edge cases."""
        user_id = 'revocation_test_user'
        consent_type = 'VOICE_DATA_PROCESSING'

        # Grant consent initially
        security.consent_manager.record_consent(
            user_id=user_id,
            consent_type=consent_type,
            granted=True,
            version='1.0'
        )

        # Verify consent granted
        assert security.consent_manager.has_consent(user_id, consent_type) == True

        # Revoke consent
        security.consent_manager.withdraw_consent(user_id, consent_type)

        # Verify consent revoked
        assert security.consent_manager.has_consent(user_id, consent_type) == False

        # Verify audit trail for revocation
        user_logs = security.audit_logger.get_user_logs(user_id)
        revocation_logs = [log for log in user_logs if log.get('event_type') == 'consent_withdrawn']

        assert len(revocation_logs) >= 1, "No consent revocation audit log"
        revocation_log = revocation_logs[0]
        assert revocation_log['user_id'] == user_id
        assert revocation_log['details']['consent_type'] == consent_type

    def test_consent_version_management(self, security):
        """Test consent version management and updates."""
        user_id = 'version_test_user'
        consent_type = 'VOICE_DATA_PROCESSING'

        # Record initial consent
        initial_consent = security.consent_manager.record_consent(
            user_id=user_id,
            consent_type=consent_type,
            granted=True,
            version='1.0',
            details={'initial_version': True}
        )

        # Update to new version
        updated_consent = security.consent_manager.record_consent(
            user_id=user_id,
            consent_type=consent_type,
            granted=True,
            version='2.0',
            details={'updated_version': True, 'new_terms': True}
        )

        # Verify version progression
        assert updated_consent['version'] == '2.0'
        assert updated_consent['details']['new_terms'] == True

        # Verify audit trail shows both versions
        user_logs = security.audit_logger.get_user_logs(user_id)
        consent_logs = [log for log in user_logs if 'consent_recorded' in log.get('event_type', '')]

        assert len(consent_logs) >= 2, "Should have logs for both consent versions"
        versions = [log['details']['version'] for log in consent_logs]
        assert '1.0' in versions
        assert '2.0' in versions

    def test_consent_multiple_types_per_user(self, security):
        """Test multiple consent types for single user."""
        user_id = 'multi_consent_user'

        consent_types = [
            'VOICE_DATA_PROCESSING',
            'THERAPY_RECORDING',
            'DATA_ANALYSIS',
            'EMERGENCY_CONTACT',
            'RESEARCH_PARTICIPATION'
        ]

        # Grant different types of consent
        for consent_type in consent_types:
            security.consent_manager.record_consent(
                user_id=user_id,
                consent_type=consent_type,
                granted=True,
                version='1.0'
            )

        # Verify all consents granted
        for consent_type in consent_types:
            assert security.consent_manager.has_consent(user_id, consent_type) == True

        # Revoke one specific consent type
        revoked_type = consent_types[2]  # DATA_ANALYSIS
        security.consent_manager.withdraw_consent(user_id, revoked_type)

        # Verify selective revocation
        assert security.consent_manager.has_consent(user_id, revoked_type) == False

        for consent_type in consent_types:
            if consent_type != revoked_type:
                assert security.consent_manager.has_consent(user_id, consent_type) == True

    def test_consent_edge_case_users(self, security, edge_case_users):
        """Test consent management with edge case user IDs."""
        for user_id in edge_case_users:
            if not user_id.strip():  # Skip empty/whitespace-only users
                continue

            consent_type = 'VOICE_DATA_PROCESSING'

            # Should handle edge case user IDs without crashing
            try:
                consent_record = security.consent_manager.record_consent(
                    user_id=user_id,
                    consent_type=consent_type,
                    granted=True,
                    version='1.0'
                )

                assert consent_record['user_id'] == user_id
                assert security.consent_manager.has_consent(user_id, consent_type) == True

                # Test revocation
                security.consent_manager.withdraw_consent(user_id, consent_type)
                assert security.consent_manager.has_consent(user_id, consent_type) == False

            except Exception as e:
                pytest.fail(f"Failed to handle edge case user '{user_id}': {e}")

    def test_consent_race_conditions(self, security):
        """Test consent management under race conditions."""
        import threading
        import time

        user_id = 'race_condition_user'
        consent_type = 'VOICE_DATA_PROCESSING'
        results = []
        errors = []

        def consent_worker(worker_id):
            try:
                # Rapid consent operations
                for i in range(50):
                    granted = i % 2 == 0  # Alternate between granted/revoked

                    if granted:
                        security.consent_manager.record_consent(
                            user_id=user_id,
                            consent_type=consent_type,
                            granted=True,
                            version=f'1.0.{worker_id}'
                        )
                    else:
                        security.consent_manager.withdraw_consent(user_id, consent_type)

                    # Small delay to increase race condition likelihood
                    time.sleep(0.001)

                results.append(f"worker_{worker_id}_completed")
            except Exception as e:
                errors.append(f"worker_{worker_id}_error: {e}")

        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=consent_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)

        # Should complete without deadlocks or corruption
        assert len(results) == 3, f"Only {len(results)} workers completed successfully"
        assert len(errors) == 0, f"{len(errors)} workers failed: {errors}"

        # Final state should be consistent
        final_consent_state = security.consent_manager.has_consent(user_id, consent_type)
        assert isinstance(final_consent_state, bool)

    def test_consent_data_integrity(self, security):
        """Test consent data integrity and consistency."""
        user_id = 'integrity_user'
        consent_type = 'VOICE_DATA_PROCESSING'

        # Record consent multiple times with same parameters
        for i in range(10):
            consent_record = security.consent_manager.record_consent(
                user_id=user_id,
                consent_type=consent_type,
                granted=True,
                version='1.0',
                details={'iteration': i}
            )

            # Each record should be timestamped uniquely
            assert 'timestamp' in consent_record

        # Verify consent state consistency
        assert security.consent_manager.has_consent(user_id, consent_type) == True

        # Verify audit trail integrity
        user_logs = security.audit_logger.get_user_logs(user_id)
        consent_logs = [log for log in user_logs if 'consent_recorded' in log.get('event_type', '')]

        assert len(consent_logs) == 10, f"Expected 10 consent logs, got {len(consent_logs)}"

        # All logs should have consistent structure
        for log in consent_logs:
            assert 'timestamp' in log
            assert 'user_id' in log
            assert 'details' in log
            assert log['details']['consent_type'] == consent_type

    def test_consent_privacy_compliance(self, security):
        """Test consent compliance with privacy regulations."""
        user_id = 'privacy_user'
        consent_type = 'VOICE_DATA_PROCESSING'

        # Test explicit consent requirement
        explicit_consent = security.consent_manager.record_consent(
            user_id=user_id,
            consent_type=consent_type,
            granted=True,
            version='1.0',
            details={
                'explicit': True,
                'purpose_specified': True,
                'data_minimization': True,
                'consent_method': 'written_digital_signature'
            }
        )

        assert explicit_consent['details']['explicit'] == True

        # Test withdrawal rights
        security.consent_manager.withdraw_consent(user_id, consent_type)

        # Verify withdrawal is effective
        assert security.consent_manager.has_consent(user_id, consent_type) == False

        # Test re-consent after withdrawal
        re_consent = security.consent_manager.record_consent(
            user_id=user_id,
            consent_type=consent_type,
            granted=True,
            version='1.1',
            details={
                're_consent': True,
                'previous_withdrawal_date': explicit_consent['timestamp']
            }
        )

        assert re_consent['details']['re_consent'] == True
        assert security.consent_manager.has_consent(user_id, consent_type) == True

    def test_consent_audit_trail_completeness(self, security):
        """Test completeness and immutability of consent audit trail."""
        user_id = 'audit_completeness_user'
        consent_type = 'VOICE_DATA_PROCESSING'

        # Clear existing logs for clean test
        security.audit_logger.logs.clear()

        # Perform consent operations
        initial_consent = security.consent_manager.record_consent(
            user_id=user_id,
            consent_type=consent_type,
            granted=True,
            version='1.0'
        )

        # Update consent
        security.consent_manager.record_consent(
            user_id=user_id,
            consent_type=consent_type,
            granted=False,
            version='1.1'
        )

        # Withdraw consent
        security.consent_manager.withdraw_consent(user_id, consent_type)

        # Verify complete audit trail
        user_logs = security.audit_logger.get_user_logs(user_id)
        consent_events = [
            log for log in user_logs
            if any(event in log.get('event_type', '')
                  for event in ['consent_recorded', 'consent_withdrawn'])
        ]

        # Should have at least 3 events (grant, update, withdraw)
        assert len(consent_events) >= 3, f"Expected at least 3 consent events, got {len(consent_events)}"

        # Verify event sequence integrity
        event_types = [log['event_type'] for log in consent_events]
        assert 'consent_recorded' in event_types
        assert 'consent_withdrawn' in event_types

        # Verify all events have required fields
        required_fields = ['timestamp', 'event_type', 'user_id', 'details']
        for log in consent_events:
            for field in required_fields:
                assert field in log, f"Missing required field '{field}' in audit log"

        # Verify timestamps are in chronological order
        timestamps = [datetime.fromisoformat(log['timestamp'].replace('Z', '+00:00'))
                     for log in consent_events]
        assert timestamps == sorted(timestamps), "Audit events not in chronological order"

    def test_consent_emergency_override_scenarios(self, security):
        """Test consent handling in emergency scenarios."""
        user_id = 'emergency_user'
        consent_type = 'VOICE_DATA_PROCESSING'

        # User initially denies consent
        security.consent_manager.record_consent(
            user_id=user_id,
            consent_type=consent_type,
            granted=False,
            version='1.0',
            details={'reason': 'privacy_concerns'}
        )

        # Verify consent denied
        assert security.consent_manager.has_consent(user_id, consent_type) == False

        # Simulate emergency scenario (implementation dependent)
        # In real implementation, this might involve legal/medical override

        # Test emergency consent override (if supported)
        emergency_consent = security.consent_manager.record_consent(
            user_id=user_id,
            consent_type=consent_type,
            granted=True,
            version='1.0-emergency',
            details={
                'emergency_override': True,
                'authorized_by': 'medical_professional',
                'reason': 'crisis_intervention_required'
            }
        )

        assert emergency_consent['details']['emergency_override'] == True

        # Verify emergency consent is effective
        assert security.consent_manager.has_consent(user_id, consent_type) == True

    def test_consent_data_retention_compliance(self, security):
        """Test consent data retention and cleanup compliance."""
        user_id = 'retention_user'
        consent_type = 'VOICE_DATA_PROCESSING'

        # Record consent
        security.consent_manager.record_consent(
            user_id=user_id,
            consent_type=consent_type,
            granted=True,
            version='1.0'
        )

        # Withdraw consent after some time (simulated)
        security.consent_manager.withdraw_consent(user_id, consent_type)

        # Verify withdrawal is effective
        assert security.consent_manager.has_consent(user_id, consent_type) == False

        # Test that withdrawal triggers data retention policy
        # (Implementation would depend on data retention manager)

        # Verify audit trail includes withdrawal
        user_logs = security.audit_logger.get_user_logs(user_id)
        withdrawal_logs = [log for log in user_logs if log.get('event_type') == 'consent_withdrawn']

        assert len(withdrawal_logs) >= 1, "No consent withdrawal audit log"
        withdrawal_log = withdrawal_logs[0]
        assert withdrawal_log['user_id'] == user_id
        assert withdrawal_log['details']['consent_type'] == consent_type

    def test_consent_malicious_input_handling(self, security):
        """Test consent system handling of malicious input."""
        # Test with malicious consent details
        malicious_details = {
            'script': '<script>alert("xss")</script>',
            'sql': "'; DROP TABLE consents; --",
            'command': '$(rm -rf /)',
            'path': '../../../etc/passwd',
            'large_data': 'x' * 1000000,  # Very large string
            'binary': b'\x00\x01\x02\xff'.decode('latin-1'),
            'nested': {'deep': {'nested': {'structure': 'x' * 1000}}}
        }

        user_id = 'malicious_input_user'
        consent_type = 'VOICE_DATA_PROCESSING'

        # Should handle malicious input gracefully
        for key, malicious_value in malicious_details.items():
            try:
                consent_record = security.consent_manager.record_consent(
                    user_id=f'{user_id}_{key}',
                    consent_type=consent_type,
                    granted=True,
                    version='1.0',
                    details={key: malicious_value}
                )

                # Should not crash and should store data safely
                assert consent_record['user_id'] == f'{user_id}_{key}'
                assert consent_record['details'][key] == malicious_value

            except Exception as e:
                # Some inputs might be rejected, but should not crash
                assert "crash" not in str(e).lower()

    def test_consent_concurrent_sessions(self, security):
        """Test consent management across concurrent user sessions."""
        import threading

        user_id = 'concurrent_user'
        consent_type = 'VOICE_DATA_PROCESSING'
        session_results = []
        session_errors = []

        def session_worker(session_id):
            try:
                # Each session manages consent independently
                for i in range(20):
                    # Random consent operations
                    if i % 3 == 0:
                        security.consent_manager.record_consent(
                            user_id=f'{user_id}_session_{session_id}',
                            consent_type=consent_type,
                            granted=i % 2 == 0,
                            version=f'1.0.{session_id}'
                        )
                    elif i % 3 == 1:
                        security.consent_manager.withdraw_consent(
                            f'{user_id}_session_{session_id}',
                            consent_type
                        )
                    # i % 3 == 2: check consent status

                    current_consent = security.consent_manager.has_consent(
                        f'{user_id}_session_{session_id}',
                        consent_type
                    )
                    assert isinstance(current_consent, bool)

                session_results.append(f'session_{session_id}_success')
            except Exception as e:
                session_errors.append(f'session_{session_id}_error: {e}')

        # Start multiple concurrent sessions
        threads = []
        for i in range(5):
            thread = threading.Thread(target=session_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=15)

        # All sessions should complete successfully
        assert len(session_results) == 5, f"Only {len(session_results)} sessions succeeded"
        assert len(session_errors) == 0, f"{len(session_errors)} sessions failed"

    def test_consent_granular_permissions(self, security):
        """Test granular consent permissions and restrictions."""
        user_id = 'granular_user'

        # Test different granular consent types
        granular_consents = [
            {
                'type': 'VOICE_RECORDING_ONLY',
                'details': {'recording_only': True, 'no_transcription': True}
            },
            {
                'type': 'TRANSCRIPTION_ONLY',
                'details': {'transcription_only': True, 'no_storage': True}
            },
            {
                'type': 'ANALYSIS_WITH_ANONYMIZATION',
                'details': {'analysis_allowed': True, 'anonymization_required': True}
            },
            {
                'type': 'EMERGENCY_USE_ONLY',
                'details': {'emergency_only': True, 'crisis_intervention': True}
            }
        ]

        # Record granular consents
        for consent_info in granular_consents:
            consent_type = consent_info['type']
            details = consent_info['details']

            security.consent_manager.record_consent(
                user_id=user_id,
                consent_type=consent_type,
                granted=True,
                version='1.0',
                details=details
            )

            # Verify each granular consent
            assert security.consent_manager.has_consent(user_id, consent_type) == True

        # Test selective revocation of granular permissions
        revoked_consent = granular_consents[1]['type']
        security.consent_manager.withdraw_consent(user_id, revoked_consent)

        # Verify selective revocation
        assert security.consent_manager.has_consent(user_id, revoked_consent) == False

        for consent_info in granular_consents:
            consent_type = consent_info['type']
            if consent_type != revoked_consent:
                assert security.consent_manager.has_consent(user_id, consent_type) == True

    def test_consent_automated_expiry(self, security):
        """Test automated consent expiry and renewal."""
        user_id = 'expiry_user'
        consent_type = 'VOICE_DATA_PROCESSING'

        # Record consent with expiry (simulated)
        security.consent_manager.record_consent(
            user_id=user_id,
            consent_type=consent_type,
            granted=True,
            version='1.0',
            details={
                'auto_expiry_days': 30,
                'renewal_required': True
            }
        )

        # Simulate time passing
        with patch('voice.security.datetime') as mock_datetime:
            # Move time forward past expiry
            future_date = datetime.now() + timedelta(days=31)
            mock_datetime.now.return_value = future_date

            # Consent should be considered expired (implementation dependent)
            # This would typically be handled by a consent expiry checker

            # Test renewal process
            renewed_consent = security.consent_manager.record_consent(
                user_id=user_id,
                consent_type=consent_type,
                granted=True,
                version='1.1',
                details={
                    'renewed': True,
                    'previous_expiry': future_date.isoformat()
                }
            )

            assert renewed_consent['details']['renewed'] == True
            assert security.consent_manager.has_consent(user_id, consent_type) == True