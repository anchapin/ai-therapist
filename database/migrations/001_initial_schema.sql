-- AI Therapist Database Initial Schema
-- Version: 001
-- Date: 2025-10-03
-- Description: Initial database schema for AI Therapist with HIPAA compliance

-- Users table for authentication and user management
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    full_name TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('patient', 'therapist', 'admin', 'guest')),
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'suspended', 'pending_verification', 'locked')),
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,  -- ISO format datetime
    updated_at TEXT NOT NULL,  -- ISO format datetime
    last_login TEXT,           -- ISO format datetime
    login_attempts INTEGER DEFAULT 0,
    account_locked_until TEXT, -- ISO format datetime
    password_reset_token TEXT,
    password_reset_expires TEXT, -- ISO format datetime
    preferences TEXT DEFAULT '{}',  -- JSON string
    medical_info TEXT DEFAULT '{}'   -- JSON string (HIPAA protected)
);

-- Sessions table for session management
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at TEXT NOT NULL,  -- ISO format datetime
    expires_at TEXT NOT NULL,  -- ISO format datetime
    ip_address TEXT,
    user_agent TEXT,
    is_active BOOLEAN DEFAULT 1,
    FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
);

-- Voice data table for encrypted voice recordings and metadata
CREATE TABLE IF NOT EXISTS voice_data (
    data_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT,
    data_type TEXT NOT NULL CHECK (data_type IN ('recording', 'transcription', 'analysis')),
    encrypted_data BLOB NOT NULL,
    metadata TEXT DEFAULT '{}',  -- JSON string with HIPAA-compliant metadata
    created_at TEXT NOT NULL,     -- ISO format datetime
    retention_until TEXT,         -- ISO format datetime
    is_deleted BOOLEAN DEFAULT 0,
    FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE SET NULL
);

-- Audit logs table for HIPAA compliance tracking
CREATE TABLE IF NOT EXISTS audit_logs (
    log_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,     -- ISO format datetime
    event_type TEXT NOT NULL,
    user_id TEXT,
    session_id TEXT,
    details TEXT DEFAULT '{}',   -- JSON string
    severity TEXT DEFAULT 'INFO' CHECK (severity IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE SET NULL,
    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE SET NULL
);

-- Consent records table for patient consent management
CREATE TABLE IF NOT EXISTS consent_records (
    consent_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    consent_type TEXT NOT NULL,
    granted BOOLEAN NOT NULL,
    timestamp TEXT NOT NULL,     -- ISO format datetime
    version TEXT NOT NULL DEFAULT '1.0',
    details TEXT DEFAULT '{}',   -- JSON string
    revoked_at TEXT,             -- ISO format datetime
    FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE,
    UNIQUE(user_id, consent_type, version)
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_status ON users(status);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_active ON sessions(is_active);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_voice_data_user_id ON voice_data(user_id);
CREATE INDEX IF NOT EXISTS idx_voice_data_session_id ON voice_data(session_id);
CREATE INDEX IF NOT EXISTS idx_voice_data_created_at ON voice_data(created_at);
CREATE INDEX IF NOT EXISTS idx_voice_data_retention ON voice_data(retention_until);
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_event_type ON audit_logs(event_type);
CREATE INDEX IF NOT EXISTS idx_consent_records_user_id ON consent_records(user_id);
CREATE INDEX IF NOT EXISTS idx_consent_records_type ON consent_records(consent_type);

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- Enable WAL mode for better concurrency
PRAGMA journal_mode = WAL;

-- Set synchronous mode for data safety
PRAGMA synchronous = NORMAL;

-- Enable auto-vacuum for space management
PRAGMA auto_vacuum = INCREMENTAL;