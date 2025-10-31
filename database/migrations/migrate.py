#!/usr/bin/env python3
"""
Database Migration Manager for AI Therapist.

Provides migration management, rollback capabilities, and schema versioning
for the SQLite database with proper error handling and logging.
"""

import os
import sys
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import json


class MigrationError(Exception):
    """Migration-related errors."""
    pass


class MigrationManager:
    """Manages database migrations with versioning and rollback support."""

    def __init__(self, db_path: str, migrations_dir: str = None):
        """Initialize migration manager."""
        self.db_path = db_path
        self.migrations_dir = Path(migrations_dir or os.path.join(os.path.dirname(__file__), "migrations"))
        self.logger = logging.getLogger(__name__)

        # Ensure migrations directory exists
        self.migrations_dir.mkdir(exist_ok=True)

        # Create migrations table if it doesn't exist
        self._initialize_migrations_table()

    def _initialize_migrations_table(self):
        """Initialize the migrations tracking table."""
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    rollback_sql TEXT
                )
            ''')
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Failed to initialize migrations table: {e}")
            raise MigrationError(f"Failed to initialize migrations table: {e}")

    def get_applied_migrations(self) -> List[Dict[str, str]]:
        """Get list of applied migrations."""
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.execute("SELECT * FROM schema_migrations ORDER BY version")
            migrations = []
            for row in cursor.fetchall():
                migrations.append({
                    'version': row[0],
                    'name': row[1],
                    'applied_at': row[2],
                    'checksum': row[3],
                    'rollback_sql': row[4]
                })
            conn.close()
            return migrations
        except Exception as e:
            self.logger.error(f"Failed to get applied migrations: {e}")
            return []

    def get_pending_migrations(self) -> List[Tuple[str, Path]]:
        """Get list of pending migrations."""
        applied_versions = {m['version'] for m in self.get_applied_migrations()}

        pending = []
        for migration_file in sorted(self.migrations_dir.glob("*.sql")):
            if migration_file.name.endswith('.sql'):
                version = migration_file.stem.split('_')[0]
                if version not in applied_versions:
                    pending.append((version, migration_file))

        return sorted(pending, key=lambda x: x[0])

    def calculate_checksum(self, sql_content: str) -> str:
        """Calculate checksum for migration content."""
        import hashlib
        return hashlib.sha256(sql_content.encode('utf-8')).hexdigest()

    def apply_migration(self, version: str, migration_file: Path) -> bool:
        """Apply a single migration."""
        try:
            # Read migration file
            with open(migration_file, 'r', encoding='utf-8') as f:
                sql_content = f.read()

            if not sql_content.strip():
                self.logger.warning(f"Migration file {migration_file} is empty")
                return False

            checksum = self.calculate_checksum(sql_content)

            # Apply migration
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("BEGIN")

            try:
                # Execute migration SQL
                conn.executescript(sql_content)

                # Record migration
                migration_name = migration_file.stem
                applied_at = datetime.now().isoformat()

                conn.execute('''
                    INSERT INTO schema_migrations (version, name, applied_at, checksum, rollback_sql)
                    VALUES (?, ?, ?, ?, ?)
                ''', (version, migration_name, applied_at, checksum, None))  # Rollback SQL can be added later

                conn.commit()
                self.logger.info(f"Applied migration {version}: {migration_name}")
                return True

            except Exception as e:
                conn.rollback()
                raise MigrationError(f"Failed to apply migration {version}: {e}")
            finally:
                conn.close()

        except Exception as e:
            self.logger.error(f"Error applying migration {version}: {e}")
            return False

    def rollback_migration(self, version: str) -> bool:
        """Rollback a specific migration."""
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("BEGIN")

            try:
                # Get rollback SQL for the migration
                cursor = conn.execute(
                    "SELECT rollback_sql FROM schema_migrations WHERE version = ?",
                    (version,)
                )
                row = cursor.fetchone()

                if not row or not row[0]:
                    raise MigrationError(f"No rollback SQL available for migration {version}")

                rollback_sql = row[0]

                # Execute rollback
                conn.executescript(rollback_sql)

                # Remove migration record
                conn.execute("DELETE FROM schema_migrations WHERE version = ?", (version,))

                conn.commit()
                self.logger.info(f"Rolled back migration {version}")
                return True

            except Exception as e:
                conn.rollback()
                raise MigrationError(f"Failed to rollback migration {version}: {e}")
            finally:
                conn.close()

        except Exception as e:
            self.logger.error(f"Error rolling back migration {version}: {e}")
            return False

    def migrate(self, target_version: Optional[str] = None) -> bool:
        """Apply all pending migrations or up to target version."""
        try:
            pending = self.get_pending_migrations()

            if not pending:
                self.logger.info("No pending migrations")
                return True

            applied_count = 0

            for version, migration_file in pending:
                if target_version and version > target_version:
                    break

                self.logger.info(f"Applying migration {version}...")
                if self.apply_migration(version, migration_file):
                    applied_count += 1
                else:
                    self.logger.error(f"Failed to apply migration {version}")
                    return False

            self.logger.info(f"Successfully applied {applied_count} migrations")
            return True

        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            return False

    def rollback(self, target_version: Optional[str] = None, steps: int = 1) -> bool:
        """Rollback migrations."""
        try:
            applied = self.get_applied_migrations()

            if not applied:
                self.logger.info("No migrations to rollback")
                return True

            # Determine which migrations to rollback
            if target_version:
                # Rollback to specific version
                to_rollback = []
                for migration in reversed(applied):
                    if migration['version'] > target_version:
                        to_rollback.append(migration['version'])
                    else:
                        break
            else:
                # Rollback specified number of steps
                to_rollback = [m['version'] for m in applied[-steps:]]

            if not to_rollback:
                self.logger.info("No migrations to rollback")
                return True

            rolled_back_count = 0

            for version in reversed(to_rollback):
                self.logger.info(f"Rolling back migration {version}...")
                if self.rollback_migration(version):
                    rolled_back_count += 1
                else:
                    self.logger.error(f"Failed to rollback migration {version}")
                    return False

            self.logger.info(f"Successfully rolled back {rolled_back_count} migrations")
            return True

        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return False

    def create_migration(self, name: str, sql_content: str = "") -> Path:
        """Create a new migration file."""
        # Generate version number
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = timestamp

        # Create filename
        safe_name = name.lower().replace(' ', '_').replace('-', '_')
        filename = f"{version}_{safe_name}.sql"
        migration_file = self.migrations_dir / filename

        # Create basic migration content
        if not sql_content:
            sql_content = f"""-- Migration: {name}
-- Version: {version}
-- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
-- Description: {name}

-- Add your migration SQL here

"""

        # Write migration file
        with open(migration_file, 'w', encoding='utf-8') as f:
            f.write(sql_content)

        self.logger.info(f"Created migration file: {migration_file}")
        return migration_file

    def get_migration_status(self) -> Dict[str, any]:
        """Get current migration status."""
        applied = self.get_applied_migrations()
        pending = self.get_pending_migrations()

        return {
            'applied_migrations': len(applied),
            'pending_migrations': len(pending),
            'current_version': applied[-1]['version'] if applied else None,
            'latest_applied': applied[-1] if applied else None,
            'next_pending': pending[0] if pending else None,
            'applied_list': applied,
            'pending_list': [v for v, _ in pending]
        }


def main():
    """Command-line interface for migration management."""
    import argparse

    parser = argparse.ArgumentParser(description="Database Migration Manager")
    parser.add_argument('--db-path', default='./data/ai_therapist.db',
                       help='Database file path')
    parser.add_argument('--migrations-dir', default='./database/migrations',
                       help='Migrations directory path')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Migrate command
    subparsers.add_parser('migrate', help='Apply all pending migrations')

    # Rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Rollback migrations')
    rollback_parser.add_argument('--version', help='Rollback to specific version')
    rollback_parser.add_argument('--steps', type=int, default=1,
                                help='Number of migrations to rollback')

    # Status command
    subparsers.add_parser('status', help='Show migration status')

    # Create command
    create_parser = subparsers.add_parser('create', help='Create new migration')
    create_parser.add_argument('name', help='Migration name')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        manager = MigrationManager(args.db_path, args.migrations_dir)

        if args.command == 'migrate':
            success = manager.migrate()
            sys.exit(0 if success else 1)

        elif args.command == 'rollback':
            success = manager.rollback(args.version, args.steps)
            sys.exit(0 if success else 1)

        elif args.command == 'status':
            status = manager.get_migration_status()
            print("Migration Status:")
            print(f"Applied migrations: {status['applied_migrations']}")
            print(f"Pending migrations: {status['pending_migrations']}")
            print(f"Current version: {status['current_version'] or 'None'}")

            if status['applied_list']:
                print("\nApplied migrations:")
                for m in status['applied_list']:
                    print(f"  {m['version']}: {m['name']} ({m['applied_at']})")

            if status['pending_list']:
                print("\nPending migrations:")
                for v in status['pending_list']:
                    print(f"  {v}")

            sys.exit(0)

        elif args.command == 'create':
            migration_file = manager.create_migration(args.name)
            print(f"Created migration: {migration_file}")
            sys.exit(0)

        else:
            parser.print_help()
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()