#!/usr/bin/env python3
"""
Dependency Validation Script for AI Therapist

This script systematically validates all required dependencies and identifies
missing packages, version conflicts, and configuration issues.
"""

import sys
import json
import importlib
import subprocess
from pathlib import Path
from datetime import datetime
import pkg_resources

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class DependencyValidator:
    """Comprehensive dependency validator."""

    def __init__(self):
        self.project_root = project_root
        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'python_version': sys.version,
                'platform': sys.platform
            },
            'dependencies': {},
            'issues': [],
            'recommendations': [],
            'summary': {}
        }

        # Required packages based on requirements.txt
        self.required_packages = {
            # Core ML/LLM
            'langchain': '0.1.0',
            'langchain-community': '0.0.10',
            'langchain-text-splitters': '0.0.1',
            'langchain-openai': '0.0.5',
            'langchain-ollama': '0.1.0',
            'openai': '1.0.0',
            'faiss-cpu': '1.7.4',
            'pypdf': '3.17.0',
            'tiktoken': '0.5.0',
            'python-dotenv': '1.0.0',
            'streamlit': '1.28.0',
            'requests': '2.28.0',

            # Audio Processing
            'pyaudio': '0.2.11',
            'librosa': '0.10.1',
            'soundfile': '0.12.1',
            'noisereduce': '2.0.0',
            'numpy': '1.21.0',
            'scipy': '1.7.0',

            # Speech Services
            'google-cloud-speech': '2.19.0',
            'elevenlabs': '0.2.28',
            'openai-whisper': '20231117',
            'piper-tts': '1.2.0',

            # Voice Analysis
            'webrtcvad': '2.0.10',
            'silero-vad': '0.3.0',

            # Async Support
            'aiofiles': '23.2.1',
            'aiohttp': '3.9.0',

            # Configuration
            'pydantic': '2.5.0',
            'pyyaml': '6.0.1',
            'jsonschema': '4.20.0',
            'python-multipart': '0.0.6',

            # Audio Formats
            'pydub': '0.25.1',
            'ffmpeg-python': '0.2.0',

            # Text Processing
            'phonenumbers': '8.13.0',
            'num2words': '0.5.12',

            # Security
            'cryptography': '41.0.0',
            'bcrypt': '4.1.0',

            # Testing
            'pytest': '8.4.0',
            'pytest-cov': '7.0.0',
            'pytest-asyncio': '1.2.0',
            'psutil': '5.9.0',

            # TTS
            'sounddevice': '0.4.6'
        }

    def validate_imports(self):
        """Validate that all required packages can be imported."""
        print("🔍 Validating package imports...")

        for package, min_version in self.required_packages.items():
            try:
                # Handle special cases
                import_name = package
                if package == 'python-dotenv':
                    import_name = 'dotenv'
                elif package == 'openai-whisper':
                    import_name = 'whisper'
                elif package == 'piper-tts':
                    import_name = 'piper_tts'
                elif package == 'sounddevice':
                    import_name = 'sounddevice'

                module = importlib.import_module(import_name)

                # Get version
                version = getattr(module, '__version__', None)
                if not version and hasattr(module, '__version_info__'):
                    version = '.'.join(map(str, module.__version_info__))
                elif not version:
                    # Try pkg_resources
                    try:
                        version = pkg_resources.get_distribution(package).version
                    except:
                        version = "unknown"

                self.results['dependencies'][package] = {
                    'status': 'SUCCESS',
                    'version': version,
                    'minimum_version': min_version,
                    'import_name': import_name
                }

                print(f"  ✅ {package} ({version})")

            except ImportError as e:
                self.results['dependencies'][package] = {
                    'status': 'FAILED',
                    'error': str(e),
                    'minimum_version': min_version,
                    'import_name': import_name if 'import_name' in locals() else package
                }

                self.results['issues'].append({
                    'type': 'IMPORT_ERROR',
                    'package': package,
                    'error': str(e),
                    'severity': 'HIGH'
                })

                print(f"  ❌ {package} - {e}")

            except Exception as e:
                self.results['dependencies'][package] = {
                    'status': 'ERROR',
                    'error': str(e),
                    'minimum_version': min_version,
                    'import_name': import_name if 'import_name' in locals() else package
                }

                self.results['issues'].append({
                    'type': 'UNEXPECTED_ERROR',
                    'package': package,
                    'error': str(e),
                    'severity': 'MEDIUM'
                })

                print(f"  ⚠️ {package} - {e}")

    def validate_system_dependencies(self):
        """Validate system-level dependencies."""
        print("\n🔍 Validating system dependencies...")

        system_deps = {
            'ffmpeg': ['ffmpeg', '-version'],
            'portaudio': ['dpkg', '-l', 'libportaudio2'],
            'alsa': ['aplay', '--version']
        }

        for dep, command in system_deps.items():
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    self.results['dependencies'][f'system:{dep}'] = {
                        'status': 'SUCCESS',
                        'version': result.stdout.split('\n')[0] if result.stdout else 'unknown'
                    }
                    print(f"  ✅ {dep}")
                else:
                    self.results['dependencies'][f'system:{dep}'] = {
                        'status': 'FAILED',
                        'error': result.stderr
                    }
                    self.results['issues'].append({
                        'type': 'SYSTEM_DEPENDENCY_ERROR',
                        'dependency': dep,
                        'error': result.stderr,
                        'severity': 'HIGH'
                    })
                    print(f"  ❌ {dep} - {result.stderr}")

            except subprocess.TimeoutExpired:
                self.results['dependencies'][f'system:{dep}'] = {
                    'status': 'ERROR',
                    'error': 'Timeout'
                }
                print(f"  ⏰ {dep} - Timeout")

            except FileNotFoundError:
                self.results['dependencies'][f'system:{dep}'] = {
                    'status': 'FAILED',
                    'error': 'Command not found'
                }
                self.results['issues'].append({
                    'type': 'SYSTEM_DEPENDENCY_MISSING',
                    'dependency': dep,
                    'error': 'Command not found',
                    'severity': 'HIGH'
                })
                print(f"  ❌ {dep} - Command not found")

    def validate_project_structure(self):
        """Validate project structure and configuration files."""
        print("\n🔍 Validating project structure...")

        required_files = [
            'requirements.txt',
            'pytest.ini',
            '.env',
            'app.py',
            'voice/__init__.py',
            'tests/__init__.py'
        ]

        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                self.results['dependencies'][f'file:{file_path}'] = {
                    'status': 'SUCCESS',
                    'path': str(full_path)
                }
                print(f"  ✅ {file_path}")
            else:
                self.results['dependencies'][f'file:{file_path}'] = {
                    'status': 'MISSING',
                    'error': 'File not found'
                }
                self.results['issues'].append({
                    'type': 'MISSING_FILE',
                    'file': file_path,
                    'severity': 'HIGH' if file_path.endswith('.py') else 'MEDIUM'
                })
                print(f"  ❌ {file_path} - Missing")

    def generate_recommendations(self):
        """Generate recommendations based on validation results."""
        print("\n💡 Generating recommendations...")

        # Group issues by type
        import_errors = [i for i in self.results['issues'] if i['type'] == 'IMPORT_ERROR']
        system_errors = [i for i in self.results['issues'] if i['type'].startswith('SYSTEM')]
        file_errors = [i for i in self.results['issues'] if i['type'] == 'MISSING_FILE']

        if import_errors:
            self.results['recommendations'].append({
                'priority': 'HIGH',
                'category': 'DEPENDENCIES',
                'issue': f'{len(import_errors)} packages failed to import',
                'action': 'pip install --upgrade -r requirements.txt',
                'packages': [e['package'] for e in import_errors]
            })

        if system_errors:
            self.results['recommendations'].append({
                'priority': 'HIGH',
                'category': 'SYSTEM_DEPENDENCIES',
                'issue': f'{len(system_errors)} system dependencies missing',
                'action': 'Install missing system packages (see Dockerfile)',
                'dependencies': [e['dependency'] for e in system_errors]
            })

        if file_errors:
            self.results['recommendations'].append({
                'priority': 'MEDIUM',
                'category': 'PROJECT_STRUCTURE',
                'issue': f'{len(file_errors)} files missing',
                'action': 'Create missing project files',
                'files': [e['file'] for e in file_errors]
            })

        # Environment setup recommendations
        if not (self.project_root / '.env').exists():
            self.results['recommendations'].append({
                'priority': 'HIGH',
                'category': 'ENVIRONMENT',
                'issue': '.env file missing',
                'action': 'cp template.env .env and configure'
            })

    def generate_summary(self):
        """Generate validation summary."""
        total_deps = len(self.required_packages)
        successful_imports = len([d for d in self.results['dependencies'].values()
                                 if d.get('status') == 'SUCCESS'])

        self.results['summary'] = {
            'total_dependencies': total_deps,
            'successful_imports': successful_imports,
            'failed_imports': total_deps - successful_imports,
            'success_rate': successful_imports / total_deps if total_deps > 0 else 0,
            'total_issues': len(self.results['issues']),
            'critical_issues': len([i for i in self.results['issues'] if i['severity'] == 'HIGH']),
            'recommendations_count': len(self.results['recommendations'])
        }

    def save_report(self):
        """Save validation report to file."""
        report_path = self.project_root / 'reports' / 'dependency-report.json'
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n📄 Report saved to: {report_path}")
        return report_path

    def run_validation(self):
        """Run complete validation process."""
        print("🚀 Starting Dependency Validation for AI Therapist")
        print("=" * 60)

        self.validate_imports()
        self.validate_system_dependencies()
        self.validate_project_structure()
        self.generate_recommendations()
        self.generate_summary()

        # Print summary
        print("\n📊 Validation Summary")
        print("-" * 30)
        print(f"Total Dependencies: {self.results['summary']['total_dependencies']}")
        print(f"Successful Imports: {self.results['summary']['successful_imports']}")
        print(f"Failed Imports: {self.results['summary']['failed_imports']}")
        print(f"Success Rate: {self.results['summary']['success_rate']:.1%}")
        print(f"Total Issues: {self.results['summary']['total_issues']}")
        print(f"Critical Issues: {self.results['summary']['critical_issues']}")

        if self.results['recommendations']:
            print(f"\n💡 Recommendations ({len(self.results['recommendations'])}):")
            for rec in self.results['recommendations']:
                icon = "🔴" if rec['priority'] == 'HIGH' else "🟡"
                print(f"  {icon} [{rec['priority']}] {rec['issue']}")
                print(f"     → {rec['action']}")

        report_path = self.save_report()

        # Return exit code based on critical issues
        critical_issues = self.results['summary']['critical_issues']
        if critical_issues > 0:
            print(f"\n❌ Validation failed with {critical_issues} critical issues")
            return 1
        else:
            print("\n✅ Validation completed successfully")
            return 0

def main():
    """Main validation function."""
    validator = DependencyValidator()
    exit_code = validator.run_validation()
    return exit_code

if __name__ == "__main__":
    sys.exit(main())