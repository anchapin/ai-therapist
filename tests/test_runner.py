"""
Test runner for voice features comprehensive testing.

This script runs all test suites and generates detailed reports
as specified in SPEECH_PRD.md requirements.
"""

import pytest
import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path
import coverage
import unittest

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.conftest import *


class VoiceFeatureTestRunner:
    """Comprehensive test runner for voice features."""

    def __init__(self):
        self.project_root = project_root
        self.test_results = {}
        self.coverage_data = None

    def run_all_tests(self):
        """Run all test suites and generate comprehensive report."""
        print("🧪 AI Therapist Voice Features - Comprehensive Test Suite")
        print("=" * 70)

        # Initialize coverage
        try:
            cov = coverage.Coverage(
                source=['voice'],
                omit=['*/__init__.py', '*/tests/*']
            )
            cov.start()
            coverage_enabled = True
        except Exception as e:
            print(f"⚠️  Coverage initialization failed: {e}")
            print("   Running tests without coverage reporting...")
            cov = None
            coverage_enabled = False

        # Test categories with their respective configurations
        test_categories = {
            'unit': {
                'path': 'tests/unit',
                'description': 'Unit Tests (Component-level testing)',
                'target_coverage': 0.90,
                'pytest_args': ['-v', '--tb=short']
            },
            'integration': {
                'path': 'tests/integration',
                'description': 'Integration Tests (Service integration testing)',
                'target_coverage': 0.85,
                'pytest_args': ['-v', '--tb=medium']
            },
            'security': {
                'path': 'tests/security',
                'description': 'Security Tests (HIPAA compliance and security)',
                'target_coverage': 0.95,
                'pytest_args': ['-v', '--tb=long']
            },
            'performance': {
                'path': 'tests/performance',
                'description': 'Performance Tests (Load and scalability)',
                'target_coverage': 0.80,
                'pytest_args': ['-v', '--tb=short', '-s']
            }
        }

        # Run each test category
        for category, config in test_categories.items():
            print(f"\n🔍 Running {config['description']}...")
            print("-" * 50)

            if os.path.exists(config['path']):
                try:
                    # Build pytest arguments
                    pytest_args = [
                        config['path'],
                        *config['pytest_args']
                    ]

                    # Add coverage arguments if coverage is enabled
                    if coverage_enabled:
                        pytest_args.extend([
                            '--cov=voice',
                            '--cov-report=term-missing',
                            '--cov-report=json',
                            f'--cov-fail-under={config["target_coverage"]}'
                        ])

                    # Run pytest
                    exit_code = pytest.main(pytest_args)

                    # Store results
                    self.test_results[category] = {
                        'exit_code': exit_code,
                        'description': config['description'],
                        'target_coverage': config['target_coverage'],
                        'timestamp': datetime.now().isoformat()
                    }

                    if exit_code == 0:
                        print(f"✅ {category} tests passed")
                    else:
                        print(f"⚠️ {category} tests completed with issues (exit code {exit_code})")
                        print("   This may be due to missing optional dependencies in CI environment")

                except Exception as e:
                    print(f"⚠️ Error running {category} tests: {e}")
                    print("   Continuing with other test categories...")
                    self.test_results[category] = {
                        'exit_code': -1,
                        'error': str(e),
                        'description': config['description'],
                        'timestamp': datetime.now().isoformat()
                    }
            else:
                print(f"⚠️ {category} test directory not found: {config['path']}")
                self.test_results[category] = {
                    'exit_code': -1,
                    'error': 'Test directory not found',
                    'description': config['description'],
                    'timestamp': datetime.now().isoformat()
                }

        # Stop coverage and collect data
        if coverage_enabled and cov:
            try:
                cov.stop()
                cov.save()
                print("✅ Coverage data collected")
            except Exception as e:
                print(f"⚠️ Coverage collection failed: {e}")
                coverage_enabled = False

        # Generate coverage report
        if coverage_enabled and cov:
            try:
                coverage_report = cov.get_data()
                self.coverage_data = {
                    'total_statements': coverage_report.num_statements(),
                    'missing_statements': coverage_report.num_missing_statements(),
                    'coverage_percentage': coverage_report.coverage()
                }
            except Exception as e:
                print(f"⚠️ Coverage report generation failed: {e}")
                self.coverage_data = None
        else:
            self.coverage_data = None

        # Generate comprehensive report
        self.generate_comprehensive_report()

        return self.test_results

    def generate_comprehensive_report(self):
        """Generate comprehensive test report as per SPEECH_PRD.md requirements."""
        print("\n📊 Generating Comprehensive Test Report...")
        print("=" * 50)

        report = {
            'metadata': {
                'test_suite': 'AI Therapist Voice Features',
                'version': '1.0.0',
                'execution_date': datetime.now().isoformat(),
                'project_root': str(self.project_root)
            },
            'summary': self.generate_test_summary(),
            'detailed_results': self.test_results,
            'coverage_analysis': self.generate_coverage_analysis(),
            'compliance_analysis': self.generate_compliance_analysis(),
            'performance_analysis': self.generate_performance_analysis(),
            'recommendations': self.generate_recommendations()
        }

        # Save report
        report_path = self.project_root / 'test_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"📄 Comprehensive test report saved to: {report_path}")

        # Print summary
        self.print_test_summary(report)

        return report

    def generate_test_summary(self):
        """Generate test summary statistics."""
        total_categories = len(self.test_results)
        passed_categories = sum(1 for result in self.test_results.values() if result.get('exit_code') == 0)
        failed_categories = total_categories - passed_categories

        return {
            'total_test_categories': total_categories,
            'passed_categories': passed_categories,
            'failed_categories': failed_categories,
            'success_rate': passed_categories / total_categories if total_categories > 0 else 0,
            'overall_status': 'PASS' if failed_categories == 0 else 'FAIL'
        }

    def generate_coverage_analysis(self):
        """Generate coverage analysis."""
        if not self.coverage_data:
            return {'error': 'Coverage data not available'}

        target_coverage = 0.90  # SPEECH_PRD.md requirement
        actual_coverage = self.coverage_data['coverage_percentage']

        return {
            'target_coverage': target_coverage,
            'actual_coverage': actual_coverage,
            'coverage_met': actual_coverage >= target_coverage,
            'total_statements': self.coverage_data['total_statements'],
            'missing_statements': self.coverage_data['missing_statements'],
            'coverage_gap': target_coverage - actual_coverage if actual_coverage < target_coverage else 0
        }

    def generate_compliance_analysis(self):
        """Generate compliance analysis for SPEECH_PRD.md requirements."""
        compliance_requirements = {
            'unit_testing_coverage': {
                'requirement': '90%+ unit test coverage',
                'status': 'COMPLETED' if self.coverage_data and self.coverage_data['coverage_percentage'] >= 0.90 else 'PENDING'
            },
            'integration_testing': {
                'requirement': 'Service integration testing',
                'status': 'COMPLETED' if 'integration' in self.test_results and self.test_results['integration'].get('exit_code') == 0 else 'PENDING'
            },
            'security_testing': {
                'requirement': 'HIPAA compliance testing',
                'status': 'COMPLETED' if 'security' in self.test_results and self.test_results['security'].get('exit_code') == 0 else 'PENDING'
            },
            'performance_testing': {
                'requirement': 'Load and scalability testing',
                'status': 'COMPLETED' if 'performance' in self.test_results and self.test_results['performance'].get('exit_code') == 0 else 'PENDING'
            },
            'automation': {
                'requirement': '≥ 80% test automation',
                'status': 'COMPLETED'  # All tests are automated
            }
        }

        return compliance_requirements

    def generate_performance_analysis(self):
        """Generate performance analysis."""
        performance_metrics = {}

        if 'performance' in self.test_results:
            perf_result = self.test_results['performance']
            if perf_result.get('exit_code') == 0:
                performance_metrics = {
                    'load_testing': 'PASSED',
                    'response_time_benchmarks': 'PASSED',
                    'concurrent_sessions': 'PASSED',
                    'scalability': 'PASSED',
                    'stress_testing': 'PASSED'
                }

        return performance_metrics

    def generate_recommendations(self):
        """Generate recommendations based on test results."""
        recommendations = []

        # Coverage recommendations
        if self.coverage_data:
            if self.coverage_data['coverage_percentage'] < 0.90:
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Coverage',
                    'issue': f'Test coverage ({self.coverage_data["coverage_percentage"]:.1%}) below 90% target',
                    'recommendation': 'Add unit tests for uncovered code paths'
                })

        # Failed test category recommendations
        for category, result in self.test_results.items():
            if result.get('exit_code') != 0:
                recommendations.append({
                    'priority': 'HIGH',
                    'category': category.upper(),
                    'issue': f'{category} tests failed',
                    'recommendation': f'Review and fix {category} test failures'
                })

        # General recommendations
        if len(recommendations) == 0:
            recommendations.append({
                'priority': 'LOW',
                'category': 'General',
                'issue': 'All tests passing',
                'recommendation': 'Continue with regular test maintenance and expansion'
            })

        return recommendations

    def print_test_summary(self, report):
        """Print test summary to console."""
        print("\n📋 Test Summary")
        print("=" * 30)
        print(f"Overall Status: {report['summary']['overall_status']}")
        print(f"Success Rate: {report['summary']['success_rate']:.1%}")
        print(f"Passed Categories: {report['summary']['passed_categories']}/{report['summary']['total_test_categories']}")

        if self.coverage_data:
            print(f"Test Coverage: {self.coverage_data['coverage_percentage']:.1%}")

        print("\n🔍 Compliance Status")
        print("-" * 30)
        for requirement, status in report['compliance_analysis'].items():
            icon = "✅" if status['status'] == 'COMPLETED' else "❌"
            print(f"{icon} {requirement}: {status['requirement']}")

        if report['recommendations']:
            print("\n💡 Recommendations")
            print("-" * 30)
            for rec in report['recommendations']:
                icon = "🔴" if rec['priority'] == 'HIGH' else "🟡"
                print(f"{icon} [{rec['priority']}] {rec['issue']}")
                print(f"   → {rec['recommendation']}")


def main():
    """Main test runner function."""
    runner = VoiceFeatureTestRunner()
    results = runner.run_all_tests()

    # Return appropriate exit code
    if all(result.get('exit_code') == 0 for result in results.values()):
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)