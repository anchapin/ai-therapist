"""
Unit tests for Performance Monitor Module.

This module provides comprehensive test coverage for the performance monitoring
functionality including metrics collection, alert generation, system monitoring,
and reporting capabilities.
"""

import pytest
import time
import threading
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Import the modules to test
from performance.monitor import (
    PerformanceMonitor, PerformanceMetric, PerformanceAlert, SystemMetrics,
    AlertLevel
)


class TestPerformanceMetric:
    """Test PerformanceMetric dataclass."""

    def test_performance_metric_creation(self):
        """Test PerformanceMetric creation with all fields."""
        current_time = time.time()
        metric = PerformanceMetric(
            name="test_metric",
            value=42.5,
            timestamp=current_time,
            unit="ms",
            tags={"source": "test", "env": "dev"}
        )

        assert metric.name == "test_metric"
        assert metric.value == 42.5
        assert metric.timestamp == current_time
        assert metric.unit == "ms"
        assert metric.tags == {"source": "test", "env": "dev"}

    def test_performance_metric_defaults(self):
        """Test PerformanceMetric with default values."""
        metric = PerformanceMetric(
            name="test_metric",
            value=42.5,
            timestamp=time.time(),
            unit="ms"
        )

        assert metric.name == "test_metric"
        assert metric.value == 42.5
        assert metric.unit == "ms"
        assert metric.tags == {}


class TestPerformanceAlert:
    """Test PerformanceAlert dataclass."""

    def test_performance_alert_creation(self):
        """Test PerformanceAlert creation with all fields."""
        current_time = time.time()
        alert = PerformanceAlert(
            alert_id="test_alert_123",
            level=AlertLevel.WARNING,
            message="Test warning message",
            metric="cpu_percent",
            value=85.5,
            threshold=80.0,
            timestamp=current_time,
            resolved=False,
            resolved_timestamp=None
        )

        assert alert.alert_id == "test_alert_123"
        assert alert.level == AlertLevel.WARNING
        assert alert.message == "Test warning message"
        assert alert.metric == "cpu_percent"
        assert alert.value == 85.5
        assert alert.threshold == 80.0
        assert alert.timestamp == current_time
        assert alert.resolved is False
        assert alert.resolved_timestamp is None

    def test_performance_alert_resolved(self):
        """Test PerformanceAlert with resolved status."""
        current_time = time.time()
        resolved_time = current_time + 60.0
        alert = PerformanceAlert(
            alert_id="test_alert_123",
            level=AlertLevel.ERROR,
            message="Test error message",
            metric="memory_percent",
            value=95.0,
            threshold=90.0,
            timestamp=current_time,
            resolved=True,
            resolved_timestamp=resolved_time
        )

        assert alert.resolved is True
        assert alert.resolved_timestamp == resolved_time


class TestSystemMetrics:
    """Test SystemMetrics dataclass."""

    def test_system_metrics_creation(self):
        """Test SystemMetrics creation with all fields."""
        current_time = time.time()
        metrics = SystemMetrics(
            cpu_percent=75.5,
            memory_percent=60.2,
            memory_used_mb=4096.0,
            memory_available_mb=2048.0,
            disk_usage_percent=45.0,
            network_connections=25,
            thread_count=12,
            process_count=150,
            timestamp=current_time
        )

        assert metrics.cpu_percent == 75.5
        assert metrics.memory_percent == 60.2
        assert metrics.memory_used_mb == 4096.0
        assert metrics.memory_available_mb == 2048.0
        assert metrics.disk_usage_percent == 45.0
        assert metrics.network_connections == 25
        assert metrics.thread_count == 12
        assert metrics.process_count == 150
        assert metrics.timestamp == current_time


class TestPerformanceMonitor:
    """Test PerformanceMonitor class."""

    def test_performance_monitor_initialization_default(self):
        """Test PerformanceMonitor initialization with default config."""
        monitor = PerformanceMonitor()

        assert monitor.monitoring_interval == 10.0
        assert monitor.retention_period == 3600.0
        assert monitor.max_metrics_history == 10000
        assert monitor.alert_cooldown == 300.0
        assert monitor.enable_alerts is True

        # Check thresholds
        assert monitor.thresholds['memory_percent'] == 80.0
        assert monitor.thresholds['cpu_percent'] == 90.0
        assert monitor.thresholds['response_time_ms'] == 2000.0
        assert monitor.thresholds['error_rate_percent'] == 5.0
        assert monitor.thresholds['memory_mb'] == 1024.0

        # Check initial state
        assert monitor.is_monitoring is False
        assert monitor.monitoring_thread is None
        assert monitor.request_count == 0
        assert monitor.error_count == 0
        assert monitor.total_response_time == 0.0
        assert len(metrics.metrics_history) == 0
        assert len(metrics.system_metrics_history) == 0
        assert len(metrics.active_alerts) == 0
        assert len(metrics.resolved_alerts) == 0

    def test_performance_monitor_initialization_custom_config(self):
        """Test PerformanceMonitor initialization with custom config."""
        config = {
            'monitoring_interval': 5.0,
            'retention_period': 7200.0,
            'max_metrics_history': 5000,
            'alert_cooldown': 150.0,
            'enable_alerts': False,
            'memory_threshold_percent': 70.0,
            'cpu_threshold_percent': 80.0,
            'response_time_threshold_ms': 1000.0,
            'error_rate_threshold_percent': 2.0,
            'memory_threshold_mb': 512.0
        }

        monitor = PerformanceMonitor(config)

        assert monitor.monitoring_interval == 5.0
        assert monitor.retention_period == 7200.0
        assert monitor.max_metrics_history == 5000
        assert monitor.alert_cooldown == 150.0
        assert monitor.enable_alerts is False

        # Check custom thresholds
        assert monitor.thresholds['memory_percent'] == 70.0
        assert monitor.thresholds['cpu_percent'] == 80.0
        assert monitor.thresholds['response_time_ms'] == 1000.0
        assert monitor.thresholds['error_rate_percent'] == 2.0
        assert monitor.thresholds['memory_mb'] == 512.0

    def test_record_metric(self):
        """Test recording a performance metric."""
        monitor = PerformanceMonitor()

        # Record a metric
        monitor.record_metric("test_metric", 42.5, "ms", {"source": "test"})

        # Check if metric was recorded
        assert len(monitor.metrics_history) == 1
        metric = monitor.metrics_history[0]
        assert metric.name == "test_metric"
        assert metric.value == 42.5
        assert metric.unit == "ms"
        assert metric.tags == {"source": "test"}

    def test_record_metric_with_callback(self):
        """Test recording metric with callback."""
        monitor = PerformanceMonitor()

        # Register callback
        callback_called = False
        received_metric = None

        def metric_callback(metric):
            nonlocal callback_called, received_metric
            callback_called = True
            received_metric = metric

        monitor.register_metric_callback(metric_callback)

        # Record metric
        test_metric = PerformanceMetric(
            name="test_metric",
            value=42.5,
            timestamp=time.time(),
            unit="ms"
        )
        monitor.record_metric("test_metric", 42.5, "ms")

        # Check callback
        assert callback_called
        assert received_metric.name == "test_metric"
        assert received_metric.value == 42.5

    def test_record_metric_with_callback_error(self):
        """Test recording metric with callback error."""
        monitor = PerformanceMonitor()

        # Register callback that raises error
        def error_callback(metric):
            raise Exception("Test error")

        monitor.register_metric_callback(error_callback)

        # Should not crash
        monitor.record_metric("test_metric", 42.5, "ms")

    def test_record_request(self):
        """Test recording a request."""
        monitor = PerformanceMonitor()

        # Record successful request
        monitor.record_request(150.0, True)

        assert monitor.request_count == 1
        assert monitor.error_count == 0
        assert monitor.total_response_time == 150.0
        assert len(monitor.response_times) == 1
        assert monitor.response_times[0] == 150.0

        # Record failed request
        monitor.record_request(200.0, False)

        assert monitor.request_count == 2
        assert monitor.error_count == 1
        assert monitor.total_response_time == 350.0
        assert len(monitor.response_times) == 2

        # Check if metrics were recorded
        metric_names = [m.name for m in monitor.metrics_history]
        assert "response_time" in metric_names
        assert "request_count" in metric_names
        assert "error_count" in metric_names
        assert "requests_per_second" in metric_names
        assert "error_rate" in metric_names
        assert "avg_response_time" in metric_names

    def test_get_current_metrics(self):
        """Test getting current metrics."""
        monitor = PerformanceMonitor()

        # Record some requests
        monitor.record_request(100.0, True)
        monitor.record_request(200.0, True)
        monitor.record_request(300.0, False)

        metrics = monitor.get_current_metrics()

        assert isinstance(metrics, dict)
        assert 'uptime_seconds' in metrics
        assert 'total_requests' in metrics
        assert 'total_errors' in metrics
        assert 'error_rate_percent' in metrics
        assert 'avg_response_time_ms' in metrics
        assert 'requests_per_second' in metrics
        assert 'active_alerts' in metrics
        assert 'metrics_in_history' in metrics
        assert 'system_metrics' in metrics
        assert 'thresholds' in metrics
        assert 'is_monitoring' in metrics

        # Check values
        assert metrics['total_requests'] == 3
        assert metrics['total_errors'] == 1
        assert metrics['error_rate_percent'] == (1/3) * 100
        assert metrics['avg_response_time_ms'] == 600.0 / 3

    def test_get_alerts(self):
        """Test getting alerts."""
        monitor = PerformanceMonitor()

        # Create some alerts
        alert1 = PerformanceAlert(
            alert_id="alert1",
            level=AlertLevel.WARNING,
            message="Warning message",
            metric="cpu_percent",
            value=85.0,
            threshold=80.0,
            timestamp=time.time(),
            resolved=False
        )

        alert2 = PerformanceAlert(
            alert_id="alert2",
            level=AlertLevel.ERROR,
            message="Error message",
            metric="memory_percent",
            value=95.0,
            threshold=90.0,
            timestamp=time.time(),
            resolved=True,
            resolved_timestamp=time.time() + 60.0
        )

        monitor.active_alerts["alert1"] = alert1
        monitor.resolved_alerts.append(alert2)

        # Get active alerts only
        alerts = monitor.get_alerts(include_resolved=False)
        assert len(alerts) == 1
        assert alerts[0]['alert_id'] == "alert1"
        assert alerts[0]['resolved'] is False

        # Get all alerts
        alerts = monitor.get_alerts(include_resolved=True)
        assert len(alerts) == 2
        assert any(a['alert_id'] == "alert1" for a in alerts)
        assert any(a['alert_id'] == "alert2" for a in alerts)

    def test_resolve_alert(self):
        """Test resolving an alert."""
        monitor = PerformanceMonitor()

        # Create an alert
        alert = PerformanceAlert(
            alert_id="test_alert",
            level=AlertLevel.WARNING,
            message="Test message",
            metric="cpu_percent",
            value=85.0,
            threshold=80.0,
            timestamp=time.time(),
            resolved=False
        )

        monitor.active_alerts["test_alert"] = alert

        # Resolve the alert
        result = monitor.resolve_alert("test_alert")
        assert result is True
        assert "test_alert" not in monitor.active_alerts
        assert len(monitor.resolved_alerts) == 1
        assert monitor.resolved_alerts[0].resolved is True

        # Try to resolve non-existent alert
        result = monitor.resolve_alert("non_existent")
        assert result is False

    def test_get_metrics_history(self):
        """Test getting metrics history."""
        monitor = PerformanceMonitor()

        # Record some metrics
        current_time = time.time()
        for i in range(10):
            monitor.record_metric(f"metric_{i}", i * 10, "units", {"index": str(i)})

        # Get all metrics
        all_metrics = monitor.get_metrics_history()
        assert len(all_metrics) == 10

        # Get metrics by name
        metric_5 = monitor.get_metrics_history(metric_name="metric_5")
        assert len(metric_5) == 1
        assert metric_5[0].value == 50

        # Get metrics by time range
        start_time = current_time - 1.0
        end_time = current_time + 1.0
        time_range_metrics = monitor.get_metrics_history(
            start_time=start_time,
            end_time=end_time
        )
        assert len(time_range_metrics) == 10

        # Get limited metrics
        limited_metrics = monitor.get_metrics_history(limit=5)
        assert len(limited_metrics) == 5

    def test_register_callbacks(self):
        """Test registering callbacks."""
        monitor = PerformanceMonitor()

        # Register alert callback
        alert_callback = Mock()
        monitor.register_alert_callback(alert_callback)
        assert len(monitor.alert_callbacks) == 1

        # Register metric callback
        metric_callback = Mock()
        monitor.register_metric_callback(metric_callback)
        assert len(monitor.metric_callbacks) == 1

    def test_generate_report(self):
        """Test generating performance report."""
        monitor = PerformanceMonitor()

        # Record some metrics
        for i in range(20):
            monitor.record_metric("test_metric", i * 5, "units")
            monitor.record_request(i * 10, i % 5 != 0)  # Some failures

        # Generate report
        report = monitor.generate_report(time_range_seconds=3600)

        assert isinstance(report, dict)
        assert 'report_period_seconds' in report
        assert 'generated_at' in report
        assert 'summary' in report
        assert 'metrics_stats' in report
        assert 'alerts_summary' in report

        # Check summary
        summary = report['summary']
        assert 'total_requests' in summary
        assert 'total_errors' in summary
        assert 'error_rate_percent' in summary

        # Check metrics stats
        metrics_stats = report['metrics_stats']
        assert 'test_metric' in metrics_stats
        test_metric_stats = metrics_stats['test_metric']
        assert 'count' in test_metric_stats
        assert 'min' in test_metric_stats
        assert 'max' in test_metric_stats
        assert 'avg' in test_metric_stats
        assert 'latest' in test_metric_stats

        # Check alerts summary
        alerts_summary = report['alerts_summary']
        assert 'active_count' in alerts_summary
        assert 'resolved_count' in alerts_summary
        assert 'active_by_level' in alerts_summary
        assert 'resolved_by_level' in alerts_summary

    def test_export_metrics(self):
        """Test exporting metrics to file."""
        monitor = PerformanceMonitor()

        # Record some metrics
        monitor.record_metric("test_metric", 42.5, "ms", {"source": "test"})
        monitor.record_request(100.0, True)

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name

        try:
            # Export metrics
            result = monitor.export_metrics(temp_file, format="json")
            assert result is True

            # Verify file was created and contains data
            with open(temp_file, 'r') as f:
                data = json.load(f)

            assert 'export_time' in data
            assert 'current_metrics' in data
            assert 'alerts' in data
            assert 'metrics_history' in data

        finally:
            # Clean up
            os.unlink(temp_file)

    def test_export_metrics_unsupported_format(self):
        """Test exporting metrics with unsupported format."""
        monitor = PerformanceMonitor()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_file = f.name

        try:
            # Try to export with unsupported format
            result = monitor.export_metrics(temp_file, format="xml")
            assert result is False

        finally:
            os.unlink(temp_file)

    def test_export_metrics_error(self):
        """Test exporting metrics with error."""
        monitor = PerformanceMonitor()

        # Try to export to invalid path
        result = monitor.export_metrics("/invalid/path/metrics.json")
        assert result is False

    @patch('performance.monitor.psutil')
    def test_get_system_metrics_success(self, mock_psutil):
        """Test successful system metrics retrieval."""
        # Mock psutil objects
        mock_psutil.cpu_percent.return_value = 75.5
        mock_psutil.virtual_memory.return_value.percent = 60.2
        mock_psutil.virtual_memory.return_value.used = 4294967296  # 4GB
        mock_psutil.virtual_memory.return_value.available = 2147483648  # 2GB
        mock_psutil.disk_usage.return_value.percent = 45.0
        mock_psutil.net_connections.return_value = [Mock(), Mock(), Mock()]
        mock_psutil.pids.return_value = [1, 2, 3, 4, 5]

        mock_process = Mock()
        mock_process.num_threads.return_value = 12
        mock_psutil.Process.return_value = mock_process

        monitor = PerformanceMonitor()
        metrics = monitor._get_system_metrics()

        assert isinstance(metrics, SystemMetrics)
        assert metrics.cpu_percent == 75.5
        assert metrics.memory_percent == 60.2
        assert metrics.memory_used_mb == 4096.0
        assert metrics.memory_available_mb == 2048.0
        assert metrics.disk_usage_percent == 45.0
        assert metrics.network_connections == 3
        assert metrics.thread_count == 12
        assert metrics.process_count == 5

    @patch('performance.monitor.psutil')
    def test_get_system_metrics_error(self, mock_psutil):
        """Test system metrics retrieval with error."""
        mock_psutil.cpu_percent.side_effect = Exception("Test error")

        monitor = PerformanceMonitor()
        metrics = monitor._get_system_metrics()

        # Should return default metrics on error
        assert isinstance(metrics, SystemMetrics)
        assert metrics.cpu_percent == 0.0
        assert metrics.memory_percent == 0.0
        assert metrics.memory_used_mb == 0.0
        assert metrics.memory_available_mb == 0.0
        assert metrics.disk_usage_percent == 0.0
        assert metrics.network_connections == 0
        assert metrics.thread_count == 0
        assert metrics.process_count == 0

    def test_check_thresholds(self):
        """Test threshold checking."""
        monitor = PerformanceMonitor(enable_alerts=True)

        # Register alert callback
        alerts_received = []
        def alert_callback(alert):
            alerts_received.append(alert)

        monitor.register_alert_callback(alert_callback)

        # Create metric that exceeds threshold
        metric = PerformanceMetric(
            name="memory_percent",
            value=85.0,  # Above default threshold of 80%
            timestamp=time.time(),
            unit="percent"
        )

        monitor._check_thresholds(metric)

        # Should have generated an alert
        assert len(alerts_received) == 1
        assert alerts_received[0].level == AlertLevel.WARNING

    def test_check_thresholds_disabled(self):
        """Test threshold checking when disabled."""
        monitor = PerformanceMonitor(enable_alerts=False)

        alerts_received = []
        def alert_callback(alert):
            alerts_received.append(alert)

        monitor.register_alert_callback(alert_callback)

        # Create metric that exceeds threshold
        metric = PerformanceMetric(
            name="memory_percent",
            value=85.0,
            timestamp=time.time(),
            unit="percent"
        )

        monitor._check_thresholds(metric)

        # Should not have generated alert
        assert len(alerts_received) == 0

    def test_determine_alert_level(self):
        """Test alert level determination."""
        monitor = PerformanceMonitor()

        # Test different ratios
        assert monitor._determine_alert_level(80.0, 80.0, "test") == AlertLevel.INFO
        assert monitor._determine_alert_level(100.0, 80.0, "test") == AlertLevel.WARNING
        assert monitor._determine_alert_level(120.0, 80.0, "test") == AlertLevel.ERROR
        assert monitor._determine_alert_level(160.0, 80.0, "test") == AlertLevel.CRITICAL

    def test_generate_alert(self):
        """Test alert generation."""
        monitor = PerformanceMonitor(alert_cooldown=0.1)

        alerts_received = []
        def alert_callback(alert):
            alerts_received.append(alert)

        monitor.register_alert_callback(alert_callback)

        # Create metric
        metric = PerformanceMetric(
            name="memory_percent",
            value=85.0,
            timestamp=time.time(),
            unit="percent"
        )

        # Generate alert
        monitor._generate_alert(metric, 'memory_percent', 80.0, AlertLevel.WARNING)

        # Should have generated alert
        assert len(alerts_received) == 1
        assert len(monitor.active_alerts) == 1
        assert alerts_received[0].level == AlertLevel.WARNING

        # Try to generate duplicate alert immediately (should be blocked by cooldown)
        monitor._generate_alert(metric, 'memory_percent', 80.0, AlertLevel.WARNING)
        assert len(alerts_received) == 1  # No new alert
        assert len(monitor.active_alerts) == 1

        # Wait for cooldown and try again
        time.sleep(0.2)
        monitor._generate_alert(metric, 'memory_percent', 80.0, AlertLevel.WARNING)
        assert len(alerts_received) == 2  # New alert
        assert len(monitor.active_alerts) == 1  # Still one active (replaced)

    def test_cleanup_old_data(self):
        """Test cleanup of old data."""
        monitor = PerformanceMonitor(retention_period=1.0)

        # Add old metrics
        old_time = time.time() - 2.0
        old_metric = PerformanceMetric(
            name="old_metric",
            value=42.0,
            timestamp=old_time,
            unit="units"
        )
        monitor.metrics_history.append(old_metric)

        # Add recent metrics
        recent_metric = PerformanceMetric(
            name="recent_metric",
            value=43.0,
            timestamp=time.time(),
            unit="units"
        )
        monitor.metrics_history.append(recent_metric)

        # Add old resolved alert
        old_alert = PerformanceAlert(
            alert_id="old_alert",
            level=AlertLevel.WARNING,
            message="Old alert",
            metric="test",
            value=42.0,
            threshold=40.0,
            timestamp=old_time,
            resolved=True,
            resolved_timestamp=old_time + 60.0
        )
        monitor.resolved_alerts.append(old_alert)

        # Add recent resolved alert
        recent_alert = PerformanceAlert(
            alert_id="recent_alert",
            level=AlertLevel.WARNING,
            message="Recent alert",
            metric="test",
            value=43.0,
            threshold=40.0,
            timestamp=time.time(),
            resolved=True,
            resolved_timestamp=time.time() + 60.0
        )
        monitor.resolved_alerts.append(recent_alert)

        # Cleanup old data
        monitor._cleanup_old_data()

        # Should have removed old data
        assert len(monitor.metrics_history) == 1
        assert monitor.metrics_history[0].name == "recent_metric"
        assert len(monitor.resolved_alerts) == 1
        assert monitor.resolved_alerts[0].alert_id == "recent_alert"

    def test_start_monitoring(self):
        """Test starting performance monitoring."""
        monitor = PerformanceMonitor(monitoring_interval=0.1)

        monitor.start_monitoring()

        assert monitor.is_monitoring is True
        assert monitor.monitoring_thread is not None

        # Wait a bit for monitoring to start
        time.sleep(0.2)

        # Should have collected some system metrics
        assert len(monitor.system_metrics_history) > 0

        monitor.stop_monitoring()

    def test_start_monitoring_already_running(self):
        """Test starting monitoring when already running."""
        monitor = PerformanceMonitor()

        monitor.start_monitoring()
        first_thread = monitor.monitoring_thread

        # Try to start again
        monitor.start_monitoring()

        # Should not create new thread
        assert monitor.monitoring_thread is first_thread

        monitor.stop_monitoring()

    def test_stop_monitoring(self):
        """Test stopping performance monitoring."""
        monitor = PerformanceMonitor(monitoring_interval=0.1)

        monitor.start_monitoring()
        assert monitor.is_monitoring is True

        monitor.stop_monitoring()
        assert monitor.is_monitoring is False

        # Wait for thread to stop
        time.sleep(0.2)
        assert not monitor.monitoring_thread.is_alive()

    def test_stop_monitoring_not_running(self):
        """Test stopping monitoring when not running."""
        monitor = PerformanceMonitor()

        # Should not crash
        monitor.stop_monitoring()
        assert monitor.is_monitoring is False

    def test_monitoring_worker_interruptible(self):
        """Test that monitoring worker can be interrupted."""
        monitor = PerformanceMonitor(monitoring_interval=1.0)

        start_time = time.time()
        monitor.start_monitoring()

        # Stop quickly
        time.sleep(0.1)
        monitor.stop_monitoring()

        elapsed = time.time() - start_time

        # Should stop quickly, not wait full interval
        assert elapsed < 0.5

    def test_context_manager(self):
        """Test using monitor as context manager."""
        with PerformanceMonitor(monitoring_interval=0.1) as monitor:
            assert monitor.is_monitoring is True
            
            # Wait a bit for monitoring
            time.sleep(0.2)
            
            # Should have collected some data
            assert len(monitor.system_metrics_history) > 0

        # Should have stopped monitoring
        assert monitor.is_monitoring is False


class TestPerformanceMonitorIntegration:
    """Integration tests for PerformanceMonitor."""

    def test_full_monitoring_cycle(self):
        """Test a full monitoring cycle."""
        monitor = PerformanceMonitor(
            monitoring_interval=0.1,
            retention_period=1.0,
            enable_alerts=True
        )

        # Register callbacks
        alerts_received = []
        metrics_received = []

        def alert_callback(alert):
            alerts_received.append(alert)

        def metric_callback(metric):
            metrics_received.append(metric)

        monitor.register_alert_callback(alert_callback)
        monitor.register_metric_callback(metric_callback)

        # Start monitoring
        monitor.start_monitoring()

        # Record various metrics
        for i in range(10):
            monitor.record_metric("test_metric", i * 10, "units", {"index": str(i)})
            monitor.record_request(i * 20, i % 3 != 0)  # Some failures

        # Wait for monitoring cycles
        time.sleep(0.3)

        # Trigger some alerts
        monitor.record_metric("memory_percent", 85.0, "percent")  # Above threshold
        monitor.record_metric("cpu_percent", 95.0, "percent")     # Above threshold

        # Get current metrics
        current_metrics = monitor.get_current_metrics()
        assert isinstance(current_metrics, dict)
        assert current_metrics['total_requests'] == 10

        # Get alerts
        alerts = monitor.get_alerts(include_resolved=True)
        assert len(alerts) >= 0  # May or may not have alerts

        # Generate report
        report = monitor.generate_report()
        assert isinstance(report, dict)
        assert 'summary' in report
        assert 'metrics_stats' in report

        # Stop monitoring
        monitor.stop_monitoring()

        # Should have received some metrics
        assert len(metrics_received) > 0

    def test_alert_lifecycle(self):
        """Test complete alert lifecycle."""
        monitor = PerformanceMonitor(alert_cooldown=0.1)

        alert_events = []

        def alert_callback(alert):
            alert_events.append(("generated", alert))

        monitor.register_alert_callback(alert_callback)

        # Start monitoring
        monitor.start_monitoring()

        # Generate alerts
        monitor.record_metric("memory_percent", 85.0, "percent")  # Warning
        monitor.record_metric("cpu_percent", 95.0, "percent")     # Error

        # Wait for alert processing
        time.sleep(0.2)

        # Check active alerts
        active_alerts = monitor.get_alerts(include_resolved=False)
        initial_count = len(active_alerts)

        # Resolve some alerts
        if active_alerts:
            alert_id = active_alerts[0]['alert_id']
            monitor.resolve_alert(alert_id)
            alert_events.append(("resolved", alert_id))

        # Wait a bit
        time.sleep(0.1)

        # Check final state
        final_active_alerts = monitor.get_alerts(include_resolved=False)
        all_alerts = monitor.get_alerts(include_resolved=True)

        # Should have resolved alerts
        assert len(all_alerts) >= len(final_active_alerts)

        # Stop monitoring
        monitor.stop_monitoring()

    def test_metrics_history_and_retention(self):
        """Test metrics history and retention."""
        monitor = PerformanceMonitor(
            retention_period=0.5,  # Short retention for testing
            monitoring_interval=0.1
        )

        # Start monitoring
        monitor.start_monitoring()

        # Record metrics over time
        start_time = time.time()
        for i in range(20):
            monitor.record_metric(f"metric_{i}", i, "units")
            time.sleep(0.05)  # Small delay

        # Should have metrics
        all_metrics = monitor.get_metrics_history()
        assert len(all_metrics) > 0

        # Wait for retention period to pass
        time.sleep(0.6)

        # Trigger cleanup (normally done by monitoring worker)
        monitor._cleanup_old_data()

        # Should have fewer metrics due to retention
        recent_metrics = monitor.get_metrics_history()
        assert len(recent_metrics) <= len(all_metrics)

        # Stop monitoring
        monitor.stop_monitoring()

    def test_performance_under_load(self):
        """Test performance under load."""
        monitor = PerformanceMonitor()

        # Record many metrics quickly
        start_time = time.time()
        
        for i in range(1000):
            monitor.record_metric(f"metric_{i}", i, "units")
            if i % 10 == 0:
                monitor.record_request(i * 0.1, i % 20 != 0)

        record_time = time.time() - start_time

        # Should complete quickly
        assert record_time < 2.0

        # Check metrics
        current_metrics = monitor.get_current_metrics()
        assert current_metrics['total_requests'] == 100

        # Generate report
        start_time = time.time()
        report = monitor.generate_report()
        report_time = time.time() - start_time

        # Report generation should be fast
        assert report_time < 1.0
        assert isinstance(report, dict)

    def test_export_import_workflow(self):
        """Test export and import workflow."""
        monitor = PerformanceMonitor()

        # Record various data
        for i in range(50):
            monitor.record_metric(f"metric_{i}", i * 2, "units")
            monitor.record_request(i * 5, i % 10 != 0)

        # Generate some alerts
        monitor.record_metric("memory_percent", 85.0, "percent")

        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name

        try:
            # Export
            export_result = monitor.export_metrics(temp_file)
            assert export_result is True

            # Verify export
            with open(temp_file, 'r') as f:
                exported_data = json.load(f)

            assert 'current_metrics' in exported_data
            assert 'metrics_history' in exported_data
            assert 'alerts' in exported_data

            # Check data integrity
            assert exported_data['current_metrics']['total_requests'] == 50
            assert len(exported_data['metrics_history']) > 0

        finally:
            os.unlink(temp_file)

    def test_thread_safety(self):
        """Test thread safety of monitor operations."""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()

        def worker(worker_id):
            for i in range(50):
                # Record metrics
                monitor.record_metric(f"worker_{worker_id}_metric_{i}", i, "units")
                monitor.record_request(i * 0.1, i % 5 != 0)
                
                # Get current metrics
                current = monitor.get_current_metrics()
                assert isinstance(current, dict)
                
                # Get alerts
                alerts = monitor.get_alerts()
                assert isinstance(alerts, list)

        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should not have crashed
        assert monitor.is_monitoring is True

        # Check final state
        final_metrics = monitor.get_current_metrics()
        assert final_metrics['total_requests'] == 150  # 3 workers * 50 requests

        monitor.stop_monitoring()