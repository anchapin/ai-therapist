"""
Performance Monitor Module

This module provides comprehensive real-time performance monitoring for the
AI Therapist voice services. It tracks memory usage, response times, resource
utilization, and provides alerts and reporting capabilities.

Features:
- Real-time performance monitoring
- Memory usage alerts and thresholds
- Response time tracking
- Resource utilization reporting
- Performance metrics collection
- Alert system for performance issues
- Historical performance data
"""

import time
import threading
import psutil
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
from enum import Enum
import json
import os
from datetime import datetime, timedelta

class AlertLevel(Enum):
    """Alert levels for performance issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    name: str
    value: float
    timestamp: float
    unit: str
    tags: Dict[str, str]

@dataclass
class PerformanceAlert:
    """Performance alert information."""
    alert_id: str
    level: AlertLevel
    message: str
    metric: str
    value: float
    threshold: float
    timestamp: float
    resolved: bool = False
    resolved_timestamp: Optional[float] = None

@dataclass
class SystemMetrics:
    """System-wide performance metrics."""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    network_connections: int
    thread_count: int
    process_count: int
    timestamp: float

class PerformanceMonitor:
    """
    Real-time performance monitoring system for voice services.

    Features:
    - Continuous system metrics collection
    - Performance threshold monitoring
    - Alert generation and management
    - Historical data retention
    - Performance reporting
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize performance monitor."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.monitoring_interval = self.config.get('monitoring_interval', 10.0)  # seconds
        self.retention_period = self.config.get('retention_period', 3600.0)  # 1 hour
        self.max_metrics_history = self.config.get('max_metrics_history', 10000)
        self.alert_cooldown = self.config.get('alert_cooldown', 300.0)  # 5 minutes
        self.enable_alerts = self.config.get('enable_alerts', True)

        # Performance thresholds
        self.thresholds = {
            'memory_percent': self.config.get('memory_threshold_percent', 80.0),
            'cpu_percent': self.config.get('cpu_threshold_percent', 90.0),
            'response_time_ms': self.config.get('response_time_threshold_ms', 2000.0),
            'error_rate_percent': self.config.get('error_rate_threshold_percent', 5.0),
            'memory_mb': self.config.get('memory_threshold_mb', 1024.0),
        }

        # Data storage
        self.metrics_history: deque[PerformanceMetric] = deque(maxlen=self.max_metrics_history)
        self.system_metrics_history: deque[SystemMetrics] = deque(maxlen=1000)
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.resolved_alerts: deque[PerformanceAlert] = deque(maxlen=1000)

        # Performance tracking
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
        self.response_times: deque[float] = deque(maxlen=1000)

        # Threading
        self.monitoring_thread = None
        self.is_monitoring = False
        self.lock = threading.RLock()

        # Callbacks
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        self.metric_callbacks: List[Callable[[PerformanceMetric], None]] = []

        # Process information
        self.process = psutil.Process()
        self.start_time = time.time()

        self.logger.info("Performance monitor initialized")

    def start_monitoring(self):
        """Start performance monitoring."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_worker,
            daemon=True,
            name="performance-monitor"
        )
        self.monitoring_thread.start()

        self.logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.is_monitoring = False

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)

        self.logger.info("Performance monitoring stopped")

    def record_metric(self, name: str, value: float, unit: str = "", tags: Optional[Dict[str, str]] = None):
        """Record a performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=time.time(),
            unit=unit,
            tags=tags or {}
        )

        with self.lock:
            self.metrics_history.append(metric)

            # Check thresholds and generate alerts
            self._check_thresholds(metric)

        # Trigger callbacks
        for callback in self.metric_callbacks:
            try:
                callback(metric)
            except Exception as e:
                self.logger.error(f"Error in metric callback: {e}")

    def record_request(self, response_time_ms: float, success: bool = True):
        """Record a request for performance tracking."""
        with self.lock:
            self.request_count += 1
            if not success:
                self.error_count += 1

            self.total_response_time += response_time_ms
            self.response_times.append(response_time_ms)

        # Record metrics
        self.record_metric("response_time", response_time_ms, "ms", {"success": str(success)})
        self.record_metric("request_count", self.request_count, "count")
        self.record_metric("error_count", self.error_count, "count")

        # Calculate and record rates
        uptime = time.time() - self.start_time
        if uptime > 0:
            requests_per_second = self.request_count / uptime
            error_rate = (self.error_count / self.request_count * 100) if self.request_count > 0 else 0
            avg_response_time = self.total_response_time / self.request_count

            self.record_metric("requests_per_second", requests_per_second, "req/s")
            self.record_metric("error_rate", error_rate, "percent")
            self.record_metric("avg_response_time", avg_response_time, "ms")

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        with self.lock:
            uptime = time.time() - self.start_time

            # Calculate averages
            avg_response_time = (
                self.total_response_time / self.request_count
                if self.request_count > 0 else 0.0
            )

            error_rate = (
                (self.error_count / self.request_count * 100)
                if self.request_count > 0 else 0.0
            )

            # Get latest system metrics
            system_metrics = self._get_system_metrics()

            return {
                'uptime_seconds': uptime,
                'total_requests': self.request_count,
                'total_errors': self.error_count,
                'error_rate_percent': error_rate,
                'avg_response_time_ms': avg_response_time,
                'requests_per_second': self.request_count / uptime if uptime > 0 else 0,
                'active_alerts': len(self.active_alerts),
                'metrics_in_history': len(self.metrics_history),
                'system_metrics': {
                    'cpu_percent': system_metrics.cpu_percent,
                    'memory_percent': system_metrics.memory_percent,
                    'memory_used_mb': system_metrics.memory_used_mb,
                    'memory_available_mb': system_metrics.memory_available_mb,
                    'thread_count': system_metrics.thread_count,
                },
                'thresholds': self.thresholds.copy(),
                'is_monitoring': self.is_monitoring
            }

    def get_alerts(self, include_resolved: bool = False, limit: int = 100) -> List[Dict[str, Any]]:
        """Get performance alerts."""
        with self.lock:
            alerts = []

            # Active alerts
            for alert in self.active_alerts.values():
                alerts.append({
                    'alert_id': alert.alert_id,
                    'level': alert.level.value,
                    'message': alert.message,
                    'metric': alert.metric,
                    'value': alert.value,
                    'threshold': alert.threshold,
                    'timestamp': alert.timestamp,
                    'resolved': False
                })

            if include_resolved:
                # Resolved alerts
                for alert in list(self.resolved_alerts)[-limit:]:
                    alerts.append({
                        'alert_id': alert.alert_id,
                        'level': alert.level.value,
                        'message': alert.message,
                        'metric': alert.metric,
                        'value': alert.value,
                        'threshold': alert.threshold,
                        'timestamp': alert.timestamp,
                        'resolved': True,
                        'resolved_timestamp': alert.resolved_timestamp
                    })

            # Sort by timestamp (newest first)
            alerts.sort(key=lambda x: x['timestamp'], reverse=True)
            return alerts[:limit]

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert."""
        with self.lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_timestamp = time.time()

                # Move to resolved alerts
                self.resolved_alerts.append(alert)
                del self.active_alerts[alert_id]

                self.logger.info(f"Resolved alert: {alert_id}")
                return True

            return False

    def get_metrics_history(self, metric_name: Optional[str] = None,
                           start_time: Optional[float] = None,
                           end_time: Optional[float] = None,
                           limit: int = 1000) -> List[PerformanceMetric]:
        """Get historical metrics data."""
        with self.lock:
            metrics = list(self.metrics_history)

            # Filter by metric name
            if metric_name:
                metrics = [m for m in metrics if m.name == metric_name]

            # Filter by time range
            if start_time is not None:
                metrics = [m for m in metrics if m.timestamp >= start_time]
            if end_time is not None:
                metrics = [m for m in metrics if m.timestamp <= end_time]

            # Sort by timestamp and limit
            metrics.sort(key=lambda m: m.timestamp, reverse=True)
            return metrics[:limit]

    def register_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Register callback for alerts."""
        self.alert_callbacks.append(callback)

    def register_metric_callback(self, callback: Callable[[PerformanceMetric], None]):
        """Register callback for metrics."""
        self.metric_callbacks.append(callback)

    def generate_report(self, time_range_seconds: int = 3600) -> Dict[str, Any]:
        """Generate a performance report for the specified time range."""
        start_time = time.time() - time_range_seconds

        # Get metrics for time range
        all_metrics = self.get_metrics_history(start_time=start_time)

        # Group metrics by name
        metrics_by_name = defaultdict(list)
        for metric in all_metrics:
            metrics_by_name[metric.name].append(metric.value)

        # Calculate statistics
        report = {
            'report_period_seconds': time_range_seconds,
            'generated_at': time.time(),
            'summary': self.get_current_metrics(),
            'metrics_stats': {}
        }

        # Calculate stats for each metric
        for name, values in metrics_by_name.items():
            if values:
                report['metrics_stats'][name] = {
                    'count': len(values),
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values),
                    'latest': values[-1] if values else None
                }

        # Alert summary
        active_alerts = self.get_alerts(include_resolved=False)
        resolved_alerts = self.get_alerts(include_resolved=True, limit=1000)
        resolved_alerts = [a for a in resolved_alerts if a['resolved']]

        report['alerts_summary'] = {
            'active_count': len(active_alerts),
            'resolved_count': len(resolved_alerts),
            'active_by_level': defaultdict(int),
            'resolved_by_level': defaultdict(int)
        }

        for alert in active_alerts:
            report['alerts_summary']['active_by_level'][alert['level']] += 1

        for alert in resolved_alerts:
            report['alerts_summary']['resolved_by_level'][alert['level']] += 1

        return report

    def export_metrics(self, filepath: str, format: str = "json"):
        """Export metrics data to file."""
        try:
            data = {
                'export_time': time.time(),
                'current_metrics': self.get_current_metrics(),
                'alerts': self.get_alerts(include_resolved=True, limit=10000),
                'metrics_history': [
                    {
                        'name': m.name,
                        'value': m.value,
                        'timestamp': m.timestamp,
                        'unit': m.unit,
                        'tags': m.tags
                    }
                    for m in self.get_metrics_history(limit=10000)
                ]
            }

            if format.lower() == "json":
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            else:
                self.logger.error(f"Unsupported export format: {format}")

            self.logger.info(f"Metrics exported to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")
            return False

    def _monitoring_worker(self):
        """Background monitoring worker."""
        last_cleanup = time.time()

        while self.is_monitoring:
            try:
                # Collect system metrics
                system_metrics = self._get_system_metrics()
                self.system_metrics_history.append(system_metrics)

                # Record system metrics
                self.record_metric("cpu_percent", system_metrics.cpu_percent, "percent")
                self.record_metric("memory_percent", system_metrics.memory_percent, "percent")
                self.record_metric("memory_used", system_metrics.memory_used_mb, "MB")
                self.record_metric("memory_available", system_metrics.memory_available_mb, "MB")
                self.record_metric("thread_count", system_metrics.thread_count, "count")

                # Periodic cleanup of old data
                current_time = time.time()
                if current_time - last_cleanup > 300:  # Every 5 minutes
                    self._cleanup_old_data()
                    last_cleanup = current_time

                time.sleep(self.monitoring_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring worker: {e}")
                time.sleep(5.0)

    def _get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_usage_percent=disk.percent,
                network_connections=len(psutil.net_connections()),
                thread_count=self.process.num_threads(),
                process_count=len(psutil.pids()),
                timestamp=time.time()
            )
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            # Return default/empty metrics
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                memory_available_mb=0.0,
                disk_usage_percent=0.0,
                network_connections=0,
                thread_count=0,
                process_count=0,
                timestamp=time.time()
            )

    def _check_thresholds(self, metric: PerformanceMetric):
        """Check if metric exceeds thresholds and generate alerts."""
        if not self.enable_alerts:
            return

        threshold_key = None
        threshold_value = None

        # Check different metric types
        if metric.name == "memory_percent" and metric.value > self.thresholds['memory_percent']:
            threshold_key = 'memory_percent'
            threshold_value = self.thresholds['memory_percent']
        elif metric.name == "cpu_percent" and metric.value > self.thresholds['cpu_percent']:
            threshold_key = 'cpu_percent'
            threshold_value = self.thresholds['cpu_percent']
        elif metric.name == "response_time" and metric.value > self.thresholds['response_time_ms']:
            threshold_key = 'response_time_ms'
            threshold_value = self.thresholds['response_time_ms']
        elif metric.name == "error_rate" and metric.value > self.thresholds['error_rate_percent']:
            threshold_key = 'error_rate_percent'
            threshold_value = self.thresholds['error_rate_percent']
        elif metric.name == "memory_used" and metric.value > self.thresholds['memory_mb']:
            threshold_key = 'memory_mb'
            threshold_value = self.thresholds['memory_mb']

        if threshold_key:
            alert_level = self._determine_alert_level(metric.value, threshold_value, metric.name)
            self._generate_alert(metric, threshold_key, threshold_value, alert_level)

    def _determine_alert_level(self, value: float, threshold: float, metric_name: str) -> AlertLevel:
        """Determine alert level based on severity."""
        ratio = value / threshold

        if ratio >= 2.0:
            return AlertLevel.CRITICAL
        elif ratio >= 1.5:
            return AlertLevel.ERROR
        elif ratio >= 1.2:
            return AlertLevel.WARNING
        else:
            return AlertLevel.INFO

    def _generate_alert(self, metric: PerformanceMetric, threshold_key: str,
                       threshold_value: float, level: AlertLevel):
        """Generate a performance alert."""
        alert_id = f"{metric.name}_{threshold_key}_{int(time.time())}"

        # Check cooldown - don't generate duplicate alerts too frequently
        current_time = time.time()
        for existing_alert in self.active_alerts.values():
            if (existing_alert.metric == metric.name and
                existing_alert.level == level and
                current_time - existing_alert.timestamp < self.alert_cooldown):
                return  # Skip duplicate alert

        alert = PerformanceAlert(
            alert_id=alert_id,
            level=level,
            message=f"{metric.name} exceeded threshold: {metric.value:.2f} {metric.unit} "
                   f"(threshold: {threshold_value:.2f})",
            metric=metric.name,
            value=metric.value,
            threshold=threshold_value,
            timestamp=current_time
        )

        with self.lock:
            self.active_alerts[alert_id] = alert

        self.logger.warning(f"Performance alert generated: {alert.message}")

        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")

    def _cleanup_old_data(self):
        """Clean up old metrics and resolved alerts."""
        cutoff_time = time.time() - self.retention_period

        with self.lock:
            # Clean up old metrics
            while self.metrics_history and self.metrics_history[0].timestamp < cutoff_time:
                self.metrics_history.popleft()

            # Clean up old resolved alerts (keep more of these)
            while (self.resolved_alerts and
                   self.resolved_alerts[0].resolved_timestamp and
                   self.resolved_alerts[0].resolved_timestamp < cutoff_time):
                self.resolved_alerts.popleft()

        self.logger.debug("Cleaned up old performance data")

    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()