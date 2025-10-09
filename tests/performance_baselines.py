"""
Performance Baselines for AI Therapist
Defines expected performance thresholds for production monitoring
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import time

@dataclass
class PerformanceThresholds:
    """Performance thresholds for different operations."""
    # Voice Processing
    voice_transcription_max_time: float = 2.0  # seconds
    voice_synthesis_max_time: float = 1.5      # seconds
    audio_processing_max_memory: float = 200.0  # MB
    
    # Security Operations
    phi_detection_max_time: float = 0.1       # seconds
    encryption_max_time: float = 0.05         # seconds
    audit_log_max_time: float = 0.02          # seconds
    
    # Performance Operations
    cache_hit_rate_min: float = 0.8          # 80%
    memory_usage_max: float = 500.0           # MB
    cpu_usage_max: float = 80.0              # percentage
    
    # UI Operations
    ui_response_max_time: float = 0.1        # seconds
    visualization_fps_target: int = 60       # frames per second

class PerformanceMonitor:
    """Monitor performance against baselines."""
    
    def __init__(self, thresholds: PerformanceThresholds = None):
        self.thresholds = thresholds or PerformanceThresholds()
        self.measurements: List[Dict] = []
    
    def measure_operation(self, operation_name: str, max_time: float = None):
        """Context manager to measure operation performance."""
        max_time = max_time or getattr(self.thresholds, f"{operation_name}_max_time", 1.0)
        
        class Measurement:
            def __init__(self, monitor, name, max_allowed):
                self.monitor = monitor
                self.name = name
                self.max_allowed = max_allowed
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.time()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                self.monitor.measurements.append({
                    'operation': self.name,
                    'duration': duration,
                    'within_threshold': duration <= self.max_allowed,
                    'timestamp': time.time()
                })
                
                if duration > self.max_allowed:
                    print(f"⚠️  Performance warning: {self.name} took {duration:.3f}s (max: {self.max_allowed:.3f}s)")
        
        return Measurement(self, operation_name, max_time)
    
    def get_performance_summary(self) -> Dict:
        """Get summary of performance measurements."""
        if not self.measurements:
            return {'message': 'No measurements recorded'}
        
        total_measurements = len(self.measurements)
        within_threshold = sum(1 for m in self.measurements if m['within_threshold'])
        
        return {
            'total_measurements': total_measurements,
            'within_threshold': within_threshold,
            'performance_score': (within_threshold / total_measurements) * 100,
            'operations': {
                m['operation']: {
                    'avg_duration': sum(
                        meas['duration'] for meas in self.measurements 
                        if meas['operation'] == m['operation']
                    ) / len([
                        meas for meas in self.measurements 
                        if meas['operation'] == m['operation']
                    ])
                }
                for m in self.measurements
            }
        }

# Baseline test cases
def run_performance_baselines():
    """Run performance baseline tests."""
    monitor = PerformanceMonitor()
    
    # Test voice processing baseline
    with monitor.measure_operation('voice_transcription'):
        time.sleep(0.1)  # Simulate transcription
    
    with monitor.measure_operation('voice_synthesis'):
        time.sleep(0.05)  # Simulate synthesis
    
    # Test security operations baseline
    with monitor.measure_operation('phi_detection'):
        time.sleep(0.01)  # Simulate PHI detection
    
    with monitor.measure_operation('encryption'):
        time.sleep(0.005)  # Simulate encryption
    
    return monitor.get_performance_summary()

if __name__ == "__main__":
    print("🏃 Running Performance Baselines...")
    summary = run_performance_baselines()
    print("📊 Performance Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")