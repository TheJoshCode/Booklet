"""
GPU Monitoring Module using NVIDIA Management Library (NVML)

Provides accurate VRAM usage, GPU utilization, temperature, power, and more.
Uses nvidia-ml-py (pynvml) for direct access to NVIDIA hardware metrics.

Install: pip install nvidia-ml-py3
"""

import logging
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime
import threading
import time

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    pynvml = None

logger = logging.getLogger("booklet.gpu_monitor")


@dataclass
class GPUStats:
    """Container for GPU statistics"""
    # Memory (bytes)
    memory_used: int
    memory_total: int
    memory_free: int
    
    # Utilization (%)
    gpu_utilization: int
    memory_utilization: int
    
    # Temperature (C)
    temperature: int
    
    # Power (W)
    power_usage: float
    power_limit: float
    
    # Clock speeds (MHz)
    gpu_clock: int
    memory_clock: int
    
    # Device info
    device_name: str
    driver_version: str
    cuda_version: str
    
    # Timestamp
    timestamp: datetime
    
    @property
    def memory_used_gb(self) -> float:
        """Memory used in GB"""
        return self.memory_used / 1024**3
    
    @property
    def memory_total_gb(self) -> float:
        """Total memory in GB"""
        return self.memory_total / 1024**3
    
    @property
    def memory_free_gb(self) -> float:
        """Free memory in GB"""
        return self.memory_free / 1024**3
    
    @property
    def memory_percent(self) -> float:
        """Memory usage percentage"""
        return (self.memory_used / self.memory_total) * 100 if self.memory_total > 0 else 0
    
    @property
    def power_percent(self) -> float:
        """Power usage percentage"""
        return (self.power_usage / self.power_limit) * 100 if self.power_limit > 0 else 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "memory": {
                "used_gb": round(self.memory_used_gb, 2),
                "total_gb": round(self.memory_total_gb, 2),
                "free_gb": round(self.memory_free_gb, 2),
                "used_percent": round(self.memory_percent, 1)
            },
            "utilization": {
                "gpu_percent": self.gpu_utilization,
                "memory_percent": self.memory_utilization
            },
            "thermal": {
                "temperature_c": self.temperature
            },
            "power": {
                "usage_w": round(self.power_usage, 1),
                "limit_w": round(self.power_limit, 1),
                "usage_percent": round(self.power_percent, 1)
            },
            "clocks": {
                "gpu_mhz": self.gpu_clock,
                "memory_mhz": self.memory_clock
            },
            "device": {
                "name": self.device_name,
                "driver": self.driver_version,
                "cuda": self.cuda_version
            },
            "timestamp": self.timestamp.isoformat()
        }
    
    def format_summary(self) -> str:
        """Format as human-readable summary"""
        return (
            f"GPU: {self.device_name} | "
            f"VRAM: {self.memory_used_gb:.2f}/{self.memory_total_gb:.2f}GB ({self.memory_percent:.1f}%) | "
            f"GPU: {self.gpu_utilization}% | "
            f"Temp: {self.temperature}°C | "
            f"Power: {self.power_usage:.1f}/{self.power_limit:.1f}W ({self.power_percent:.1f}%)"
        )


class GPUMonitor:
    """
    GPU Monitor using NVIDIA Management Library.
    
    Provides accurate real-time GPU metrics including:
    - VRAM usage (used, free, total)
    - GPU utilization
    - Memory controller utilization
    - Temperature
    - Power consumption
    - Clock speeds
    
    Example:
        monitor = GPUMonitor()
        stats = monitor.get_stats()
        print(f"VRAM: {stats.memory_used_gb:.2f}GB / {stats.memory_total_gb:.2f}GB")
    """
    
    def __init__(self, device_index: int = 0):
        """
        Initialize GPU monitor.
        
        Args:
            device_index: GPU device index (default: 0)
        """
        self.device_index = device_index
        self.handle = None
        self.initialized = False
        
        if not NVML_AVAILABLE:
            logger.warning("nvidia-ml-py3 not installed. GPU monitoring disabled.")
            logger.warning("Install with: pip install nvidia-ml-py3")
            return
        
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self.initialized = True
            
            # Get static info
            self.device_name = pynvml.nvmlDeviceGetName(self.handle)
            if isinstance(self.device_name, bytes):
                self.device_name = self.device_name.decode('utf-8')
            
            self.driver_version = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(self.driver_version, bytes):
                self.driver_version = self.driver_version.decode('utf-8')
            
            cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
            self.cuda_version = f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"
            
            logger.info(f"GPU Monitor initialized: {self.device_name}")
            logger.info(f"Driver: {self.driver_version}, CUDA: {self.cuda_version}")
            
        except Exception as e:
            logger.error(f"Failed to initialize GPU monitor: {e}")
            self.initialized = False
    
    def get_stats(self) -> Optional[GPUStats]:
        """
        Get current GPU statistics.
        
        Returns:
            GPUStats object with current metrics, or None if monitoring unavailable
        """
        if not self.initialized:
            return None
        
        try:
            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            
            # Utilization
            util_rates = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            
            # Temperature
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temperature = 0
            
            # Power
            try:
                power_usage = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # mW to W
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(self.handle) / 1000.0
            except:
                power_usage = 0.0
                power_limit = 0.0
            
            # Clock speeds
            try:
                gpu_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_GRAPHICS)
                memory_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)
            except:
                gpu_clock = 0
                memory_clock = 0
            
            return GPUStats(
                memory_used=mem_info.used,
                memory_total=mem_info.total,
                memory_free=mem_info.free,
                gpu_utilization=util_rates.gpu,
                memory_utilization=util_rates.memory,
                temperature=temperature,
                power_usage=power_usage,
                power_limit=power_limit,
                gpu_clock=gpu_clock,
                memory_clock=memory_clock,
                device_name=self.device_name,
                driver_version=self.driver_version,
                cuda_version=self.cuda_version,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to get GPU stats: {e}")
            return None
    
    def get_memory_info(self) -> Dict[str, float]:
        """
        Get just memory info (lightweight).
        
        Returns:
            Dict with used_gb, total_gb, free_gb, percent
        """
        stats = self.get_stats()
        if stats is None:
            return {"used_gb": 0, "total_gb": 0, "free_gb": 0, "percent": 0}
        
        return {
            "used_gb": stats.memory_used_gb,
            "total_gb": stats.memory_total_gb,
            "free_gb": stats.memory_free_gb,
            "percent": stats.memory_percent
        }
    
    def log_stats(self, prefix: str = ""):
        """
        Log current GPU stats.
        
        Args:
            prefix: Optional prefix for log message
        """
        stats = self.get_stats()
        if stats:
            msg = f"{prefix}{stats.format_summary()}" if prefix else stats.format_summary()
            logger.info(msg)
    
    def __del__(self):
        """Cleanup on destruction"""
        if self.initialized:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


class GPUMemoryTracker:
    """
    Track GPU memory usage over time with leak detection.
    
    Monitors VRAM usage and detects memory leaks by comparing
    baseline usage with current usage.
    """
    
    def __init__(self, device_index: int = 0):
        self.monitor = GPUMonitor(device_index)
        self.baseline_memory = None
        self.peak_memory = 0
        self.measurements: List[float] = []
        self.lock = threading.Lock()
    
    def set_baseline(self):
        """Set current memory as baseline"""
        stats = self.monitor.get_stats()
        if stats:
            with self.lock:
                self.baseline_memory = stats.memory_used_gb
                self.peak_memory = stats.memory_used_gb
            logger.info(f"Memory baseline set: {self.baseline_memory:.2f}GB")
    
    def record_measurement(self):
        """Record current memory usage"""
        stats = self.monitor.get_stats()
        if stats:
            with self.lock:
                self.measurements.append(stats.memory_used_gb)
                if stats.memory_used_gb > self.peak_memory:
                    self.peak_memory = stats.memory_used_gb
    
    def get_leak_estimate(self) -> Optional[float]:
        """
        Estimate memory leak in GB.
        
        Returns:
            Estimated leak in GB, or None if not enough data
        """
        stats = self.monitor.get_stats()
        if stats is None or self.baseline_memory is None:
            return None
        
        current_memory = stats.memory_used_gb
        leak = current_memory - self.baseline_memory
        return max(0, leak)  # Don't report negative leaks
    
    def check_for_leak(self, threshold_gb: float = 0.5) -> bool:
        """
        Check if there's a memory leak.
        
        Args:
            threshold_gb: Leak threshold in GB (default: 0.5GB)
            
        Returns:
            True if leak detected
        """
        leak = self.get_leak_estimate()
        if leak is not None and leak > threshold_gb:
            logger.warning(f"⚠️  Memory leak detected: {leak:.2f}GB above baseline")
            return True
        return False
    
    def get_summary(self) -> Dict:
        """Get tracking summary"""
        stats = self.monitor.get_stats()
        leak = self.get_leak_estimate()
        
        return {
            "current_gb": stats.memory_used_gb if stats else 0,
            "baseline_gb": self.baseline_memory or 0,
            "peak_gb": self.peak_memory,
            "leak_gb": leak or 0,
            "measurements": len(self.measurements)
        }


class GPUMonitorThread:
    """
    Background thread for continuous GPU monitoring.
    
    Monitors GPU stats in the background and logs warnings
    if VRAM usage exceeds thresholds.
    """
    
    def __init__(self, interval: float = 5.0, vram_warning_percent: float = 80.0):
        """
        Args:
            interval: Monitoring interval in seconds
            vram_warning_percent: VRAM usage % to trigger warning
        """
        self.monitor = GPUMonitor()
        self.interval = interval
        self.vram_warning_percent = vram_warning_percent
        self.running = False
        self.thread = None
        
        self.history: List[GPUStats] = []
        self.max_history = 100
    
    def start(self):
        """Start background monitoring"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info(f"GPU monitoring thread started (interval: {self.interval}s)")
    
    def stop(self):
        """Stop background monitoring"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        logger.info("GPU monitoring thread stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.running:
            stats = self.monitor.get_stats()
            
            if stats:
                # Store in history
                self.history.append(stats)
                if len(self.history) > self.max_history:
                    self.history.pop(0)
                
                # Check thresholds
                if stats.memory_percent > self.vram_warning_percent:
                    logger.warning(
                        f"⚠️  High VRAM usage: {stats.memory_used_gb:.2f}GB "
                        f"({stats.memory_percent:.1f}%)"
                    )
                
                if stats.temperature > 85:
                    logger.warning(f"⚠️  High GPU temperature: {stats.temperature}°C")
            
            time.sleep(self.interval)
    
    def get_latest(self) -> Optional[GPUStats]:
        """Get latest stats"""
        return self.history[-1] if self.history else None
    
    def get_history(self, last_n: int = 10) -> List[GPUStats]:
        """Get recent history"""
        return self.history[-last_n:]


# Global monitor instance (singleton)
_global_monitor: Optional[GPUMonitor] = None


def get_gpu_monitor() -> GPUMonitor:
    """
    Get global GPU monitor instance.
    
    Returns:
        Singleton GPUMonitor instance
    """
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = GPUMonitor()
    return _global_monitor


def log_gpu_stats(prefix: str = ""):
    """
    Convenience function to log GPU stats.
    
    Args:
        prefix: Optional prefix for log message
    """
    monitor = get_gpu_monitor()
    monitor.log_stats(prefix)


def get_vram_usage() -> Dict[str, float]:
    """
    Convenience function to get VRAM usage.
    
    Returns:
        Dict with used_gb, total_gb, free_gb, percent
    """
    monitor = get_gpu_monitor()
    return monitor.get_memory_info()


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("GPU Monitor Test")
    print("=" * 60)
    
    monitor = GPUMonitor()
    
    if monitor.initialized:
        stats = monitor.get_stats()
        print("\nCurrent GPU Stats:")
        print(stats.format_summary())
        print("\nDetailed Stats:")
        import json
        print(json.dumps(stats.to_dict(), indent=2))
        
        print("\nTesting memory tracker...")
        tracker = GPUMemoryTracker()
        tracker.set_baseline()
        
        import torch
        print("Allocating 1GB tensor...")
        tensor = torch.randn(256, 256, 256, device='cuda')
        
        tracker.record_measurement()
        print(f"Leak estimate: {tracker.get_leak_estimate():.2f}GB")
        
        del tensor
        torch.cuda.empty_cache()
        
        tracker.record_measurement()
        print(f"After cleanup: {tracker.get_leak_estimate():.2f}GB")
    else:
        print("GPU monitoring not available")