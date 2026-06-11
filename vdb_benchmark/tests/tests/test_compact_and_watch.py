"""
Unit tests for compaction and monitoring functionality in vdb-bench
"""
import pytest
import time
from unittest.mock import Mock, MagicMock, patch, call
import threading
from typing import Dict, Any, List
import json
from datetime import datetime, timedelta


class TestCompactionOperations:
    """Test database compaction operations."""
    
    def test_manual_compaction_trigger(self, mock_collection):
        """Test manually triggering compaction."""
        mock_collection.compact.return_value = 1234  # Compaction ID
        
        def trigger_compaction(collection):
            """Trigger manual compaction."""
            try:
                compaction_id = collection.compact()
                return {
                    "success": True,
                    "compaction_id": compaction_id,
                    "timestamp": time.time()
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e)
                }
        
        result = trigger_compaction(mock_collection)
        
        assert result["success"] is True
        assert result["compaction_id"] == 1234
        assert "timestamp" in result
        mock_collection.compact.assert_called_once()
    
    def test_compaction_state_monitoring(self, mock_collection):
        """Test monitoring compaction state."""
        # Mock compaction state progression
        states = ["Executing", "Executing", "Completed"]
        state_iter = iter(states)
        
        def get_compaction_state(compaction_id):
            try:
                return next(state_iter)
            except StopIteration:
                return "Completed"
        
        mock_collection.get_compaction_state = Mock(side_effect=get_compaction_state)
        
        def monitor_compaction(collection, compaction_id, timeout=60):
            """Monitor compaction until completion."""
            start_time = time.time()
            states = []
            
            while time.time() - start_time < timeout:
                state = collection.get_compaction_state(compaction_id)
                states.append({
                    "state": state,
                    "timestamp": time.time() - start_time
                })
                
                if state == "Completed":
                    return {
                        "success": True,
                        "duration": time.time() - start_time,
                        "states": states
                    }
                elif state == "Failed":
                    return {
                        "success": False,
                        "error": "Compaction failed",
                        "states": states
                    }
                
                time.sleep(0.1)  # Check interval
            
            return {
                "success": False,
                "error": "Compaction timeout",
                "states": states
            }
        
        with patch('time.sleep'):  # Speed up test
            result = monitor_compaction(mock_collection, 1234)
        
        assert result["success"] is True
        assert len(result["states"]) == 3
        assert result["states"][-1]["state"] == "Completed"
    
    def test_automatic_compaction_scheduling(self):
        """Test automatic compaction scheduling based on conditions."""
        class CompactionScheduler:
            def __init__(self, collection):
                self.collection = collection
                self.last_compaction = None
                self.compaction_history = []
            
            def should_compact(self, num_segments, deleted_ratio, time_since_last):
                """Determine if compaction should be triggered."""
                # Compact if:
                # - More than 10 segments
                # - Deleted ratio > 20%
                # - More than 1 hour since last compaction
                
                if num_segments > 10:
                    return True, "Too many segments"
                
                if deleted_ratio > 0.2:
                    return True, "High deletion ratio"
                
                if self.last_compaction and time_since_last > 3600:
                    return True, "Time-based compaction"
                
                return False, None
            
            def check_and_compact(self):
                """Check conditions and trigger compaction if needed."""
                # Get collection stats (mocked here)
                stats = {
                    "num_segments": 12,
                    "deleted_ratio": 0.15,
                    "last_compaction": self.last_compaction
                }
                
                time_since_last = (
                    time.time() - self.last_compaction 
                    if self.last_compaction else float('inf')
                )
                
                should_compact, reason = self.should_compact(
                    stats["num_segments"],
                    stats["deleted_ratio"],
                    time_since_last
                )
                
                if should_compact:
                    compaction_id = self.collection.compact()
                    self.last_compaction = time.time()
                    self.compaction_history.append({
                        "id": compaction_id,
                        "reason": reason,
                        "timestamp": self.last_compaction
                    })
                    return True, reason
                
                return False, None
        
        mock_collection = Mock()
        mock_collection.compact.return_value = 5678
        
        scheduler = CompactionScheduler(mock_collection)
        
        # Should trigger compaction (too many segments)
        compacted, reason = scheduler.check_and_compact()
        
        assert compacted is True
        assert reason == "Too many segments"
        assert len(scheduler.compaction_history) == 1
        mock_collection.compact.assert_called_once()
    
    def test_compaction_with_resource_monitoring(self):
        """Test compaction with system resource monitoring."""
        import psutil
        
        class ResourceAwareCompaction:
            def __init__(self, collection):
                self.collection = collection
                self.resource_thresholds = {
                    "cpu_percent": 80,
                    "memory_percent": 85,
                    "disk_io_rate": 100  # MB/s
                }
            
            def check_resources(self):
                """Check if system resources allow compaction."""
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                
                # Mock disk I/O rate
                disk_io_rate = 50  # MB/s
                
                return {
                    "cpu_ok": cpu_percent < self.resource_thresholds["cpu_percent"],
                    "memory_ok": memory_percent < self.resource_thresholds["memory_percent"],
                    "disk_ok": disk_io_rate < self.resource_thresholds["disk_io_rate"],
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "disk_io_rate": disk_io_rate
                }
            
            def compact_with_resource_check(self):
                """Perform compaction only if resources are available."""
                resource_status = self.check_resources()
                
                if all([resource_status["cpu_ok"], 
                       resource_status["memory_ok"], 
                       resource_status["disk_ok"]]):
                    
                    compaction_id = self.collection.compact()
                    return {
                        "success": True,
                        "compaction_id": compaction_id,
                        "resource_status": resource_status
                    }
                else:
                    return {
                        "success": False,
                        "reason": "Resource constraints",
                        "resource_status": resource_status
                    }
        
        with patch('psutil.cpu_percent', return_value=50):
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value = Mock(percent=60)
                
                mock_collection = Mock()
                mock_collection.compact.return_value = 9999
                
                compactor = ResourceAwareCompaction(mock_collection)
                result = compactor.compact_with_resource_check()
                
                assert result["success"] is True
                assert result["compaction_id"] == 9999
                assert result["resource_status"]["cpu_ok"] is True


class TestMonitoring:
    """Test monitoring functionality."""
    
    def test_collection_stats_monitoring(self, mock_collection):
        """Test monitoring collection statistics."""
        mock_collection.num_entities = 1000000
        
        # Mock getting collection stats
        def get_stats():
            return {
                "num_entities": mock_collection.num_entities,
                "num_segments": 10,
                "index_building_progress": 95
            }
        
        mock_collection.get_stats = get_stats
        
        class StatsMonitor:
            def __init__(self, collection):
                self.collection = collection
                self.stats_history = []
            
            def collect_stats(self):
                """Collect current statistics."""
                stats = self.collection.get_stats()
                stats["timestamp"] = time.time()
                self.stats_history.append(stats)
                return stats
            
            def get_trends(self, window_size=10):
                """Calculate trends from recent stats."""
                if len(self.stats_history) < 2:
                    return None
                
                recent = self.stats_history[-window_size:]
                
                # Calculate entity growth rate
                if len(recent) >= 2:
                    time_diff = recent[-1]["timestamp"] - recent[0]["timestamp"]
                    entity_diff = recent[-1]["num_entities"] - recent[0]["num_entities"]
                    
                    growth_rate = entity_diff / time_diff if time_diff > 0 else 0
                    
                    return {
                        "entity_growth_rate": growth_rate,
                        "avg_segments": sum(s["num_segments"] for s in recent) / len(recent),
                        "current_entities": recent[-1]["num_entities"]
                    }
                
                return None
        
        monitor = StatsMonitor(mock_collection)
        
        # Collect stats over time
        for i in range(5):
            mock_collection.num_entities += 10000
            stats = monitor.collect_stats()
            time.sleep(0.01)  # Small delay
        
        trends = monitor.get_trends()
        
        assert trends is not None
        assert trends["current_entities"] == 1050000  # 1000000 + (5 * 10000)
        assert len(monitor.stats_history) == 5
    
    def test_periodic_monitoring(self):
        """Test periodic monitoring with configurable intervals."""
        class PeriodicMonitor:
            def __init__(self, collection, interval=5):
                self.collection = collection
                self.interval = interval
                self.running = False
                self.thread = None
                self.data = []
            
            def monitor_function(self):
                """Function to run periodically."""
                stats = {
                    "timestamp": time.time(),
                    "num_entities": self.collection.num_entities,
                    "status": "healthy"
                }
                self.data.append(stats)
                return stats
            
            def start(self):
                """Start periodic monitoring."""
                self.running = True
                
                def run():
                    while self.running:
                        self.monitor_function()
                        time.sleep(self.interval)
                
                self.thread = threading.Thread(target=run)
                self.thread.daemon = True
                self.thread.start()
            
            def stop(self):
                """Stop periodic monitoring."""
                self.running = False
                if self.thread:
                    self.thread.join(timeout=1)
            
            def get_latest(self, n=5):
                """Get latest n monitoring results."""
                return self.data[-n:] if self.data else []
        
        mock_collection = Mock()
        mock_collection.num_entities = 1000000
        
        monitor = PeriodicMonitor(mock_collection, interval=0.01)  # Fast interval for testing
        
        monitor.start()
        time.sleep(0.05)  # Let it collect some data
        monitor.stop()
        
        latest = monitor.get_latest()
        
        assert len(latest) > 0
        assert all("timestamp" in item for item in latest)
    
    def test_alert_system(self):
        """Test alert system for monitoring thresholds."""
        class AlertSystem:
            def __init__(self):
                self.alerts = []
                self.thresholds = {
                    "high_latency": 100,  # ms
                    "low_qps": 50,
                    "high_error_rate": 0.05,
                    "segment_count": 20
                }
                self.alert_callbacks = []
            
            def check_metric(self, metric_name, value):
                """Check if metric exceeds threshold."""
                if metric_name == "latency" and value > self.thresholds["high_latency"]:
                    self.trigger_alert("HIGH_LATENCY", f"Latency {value}ms exceeds threshold")
                
                elif metric_name == "qps" and value < self.thresholds["low_qps"]:
                    self.trigger_alert("LOW_QPS", f"QPS {value} below threshold")
                
                elif metric_name == "error_rate" and value > self.thresholds["high_error_rate"]:
                    self.trigger_alert("HIGH_ERROR_RATE", f"Error rate {value:.2%} exceeds threshold")
                
                elif metric_name == "segments" and value > self.thresholds["segment_count"]:
                    self.trigger_alert("TOO_MANY_SEGMENTS", f"Segment count {value} exceeds threshold")
            
            def trigger_alert(self, alert_type, message):
                """Trigger an alert."""
                alert = {
                    "type": alert_type,
                    "message": message,
                    "timestamp": time.time(),
                    "resolved": False
                }
                
                self.alerts.append(alert)
                
                # Call registered callbacks
                for callback in self.alert_callbacks:
                    callback(alert)
                
                return alert
            
            def resolve_alert(self, alert_type):
                """Mark alerts of given type as resolved."""
                for alert in self.alerts:
                    if alert["type"] == alert_type and not alert["resolved"]:
                        alert["resolved"] = True
                        alert["resolved_time"] = time.time()
            
            def register_callback(self, callback):
                """Register callback for alerts."""
                self.alert_callbacks.append(callback)
            
            def get_active_alerts(self):
                """Get list of active (unresolved) alerts."""
                return [a for a in self.alerts if not a["resolved"]]
        
        alert_system = AlertSystem()
        
        # Register a callback
        received_alerts = []
        alert_system.register_callback(lambda alert: received_alerts.append(alert))
        
        # Test various metrics
        alert_system.check_metric("latency", 150)  # Should trigger
        alert_system.check_metric("qps", 100)  # Should not trigger
        alert_system.check_metric("error_rate", 0.1)  # Should trigger
        alert_system.check_metric("segments", 25)  # Should trigger
        
        active = alert_system.get_active_alerts()
        
        assert len(active) == 3
        assert len(received_alerts) == 3
        assert any(a["type"] == "HIGH_LATENCY" for a in active)
        
        # Resolve an alert
        alert_system.resolve_alert("HIGH_LATENCY")
        active = alert_system.get_active_alerts()
        
        assert len(active) == 2
    
    def test_monitoring_data_aggregation(self):
        """Test aggregating monitoring data over time windows."""
        class DataAggregator:
            def __init__(self):
                self.raw_data = []
            
            def add_data_point(self, timestamp, metrics):
                """Add a data point."""
                self.raw_data.append({
                    "timestamp": timestamp,
                    **metrics
                })
            
            def aggregate_window(self, start_time, end_time, aggregation="avg"):
                """Aggregate data within a time window."""
                window_data = [
                    d for d in self.raw_data 
                    if start_time <= d["timestamp"] <= end_time
                ]
                
                if not window_data:
                    return None
                
                if aggregation == "avg":
                    return self._average_aggregation(window_data)
                elif aggregation == "max":
                    return self._max_aggregation(window_data)
                elif aggregation == "min":
                    return self._min_aggregation(window_data)
                else:
                    return window_data
            
            def _average_aggregation(self, data):
                """Calculate average of metrics."""
                result = {"count": len(data)}
                
                # Get all metric keys (excluding timestamp)
                metric_keys = [k for k in data[0].keys() if k != "timestamp"]
                
                for key in metric_keys:
                    values = [d[key] for d in data if key in d]
                    result[f"{key}_avg"] = sum(values) / len(values) if values else 0
                
                return result
            
            def _max_aggregation(self, data):
                """Get maximum values of metrics."""
                result = {"count": len(data)}
                
                metric_keys = [k for k in data[0].keys() if k != "timestamp"]
                
                for key in metric_keys:
                    values = [d[key] for d in data if key in d]
                    result[f"{key}_max"] = max(values) if values else 0
                
                return result
            
            def _min_aggregation(self, data):
                """Get minimum values of metrics."""
                result = {"count": len(data)}
                
                metric_keys = [k for k in data[0].keys() if k != "timestamp"]
                
                for key in metric_keys:
                    values = [d[key] for d in data if key in d]
                    result[f"{key}_min"] = min(values) if values else 0
                
                return result
            
            def create_time_series(self, metric_name, interval=60):
                """Create time series data for a specific metric."""
                if not self.raw_data:
                    return []
                
                min_time = min(d["timestamp"] for d in self.raw_data)
                max_time = max(d["timestamp"] for d in self.raw_data)
                
                time_series = []
                current_time = min_time
                
                while current_time <= max_time:
                    window_end = current_time + interval
                    window_data = [
                        d for d in self.raw_data
                        if current_time <= d["timestamp"] < window_end
                        and metric_name in d
                    ]
                    
                    if window_data:
                        avg_value = sum(d[metric_name] for d in window_data) / len(window_data)
                        time_series.append({
                            "timestamp": current_time,
                            "value": avg_value
                        })
                    
                    current_time = window_end
                
                return time_series
        
        aggregator = DataAggregator()
        
        # Add sample data points
        base_time = time.time()
        for i in range(100):
            aggregator.add_data_point(
                base_time + i,
                {
                    "qps": 100 + i % 20,
                    "latency": 10 + i % 5,
                    "error_count": i % 3
                }
            )
        
        # Test aggregation
        avg_metrics = aggregator.aggregate_window(base_time, base_time + 50, "avg")
        assert avg_metrics is not None
        assert "qps_avg" in avg_metrics
        assert avg_metrics["count"] == 51
        
        # Test time series creation
        time_series = aggregator.create_time_series("qps", interval=10)
        assert len(time_series) > 0
        assert all("timestamp" in point and "value" in point for point in time_series)


class TestWatchOperations:
    """Test watch operations for monitoring database state."""
    
    def test_index_building_watch(self, mock_collection):
        """Test watching index building progress."""
        progress_values = [0, 25, 50, 75, 100]
        progress_iter = iter(progress_values)
        
        def get_index_progress():
            try:
                return next(progress_iter)
            except StopIteration:
                return 100
        
        mock_collection.index.get_build_progress = Mock(side_effect=get_index_progress)
        
        class IndexWatcher:
            def __init__(self, collection):
                self.collection = collection
                self.progress_history = []
            
            def watch_build(self, check_interval=1):
                """Watch index building until completion."""
                while True:
                    progress = self.collection.index.get_build_progress()
                    self.progress_history.append({
                        "progress": progress,
                        "timestamp": time.time()
                    })
                    
                    if progress >= 100:
                        return {
                            "completed": True,
                            "final_progress": progress,
                            "history": self.progress_history
                        }
                    
                    time.sleep(check_interval)
        
        mock_collection.index = Mock()
        mock_collection.index.get_build_progress = Mock(side_effect=get_index_progress)
        
        watcher = IndexWatcher(mock_collection)
        
        with patch('time.sleep'):  # Speed up test
            result = watcher.watch_build()
        
        assert result["completed"] is True
        assert result["final_progress"] == 100
        assert len(result["history"]) == 5
    
    def test_segment_merge_watch(self):
        """Test watching segment merge operations."""
        class SegmentMergeWatcher:
            def __init__(self):
                self.merge_operations = []
                self.active_merges = {}
            
            def start_merge(self, segments):
                """Start watching a segment merge."""
                merge_id = f"merge_{len(self.merge_operations)}"
                
                merge_op = {
                    "id": merge_id,
                    "segments": segments,
                    "start_time": time.time(),
                    "status": "running",
                    "progress": 0
                }
                
                self.merge_operations.append(merge_op)
                self.active_merges[merge_id] = merge_op
                
                return merge_id
            
            def update_progress(self, merge_id, progress):
                """Update merge progress."""
                if merge_id in self.active_merges:
                    self.active_merges[merge_id]["progress"] = progress
                    
                    if progress >= 100:
                        self.complete_merge(merge_id)
            
            def complete_merge(self, merge_id):
                """Mark merge as completed."""
                if merge_id in self.active_merges:
                    merge_op = self.active_merges[merge_id]
                    merge_op["status"] = "completed"
                    merge_op["end_time"] = time.time()
                    merge_op["duration"] = merge_op["end_time"] - merge_op["start_time"]
                    
                    del self.active_merges[merge_id]
                    
                    return merge_op
                
                return None
            
            def get_active_merges(self):
                """Get list of active merge operations."""
                return list(self.active_merges.values())
            
            def get_merge_stats(self):
                """Get statistics about merge operations."""
                completed = [m for m in self.merge_operations if m["status"] == "completed"]
                
                if not completed:
                    return None
                
                durations = [m["duration"] for m in completed]
                
                return {
                    "total_merges": len(self.merge_operations),
                    "completed_merges": len(completed),
                    "active_merges": len(self.active_merges),
                    "avg_duration": sum(durations) / len(durations) if durations else 0,
                    "min_duration": min(durations) if durations else 0,
                    "max_duration": max(durations) if durations else 0
                }
        
        watcher = SegmentMergeWatcher()
        
        # Start multiple merges
        merge1 = watcher.start_merge(["seg1", "seg2"])
        merge2 = watcher.start_merge(["seg3", "seg4"])
        
        assert len(watcher.get_active_merges()) == 2
        
        # Update progress
        watcher.update_progress(merge1, 50)
        watcher.update_progress(merge2, 100)  # Complete this one
        
        assert len(watcher.get_active_merges()) == 1
        
        # Complete remaining merge
        watcher.update_progress(merge1, 100)
        
        stats = watcher.get_merge_stats()
        assert stats["completed_merges"] == 2
        assert stats["active_merges"] == 0
