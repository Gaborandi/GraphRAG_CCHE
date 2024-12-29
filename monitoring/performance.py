# monitoring/performance.py
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'query_latency': Histogram(buckets=[10, 50, 100, 500]),
            'cache_hits': Counter(),
            'graph_operations': Timer(),
            'memory_usage': Gauge()
        }
        
    def start_operation(self, operation_type):
        return self.metrics['graph_operations'].time()

    def record_query(self, duration_ms):
        self.metrics['query_latency'].observe(duration_ms)

    def export_metrics(self):
        return {name: metric.value() for name, metric in self.metrics.items()}