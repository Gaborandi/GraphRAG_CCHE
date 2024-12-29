# analytics/graph_analytics.py
class GraphAnalytics:
    def __init__(self, neo4j_connection):
        self.neo4j = neo4j_connection
        self.metrics = {
            'centrality': CentralityAnalyzer(),
            'clustering': ClusteringAnalyzer(),
            'paths': PathAnalyzer()
        }

    def analyze_subgraph(self, nodes):
        results = {}
        for name, analyzer in self.metrics.items():
            results[name] = analyzer.analyze(self.neo4j, nodes)
        return AnalyticsReport(results)

    def export_metrics(self):
        return {
            name: analyzer.get_metrics()
            for name, analyzer in self.metrics.items()
        }