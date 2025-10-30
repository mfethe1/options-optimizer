"""
Tests for Performance Benchmark Script

Tests the performance benchmark script with:
- Mock mode benchmarking
- Result structure validation
- Bottleneck identification
- Recommendation generation
- Report generation
"""

import pytest
import asyncio
import json
from pathlib import Path
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from performance_benchmark import PerformanceBenchmark


class TestPerformanceBenchmark:
    """Test performance benchmark functionality"""
    
    @pytest.fixture
    def benchmark(self):
        """Create benchmark instance in mock mode"""
        return PerformanceBenchmark(mock_mode=True)
    
    @pytest.mark.asyncio
    async def test_benchmark_api_cached(self, benchmark):
        """Test API cached benchmark"""
        result = await benchmark.benchmark_api_cached(num_requests=10)
        
        # Verify result structure
        assert 'p50' in result
        assert 'p95' in result
        assert 'p99' in result
        assert 'mean' in result
        assert 'min' in result
        assert 'max' in result
        assert 'throughput' in result
        
        # Verify values are reasonable
        assert result['p50'] > 0
        assert result['p95'] > result['p50']
        assert result['p99'] > result['p95']
        assert result['throughput'] > 0
    
    @pytest.mark.asyncio
    async def test_benchmark_websocket_latency(self, benchmark):
        """Test WebSocket latency benchmark"""
        result = await benchmark.benchmark_websocket_latency(num_messages=10)
        
        # Verify result structure
        assert 'p50' in result
        assert 'p95' in result
        assert 'p99' in result
        assert 'mean' in result
        assert 'min' in result
        assert 'max' in result
        
        # Verify values are reasonable
        assert result['p50'] > 0
        assert result['p95'] > result['p50']
    
    @pytest.mark.asyncio
    async def test_benchmark_phase4_computation(self, benchmark):
        """Test Phase 4 computation benchmark"""
        result = await benchmark.benchmark_phase4_computation(num_assets=5)
        
        # Verify result structure
        assert 'p50' in result
        assert 'p95' in result
        assert 'p99' in result
        assert 'mean' in result
        assert 'min' in result
        assert 'max' in result
        
        # Verify values are reasonable
        assert result['p50'] > 0
        assert result['mean'] > 0
    
    def test_measure_resource_utilization(self, benchmark):
        """Test resource utilization measurement"""
        result = benchmark.measure_resource_utilization()
        
        # Verify result structure
        assert 'cpu_percent' in result
        assert 'memory_percent' in result
        assert 'memory_used_mb' in result
        assert 'memory_available_mb' in result
        assert 'network_bytes_sent_mb' in result
        assert 'network_bytes_recv_mb' in result
        
        # Verify values are reasonable
        assert 0 <= result['cpu_percent'] <= 100
        assert 0 <= result['memory_percent'] <= 100
        assert result['memory_used_mb'] > 0
        assert result['memory_available_mb'] > 0
    
    def test_identify_bottlenecks(self, benchmark):
        """Test bottleneck identification"""
        # Set up test data
        benchmark.results['api_latency'] = {'p95': 600}  # Exceeds 500ms target
        benchmark.results['websocket_latency'] = {'p95': 60}  # Exceeds 50ms target
        benchmark.results['phase4_latency'] = {'mean': 250}  # Exceeds 200ms target
        benchmark.results['cache_hit_rate'] = 0.5  # Below 80% target
        benchmark.results['resource_utilization'] = {
            'cpu_percent': 85,  # Exceeds 80%
            'memory_percent': 85  # Exceeds 80%
        }
        
        bottlenecks = benchmark.identify_bottlenecks()
        
        # Verify bottlenecks are identified
        assert len(bottlenecks) == 6
        assert any('API P95 latency' in b for b in bottlenecks)
        assert any('WebSocket P95 latency' in b for b in bottlenecks)
        assert any('Phase 4 mean latency' in b for b in bottlenecks)
        assert any('Cache hit rate' in b for b in bottlenecks)
        assert any('CPU utilization' in b for b in bottlenecks)
        assert any('Memory utilization' in b for b in bottlenecks)
    
    def test_generate_recommendations(self, benchmark):
        """Test recommendation generation"""
        # Set up test data
        benchmark.results['api_latency'] = {'p95': 600}
        benchmark.results['websocket_latency'] = {'p95': 60}
        benchmark.results['phase4_latency'] = {'mean': 250}
        benchmark.results['cache_hit_rate'] = 0.5
        benchmark.results['resource_utilization'] = {
            'cpu_percent': 85,
            'memory_percent': 85
        }
        
        recommendations = benchmark.generate_recommendations()
        
        # Verify recommendations are generated
        assert len(recommendations) == 6
        assert any('API response time' in r for r in recommendations)
        assert any('WebSocket latency' in r for r in recommendations)
        assert any('Phase 4 computation' in r for r in recommendations)
        assert any('cache hit rate' in r for r in recommendations)
        assert any('CPU utilization' in r for r in recommendations)
        assert any('memory utilization' in r for r in recommendations)
    
    @pytest.mark.asyncio
    async def test_run_all_benchmarks(self, benchmark):
        """Test running all benchmarks"""
        results = await benchmark.run_all_benchmarks()
        
        # Verify all results are present
        assert 'timestamp' in results
        assert 'api_latency' in results
        assert 'websocket_latency' in results
        assert 'phase4_latency' in results
        assert 'cache_hit_rate' in results
        assert 'resource_utilization' in results
        assert 'bottlenecks' in results
        assert 'recommendations' in results
        
        # Verify results have correct structure
        assert isinstance(results['api_latency'], dict)
        assert isinstance(results['websocket_latency'], dict)
        assert isinstance(results['phase4_latency'], dict)
        assert isinstance(results['bottlenecks'], list)
        assert isinstance(results['recommendations'], list)
    
    def test_print_report(self, benchmark, capsys):
        """Test report printing"""
        # Set up test data
        benchmark.results['api_latency'] = {'p95': 100}
        benchmark.results['websocket_latency'] = {'p95': 20}
        benchmark.results['phase4_latency'] = {'mean': 150}
        benchmark.results['cache_hit_rate'] = 0.85
        benchmark.results['bottlenecks'] = ['Test bottleneck']
        benchmark.results['recommendations'] = ['Test recommendation']
        
        benchmark.print_report()
        
        # Verify output
        captured = capsys.readouterr()
        assert 'PERFORMANCE BENCHMARK REPORT' in captured.out
        assert 'Performance Targets:' in captured.out
        assert 'Bottlenecks Identified:' in captured.out
        assert 'Optimization Recommendations:' in captured.out


class TestPerformanceBenchmarkIntegration:
    """Integration tests for performance benchmark"""
    
    @pytest.mark.asyncio
    async def test_full_benchmark_run(self):
        """Test full benchmark run in mock mode"""
        benchmark = PerformanceBenchmark(mock_mode=True)
        results = await benchmark.run_all_benchmarks()
        
        # Verify all benchmarks completed
        assert results['api_latency']['p50'] > 0
        assert results['websocket_latency']['p50'] > 0
        assert results['phase4_latency']['mean'] > 0
        assert results['resource_utilization']['cpu_percent'] >= 0
        
        # Verify report can be printed
        benchmark.print_report()
    
    def test_performance_targets_validation(self):
        """Test that performance targets are correctly validated"""
        benchmark = PerformanceBenchmark(mock_mode=True)
        
        # Set up passing results
        benchmark.results['api_latency'] = {'p95': 100}
        benchmark.results['websocket_latency'] = {'p95': 20}
        benchmark.results['phase4_latency'] = {'mean': 150}
        benchmark.results['cache_hit_rate'] = 0.85
        benchmark.results['resource_utilization'] = {
            'cpu_percent': 50,
            'memory_percent': 60
        }
        
        bottlenecks = benchmark.identify_bottlenecks()
        
        # Verify no bottlenecks for passing results
        assert len(bottlenecks) == 0
    
    def test_performance_targets_failure(self):
        """Test that performance targets correctly identify failures"""
        benchmark = PerformanceBenchmark(mock_mode=True)
        
        # Set up failing results
        benchmark.results['api_latency'] = {'p95': 600}
        benchmark.results['websocket_latency'] = {'p95': 60}
        benchmark.results['phase4_latency'] = {'mean': 250}
        benchmark.results['cache_hit_rate'] = 0.5
        benchmark.results['resource_utilization'] = {
            'cpu_percent': 85,
            'memory_percent': 85
        }
        
        bottlenecks = benchmark.identify_bottlenecks()
        recommendations = benchmark.generate_recommendations()
        
        # Verify bottlenecks and recommendations are generated
        assert len(bottlenecks) > 0
        assert len(recommendations) > 0

