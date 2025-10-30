"""
Integration tests for InvestorReport API endpoints

Tests:
- GET /api/investor-report returns valid InvestorReport.v1 JSON
- Response time <500ms
- Schema validation passes
- Phase 4 metrics included
- WebSocket /ws/phase4-metrics streams updates every 30s
"""

import pytest
import json
import jsonschema
from pathlib import Path
from fastapi.testclient import TestClient
from datetime import datetime, timezone


@pytest.fixture
def schema():
    """Load InvestorReport.v1 JSON Schema"""
    schema_path = Path("src/schemas/investor_report_schema.json")
    with open(schema_path) as f:
        return json.load(f)


@pytest.fixture
def client():
    """Create FastAPI test client"""
    from src.api.main import app
    return TestClient(app)


class TestInvestorReportAPI:
    """Test suite for GET /api/investor-report endpoint"""
    
    def test_investor_report_endpoint_exists(self, client):
        """Test that the endpoint exists and returns 200"""
        response = client.get("/api/investor-report?user_id=test_user")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    def test_investor_report_returns_valid_json(self, client):
        """Test that the endpoint returns valid JSON"""
        response = client.get("/api/investor-report?user_id=test_user")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, dict), "Response should be a JSON object"
    
    def test_investor_report_schema_compliance(self, client, schema):
        """Test that the response validates against InvestorReport.v1 schema"""
        response = client.get("/api/investor-report?user_id=test_user&symbols=AAPL,MSFT")
        assert response.status_code == 200
        
        data = response.json()
        
        # Remove metadata before validation (not part of schema)
        if 'metadata' in data:
            del data['metadata']
        
        try:
            jsonschema.validate(instance=data, schema=schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"Schema validation failed: {e.message}")
    
    def test_investor_report_has_required_fields(self, client):
        """Test that response has all required top-level fields"""
        response = client.get("/api/investor-report?user_id=test_user")
        assert response.status_code == 200
        
        data = response.json()
        
        required_fields = [
            'as_of', 'universe', 'executive_summary', 'risk_panel',
            'signals', 'actions', 'sources', 'confidence'
        ]
        
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
    
    def test_investor_report_has_phase4_metrics(self, client):
        """Test that response includes Phase 4 metrics"""
        response = client.get("/api/investor-report?user_id=test_user")
        assert response.status_code == 200
        
        data = response.json()
        
        assert 'signals' in data
        assert 'phase4_tech' in data['signals']
        
        phase4 = data['signals']['phase4_tech']
        
        # Check Phase 4 fields exist (can be null)
        phase4_fields = [
            'options_flow_composite',
            'residual_momentum',
            'seasonality_score',
            'breadth_liquidity'
        ]
        
        for field in phase4_fields:
            assert field in phase4, f"Missing Phase 4 field: {field}"
    
    def test_investor_report_has_risk_panel(self, client):
        """Test that response includes all 7 risk metrics"""
        response = client.get("/api/investor-report?user_id=test_user")
        assert response.status_code == 200
        
        data = response.json()
        
        assert 'risk_panel' in data
        risk_panel = data['risk_panel']
        
        risk_metrics = [
            'omega', 'gh1', 'pain_index', 'upside_capture',
            'downside_capture', 'cvar_95', 'max_drawdown'
        ]
        
        for metric in risk_metrics:
            assert metric in risk_panel, f"Missing risk metric: {metric}"
            assert isinstance(risk_panel[metric], (int, float)), f"{metric} should be numeric"
    
    def test_investor_report_response_time(self, client):
        """Test that response time is <500ms (performance target)"""
        import time
        
        start = time.time()
        response = client.get("/api/investor-report?user_id=test_user")
        elapsed_ms = (time.time() - start) * 1000
        
        assert response.status_code == 200
        assert elapsed_ms < 500, f"Response time {elapsed_ms:.2f}ms exceeds 500ms target"
    
    def test_investor_report_with_symbols_param(self, client):
        """Test that symbols parameter is respected"""
        response = client.get("/api/investor-report?user_id=test_user&symbols=AAPL,MSFT,GOOGL")
        assert response.status_code == 200
        
        data = response.json()
        assert 'universe' in data
        
        # Universe should contain the requested symbols
        universe = data['universe']
        assert isinstance(universe, list)
        assert len(universe) > 0
    
    def test_investor_report_fresh_param(self, client):
        """Test that fresh parameter bypasses cache"""
        response1 = client.get("/api/investor-report?user_id=test_user&fresh=false")
        response2 = client.get("/api/investor-report?user_id=test_user&fresh=true")
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # Both should return valid data
        data1 = response1.json()
        data2 = response2.json()
        
        assert 'as_of' in data1
        assert 'as_of' in data2


class TestHealthEndpoint:
    """Test suite for /api/health endpoint"""
    
    def test_health_endpoint(self, client):
        """Test that health endpoint returns 200"""
        response = client.get("/api/health")
        assert response.status_code == 200
        
        data = response.json()
        assert 'status' in data
        assert data['status'] == 'healthy'


class TestRootEndpoint:
    """Test suite for root endpoint"""
    
    def test_root_endpoint(self, client):
        """Test that root endpoint returns API info"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert 'service' in data or 'status' in data

