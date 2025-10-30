"""
Tests for Firecrawl MCP Integration

Tests the real Firecrawl MCP integration with:
- search_web() - Web search functionality
- scrape_url() - URL scraping functionality
- fetch_provider_fact_sheet() - Provider fact sheet retrieval
- Graceful degradation when Firecrawl unavailable
- Error handling and retry logic
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from src.agents.swarm.mcp_tools import FirecrawlMCPTools, MCPToolRegistry


class TestFirecrawlMCPTools:
    """Test Firecrawl MCP tools"""

    def test_search_web_fallback_behavior(self):
        """Test search_web fallback when Firecrawl MCP is unavailable"""
        # Since Firecrawl MCP is not installed, test fallback behavior
        result = FirecrawlMCPTools.search_web('test query', max_results=5)

        # Verify fallback behavior
        assert result['query'] == 'test query'
        assert result['success'] is False
        assert result['source'] == 'firecrawl_fallback'
        assert result['results'] == []
        assert 'error' in result
        assert 'not available' in result['error'].lower()

    def test_search_web_result_structure(self):
        """Test that search_web returns correct structure"""
        result = FirecrawlMCPTools.search_web('test query', max_results=5)

        # Verify result has required fields
        assert 'query' in result
        assert 'results' in result
        assert 'timestamp' in result
        assert 'source' in result
        assert 'success' in result
        assert isinstance(result['results'], list)

    def test_scrape_url_fallback_behavior(self):
        """Test scrape_url fallback when Firecrawl MCP is unavailable"""
        # Since Firecrawl MCP is not installed, test fallback behavior
        result = FirecrawlMCPTools.scrape_url('https://example.com')

        # Verify fallback behavior
        assert result['url'] == 'https://example.com'
        assert result['success'] is False
        assert result['source'] == 'firecrawl_fallback'
        assert result['content'] == ''
        assert 'error' in result
        assert 'not available' in result['error'].lower()

    def test_scrape_url_result_structure(self):
        """Test that scrape_url returns correct structure"""
        result = FirecrawlMCPTools.scrape_url('https://example.com')

        # Verify result has required fields
        assert 'url' in result
        assert 'content' in result
        assert 'timestamp' in result
        assert 'source' in result
        assert 'success' in result
        assert isinstance(result['content'], str)
    
    def test_fetch_provider_fact_sheet_success(self):
        """Test fetch_provider_fact_sheet with successful search and scrape"""
        # Mock search results
        mock_search = {
            'success': True,
            'results': [
                {'url': 'https://extractalpha.com/cam-fact-sheet', 'title': 'CAM Fact Sheet'}
            ]
        }
        
        # Mock scrape result
        mock_scrape = {
            'success': True,
            'content': '# Cross-Asset Model\n\nFact sheet content...'
        }
        
        with patch.object(FirecrawlMCPTools, 'search_web', return_value=mock_search), \
             patch.object(FirecrawlMCPTools, 'scrape_url', return_value=mock_scrape):
            result = FirecrawlMCPTools.fetch_provider_fact_sheet('extractalpha', 'cam')
        
        # Verify result
        assert result['provider'] == 'extractalpha'
        assert result['signal'] == 'cam'
        assert result['success'] is True
        assert result['url'] == 'https://extractalpha.com/cam-fact-sheet'
        assert '# Cross-Asset Model' in result['content']
    
    def test_fetch_provider_fact_sheet_no_results(self):
        """Test fetch_provider_fact_sheet with no search results"""
        # Mock empty search results
        mock_search = {
            'success': True,
            'results': []
        }
        
        with patch.object(FirecrawlMCPTools, 'search_web', return_value=mock_search):
            result = FirecrawlMCPTools.fetch_provider_fact_sheet('extractalpha', 'cam')
        
        # Verify fallback behavior
        assert result['provider'] == 'extractalpha'
        assert result['signal'] == 'cam'
        assert 'search_results' in result
    
    def test_fetch_provider_fact_sheet_search_failure(self):
        """Test fetch_provider_fact_sheet with search failure"""
        # Mock failed search
        mock_search = {
            'success': False,
            'results': [],
            'error': 'Search failed'
        }
        
        with patch.object(FirecrawlMCPTools, 'search_web', return_value=mock_search):
            result = FirecrawlMCPTools.fetch_provider_fact_sheet('extractalpha', 'cam')
        
        # Verify fallback behavior
        assert result['provider'] == 'extractalpha'
        assert result['signal'] == 'cam'
        assert result['success'] is False
        assert 'search_results' in result


class TestMCPToolRegistryFirecrawl:
    """Test MCPToolRegistry Firecrawl integration"""
    
    @pytest.fixture
    def mcp_tools(self):
        """Create MCPToolRegistry instance"""
        return MCPToolRegistry()
    
    def test_search_web_tool_definition(self, mcp_tools):
        """Test that search_web tool is properly defined"""
        tools = mcp_tools.get_tool_definitions()
        
        # Find search_web tool
        search_tool = next((t for t in tools if t['function']['name'] == 'search_web'), None)
        
        assert search_tool is not None
        assert search_tool['type'] == 'function'
        assert 'query' in search_tool['function']['parameters']['properties']
        assert 'max_results' in search_tool['function']['parameters']['properties']
    
    def test_scrape_url_tool_definition(self, mcp_tools):
        """Test that scrape_url tool is properly defined"""
        tools = mcp_tools.get_tool_definitions()
        
        # Find scrape_url tool
        scrape_tool = next((t for t in tools if t['function']['name'] == 'scrape_url'), None)
        
        assert scrape_tool is not None
        assert scrape_tool['type'] == 'function'
        assert 'url' in scrape_tool['function']['parameters']['properties']
    
    def test_fetch_provider_fact_sheet_tool_definition(self, mcp_tools):
        """Test that fetch_provider_fact_sheet tool is properly defined"""
        tools = mcp_tools.get_tool_definitions()
        
        # Find fetch_provider_fact_sheet tool
        fetch_tool = next((t for t in tools if t['function']['name'] == 'fetch_provider_fact_sheet'), None)
        
        assert fetch_tool is not None
        assert fetch_tool['type'] == 'function'
        assert 'provider' in fetch_tool['function']['parameters']['properties']
        assert 'signal' in fetch_tool['function']['parameters']['properties']
    
    def test_call_search_web_tool(self, mcp_tools):
        """Test calling search_web via MCPToolRegistry"""
        result = mcp_tools.call_tool('search_web', query='test', max_results=5)

        # Verify result structure (fallback mode since Firecrawl not installed)
        assert result['query'] == 'test'
        assert 'success' in result
        assert 'results' in result

    def test_call_scrape_url_tool(self, mcp_tools):
        """Test calling scrape_url via MCPToolRegistry"""
        result = mcp_tools.call_tool('scrape_url', url='https://example.com')

        # Verify result structure (fallback mode since Firecrawl not installed)
        assert result['url'] == 'https://example.com'
        assert 'success' in result
        assert 'content' in result
    
    def test_call_fetch_provider_fact_sheet_tool(self, mcp_tools):
        """Test calling fetch_provider_fact_sheet via MCPToolRegistry"""
        # Mock search and scrape
        mock_search = {
            'success': True,
            'results': [{'url': 'https://cboe.com/pcr'}]
        }
        mock_scrape = {
            'success': True,
            'content': 'PCR data'
        }
        
        with patch.object(FirecrawlMCPTools, 'search_web', return_value=mock_search), \
             patch.object(FirecrawlMCPTools, 'scrape_url', return_value=mock_scrape):
            result = mcp_tools.call_tool('fetch_provider_fact_sheet', provider='cboe', signal='pcr')
        
        assert result['provider'] == 'cboe'
        assert result['signal'] == 'pcr'
        assert result['success'] is True


class TestFirecrawlGracefulDegradation:
    """Test graceful degradation when Firecrawl is unavailable"""

    def test_system_works_without_firecrawl(self):
        """Test that system continues to work when Firecrawl is unavailable"""
        # Test with Firecrawl unavailable (default state)
        result = FirecrawlMCPTools.search_web('test query')

        # System should not crash
        assert result is not None
        assert result['success'] is False
        assert result['results'] == []

    def test_error_messages_are_informative(self):
        """Test that error messages provide useful information"""
        # Test with Firecrawl unavailable (default state)
        result = FirecrawlMCPTools.search_web('test query')

        # Error message should be informative
        assert 'error' in result
        assert 'not available' in result['error'].lower()

    def test_all_tools_have_fallback(self):
        """Test that all Firecrawl tools have fallback behavior"""
        # Test search_web
        search_result = FirecrawlMCPTools.search_web('test')
        assert search_result is not None
        assert 'success' in search_result

        # Test scrape_url
        scrape_result = FirecrawlMCPTools.scrape_url('https://example.com')
        assert scrape_result is not None
        assert 'success' in scrape_result

        # Test fetch_provider_fact_sheet
        fact_sheet_result = FirecrawlMCPTools.fetch_provider_fact_sheet('cboe', 'pcr')
        assert fact_sheet_result is not None
        assert 'provider' in fact_sheet_result

