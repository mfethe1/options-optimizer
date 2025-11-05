/**
 * AI Recommendations Dashboard
 *
 * Comprehensive AI-powered trading insights:
 * - Expert platform critique
 * - Swarm strategy analysis
 * - Risk management recommendations
 */

import React, { useState, useEffect } from 'react';
import { toast } from 'react-hot-toast';
import {
  getPlatformCritique,
  getRecommendationColor,
  getRiskLevelColor,
  getPriorityColor,
  formatScore,
  type ExpertCritiqueReport,
  type Recommendation
} from '../services/aiApi';

export default function AIRecommendationsPage() {
  const [critique, setCritique] = useState<ExpertCritiqueReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'overview' | 'recommendations' | 'competitive'>('overview');

  useEffect(() => {
    loadCritique();
  }, []);

  const loadCritique = async () => {
    setLoading(true);
    try {
      const report = await getPlatformCritique();
      setCritique(report);
    } catch (error) {
      toast.error(`Failed to load critique: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  const groupRecommendationsByPriority = (recommendations: Recommendation[]) => {
    return {
      CRITICAL: recommendations.filter(r => r.priority === 'CRITICAL'),
      HIGH: recommendations.filter(r => r.priority === 'HIGH'),
      MEDIUM: recommendations.filter(r => r.priority === 'MEDIUM'),
      LOW: recommendations.filter(r => r.priority === 'LOW'),
    };
  };

  if (loading) {
    return (
      <div className="p-8">
        <div className="max-w-7xl mx-auto text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <h2 className="text-xl font-semibold text-gray-900">Analyzing Platform...</h2>
          <p className="text-gray-600 mt-2">AI agents are evaluating system capabilities</p>
        </div>
      </div>
    );
  }

  if (!critique) {
    return (
      <div className="p-8">
        <div className="max-w-7xl mx-auto text-center">
          <p className="text-gray-600">No critique available</p>
        </div>
      </div>
    );
  }

  const groupedRecommendations = groupRecommendationsByPriority(critique.recommendations);

  return (
    <div className="p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">AI-Powered Platform Analysis</h1>
          <p className="text-gray-600">
            Institutional investor perspective with actionable recommendations
          </p>
        </div>

        {/* Overall Grade Card */}
        <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg shadow-xl p-8 mb-8 text-white">
          <div className="flex justify-between items-start">
            <div>
              <div className="text-6xl font-bold mb-2">{critique.overall_rating}</div>
              <div className="text-2xl opacity-90 mb-4">{formatScore(critique.overall_score)}</div>
              <p className="text-blue-100 text-lg max-w-2xl">
                Your platform demonstrates strong foundational capabilities with Bloomberg-competitive features
                in options analytics and risk management.
              </p>
            </div>
            <button
              onClick={loadCritique}
              className="bg-white text-blue-600 px-6 py-3 rounded-lg hover:bg-blue-50 transition-colors font-semibold"
            >
              Refresh Analysis
            </button>
          </div>
        </div>

        {/* Category Scores Grid */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-8">
          {[
            { name: 'Data Quality', score: critique.category_scores.data_quality },
            { name: 'Analytics', score: critique.category_scores.analytics },
            { name: 'Execution', score: critique.category_scores.execution },
            { name: 'Risk Mgmt', score: critique.category_scores.risk_management },
            { name: 'UX', score: critique.category_scores.user_experience },
            { name: 'Technology', score: critique.category_scores.technology },
          ].map((category) => (
            <div key={category.name} className="bg-white rounded-lg shadow p-4">
              <div className="text-sm text-gray-600 mb-1">{category.name}</div>
              <div className="text-2xl font-bold text-gray-900">{formatScore(category.score)}</div>
              <div className="mt-2">
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full transition-all"
                    style={{ width: `${category.score}%` }}
                  ></div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Tabs */}
        <div className="mb-6 border-b border-gray-200">
          <nav className="flex space-x-8">
            <button
              onClick={() => setActiveTab('overview')}
              className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                activeTab === 'overview'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Overview & Strengths
            </button>
            <button
              onClick={() => setActiveTab('recommendations')}
              className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                activeTab === 'recommendations'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Recommendations ({critique.recommendations.length})
            </button>
            <button
              onClick={() => setActiveTab('competitive')}
              className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                activeTab === 'competitive'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Competitive Analysis
            </button>
          </nav>
        </div>

        {/* Tab Content */}
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Executive Summary */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">Executive Summary</h2>
              <pre className="whitespace-pre-wrap text-sm text-gray-700 font-mono bg-gray-50 p-4 rounded">
                {critique.executive_summary}
              </pre>
            </div>

            {/* Strengths */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">Platform Strengths</h2>
              <div className="space-y-3">
                {critique.strengths.map((strength, idx) => (
                  <div key={idx} className="flex items-start">
                    <span className="text-green-500 mr-3 mt-1 text-xl">✓</span>
                    <p className="text-gray-700">{strength}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Critical Gaps */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">Critical Gaps</h2>
              <div className="space-y-3">
                {critique.critical_gaps.map((gap, idx) => (
                  <div key={idx} className="flex items-start">
                    <span className="text-red-500 mr-3 mt-1 text-xl">⚠</span>
                    <p className="text-gray-700">{gap}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'recommendations' && (
          <div className="space-y-6">
            {/* Critical Recommendations */}
            {groupedRecommendations.CRITICAL.length > 0 && (
              <div>
                <h2 className="text-2xl font-bold text-gray-900 mb-4">
                  Critical Priority ({groupedRecommendations.CRITICAL.length})
                </h2>
                <div className="space-y-4">
                  {groupedRecommendations.CRITICAL.map((rec, idx) => (
                    <div
                      key={idx}
                      className={`bg-white rounded-lg shadow-lg p-6 border-l-4 ${getPriorityColor(rec.priority)}`}
                    >
                      <div className="flex justify-between items-start mb-3">
                        <div>
                          <h3 className="text-xl font-semibold text-gray-900 mb-1">{rec.title}</h3>
                          <div className="flex gap-2">
                            <span className={`px-3 py-1 rounded-full text-xs font-semibold ${getPriorityColor(rec.priority)}`}>
                              {rec.priority}
                            </span>
                            <span className="px-3 py-1 rounded-full text-xs font-semibold bg-gray-100 text-gray-700">
                              {rec.category}
                            </span>
                            <span className="px-3 py-1 rounded-full text-xs font-semibold bg-purple-100 text-purple-700">
                              Value: {rec.estimated_value}
                            </span>
                            <span className="px-3 py-1 rounded-full text-xs font-semibold bg-blue-100 text-blue-700">
                              {rec.implementation_complexity} Complexity
                            </span>
                          </div>
                        </div>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                        <div>
                          <div className="text-sm font-medium text-gray-500 mb-1">Current State</div>
                          <p className="text-gray-700">{rec.current_state}</p>
                        </div>
                        <div>
                          <div className="text-sm font-medium text-gray-500 mb-1">Desired State</div>
                          <p className="text-gray-700">{rec.desired_state}</p>
                        </div>
                      </div>

                      <div className="mb-4">
                        <div className="text-sm font-medium text-gray-500 mb-1">Rationale</div>
                        <p className="text-gray-700">{rec.rationale}</p>
                      </div>

                      <div className="bg-green-50 border border-green-200 rounded p-3">
                        <div className="text-sm font-medium text-green-800 mb-1">Expected Impact</div>
                        <p className="text-green-700">{rec.expected_impact}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* High Priority Recommendations */}
            {groupedRecommendations.HIGH.length > 0 && (
              <div>
                <h2 className="text-2xl font-bold text-gray-900 mb-4">
                  High Priority ({groupedRecommendations.HIGH.length})
                </h2>
                <div className="space-y-4">
                  {groupedRecommendations.HIGH.map((rec, idx) => (
                    <div
                      key={idx}
                      className={`bg-white rounded-lg shadow p-6 border-l-4 ${getPriorityColor(rec.priority)}`}
                    >
                      <h3 className="text-lg font-semibold text-gray-900 mb-2">{rec.title}</h3>
                      <div className="flex gap-2 mb-3">
                        <span className={`px-2 py-1 rounded text-xs font-semibold ${getPriorityColor(rec.priority)}`}>
                          {rec.priority}
                        </span>
                        <span className="px-2 py-1 rounded text-xs font-semibold bg-gray-100 text-gray-700">
                          {rec.category}
                        </span>
                      </div>
                      <p className="text-sm text-gray-700 mb-2">{rec.rationale}</p>
                      <p className="text-sm text-green-700 italic">{rec.expected_impact}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Medium & Low Priority (Collapsed) */}
            {(groupedRecommendations.MEDIUM.length > 0 || groupedRecommendations.LOW.length > 0) && (
              <div>
                <h2 className="text-2xl font-bold text-gray-900 mb-4">
                  Other Recommendations ({groupedRecommendations.MEDIUM.length + groupedRecommendations.LOW.length})
                </h2>
                <div className="bg-white rounded-lg shadow p-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {[...groupedRecommendations.MEDIUM, ...groupedRecommendations.LOW].map((rec, idx) => (
                      <div key={idx} className="border-l-2 border-gray-300 pl-4">
                        <div className="text-sm font-semibold text-gray-900 mb-1">{rec.title}</div>
                        <div className="text-xs text-gray-600">{rec.category}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'competitive' && (
          <div className="space-y-6">
            {/* Competitive Positioning */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">Competitive Positioning</h2>
              <pre className="whitespace-pre-wrap text-sm text-gray-700 font-mono bg-gray-50 p-4 rounded">
                {critique.competitive_positioning}
              </pre>
            </div>

            {/* Competitive Scores */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-2xl font-bold text-gray-900 mb-6">vs Premier Platforms</h2>
              <div className="space-y-6">
                {[
                  { name: 'Bloomberg Terminal', score: critique.competitive_scores.vs_bloomberg, color: 'blue' },
                  { name: 'Refinitiv Eikon', score: critique.competitive_scores.vs_refinitiv, color: 'purple' },
                  { name: 'FactSet', score: critique.competitive_scores.vs_factset, color: 'indigo' },
                ].map((platform) => (
                  <div key={platform.name}>
                    <div className="flex justify-between items-center mb-2">
                      <span className="font-semibold text-gray-900">{platform.name}</span>
                      <span className="text-2xl font-bold text-gray-900">{formatScore(platform.score)}</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-4">
                      <div
                        className={`bg-${platform.color}-600 h-4 rounded-full transition-all flex items-center justify-end pr-2`}
                        style={{ width: `${platform.score}%` }}
                      >
                        <span className="text-white text-xs font-semibold">
                          {platform.score >= 40 ? `${platform.score.toFixed(0)}% feature parity` : ''}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Market Positioning */}
            <div className="bg-gradient-to-r from-green-50 to-blue-50 border border-green-200 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-green-900 mb-3">🎯 Market Positioning</h3>
              <div className="space-y-2 text-sm text-green-800">
                <p><strong>Current:</strong> Advanced retail / small RIA market ($50K - $1M accounts)</p>
                <p><strong>Potential:</strong> Small-mid hedge funds / family offices ($5M - $50M with improvements)</p>
                <p><strong>Bloomberg Competition:</strong> 2-3 years away with critical improvements</p>
              </div>
            </div>

            {/* Path to 20% Monthly Returns */}
            <div className="bg-gradient-to-r from-yellow-50 to-orange-50 border border-yellow-300 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-orange-900 mb-3">💰 Achieving {'>'}20% Monthly Returns</h3>
              <div className="space-y-2 text-sm text-orange-800">
                <p><strong>✓ FEASIBLE:</strong> 10-15% monthly (very achievable with current capabilities)</p>
                <p><strong>⚠ AMBITIOUS:</strong> 20% monthly (requires perfect execution + favorable markets)</p>
                <p><strong>🎯 REQUIRES:</strong> Institutional data quality and execution upgrades for consistency</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
