import React, { useState } from 'react';
import './InvestorReportViewer.css';

// TypeScript interfaces
interface Recommendation {
  rating: 'BUY' | 'SELL' | 'HOLD';
  conviction: 'HIGH' | 'MEDIUM' | 'LOW';
  price_target?: number;
  rationale: string[];
}

interface RiskFactor {
  risk: string;
  severity: 'HIGH' | 'MEDIUM' | 'LOW';
  probability: number;
  mitigation: string;
}

interface RiskAssessment {
  primary_risks: RiskFactor[] | string[];
}

interface FutureOutlook {
  projections?: {
    '3_month': string;
    '6_month': string;
    '12_month': string;
  };
  summary?: string;
  catalysts?: string[];
  scenarios?: {
    bull: string;
    base: string;
    bear: string;
  };
}

interface InvestorReport {
  executive_summary: string;
  recommendation: Recommendation;
  risk_assessment: RiskAssessment;
  future_outlook: FutureOutlook;
  next_steps: string[];
  metadata?: {
    generated_at: string;
    agent: string;
    total_insights: number;
    categories: Record<string, number>;
  };
}

interface InvestorReportViewerProps {
  report: InvestorReport;
}

export const InvestorReportViewer: React.FC<InvestorReportViewerProps> = ({ report }) => {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    new Set(['executive_summary', 'recommendation'])
  );

  const toggleSection = (section: string) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(section)) {
      newExpanded.delete(section);
    } else {
      newExpanded.add(section);
    }
    setExpandedSections(newExpanded);
  };

  const getRatingColor = (rating: string): string => {
    switch (rating) {
      case 'BUY': return '#10b981';
      case 'SELL': return '#ef4444';
      case 'HOLD': return '#f59e0b';
      default: return '#6b7280';
    }
  };

  const getConvictionBadge = (conviction: string): string => {
    switch (conviction) {
      case 'HIGH': return '🔥';
      case 'MEDIUM': return '⚡';
      case 'LOW': return '💡';
      default: return '📊';
    }
  };

  const getSeverityColor = (severity: string): string => {
    switch (severity) {
      case 'HIGH': return '#ef4444';
      case 'MEDIUM': return '#f59e0b';
      case 'LOW': return '#10b981';
      default: return '#6b7280';
    }
  };

  return (
    <div className="investor-report">
      <div className="report-header">
        <h1>📊 Investment Analysis Report</h1>
        {report.metadata && (
          <div className="report-metadata">
            <span>Generated: {new Date(report.metadata.generated_at).toLocaleString()}</span>
            <span>Total Insights: {report.metadata.total_insights}</span>
          </div>
        )}
      </div>

      {/* Executive Summary */}
      <section className={`report-section ${expandedSections.has('executive_summary') ? 'expanded' : ''}`}>
        <div className="section-header" onClick={() => toggleSection('executive_summary')}>
          <h2>📝 Executive Summary</h2>
          <span className="toggle-icon">{expandedSections.has('executive_summary') ? '▼' : '▶'}</span>
        </div>
        {expandedSections.has('executive_summary') && (
          <div className="section-content">
            <p className="executive-summary-text">{report.executive_summary}</p>
          </div>
        )}
      </section>

      {/* Investment Recommendation */}
      <section className={`report-section ${expandedSections.has('recommendation') ? 'expanded' : ''}`}>
        <div className="section-header" onClick={() => toggleSection('recommendation')}>
          <h2>🎯 Investment Recommendation</h2>
          <span className="toggle-icon">{expandedSections.has('recommendation') ? '▼' : '▶'}</span>
        </div>
        {expandedSections.has('recommendation') && (
          <div className="section-content">
            <div className="recommendation-card">
              <div className="rating-badge" style={{ backgroundColor: getRatingColor(report.recommendation.rating) }}>
                <span className="rating-text">{report.recommendation.rating}</span>
                <span className="conviction-badge">{getConvictionBadge(report.recommendation.conviction)}</span>
              </div>
              <div className="recommendation-details">
                <div className="conviction-level">
                  Conviction: <strong>{report.recommendation.conviction}</strong>
                </div>
                {report.recommendation.price_target && (
                  <div className="price-target">
                    Price Target: <strong>${report.recommendation.price_target.toFixed(2)}</strong>
                  </div>
                )}
                <div className="rationale">
                  <h4>Rationale:</h4>
                  <ul>
                    {report.recommendation.rationale.map((reason, idx) => (
                      <li key={idx}>{reason}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </div>
        )}
      </section>

      {/* Risk Assessment */}
      <section className={`report-section ${expandedSections.has('risk_assessment') ? 'expanded' : ''}`}>
        <div className="section-header" onClick={() => toggleSection('risk_assessment')}>
          <h2>⚠️ Risk Assessment</h2>
          <span className="toggle-icon">{expandedSections.has('risk_assessment') ? '▼' : '▶'}</span>
        </div>
        {expandedSections.has('risk_assessment') && (
          <div className="section-content">
            <div className="risk-list">
              {report.risk_assessment.primary_risks.map((risk, idx) => {
                if (typeof risk === 'string') {
                  return (
                    <div key={idx} className="risk-item">
                      <div className="risk-header">
                        <span className="risk-number">{idx + 1}</span>
                        <span className="risk-text">{risk}</span>
                      </div>
                    </div>
                  );
                } else {
                  return (
                    <div key={idx} className="risk-item">
                      <div className="risk-header">
                        <span className="risk-number">{idx + 1}</span>
                        <span className="risk-text">{risk.risk}</span>
                        <span 
                          className="severity-badge" 
                          style={{ backgroundColor: getSeverityColor(risk.severity) }}
                        >
                          {risk.severity}
                        </span>
                      </div>
                      {risk.probability !== undefined && (
                        <div className="risk-probability">
                          Probability: {(risk.probability * 100).toFixed(0)}%
                        </div>
                      )}
                      {risk.mitigation && (
                        <div className="risk-mitigation">
                          <strong>Mitigation:</strong> {risk.mitigation}
                        </div>
                      )}
                    </div>
                  );
                }
              })}
            </div>
          </div>
        )}
      </section>

      {/* Future Outlook */}
      <section className={`report-section ${expandedSections.has('future_outlook') ? 'expanded' : ''}`}>
        <div className="section-header" onClick={() => toggleSection('future_outlook')}>
          <h2>🔮 Future Outlook</h2>
          <span className="toggle-icon">{expandedSections.has('future_outlook') ? '▼' : '▶'}</span>
        </div>
        {expandedSections.has('future_outlook') && (
          <div className="section-content">
            {report.future_outlook.summary && (
              <p className="outlook-summary">{report.future_outlook.summary}</p>
            )}
            
            {report.future_outlook.projections && (
              <div className="projections">
                <h4>Projections:</h4>
                <div className="projection-grid">
                  <div className="projection-item">
                    <span className="projection-label">3 Month</span>
                    <span className="projection-value">{report.future_outlook.projections['3_month']}</span>
                  </div>
                  <div className="projection-item">
                    <span className="projection-label">6 Month</span>
                    <span className="projection-value">{report.future_outlook.projections['6_month']}</span>
                  </div>
                  <div className="projection-item">
                    <span className="projection-label">12 Month</span>
                    <span className="projection-value">{report.future_outlook.projections['12_month']}</span>
                  </div>
                </div>
              </div>
            )}

            {report.future_outlook.catalysts && report.future_outlook.catalysts.length > 0 && (
              <div className="catalysts">
                <h4>Key Catalysts:</h4>
                <ul>
                  {report.future_outlook.catalysts.map((catalyst, idx) => (
                    <li key={idx}>{catalyst}</li>
                  ))}
                </ul>
              </div>
            )}

            {report.future_outlook.scenarios && (
              <div className="scenarios">
                <h4>Scenarios:</h4>
                <div className="scenario-grid">
                  <div className="scenario-item bull">
                    <h5>🐂 Bull Case</h5>
                    <p>{report.future_outlook.scenarios.bull}</p>
                  </div>
                  <div className="scenario-item base">
                    <h5>📊 Base Case</h5>
                    <p>{report.future_outlook.scenarios.base}</p>
                  </div>
                  <div className="scenario-item bear">
                    <h5>🐻 Bear Case</h5>
                    <p>{report.future_outlook.scenarios.bear}</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </section>

      {/* Actionable Next Steps */}
      <section className={`report-section ${expandedSections.has('next_steps') ? 'expanded' : ''}`}>
        <div className="section-header" onClick={() => toggleSection('next_steps')}>
          <h2>✅ Actionable Next Steps</h2>
          <span className="toggle-icon">{expandedSections.has('next_steps') ? '▼' : '▶'}</span>
        </div>
        {expandedSections.has('next_steps') && (
          <div className="section-content">
            <ol className="next-steps-list">
              {report.next_steps.map((step, idx) => (
                <li key={idx} className="next-step-item">
                  <span className="step-number">{idx + 1}</span>
                  <span className="step-text">{step}</span>
                </li>
              ))}
            </ol>
          </div>
        )}
      </section>
    </div>
  );
};

