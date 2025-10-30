import React from 'react';
import './InvestorReportSynopsis.css';
import { InvestorReport, formatTimestamp } from '../types/investor-report';

interface Props {
  report: InvestorReport;
}

function titleCase(s: string): string {
  if (!s) return s;
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function groupActions(actions: InvestorReport['actions']) {
  const buckets: Record<'buy'|'sell'|'hold'|'watch', {ticker: string, sizing: string, risk_controls: string}[]> = {
    buy: [], sell: [], hold: [], watch: []
  };
  actions.forEach(a => {
    const key = a.action as 'buy'|'sell'|'hold'|'watch';
    buckets[key].push({ ticker: a.ticker, sizing: a.sizing, risk_controls: a.risk_controls });
  });
  return buckets;
}

export const InvestorReportSynopsis: React.FC<Props> = ({ report }) => {
  const { executive_summary, actions, confidence, signals, risk_panel, as_of, universe } = report;
  const buckets = groupActions(actions || []);

  // Build short narrative snippets
  const regime = signals?.regime ? `${signals.regime}` : undefined;
  const sentiment = typeof signals?.sentiment?.level === 'number'
    ? (signals.sentiment.level > 0.3 ? 'bullish' : signals.sentiment.level < -0.3 ? 'bearish' : 'neutral')
    : undefined;
  const riskBrief = (() => {
    if (!risk_panel) return undefined;
    const dd = risk_panel.max_drawdown;
    const cvar = risk_panel.cvar_95;
    return `Max drawdown ${Math.round(dd*100)/100}%, CVaR@95 ${Math.round(cvar*100)/100}%`;
  })();

  const topPicks = executive_summary?.top_picks || [];

  return (
    <article className="synopsis">
      <header className="synopsis__header">
        <h1>Portfolio Synopsis</h1>
        <div className="synopsis__meta">
          <span>As of {formatTimestamp(as_of)}</span>
          {universe?.length ? <span>Universe: {universe.join(', ')}</span> : null}
          <span>Confidence: {(confidence?.overall ?? 0)*100 | 0}%</span>
        </div>
      </header>

      {/* 1. Recommendations */}
      <section className="synopsis__section">
        <h2>Recommendations</h2>
        {topPicks.length > 0 ? (
          <ul className="synopsis__list">
            {topPicks.map((p, idx) => (
              <li key={idx}>
                <strong>{p.ticker}</strong> — {p.rationale}
                {p.expected_horizon_days ? (
                  <> (horizon ~{p.expected_horizon_days} days)</>
                ) : null}
              </li>
            ))}
          </ul>
        ) : (
          <p>No specific top picks highlighted in this run.</p>
        )}
      </section>

      {/* 2. What to buy/sell/hold */}
      <section className="synopsis__section">
        <h2>What to buy / sell / hold</h2>
        <div className="synopsis__actions">
          {(['buy','hold','sell','watch'] as const).map(kind => (
            buckets[kind]?.length ? (
              <div key={kind} className={`synopsis__action-bucket synopsis__action-bucket--${kind}`}>
                <h3>{titleCase(kind)}</h3>
                <ul>
                  {buckets[kind].map((a, i) => (
                    <li key={i}><strong>{a.ticker}</strong>
  <span className="badge badge--sizing" aria-label={`Sizing: ${a.sizing}`}>{a.sizing}</span>
  {a.risk_controls ? (<span className="badge badge--risk" aria-label={`Risk controls: ${a.risk_controls}`}>{a.risk_controls}</span>) : null}
</li>
                  ))}
                </ul>
              </div>
            ) : null
          ))}
        </div>
      </section>

      {/* 3. Summary and outlook */}
      <section className="synopsis__section">
        <h2>Summary and outlook</h2>
        <div className="synopsis__prose">
          {executive_summary?.thesis ? (
            <p>{executive_summary.thesis}</p>
          ) : (
            <p>We synthesized the latest cross-signal read on your portfolio and market conditions.</p>
          )}
          <ul>
            {regime && <li>Market regime: <strong>{regime}</strong></li>}
            {sentiment && <li>Sentiment skew: <strong>{sentiment}</strong></li>}
            {signals?.ml_alpha?.score != null && (
              <li>ML-Alpha: {signals.ml_alpha.score.toFixed(2)}</li>
            )}
            {riskBrief && <li>Risk posture: {riskBrief}</li>}
          </ul>
          {executive_summary?.key_risks?.length ? (
            <>
              <h4>Key risks</h4>
              <ul>
                {executive_summary.key_risks.map((r, idx) => <li key={idx}>{r}</li>)}
              </ul>
            </>
          ) : null}
        </div>
      </section>
    </article>
  );
};

export default InvestorReportSynopsis;

