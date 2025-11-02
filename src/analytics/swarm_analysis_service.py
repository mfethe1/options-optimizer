"""
AI Swarm Analysis Service

Multi-agent consensus system for strategy evaluation and recommendation.
Uses swarm intelligence to analyze backtest results and make institutional-grade
trading decisions with risk management guardrails.
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class BacktestResult:
    """Backtest result for a strategy"""
    strategy_name: str
    symbol: str
    timeframe: str
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_trades: int
    kelly_criterion: float
    var_95: float
    expected_value: float


@dataclass
class AgentAnalysis:
    """Analysis from a single agent"""
    agent_name: str
    agent_type: str  # conservative, aggressive, balanced, risk_manager, quant
    score: float  # 0-100
    recommendation: str  # STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
    confidence: float  # 0-1
    reasoning: str
    risk_concerns: List[str]
    opportunity_highlights: List[str]


@dataclass
class SwarmConsensus:
    """Consensus from swarm analysis"""
    strategy: str
    symbol: str
    overall_score: float  # 0-100
    consensus_recommendation: str
    consensus_confidence: float
    expected_value: float
    risk_adjusted_return: float
    suggested_position_size: float  # Percentage of portfolio
    stop_loss: float
    take_profit: float
    max_loss_per_trade: float
    agent_votes: Dict[str, int]  # recommendation -> count
    agent_analyses: List[AgentAnalysis]
    risk_warnings: List[str]
    go_decision: bool  # Final go/no-go decision
    reasoning_summary: str


# ============================================================================
# Agent Types
# ============================================================================

class TradingAgent:
    """Base trading agent"""

    def __init__(self, name: str, agent_type: str, risk_tolerance: float):
        self.name = name
        self.agent_type = agent_type
        self.risk_tolerance = risk_tolerance  # 0-1

    async def analyze(self, backtest: BacktestResult) -> AgentAnalysis:
        """Analyze backtest result and provide recommendation"""
        raise NotImplementedError


class ConservativeAgent(TradingAgent):
    """Conservative agent focused on capital preservation"""

    def __init__(self):
        super().__init__("Conservative Carl", "conservative", 0.3)

    async def analyze(self, backtest: BacktestResult) -> AgentAnalysis:
        score = 0
        risk_concerns = []
        opportunities = []

        # Prioritize low drawdown
        if backtest.max_drawdown < 10:
            score += 30
            opportunities.append(f"Excellent drawdown control at {backtest.max_drawdown:.1f}%")
        elif backtest.max_drawdown < 20:
            score += 15
        else:
            risk_concerns.append(f"High drawdown risk: {backtest.max_drawdown:.1f}%")

        # Require high Sharpe ratio
        if backtest.sharpe_ratio > 2.0:
            score += 25
            opportunities.append(f"Strong risk-adjusted returns (Sharpe: {backtest.sharpe_ratio:.2f})")
        elif backtest.sharpe_ratio > 1.0:
            score += 10
        else:
            risk_concerns.append(f"Insufficient risk-adjusted returns (Sharpe: {backtest.sharpe_ratio:.2f})")

        # Require high win rate
        if backtest.win_rate > 0.60:
            score += 25
            opportunities.append(f"High win consistency ({backtest.win_rate*100:.1f}%)")
        elif backtest.win_rate > 0.50:
            score += 10
        else:
            risk_concerns.append(f"Low win rate ({backtest.win_rate*100:.1f}%)")

        # Check profit factor
        if backtest.profit_factor > 2.0:
            score += 20
        elif backtest.profit_factor > 1.5:
            score += 10
        else:
            risk_concerns.append(f"Weak profit factor: {backtest.profit_factor:.2f}")

        # Determine recommendation
        if score >= 80:
            rec = "BUY"
            confidence = 0.9
        elif score >= 60:
            rec = "HOLD"
            confidence = 0.6
        else:
            rec = "SELL"
            confidence = 0.8

        reasoning = f"Conservative analysis focused on capital preservation. Score: {score}/100. "
        reasoning += f"Strategy shows {'acceptable' if score >= 60 else 'concerning'} risk metrics."

        return AgentAnalysis(
            agent_name=self.name,
            agent_type=self.agent_type,
            score=score,
            recommendation=rec,
            confidence=confidence,
            reasoning=reasoning,
            risk_concerns=risk_concerns,
            opportunity_highlights=opportunities
        )


class AggressiveAgent(TradingAgent):
    """Aggressive agent focused on maximum returns"""

    def __init__(self):
        super().__init__("Aggressive Alex", "aggressive", 0.8)

    async def analyze(self, backtest: BacktestResult) -> AgentAnalysis:
        score = 0
        risk_concerns = []
        opportunities = []

        # Prioritize total return
        if backtest.total_return > 50:
            score += 40
            opportunities.append(f"Exceptional returns: {backtest.total_return:.1f}%")
        elif backtest.total_return > 30:
            score += 25
            opportunities.append(f"Strong returns: {backtest.total_return:.1f}%")
        elif backtest.total_return > 20:
            score += 15
        else:
            risk_concerns.append(f"Insufficient returns: {backtest.total_return:.1f}%")

        # Expected value
        if backtest.expected_value > 3.0:
            score += 30
            opportunities.append(f"High EV per trade: ${backtest.expected_value:.2f}")
        elif backtest.expected_value > 1.5:
            score += 15
        else:
            risk_concerns.append(f"Low expected value: ${backtest.expected_value:.2f}")

        # Kelly criterion (aggressive position sizing)
        if backtest.kelly_criterion > 0.15:
            score += 20
            opportunities.append(f"Strong Kelly signal: {backtest.kelly_criterion*100:.1f}%")
        elif backtest.kelly_criterion > 0.08:
            score += 10

        # Profit factor
        if backtest.profit_factor > 2.5:
            score += 10
        elif backtest.profit_factor < 1.3:
            risk_concerns.append(f"Weak profit factor: {backtest.profit_factor:.2f}")

        # Determine recommendation
        if score >= 70:
            rec = "STRONG_BUY"
            confidence = 0.85
        elif score >= 50:
            rec = "BUY"
            confidence = 0.7
        else:
            rec = "HOLD"
            confidence = 0.5

        reasoning = f"Aggressive growth analysis. Score: {score}/100. "
        reasoning += f"Strategy shows {'excellent' if score >= 70 else 'moderate'} profit potential."

        return AgentAnalysis(
            agent_name=self.name,
            agent_type=self.agent_type,
            score=score,
            recommendation=rec,
            confidence=confidence,
            reasoning=reasoning,
            risk_concerns=risk_concerns,
            opportunity_highlights=opportunities
        )


class BalancedAgent(TradingAgent):
    """Balanced agent evaluating both risk and return"""

    def __init__(self):
        super().__init__("Balanced Betty", "balanced", 0.5)

    async def analyze(self, backtest: BacktestResult) -> AgentAnalysis:
        score = 0
        risk_concerns = []
        opportunities = []

        # Balance return and risk
        risk_adjusted_return = backtest.total_return / (backtest.max_drawdown + 1)
        if risk_adjusted_return > 2.0:
            score += 30
            opportunities.append(f"Excellent risk/return ratio: {risk_adjusted_return:.2f}")
        elif risk_adjusted_return > 1.0:
            score += 15
        else:
            risk_concerns.append(f"Poor risk/return balance: {risk_adjusted_return:.2f}")

        # Sharpe and Sortino
        avg_ratio = (backtest.sharpe_ratio + backtest.sortino_ratio) / 2
        if avg_ratio > 1.5:
            score += 25
            opportunities.append(f"Strong risk-adjusted metrics (Avg: {avg_ratio:.2f})")
        elif avg_ratio > 1.0:
            score += 15
        else:
            risk_concerns.append(f"Weak risk-adjusted performance: {avg_ratio:.2f}")

        # Win rate and profit factor balance
        if backtest.win_rate > 0.55 and backtest.profit_factor > 1.8:
            score += 25
            opportunities.append("Balanced win rate and profit factor")
        elif backtest.win_rate > 0.50 and backtest.profit_factor > 1.5:
            score += 15

        # Kelly criterion (moderate sizing)
        if 0.10 <= backtest.kelly_criterion <= 0.25:
            score += 20
            opportunities.append(f"Optimal position sizing range: {backtest.kelly_criterion*100:.1f}%")
        elif backtest.kelly_criterion < 0.05:
            risk_concerns.append("Position size too conservative")
        elif backtest.kelly_criterion > 0.30:
            risk_concerns.append("Position size may be too aggressive")

        # Determine recommendation
        if score >= 75:
            rec = "BUY"
            confidence = 0.85
        elif score >= 50:
            rec = "HOLD"
            confidence = 0.65
        else:
            rec = "SELL"
            confidence = 0.7

        reasoning = f"Balanced risk/return analysis. Score: {score}/100. "
        reasoning += f"Strategy shows {'well-balanced' if score >= 60 else 'imbalanced'} characteristics."

        return AgentAnalysis(
            agent_name=self.name,
            agent_type=self.agent_type,
            score=score,
            recommendation=rec,
            confidence=confidence,
            reasoning=reasoning,
            risk_concerns=risk_concerns,
            opportunity_highlights=opportunities
        )


class RiskManagerAgent(TradingAgent):
    """Risk manager agent enforcing strict risk controls"""

    def __init__(self):
        super().__init__("Risk Manager Rita", "risk_manager", 0.2)

    async def analyze(self, backtest: BacktestResult) -> AgentAnalysis:
        score = 100  # Start at max, deduct for violations
        risk_concerns = []
        opportunities = []

        # Maximum drawdown hard limit
        if backtest.max_drawdown > 25:
            score -= 50
            risk_concerns.append(f"CRITICAL: Drawdown exceeds 25% limit ({backtest.max_drawdown:.1f}%)")
        elif backtest.max_drawdown > 20:
            score -= 30
            risk_concerns.append(f"WARNING: High drawdown ({backtest.max_drawdown:.1f}%)")
        elif backtest.max_drawdown < 15:
            opportunities.append(f"Acceptable drawdown: {backtest.max_drawdown:.1f}%")

        # VaR check
        if backtest.var_95 > 5000:
            score -= 20
            risk_concerns.append(f"High VaR(95): ${backtest.var_95:.2f}")
        elif backtest.var_95 < 2000:
            opportunities.append(f"Controlled VaR: ${backtest.var_95:.2f}")

        # Minimum Sharpe requirement
        if backtest.sharpe_ratio < 1.0:
            score -= 25
            risk_concerns.append(f"Below minimum Sharpe (1.0): {backtest.sharpe_ratio:.2f}")
        elif backtest.sharpe_ratio > 1.5:
            opportunities.append(f"Strong Sharpe ratio: {backtest.sharpe_ratio:.2f}")

        # Loss management
        if abs(backtest.avg_loss) > backtest.avg_win * 1.5:
            score -= 20
            risk_concerns.append(f"Losses exceed wins by 50%: ${abs(backtest.avg_loss):.2f} vs ${backtest.avg_win:.2f}")

        # Sample size check
        if backtest.total_trades < 30:
            score -= 15
            risk_concerns.append(f"Insufficient sample size: {backtest.total_trades} trades")
        elif backtest.total_trades > 50:
            opportunities.append(f"Adequate sample: {backtest.total_trades} trades")

        # Determine recommendation with strict criteria
        if score >= 80 and len(risk_concerns) == 0:
            rec = "BUY"
            confidence = 0.95
        elif score >= 60 and "CRITICAL" not in str(risk_concerns):
            rec = "HOLD"
            confidence = 0.75
        else:
            rec = "STRONG_SELL"
            confidence = 0.9

        reasoning = f"Risk management assessment. Score: {score}/100. "
        reasoning += f"Found {len(risk_concerns)} risk violations. "
        reasoning += "Strategy is " + ("APPROVED" if score >= 60 else "REJECTED") + " from risk perspective."

        return AgentAnalysis(
            agent_name=self.name,
            agent_type=self.agent_type,
            score=score,
            recommendation=rec,
            confidence=confidence,
            reasoning=reasoning,
            risk_concerns=risk_concerns,
            opportunity_highlights=opportunities
        )


class QuantAgent(TradingAgent):
    """Quantitative agent using statistical analysis"""

    def __init__(self):
        super().__init__("Quant Quinn", "quant", 0.6)

    async def analyze(self, backtest: BacktestResult) -> AgentAnalysis:
        score = 0
        risk_concerns = []
        opportunities = []

        # Statistical significance of Sharpe
        sharpe_tstat = backtest.sharpe_ratio * np.sqrt(backtest.total_trades)
        if sharpe_tstat > 2.0:  # Statistically significant at 95%
            score += 30
            opportunities.append(f"Statistically significant Sharpe (t-stat: {sharpe_tstat:.2f})")
        elif sharpe_tstat > 1.5:
            score += 15
        else:
            risk_concerns.append(f"Sharpe not statistically significant (t-stat: {sharpe_tstat:.2f})")

        # Sortino ratio (downside risk)
        if backtest.sortino_ratio > 2.0:
            score += 25
            opportunities.append(f"Excellent downside protection (Sortino: {backtest.sortino_ratio:.2f})")
        elif backtest.sortino_ratio > 1.5:
            score += 15

        # Expected value with confidence interval
        if backtest.expected_value > 2.0 and backtest.total_trades > 30:
            score += 25
            opportunities.append(f"High EV with adequate sample: ${backtest.expected_value:.2f}")
        elif backtest.expected_value > 1.0:
            score += 10

        # Kelly criterion optimization
        optimal_kelly = backtest.kelly_criterion / 2  # Half-Kelly for safety
        if 0.05 <= optimal_kelly <= 0.15:
            score += 20
            opportunities.append(f"Optimal half-Kelly position: {optimal_kelly*100:.1f}%")
        elif optimal_kelly < 0.03:
            risk_concerns.append("Kelly suggests minimal position sizing")

        # Determine recommendation
        if score >= 70 and sharpe_tstat > 1.5:
            rec = "STRONG_BUY"
            confidence = 0.9
        elif score >= 50:
            rec = "BUY"
            confidence = 0.75
        elif score >= 30:
            rec = "HOLD"
            confidence = 0.6
        else:
            rec = "SELL"
            confidence = 0.8

        reasoning = f"Quantitative statistical analysis. Score: {score}/100. "
        reasoning += f"Strategy shows {'strong' if score >= 60 else 'weak'} statistical validity."

        return AgentAnalysis(
            agent_name=self.name,
            agent_type=self.agent_type,
            score=score,
            recommendation=rec,
            confidence=confidence,
            reasoning=reasoning,
            risk_concerns=risk_concerns,
            opportunity_highlights=opportunities
        )


# ============================================================================
# Swarm Analysis Service
# ============================================================================

class SwarmAnalysisService:
    """Multi-agent swarm intelligence for strategy evaluation"""

    def __init__(self):
        self.agents = [
            ConservativeAgent(),
            AggressiveAgent(),
            BalancedAgent(),
            RiskManagerAgent(),
            QuantAgent()
        ]

    async def analyze_strategy(self, backtest: BacktestResult) -> SwarmConsensus:
        """
        Perform multi-agent swarm analysis on a strategy.

        Returns consensus recommendation with confidence and position sizing.
        """
        # Get analysis from all agents in parallel
        analyses = await asyncio.gather(*[
            agent.analyze(backtest) for agent in self.agents
        ])

        # Calculate consensus
        total_score = sum(a.score * a.confidence for a in analyses)
        total_weight = sum(a.confidence for a in analyses)
        overall_score = total_score / total_weight if total_weight > 0 else 0

        # Vote on recommendation
        vote_weights = {
            "STRONG_BUY": 2,
            "BUY": 1,
            "HOLD": 0,
            "SELL": -1,
            "STRONG_SELL": -2
        }

        weighted_votes = sum(vote_weights.get(a.recommendation, 0) * a.confidence for a in analyses)
        consensus_confidence = np.mean([a.confidence for a in analyses])

        # Determine consensus recommendation
        if weighted_votes >= 1.5:
            consensus_rec = "STRONG_BUY"
        elif weighted_votes >= 0.5:
            consensus_rec = "BUY"
        elif weighted_votes >= -0.5:
            consensus_rec = "HOLD"
        elif weighted_votes >= -1.5:
            consensus_rec = "SELL"
        else:
            consensus_rec = "STRONG_SELL"

        # Count votes
        agent_votes = {}
        for analysis in analyses:
            agent_votes[analysis.recommendation] = agent_votes.get(analysis.recommendation, 0) + 1

        # Collect all risk warnings
        all_risks = []
        for analysis in analyses:
            all_risks.extend(analysis.risk_concerns)

        # Calculate risk-adjusted return
        risk_adjusted_return = backtest.total_return / (backtest.max_drawdown + 1)

        # Calculate suggested position size (Half-Kelly with cap)
        suggested_position = min(backtest.kelly_criterion / 2, 0.15) * 100  # Cap at 15%

        # Calculate stop loss and take profit
        stop_loss = backtest.max_drawdown * 0.6  # 60% of historical max drawdown
        take_profit = backtest.total_return * 0.5  # 50% of historical return

        # Max loss per trade (2% rule)
        max_loss_per_trade = 2.0

        # Go/No-Go decision (requires majority approval and no critical risks)
        risk_manager_analysis = next((a for a in analyses if a.agent_type == "risk_manager"), None)
        go_decision = (
            consensus_rec in ["BUY", "STRONG_BUY"] and
            overall_score >= 60 and
            risk_manager_analysis and
            risk_manager_analysis.recommendation != "STRONG_SELL" and
            not any("CRITICAL" in str(risk) for risk in all_risks)
        )

        # Generate reasoning summary
        reasoning_summary = self._generate_reasoning_summary(
            analyses, consensus_rec, overall_score, go_decision
        )

        return SwarmConsensus(
            strategy=backtest.strategy_name,
            symbol=backtest.symbol,
            overall_score=overall_score,
            consensus_recommendation=consensus_rec,
            consensus_confidence=consensus_confidence,
            expected_value=backtest.expected_value,
            risk_adjusted_return=risk_adjusted_return,
            suggested_position_size=suggested_position,
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_loss_per_trade=max_loss_per_trade,
            agent_votes=agent_votes,
            agent_analyses=analyses,
            risk_warnings=list(set(all_risks)),  # Deduplicate
            go_decision=go_decision,
            reasoning_summary=reasoning_summary
        )

    def _generate_reasoning_summary(
        self,
        analyses: List[AgentAnalysis],
        consensus: str,
        score: float,
        go_decision: bool
    ) -> str:
        """Generate human-readable reasoning summary"""
        summary = f"SWARM CONSENSUS: {consensus} (Score: {score:.1f}/100)\n\n"

        summary += f"DECISION: {'✅ GO - Strategy Approved' if go_decision else '❌ NO-GO - Strategy Rejected'}\n\n"

        summary += "AGENT VOTES:\n"
        for analysis in analyses:
            summary += f"  • {analysis.agent_name} ({analysis.agent_type}): "
            summary += f"{analysis.recommendation} ({analysis.confidence*100:.0f}% confident)\n"

        summary += f"\nKEY INSIGHTS:\n"
        all_opportunities = []
        for analysis in analyses:
            all_opportunities.extend(analysis.opportunity_highlights)
        if all_opportunities:
            for opp in list(set(all_opportunities))[:5]:
                summary += f"  ✓ {opp}\n"

        return summary

    async def compare_strategies(
        self,
        backtests: List[BacktestResult]
    ) -> List[SwarmConsensus]:
        """
        Compare multiple strategies and rank by consensus score.

        Returns ranked list of swarm consensus results.
        """
        # Analyze all strategies in parallel
        consensuses = await asyncio.gather(*[
            self.analyze_strategy(bt) for bt in backtests
        ])

        # Sort by overall score
        ranked = sorted(consensuses, key=lambda c: c.overall_score, reverse=True)

        return ranked
