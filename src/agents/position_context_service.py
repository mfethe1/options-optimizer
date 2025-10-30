"""
Position Context Service for AI Agents
Provides position data and conversation memory to agents
"""
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from ..data.position_manager import PositionManager
from ..data.position_enrichment_service import PositionEnrichmentService

logger = logging.getLogger(__name__)


class ConversationMemory:
    """Manages conversation history and context for AI agents"""
    
    def __init__(self, storage_file: str = "data/conversation_memory.json"):
        self.storage_file = storage_file
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
        self.load_memory()
    
    def add_message(self, conversation_id: str, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add a message to conversation history"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        message = {
            'timestamp': datetime.now().isoformat(),
            'role': role,  # 'user', 'assistant', 'system'
            'content': content,
            'metadata': metadata or {}
        }
        
        self.conversations[conversation_id].append(message)
        self.save_memory()
    
    def get_conversation(self, conversation_id: str, limit: int = None) -> List[Dict[str, Any]]:
        """Get conversation history"""
        messages = self.conversations.get(conversation_id, [])
        if limit:
            return messages[-limit:]
        return messages
    
    def get_recent_conversations(self, limit: int = 5) -> List[str]:
        """Get list of recent conversation IDs"""
        # Sort by most recent message
        sorted_convs = sorted(
            self.conversations.items(),
            key=lambda x: x[1][-1]['timestamp'] if x[1] else '',
            reverse=True
        )
        return [conv_id for conv_id, _ in sorted_convs[:limit]]
    
    def clear_conversation(self, conversation_id: str):
        """Clear a specific conversation"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            self.save_memory()
    
    def save_memory(self):
        """Save conversation memory to file"""
        try:
            Path(self.storage_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_file, 'w') as f:
                json.dump(self.conversations, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving conversation memory: {e}")
    
    def load_memory(self):
        """Load conversation memory from file"""
        try:
            if Path(self.storage_file).exists():
                with open(self.storage_file, 'r') as f:
                    self.conversations = json.load(f)
                logger.info(f"Loaded {len(self.conversations)} conversations")
        except Exception as e:
            logger.error(f"Error loading conversation memory: {e}")


class PositionContextService:
    """Service to provide position context to AI agents"""
    
    def __init__(
        self,
        position_manager: PositionManager = None,
        enrichment_service: PositionEnrichmentService = None,
        conversation_memory: ConversationMemory = None
    ):
        self.position_manager = position_manager or PositionManager()
        self.enrichment_service = enrichment_service or PositionEnrichmentService(self.position_manager)
        self.conversation_memory = conversation_memory or ConversationMemory()
    
    def get_position_summary_for_agent(self) -> str:
        """Get formatted position summary for AI agent context"""
        try:
            # Get enriched summary
            summary = self.enrichment_service.get_enriched_portfolio_summary()
            
            # Format for agent
            text = f"""
# Portfolio Summary

**Total Positions**: {summary['total_stocks']} stocks, {summary['total_options']} options
**Unique Symbols**: {summary['unique_symbols']} ({', '.join(summary['symbols'][:10])})
**Total Value**: ${summary.get('total_current_value', 0):,.2f}
**Total P&L**: ${summary.get('total_pnl', 0):,.2f} ({summary.get('total_pnl_pct', 0):.2f}%)

## Stock Positions
- Total Value: ${summary.get('total_stock_current_value', 0):,.2f}
- P&L: ${summary.get('total_stock_pnl', 0):,.2f}

## Option Positions
- Total Value: ${summary.get('total_option_current_value', 0):,.2f}
- P&L: ${summary.get('total_option_pnl', 0):,.2f}
"""
            return text.strip()
        except Exception as e:
            logger.error(f"Error generating position summary: {e}")
            return "Error loading position data"
    
    def get_detailed_positions_for_agent(self, symbol: str = None) -> str:
        """Get detailed position information for AI agent"""
        try:
            text = "# Detailed Positions\n\n"
            
            # Get positions
            if symbol:
                positions = self.position_manager.get_positions_by_symbol(symbol.upper())
                stocks = positions['stocks']
                options = positions['options']
            else:
                stocks = self.position_manager.get_all_stock_positions()
                options = self.position_manager.get_all_option_positions()
            
            # Format stock positions
            if stocks:
                text += "## Stock Positions\n\n"
                for pos in stocks:
                    # Enrich position
                    self.enrichment_service.enrich_stock_position(pos)
                    
                    text += f"### {pos.symbol}\n"
                    text += f"- **Quantity**: {pos.quantity} shares\n"
                    text += f"- **Entry**: ${pos.entry_price:.2f} on {pos.entry_date}\n"
                    text += f"- **Current**: ${pos.current_price:.2f}\n" if pos.current_price else ""
                    text += f"- **P&L**: ${pos.unrealized_pnl:.2f} ({pos.unrealized_pnl_pct:.2f}%)\n" if pos.unrealized_pnl else ""
                    text += f"- **Status**: {pos.get_status()}\n"
                    if pos.target_price:
                        text += f"- **Target**: ${pos.target_price:.2f}\n"
                    if pos.stop_loss:
                        text += f"- **Stop Loss**: ${pos.stop_loss:.2f}\n"
                    if pos.notes:
                        text += f"- **Notes**: {pos.notes}\n"
                    text += "\n"
            
            # Format option positions
            if options:
                text += "## Option Positions\n\n"
                for pos in options:
                    # Enrich position
                    self.enrichment_service.enrich_option_position(pos)
                    
                    text += f"### {pos.symbol} ${pos.strike} {pos.option_type.upper()} exp {pos.expiration_date}\n"
                    text += f"- **Quantity**: {pos.quantity} contracts\n"
                    text += f"- **Premium Paid**: ${pos.premium_paid:.2f}\n"
                    text += f"- **Current Price**: ${pos.current_price:.2f}\n" if pos.current_price else ""
                    text += f"- **Underlying**: ${pos.underlying_price:.2f}\n" if pos.underlying_price else ""
                    text += f"- **P&L**: ${pos.unrealized_pnl:.2f} ({pos.unrealized_pnl_pct:.2f}%)\n" if pos.unrealized_pnl else ""
                    text += f"- **Days to Expiry**: {pos.days_to_expiry()}\n"
                    text += f"- **Status**: {pos.get_status()}\n"
                    text += f"- **Risk Level**: {pos.get_risk_level()}\n"
                    
                    # Greeks
                    if pos.delta:
                        text += f"- **Greeks**: Delta={pos.delta:.3f}, Gamma={pos.gamma:.3f}, Theta={pos.theta:.3f}, Vega={pos.vega:.3f}\n"
                    
                    # Volatility
                    if pos.implied_volatility:
                        text += f"- **IV**: {pos.implied_volatility*100:.1f}%"
                        if pos.historical_volatility:
                            text += f" (HV: {pos.historical_volatility*100:.1f}%)"
                        text += "\n"
                    
                    # Probability
                    if pos.probability_of_profit:
                        text += f"- **Probability of Profit**: {pos.probability_of_profit:.1f}%\n"
                    
                    if pos.notes:
                        text += f"- **Notes**: {pos.notes}\n"
                    text += "\n"
            
            if not stocks and not options:
                text += "No positions found.\n"
            
            return text.strip()
        except Exception as e:
            logger.error(f"Error generating detailed positions: {e}")
            return "Error loading position details"
    
    def get_expiring_options_for_agent(self, days: int = 7) -> str:
        """Get options expiring soon for AI agent"""
        try:
            expiring = self.position_manager.get_expiring_soon(days)
            
            if not expiring:
                return f"No options expiring in the next {days} days."
            
            text = f"# Options Expiring in Next {days} Days\n\n"
            
            for pos in expiring:
                # Enrich position
                self.enrichment_service.enrich_option_position(pos)
                
                days_left = pos.days_to_expiry()
                text += f"## {pos.symbol} ${pos.strike} {pos.option_type.upper()} - {days_left} days left\n"
                text += f"- **Expiration**: {pos.expiration_date}\n"
                text += f"- **Current Price**: ${pos.current_price:.2f}\n" if pos.current_price else ""
                text += f"- **P&L**: ${pos.unrealized_pnl:.2f} ({pos.unrealized_pnl_pct:.2f}%)\n" if pos.unrealized_pnl else ""
                text += f"- **Status**: {pos.get_status()}\n"
                text += f"- **Risk Level**: {pos.get_risk_level()}\n"
                text += "\n"
            
            return text.strip()
        except Exception as e:
            logger.error(f"Error getting expiring options: {e}")
            return "Error loading expiring options"
    
    def create_agent_context(
        self,
        conversation_id: str,
        include_summary: bool = True,
        include_positions: bool = False,
        symbol: str = None,
        include_expiring: bool = False
    ) -> str:
        """Create comprehensive context for AI agent"""
        context_parts = []
        
        # Add conversation history
        recent_messages = self.conversation_memory.get_conversation(conversation_id, limit=10)
        if recent_messages:
            context_parts.append("# Recent Conversation\n")
            for msg in recent_messages[-5:]:  # Last 5 messages
                role = msg['role'].capitalize()
                content = msg['content'][:200]  # Truncate long messages
                context_parts.append(f"**{role}**: {content}\n")
        
        # Add portfolio summary
        if include_summary:
            context_parts.append("\n" + self.get_position_summary_for_agent())
        
        # Add detailed positions
        if include_positions:
            context_parts.append("\n" + self.get_detailed_positions_for_agent(symbol))
        
        # Add expiring options
        if include_expiring:
            context_parts.append("\n" + self.get_expiring_options_for_agent())
        
        return "\n".join(context_parts)
    
    def log_agent_interaction(
        self,
        conversation_id: str,
        user_query: str,
        agent_response: str,
        positions_accessed: List[str] = None
    ):
        """Log an agent interaction with position context"""
        # Add user message
        self.conversation_memory.add_message(
            conversation_id,
            'user',
            user_query
        )
        
        # Add agent response
        self.conversation_memory.add_message(
            conversation_id,
            'assistant',
            agent_response,
            metadata={
                'positions_accessed': positions_accessed or [],
                'timestamp': datetime.now().isoformat()
            }
        )

