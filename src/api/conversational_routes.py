"""
Conversational Trading API Routes

Natural language interface for trading and analysis.
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import json

from ..agents.conversational.conversation_coordinator import ConversationCoordinatorAgent
from ..agents.conversational.explanation_agent import ExplanationAgent

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/conversation",
    tags=["Conversational Trading"]
)

# Initialize agents
conversation_coordinator = ConversationCoordinatorAgent()
explanation_agent = ExplanationAgent()


# Request/Response models
class ConversationMessage(BaseModel):
    """User message in conversation"""
    message: str = Field(..., description="User's message")
    user_id: str = Field(..., description="User identifier")
    session_id: Optional[str] = Field(None, description="Conversation session ID")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context (positions, market data, etc.)")


class ConversationResponse(BaseModel):
    """Response from conversation agent"""
    response: str = Field(..., description="Agent's response text")
    intent: str = Field(..., description="Detected intent")
    confidence: float = Field(..., description="Intent confidence (0-1)")
    actions: Optional[List[Dict[str, Any]]] = Field(None, description="Suggested actions")
    data: Optional[Dict[str, Any]] = Field(None, description="Supporting data")
    session_id: str = Field(..., description="Conversation session ID")
    turn_number: int = Field(..., description="Conversation turn number")
    timestamp: str = Field(..., description="Response timestamp")


class ExplanationRequest(BaseModel):
    """Request for explanation of a topic"""
    topic: str = Field(..., description="Topic to explain")
    complexity: str = Field("medium", description="Complexity level: beginner, medium, advanced")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class ExplanationResponse(BaseModel):
    """Response with explanation"""
    topic: str
    simple_explanation: str
    detailed_explanation: str
    example: Optional[str]
    misconceptions: Optional[List[str]]
    related_topics: Optional[List[str]]


# Routes

@router.post("/message", response_model=ConversationResponse)
async def send_message(request: ConversationMessage):
    """
    Send a message to the conversational trading agent

    Supports natural language queries like:
    - "What's the risk/reward on selling NVDA 950 puts expiring next Friday?"
    - "Show me high IV stocks in the tech sector"
    - "What happens if AAPL drops 10% before my calls expire?"
    - "Help me understand iron condors"

    The agent will:
    1. Classify intent (trade execution, risk analysis, research, education, etc.)
    2. Route to specialized workflow
    3. Return actionable response with data

    Args:
        request: Conversation message with user_id and optional context

    Returns:
        Response with agent's answer, detected intent, and actions
    """
    try:
        logger.info(f"Processing message from user {request.user_id}: {request.message[:100]}")

        # Process message through coordinator
        result = await conversation_coordinator.process_message(
            user_message=request.message,
            user_id=request.user_id,
            context=request.context
        )

        # Extract session info
        session_id = result.get('session_id', request.session_id or f"session_{request.user_id}_{datetime.now().timestamp()}")
        turn_number = result.get('turn_number', 1)

        # Build response
        return ConversationResponse(
            response=result['response'],
            intent=result['intent'],
            confidence=result['confidence'],
            actions=result.get('actions'),
            data=result.get('data'),
            session_id=session_id,
            turn_number=turn_number,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Error processing conversation message: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process message: {str(e)}")


@router.post("/explain", response_model=ExplanationResponse)
async def explain_topic(request: ExplanationRequest):
    """
    Get educational explanation of options trading concepts

    Topics supported:
    - Greeks: delta, gamma, theta, vega, rho
    - Strategies: iron condor, butterfly, straddle, strangle, etc.
    - Risk metrics: IV, IV Rank, IV Percentile, expected move
    - Market concepts: open interest, volume, max pain, pin risk

    Complexity levels:
    - beginner: ELI5 explanations with simple analogies
    - medium: Balanced technical depth
    - advanced: Deep technical details and mathematics

    Args:
        request: Topic and complexity level

    Returns:
        Multi-level explanation with examples and related topics
    """
    try:
        logger.info(f"Explaining topic: {request.topic} (complexity: {request.complexity})")

        # Get explanation
        result = await explanation_agent.explain(
            topic=request.topic,
            complexity=request.complexity
        )

        return ExplanationResponse(**result)

    except Exception as e:
        logger.error(f"Error generating explanation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate explanation: {str(e)}")


@router.get("/history/{user_id}")
async def get_conversation_history(
    user_id: str,
    session_id: Optional[str] = None,
    limit: int = 50
):
    """
    Get conversation history for a user

    Args:
        user_id: User identifier
        session_id: Optional session filter
        limit: Max messages to return (default 50)

    Returns:
        List of conversation messages
    """
    try:
        # Get history from coordinator
        history = await conversation_coordinator.get_conversation_history(
            user_id=user_id,
            session_id=session_id,
            limit=limit
        )

        return {
            "user_id": user_id,
            "session_id": session_id,
            "messages": history,
            "count": len(history)
        }

    except Exception as e:
        logger.error(f"Error fetching conversation history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")


@router.delete("/history/{user_id}")
async def clear_conversation_history(user_id: str, session_id: Optional[str] = None):
    """
    Clear conversation history for a user

    Args:
        user_id: User identifier
        session_id: Optional session to clear (if None, clears all)

    Returns:
        Success message
    """
    try:
        await conversation_coordinator.clear_history(
            user_id=user_id,
            session_id=session_id
        )

        return {
            "message": f"Conversation history cleared for user {user_id}",
            "session_id": session_id
        }

    except Exception as e:
        logger.error(f"Error clearing conversation history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")


@router.get("/intents")
async def get_supported_intents():
    """
    Get list of supported conversation intents

    Returns:
        Dictionary of intents with descriptions and examples
    """
    return {
        "intents": {
            "trade_execution": {
                "description": "Execute or plan trades",
                "examples": [
                    "Buy 5 AAPL 180 calls expiring 12/15",
                    "Sell 10 TSLA 250 puts",
                    "Close my NVDA position"
                ]
            },
            "risk_analysis": {
                "description": "Analyze risk/reward and scenario analysis",
                "examples": [
                    "What's my max loss on this iron condor?",
                    "What happens if SPY drops 5% tomorrow?",
                    "Show me my portfolio Greeks"
                ]
            },
            "research": {
                "description": "Research stocks and options opportunities",
                "examples": [
                    "Find high IV stocks in tech sector",
                    "Show me stocks with earnings this week",
                    "Which tech stocks have unusual options activity?"
                ]
            },
            "portfolio_review": {
                "description": "Review and optimize portfolio",
                "examples": [
                    "Review my current positions",
                    "What should I close today?",
                    "Show me my P&L by strategy"
                ]
            },
            "education": {
                "description": "Learn about options trading concepts",
                "examples": [
                    "Explain theta decay",
                    "How do iron condors work?",
                    "What is implied volatility?"
                ]
            },
            "market_data": {
                "description": "Get market data and quotes",
                "examples": [
                    "What's AAPL trading at?",
                    "Show me SPY options chain",
                    "What's the IV on TSLA?"
                ]
            },
            "general": {
                "description": "General conversation and help",
                "examples": [
                    "What can you help me with?",
                    "How do I use this platform?",
                    "Tell me about your features"
                ]
            }
        }
    }
