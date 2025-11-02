"""
Vision Analysis API Routes

Chart image analysis with GPT-4 Vision and Claude 3.5 Sonnet.
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import os
import tempfile
from pathlib import Path

from ..agents.vision.chart_analysis_agent import ChartAnalysisAgent

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/vision",
    tags=["Vision Analysis"]
)

# Initialize agent
# Supports both OpenAI and Anthropic - will use whatever API keys are available
chart_analyzer = ChartAnalysisAgent(
    preferred_provider=os.getenv("VISION_PROVIDER", "anthropic")  # Default to Claude 3.5 Sonnet
)


# Request/Response models
class ChartAnalysisResponse(BaseModel):
    """Response from chart analysis"""
    analysis: Dict[str, Any] = Field(..., description="Analysis results")
    provider: str = Field(..., description="Provider used (openai or anthropic)")
    timestamp: str = Field(..., description="Analysis timestamp")


class ChartComparisonResponse(BaseModel):
    """Response from multi-chart comparison"""
    comparison: Dict[str, Any] = Field(..., description="Comparison analysis")
    chart_count: int = Field(..., description="Number of charts compared")
    timestamp: str = Field(..., description="Analysis timestamp")


# Routes

@router.post("/analyze-chart", response_model=ChartAnalysisResponse)
async def analyze_chart(
    image: UploadFile = File(..., description="Chart image file (PNG, JPG, WEBP)"),
    analysis_type: str = Form("comprehensive", description="Analysis type: comprehensive, pattern, levels, flow"),
    question: Optional[str] = Form(None, description="Specific question about the chart")
):
    """
    Analyze a chart image with AI vision

    **COMPETITIVE ADVANTAGE**: First options platform with AI-powered chart image analysis

    Supported image formats: PNG, JPG, JPEG, WEBP, GIF

    Analysis types:
    - **comprehensive**: Full analysis including patterns, levels, indicators, options flow
    - **pattern**: Focus on chart patterns (head & shoulders, triangles, flags, etc.)
    - **levels**: Identify support/resistance levels
    - **flow**: Analyze options flow and unusual activity (if visible)

    Use cases:
    1. Upload chart screenshots from TradingView, ThinkOrSwim, Webull
    2. Analyze charts from Twitter/Discord/FinTwit
    3. Verify influencer claims with chart analysis
    4. Extract insights from YouTube video thumbnails

    Args:
        image: Chart image file
        analysis_type: Type of analysis to perform
        question: Optional specific question about the chart

    Returns:
        AI-powered analysis with patterns, levels, indicators, and trading recommendations
    """
    temp_path = None
    try:
        logger.info(f"Analyzing chart: {image.filename} (type: {analysis_type})")

        # Validate file type
        allowed_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.gif'}
        file_ext = Path(image.filename).suffix.lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}. Supported: {', '.join(allowed_extensions)}"
            )

        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            content = await image.read()
            tmp.write(content)
            temp_path = tmp.name

        # Analyze chart
        result = await chart_analyzer.analyze_chart(
            image_path=temp_path,
            question=question,
            analysis_type=analysis_type
        )

        # Check for errors
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return ChartAnalysisResponse(
            analysis=result,
            provider=result.get('provider', 'unknown'),
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing chart: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to analyze chart: {str(e)}")
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")


@router.post("/compare-charts", response_model=ChartComparisonResponse)
async def compare_charts(
    charts: List[UploadFile] = File(..., description="2-4 chart images to compare"),
    comparison_type: str = Form("relative_strength", description="Comparison type: relative_strength, divergence, correlation")
):
    """
    Compare multiple charts side-by-side

    **Use cases**:
    - Compare stock vs sector performance
    - Identify divergences between price and indicators
    - Analyze correlation between related stocks
    - Compare different timeframes of same stock

    Comparison types:
    - **relative_strength**: Which chart is stronger?
    - **divergence**: Are there divergences between the charts?
    - **correlation**: How correlated are the movements?

    Args:
        charts: 2-4 chart images (max 4)
        comparison_type: Type of comparison analysis

    Returns:
        Multi-chart comparison analysis with trading insights
    """
    temp_paths = []
    try:
        # Validate number of charts
        if len(charts) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 charts for comparison")
        if len(charts) > 4:
            raise HTTPException(status_code=400, detail="Maximum 4 charts for comparison")

        logger.info(f"Comparing {len(charts)} charts (type: {comparison_type})")

        # Save all uploaded files
        for idx, chart in enumerate(charts):
            file_ext = Path(chart.filename).suffix.lower()

            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                content = await chart.read()
                tmp.write(content)
                temp_paths.append(tmp.name)

        # Compare charts
        result = await chart_analyzer.compare_charts(
            image_paths=temp_paths,
            comparison_type=comparison_type
        )

        # Check for errors
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return ChartComparisonResponse(
            comparison=result,
            chart_count=len(temp_paths),
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing charts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to compare charts: {str(e)}")
    finally:
        # Clean up temp files
        for temp_path in temp_paths:
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file: {e}")


@router.get("/providers")
async def get_available_providers():
    """
    Get list of available vision providers

    Returns:
        Available providers and their capabilities
    """
    providers = []

    if chart_analyzer.openai_client:
        providers.append({
            "name": "openai",
            "model": "gpt-4-vision-preview",
            "capabilities": ["single_image", "pattern_recognition", "text_extraction"]
        })

    if chart_analyzer.anthropic_client:
        providers.append({
            "name": "anthropic",
            "model": "claude-3-5-sonnet-20241022",
            "capabilities": ["single_image", "multiple_images", "pattern_recognition", "text_extraction"]
        })

    return {
        "providers": providers,
        "count": len(providers),
        "preferred": chart_analyzer.preferred_provider
    }


@router.get("/analysis-types")
async def get_analysis_types():
    """
    Get supported analysis types

    Returns:
        Dictionary of analysis types with descriptions
    """
    return {
        "analysis_types": {
            "comprehensive": {
                "description": "Full chart analysis including patterns, levels, indicators, and options flow",
                "includes": [
                    "Chart pattern recognition",
                    "Support/resistance levels",
                    "Trend analysis",
                    "Technical indicators",
                    "Options flow (if visible)",
                    "Trading recommendations",
                    "Risk assessment"
                ]
            },
            "pattern": {
                "description": "Focus on identifying chart patterns",
                "includes": [
                    "Head & shoulders",
                    "Double tops/bottoms",
                    "Triangles, wedges, flags",
                    "Pattern bias (bullish/bearish)",
                    "Key price levels",
                    "Expected move",
                    "Pattern invalidation levels"
                ]
            },
            "levels": {
                "description": "Identify support and resistance levels",
                "includes": [
                    "Major support levels",
                    "Major resistance levels",
                    "Current price position",
                    "Next key levels",
                    "Volume profile (if visible)"
                ]
            },
            "flow": {
                "description": "Analyze options flow and unusual activity",
                "includes": [
                    "Unusual volume detection",
                    "Large block trades",
                    "Put/call ratio",
                    "Expiration concentration",
                    "Smart money indicators",
                    "Trading implications"
                ]
            }
        },
        "comparison_types": {
            "relative_strength": "Compare performance between charts",
            "divergence": "Identify divergences and disagreements",
            "correlation": "Analyze correlation between movements"
        }
    }
