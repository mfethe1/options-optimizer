"""
Multi-Model Configuration for Agentic System

Configures multiple LLM providers:
- OpenAI GPT-4
- Anthropic Claude Sonnet 4.5
- LM Studio (local)
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ModelProvider(Enum):
    """Supported model providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LMSTUDIO = "lmstudio"


@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    provider: ModelProvider
    model_name: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'provider': self.provider.value,
            'model_name': self.model_name,
            'api_key': self.api_key,
            'api_base': self.api_base,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }


# Model configurations
MODELS = {
    'gpt4': ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name='gpt-4',
        api_key=os.getenv('OPENAI_API_KEY'),
        temperature=0.7,
        max_tokens=4000
    ),
    'claude': ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name='claude-sonnet-4.5',
        api_key=os.getenv('ANTHROPIC_API_KEY'),
        temperature=0.7,
        max_tokens=4000
    ),
    'lmstudio': ModelConfig(
        provider=ModelProvider.LMSTUDIO,
        model_name='local-model',
        api_base=os.getenv('LMSTUDIO_API_BASE', 'http://localhost:1234/v1'),
        temperature=0.7,
        max_tokens=4000
    )
}


# Agent-to-Model assignments
AGENT_MODEL_ASSIGNMENTS = {
    'market_intelligence': 'gpt4',      # GPT-4 for market analysis
    'risk_analysis': 'claude',          # Claude for risk assessment
    'quant_analysis': 'lmstudio',       # LM Studio for quantitative analysis
    'sentiment_research': 'gpt4',       # GPT-4 for sentiment analysis
    'technical_analysis': 'claude',     # Claude for technical analysis
    'fundamental_analysis': 'lmstudio', # LM Studio for fundamental analysis
    'coordinator': 'gpt4'               # GPT-4 for coordination
}


# Discussion configuration
DISCUSSION_CONFIG = {
    'rounds': 5,
    'agents_per_round': 3,
    'use_firecrawl': True,
    'aggregate_insights': True,
    'consensus_threshold': 0.7
}


def get_model_config(model_key: str) -> ModelConfig:
    """Get model configuration by key"""
    if model_key not in MODELS:
        raise ValueError(f"Unknown model key: {model_key}")
    return MODELS[model_key]


def get_agent_model(agent_name: str) -> ModelConfig:
    """Get model configuration for an agent"""
    model_key = AGENT_MODEL_ASSIGNMENTS.get(agent_name, 'gpt4')
    return get_model_config(model_key)


def validate_configurations() -> Dict[str, bool]:
    """Validate all model configurations"""
    results = {}
    
    for key, config in MODELS.items():
        if config.provider == ModelProvider.OPENAI:
            results[key] = bool(config.api_key)
        elif config.provider == ModelProvider.ANTHROPIC:
            results[key] = bool(config.api_key)
        elif config.provider == ModelProvider.LMSTUDIO:
            results[key] = bool(config.api_base)
        else:
            results[key] = False
    
    return results

