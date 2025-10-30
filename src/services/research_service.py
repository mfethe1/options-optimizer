"""
Research Service - Aggregate news, social media, YouTube, and GitHub research
"""
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import pandas as pd

logger = logging.getLogger(__name__)


class ResearchService:
    """
    Centralized research service using Firecrawl, Reddit, YouTube, and GitHub.
    Caches results to Parquet for historical analysis.
    """
    
    def __init__(self):
        self.cache_dir = Path("data/research")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # API keys from environment
        self.firecrawl_api_key = os.getenv('FIRECRAWL_API_KEY')
        self.firecrawl_base_url = os.getenv('FIRECRAWL_BASE_URL', 'https://api.firecrawl.dev')
        self.youtube_api_key = os.getenv('YOUTUBE_API_KEY')
        self.github_token = os.getenv('GITHUB_TOKEN')
        
        # Reddit credentials
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.reddit_username = os.getenv('REDDIT_USERNAME')
        self.reddit_password = os.getenv('REDDIT_PASSWORD')
        self.reddit_user_agent = os.getenv('REDDIT_USER_AGENT', 'options-probability/1.0')
        
        self.reddit = None
        self._init_reddit()
    
    def _init_reddit(self):
        """Initialize Reddit client if credentials available"""
        try:
            if all([self.reddit_client_id, self.reddit_client_secret, 
                   self.reddit_username, self.reddit_password]):
                import praw
                self.reddit = praw.Reddit(
                    client_id=self.reddit_client_id,
                    client_secret=self.reddit_client_secret,
                    username=self.reddit_username,
                    password=self.reddit_password,
                    user_agent=self.reddit_user_agent
                )
                logger.info("Reddit client initialized")
        except Exception as e:
            logger.warning(f"Could not initialize Reddit client: {e}")
    
    def research_symbol(self, symbol: str, max_age_hours: int = 24) -> Dict[str, Any]:
        """
        Comprehensive research for a symbol.
        Returns cached data if fresh, otherwise fetches new data.
        """
        cache_file = self.cache_dir / f"{symbol}_latest.json"
        
        # Check cache
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cached = json.load(f)
                cache_time = datetime.fromisoformat(cached['timestamp'])
                if datetime.now() - cache_time < timedelta(hours=max_age_hours):
                    logger.info(f"Using cached research for {symbol}")
                    return cached
        
        # Fetch fresh data
        logger.info(f"Fetching fresh research for {symbol}")
        research = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'news': self.get_news(symbol),
            'social': self.get_social_sentiment(symbol),
            'youtube': self.get_youtube_sentiment(symbol),
            'summary': None
        }
        
        # Generate summary
        research['summary'] = self._generate_summary(research)
        
        # Cache results
        with open(cache_file, 'w') as f:
            json.dump(research, f, indent=2)
        
        # Also append to historical parquet
        self._append_to_history(symbol, research)
        
        return research
    
    def get_news(self, symbol: str, max_results: int = 10) -> Dict[str, Any]:
        """Get news using Firecrawl"""
        if not self.firecrawl_api_key:
            logger.warning("Firecrawl API key not set, using placeholder")
            return {
                'articles': [],
                'sentiment': 'neutral',
                'score': 0.0,
                'source': 'placeholder'
            }
        
        try:
            # Use Firecrawl search
            # TODO: Integrate with actual Firecrawl MCP when available
            # For now, placeholder
            return {
                'articles': [
                    {
                        'title': f'{symbol} market update',
                        'url': 'https://example.com',
                        'published': datetime.now().isoformat(),
                        'sentiment': 'neutral',
                        'source': 'placeholder'
                    }
                ],
                'sentiment': 'neutral',
                'score': 0.0,
                'source': 'firecrawl'
            }
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return {'articles': [], 'sentiment': 'neutral', 'score': 0.0, 'source': 'error'}
    
    def get_social_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get social media sentiment from Reddit"""
        if not self.reddit:
            logger.warning("Reddit not initialized, using placeholder")
            return {
                'reddit': {'sentiment': 'neutral', 'volume': 0, 'posts': []},
                'overall_sentiment': 'neutral',
                'score': 0.0
            }
        
        try:
            # Search relevant subreddits
            subreddits = ['stocks', 'options', 'wallstreetbets', 'investing']
            posts = []
            
            for sub_name in subreddits:
                try:
                    subreddit = self.reddit.subreddit(sub_name)
                    for post in subreddit.search(symbol, time_filter='day', limit=5):
                        posts.append({
                            'title': post.title,
                            'score': post.score,
                            'num_comments': post.num_comments,
                            'created': datetime.fromtimestamp(post.created_utc).isoformat(),
                            'url': f"https://reddit.com{post.permalink}",
                            'subreddit': sub_name
                        })
                except Exception as e:
                    logger.warning(f"Error searching r/{sub_name}: {e}")
            
            # Calculate sentiment from scores and volume
            total_score = sum(p['score'] for p in posts)
            sentiment = 'bullish' if total_score > 100 else 'bearish' if total_score < -50 else 'neutral'
            
            return {
                'reddit': {
                    'sentiment': sentiment,
                    'volume': len(posts),
                    'posts': posts[:10],  # Top 10
                    'total_score': total_score
                },
                'overall_sentiment': sentiment,
                'score': min(max(total_score / 1000, -1), 1)  # Normalize to -1 to 1
            }
        except Exception as e:
            logger.error(f"Error fetching Reddit sentiment for {symbol}: {e}")
            return {
                'reddit': {'sentiment': 'neutral', 'volume': 0, 'posts': []},
                'overall_sentiment': 'neutral',
                'score': 0.0
            }
    
    def get_youtube_sentiment(self, symbol: str, max_results: int = 5) -> Dict[str, Any]:
        """Get YouTube sentiment"""
        if not self.youtube_api_key:
            logger.warning("YouTube API key not set, using placeholder")
            return {
                'videos': [],
                'sentiment': 'neutral',
                'score': 0.0
            }
        
        try:
            from googleapiclient.discovery import build
            
            youtube = build('youtube', 'v3', developerKey=self.youtube_api_key)
            
            # Search for videos
            search_response = youtube.search().list(
                q=f"{symbol} stock analysis",
                part='snippet',
                maxResults=max_results,
                type='video',
                order='relevance'
            ).execute()
            
            videos = []
            for item in search_response.get('items', []):
                videos.append({
                    'title': item['snippet']['title'],
                    'description': item['snippet']['description'],
                    'published': item['snippet']['publishedAt'],
                    'channel': item['snippet']['channelTitle'],
                    'video_id': item['id']['videoId'],
                    'url': f"https://www.youtube.com/watch?v={item['id']['videoId']}"
                })
            
            # Simple sentiment from title keywords
            bullish_keywords = ['buy', 'bullish', 'moon', 'rally', 'breakout']
            bearish_keywords = ['sell', 'bearish', 'crash', 'drop', 'warning']
            
            bullish_count = sum(1 for v in videos if any(k in v['title'].lower() for k in bullish_keywords))
            bearish_count = sum(1 for v in videos if any(k in v['title'].lower() for k in bearish_keywords))
            
            if bullish_count > bearish_count:
                sentiment = 'bullish'
                score = 0.5
            elif bearish_count > bullish_count:
                sentiment = 'bearish'
                score = -0.5
            else:
                sentiment = 'neutral'
                score = 0.0
            
            return {
                'videos': videos,
                'sentiment': sentiment,
                'score': score,
                'bullish_count': bullish_count,
                'bearish_count': bearish_count
            }
        except Exception as e:
            logger.error(f"Error fetching YouTube data for {symbol}: {e}")
            return {'videos': [], 'sentiment': 'neutral', 'score': 0.0}
    
    def search_github_repos(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search GitHub for relevant repositories"""
        if not self.github_token:
            logger.warning("GitHub token not set, using placeholder")
            return []
        
        try:
            from github import Github
            
            g = Github(self.github_token)
            repos = g.search_repositories(query=query, sort='stars', order='desc')
            
            results = []
            for repo in repos[:max_results]:
                results.append({
                    'name': repo.full_name,
                    'description': repo.description,
                    'stars': repo.stargazers_count,
                    'language': repo.language,
                    'url': repo.html_url,
                    'topics': repo.get_topics(),
                    'updated': repo.updated_at.isoformat()
                })
            
            return results
        except Exception as e:
            logger.error(f"Error searching GitHub: {e}")
            return []
    
    def _generate_summary(self, research: Dict[str, Any]) -> str:
        """Generate natural language summary of research"""
        symbol = research['symbol']
        news = research.get('news', {})
        social = research.get('social', {})
        youtube = research.get('youtube', {})
        
        summary = f"**Research Summary for {symbol}**\n\n"
        
        # News sentiment
        news_sentiment = news.get('sentiment', 'neutral')
        summary += f"**News Sentiment:** {news_sentiment.upper()}\n"
        summary += f"- {len(news.get('articles', []))} articles analyzed\n\n"
        
        # Social sentiment
        social_sentiment = social.get('overall_sentiment', 'neutral')
        reddit_volume = social.get('reddit', {}).get('volume', 0)
        summary += f"**Social Media Sentiment:** {social_sentiment.upper()}\n"
        summary += f"- Reddit: {reddit_volume} posts found\n\n"
        
        # YouTube sentiment
        youtube_sentiment = youtube.get('sentiment', 'neutral')
        video_count = len(youtube.get('videos', []))
        summary += f"**YouTube Sentiment:** {youtube_sentiment.upper()}\n"
        summary += f"- {video_count} videos analyzed\n\n"
        
        # Overall
        sentiments = [news_sentiment, social_sentiment, youtube_sentiment]
        bullish_count = sentiments.count('bullish')
        bearish_count = sentiments.count('bearish')
        
        if bullish_count > bearish_count:
            overall = 'BULLISH'
        elif bearish_count > bullish_count:
            overall = 'BEARISH'
        else:
            overall = 'NEUTRAL'
        
        summary += f"**Overall Consensus:** {overall}\n"
        
        return summary
    
    def _append_to_history(self, symbol: str, research: Dict[str, Any]):
        """Append research to historical parquet file"""
        try:
            history_file = self.cache_dir / f"{symbol}_history.parquet"
            
            # Flatten research for parquet
            row = {
                'timestamp': research['timestamp'],
                'symbol': symbol,
                'news_sentiment': research['news'].get('sentiment'),
                'news_score': research['news'].get('score'),
                'news_count': len(research['news'].get('articles', [])),
                'social_sentiment': research['social'].get('overall_sentiment'),
                'social_score': research['social'].get('score'),
                'social_volume': research['social'].get('reddit', {}).get('volume', 0),
                'youtube_sentiment': research['youtube'].get('sentiment'),
                'youtube_score': research['youtube'].get('score'),
                'youtube_count': len(research['youtube'].get('videos', []))
            }
            
            df_new = pd.DataFrame([row])
            
            if history_file.exists():
                df_existing = pd.read_parquet(history_file)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df_combined = df_new
            
            df_combined.to_parquet(history_file, index=False)
            logger.info(f"Appended research to {history_file}")
        except Exception as e:
            logger.error(f"Error appending to history: {e}")

