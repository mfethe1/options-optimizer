"""
Start FastAPI server with rate limiting enabled
"""

import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting FastAPI server with rate limiting...")
    logger.info("Server will be available at http://localhost:8000")
    logger.info("API docs at http://localhost:8000/docs")
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

