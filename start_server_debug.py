"""
Start server with enhanced debug logging
"""

import logging
import sys

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('server_debug.log')
    ]
)

# Set specific loggers to DEBUG
logging.getLogger('src.api.swarm_routes').setLevel(logging.DEBUG)
logging.getLogger('src.agents.swarm').setLevel(logging.DEBUG)
logging.getLogger('uvicorn').setLevel(logging.INFO)
logging.getLogger('uvicorn.access').setLevel(logging.INFO)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug"
    )

