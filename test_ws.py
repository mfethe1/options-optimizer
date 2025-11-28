
import asyncio
import websockets
import json

async def test_ws():
    uri = "ws://127.0.0.1:8000/ws/phase4-metrics/demo_user"
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected!")
            
            # Wait for initial message
            msg = await websocket.recv()
            print(f"Received: {msg}")
            
            # Wait for another one (should be heartbeat or update)
            # msg = await websocket.recv()
            # print(f"Received: {msg}")
            
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_ws())
