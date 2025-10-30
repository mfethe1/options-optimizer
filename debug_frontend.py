"""Debug frontend - get actual HTML and console errors"""
import asyncio
from playwright.async_api import async_playwright


async def debug_frontend():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        
        # Collect all messages
        messages = []
        page.on("console", lambda msg: messages.append(f"[{msg.type}] {msg.text}"))
        page.on("pageerror", lambda exc: messages.append(f"[ERROR] {exc}"))
        
        print("Navigating to http://localhost:3000...")
        await page.goto("http://localhost:3000")
        await page.wait_for_timeout(5000)  # Wait 5 seconds
        
        # Get HTML
        html = await page.content()
        print(f"\n=== HTML ({len(html)} chars) ===")
        print(html[:1000])  # First 1000 chars
        
        # Get console messages
        print(f"\n=== Console Messages ({len(messages)}) ===")
        for msg in messages:
            print(msg)
        
        # Check if root div has content
        root_html = await page.locator("#root").inner_html()
        print(f"\n=== Root div content ({len(root_html)} chars) ===")
        print(root_html[:500] if root_html else "EMPTY!")
        
        await browser.close()


if __name__ == "__main__":
    asyncio.run(debug_frontend())

