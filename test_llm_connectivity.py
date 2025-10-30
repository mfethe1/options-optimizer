"""
Test LLM API Connectivity

Verifies that OpenAI, Anthropic, and LMStudio APIs are accessible
and responding correctly.
"""

import os
import requests
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("\n" + "="*80)
print("LLM API CONNECTIVITY TEST")
print("="*80)
print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n")

results = {
    'openai': {'status': 'unknown', 'error': None, 'response': None},
    'anthropic': {'status': 'unknown', 'error': None, 'response': None},
    'lmstudio': {'status': 'unknown', 'error': None, 'response': None}
}

# ============================================================================
# Test OpenAI API
# ============================================================================
print("-" * 80)
print("Testing OpenAI API (GPT-4)...")
print("-" * 80)

openai_key = os.getenv('OPENAI_API_KEY')
if not openai_key:
    print("✗ OPENAI_API_KEY not found in .env")
    results['openai']['status'] = 'missing_key'
else:
    print(f"✓ API Key found: {openai_key[:20]}...")
    
    try:
        headers = {
            'Authorization': f'Bearer {openai_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': 'gpt-4',
            'messages': [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'Say "OpenAI API is working!" in exactly 5 words.'}
            ],
            'temperature': 0.7,
            'max_tokens': 50
        }
        
        print("  Sending test request...")
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            message = result['choices'][0]['message']['content']
            print(f"✓ OpenAI API is working!")
            print(f"  Response: {message}")
            results['openai']['status'] = 'success'
            results['openai']['response'] = message
        else:
            print(f"✗ OpenAI API error: {response.status_code}")
            print(f"  Response: {response.text[:200]}")
            results['openai']['status'] = 'error'
            results['openai']['error'] = f"HTTP {response.status_code}"
            
    except Exception as e:
        print(f"✗ OpenAI API exception: {e}")
        results['openai']['status'] = 'exception'
        results['openai']['error'] = str(e)

print()

# ============================================================================
# Test Anthropic API
# ============================================================================
print("-" * 80)
print("Testing Anthropic API (Claude)...")
print("-" * 80)

anthropic_key = os.getenv('ANTHROPIC_API_KEY')
if not anthropic_key:
    print("✗ ANTHROPIC_API_KEY not found in .env")
    results['anthropic']['status'] = 'missing_key'
else:
    print(f"✓ API Key found: {anthropic_key[:20]}...")
    
    try:
        headers = {
            'x-api-key': anthropic_key,
            'anthropic-version': '2023-06-01',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': 'claude-3-5-sonnet-20241022',
            'messages': [
                {'role': 'user', 'content': 'Say "Anthropic API is working!" in exactly 5 words.'}
            ],
            'system': 'You are a helpful assistant.',
            'temperature': 0.7,
            'max_tokens': 50
        }
        
        print("  Sending test request...")
        response = requests.post(
            'https://api.anthropic.com/v1/messages',
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            message = result['content'][0]['text']
            print(f"✓ Anthropic API is working!")
            print(f"  Response: {message}")
            results['anthropic']['status'] = 'success'
            results['anthropic']['response'] = message
        else:
            print(f"✗ Anthropic API error: {response.status_code}")
            print(f"  Response: {response.text[:200]}")
            results['anthropic']['status'] = 'error'
            results['anthropic']['error'] = f"HTTP {response.status_code}"
            
    except Exception as e:
        print(f"✗ Anthropic API exception: {e}")
        results['anthropic']['status'] = 'exception'
        results['anthropic']['error'] = str(e)

print()

# ============================================================================
# Test LMStudio API
# ============================================================================
print("-" * 80)
print("Testing LMStudio API (Local)...")
print("-" * 80)

lmstudio_base = os.getenv('LMSTUDIO_API_BASE', 'http://localhost:1234/v1')
print(f"  Base URL: {lmstudio_base}")

try:
    headers = {'Content-Type': 'application/json'}
    
    data = {
        'model': 'local-model',
        'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'Say "LMStudio API is working!" in exactly 5 words.'}
        ],
        'temperature': 0.7,
        'max_tokens': 50
    }
    
    print("  Sending test request...")
    response = requests.post(
        f'{lmstudio_base}/chat/completions',
        headers=headers,
        json=data,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        message = result['choices'][0]['message']['content']
        print(f"✓ LMStudio API is working!")
        print(f"  Response: {message}")
        results['lmstudio']['status'] = 'success'
        results['lmstudio']['response'] = message
    else:
        print(f"✗ LMStudio API error: {response.status_code}")
        print(f"  Response: {response.text[:200]}")
        results['lmstudio']['status'] = 'error'
        results['lmstudio']['error'] = f"HTTP {response.status_code}"
        
except Exception as e:
    print(f"✗ LMStudio API exception: {e}")
    print(f"  Note: LMStudio is optional. Make sure it's running if you want to use it.")
    results['lmstudio']['status'] = 'exception'
    results['lmstudio']['error'] = str(e)

print()

# ============================================================================
# Summary
# ============================================================================
print("="*80)
print("CONNECTIVITY TEST SUMMARY")
print("="*80)

working_providers = []
for provider, result in results.items():
    status_icon = "✓" if result['status'] == 'success' else "✗"
    print(f"{status_icon} {provider.upper()}: {result['status']}")
    if result['status'] == 'success':
        working_providers.append(provider)

print()
print(f"Working Providers: {len(working_providers)}/3")

if working_providers:
    print(f"✓ Ready to use: {', '.join(working_providers)}")
    print("\nYou can now use LLM-powered agents with these providers!")
else:
    print("✗ No LLM providers are working. Please check your API keys.")

print()

# Save results
with open('llm_connectivity_test_results.json', 'w') as f:
    json.dump({
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'working_providers': working_providers
    }, f, indent=2)

print(f"Results saved to: llm_connectivity_test_results.json")
print()

