"""
Quick test to check monitoring status after analysis
"""

import requests
import json

BACKEND_URL = "http://localhost:8000"

print("=" * 80)
print("MONITORING STATUS CHECK")
print("=" * 80)

# 1. System Health
print("\n1. SYSTEM HEALTH")
print("-" * 80)
response = requests.get(f"{BACKEND_URL}/api/monitoring/health")
data = response.json()
print(json.dumps(data, indent=2))

# 2. Agent Statistics
print("\n2. AGENT STATISTICS")
print("-" * 80)
response = requests.get(f"{BACKEND_URL}/api/monitoring/agents/statistics")
data = response.json()
print(f"Summary:")
print(json.dumps(data.get('summary', {}), indent=2))

print(f"\nTop 5 Agents by Calls:")
agents = data.get('agents', {})
sorted_agents = sorted(
    agents.items(),
    key=lambda x: x[1].get('total_calls', 0),
    reverse=True
)[:5]

for agent_id, stats in sorted_agents:
    print(f"\n{agent_id}:")
    print(f"  Total calls: {stats.get('total_calls', 0)}")
    print(f"  Successful: {stats.get('successful_calls', 0)}")
    print(f"  Failed: {stats.get('failed_calls', 0)}")
    print(f"  Avg time: {stats.get('avg_time', 0):.2f}s")
    if stats.get('last_error'):
        print(f"  Last error: {stats['last_error'][:100]}")

# 3. Diagnostics
print("\n3. DIAGNOSTICS")
print("-" * 80)
response = requests.get(f"{BACKEND_URL}/api/monitoring/diagnostics")
data = response.json()

print(f"Active analyses: {data.get('active_analyses_count', 0)}")
print(f"Recent errors: {data.get('recent_errors_count', 0)}")
print(f"Problematic agents: {data.get('problematic_agents_count', 0)}")

if data.get('problematic_agents'):
    print(f"\nProblematic Agents:")
    for agent in data['problematic_agents']:
        print(f"  {agent['agent_id']}: {agent['failure_rate']:.1f}% failure rate")
        if agent.get('last_error'):
            print(f"    Last error: {agent['last_error'][:100]}")

if data.get('recent_errors'):
    print(f"\nRecent Errors:")
    for error in data['recent_errors'][:5]:
        print(f"  [{error['timestamp']}] {error['error'][:100]}")

print("\n" + "=" * 80)
print("MONITORING CHECK COMPLETE")
print("=" * 80)

