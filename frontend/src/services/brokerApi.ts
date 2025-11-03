/**
 * Multi-Broker Connectivity API Client
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface BrokerHealth {
  broker_type: string;
  status: 'healthy' | 'degraded' | 'offline' | 'maintenance';
  latency_ms: number;
  last_check: string;
  error_count: number;
  error_message: string | null;
  uptime_pct: number;
}

export interface BrokerCredentials {
  broker_type: string;
  api_key?: string;
  api_secret?: string;
  account_id?: string;
  client_id?: string;
  client_secret?: string;
}

export interface BrokerStatus {
  total_brokers: number;
  healthy_brokers: number;
  primary_broker: string | null;
  timestamp: string;
}

export async function connectBroker(credentials: BrokerCredentials): Promise<{ status: string; message: string }> {
  const response = await fetch(`${API_BASE_URL}/api/brokers/connect`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(credentials),
  });
  if (!response.ok) throw new Error('Failed to connect broker');
  return response.json();
}

export async function disconnectBroker(brokerType: string): Promise<{ status: string; message: string }> {
  const response = await fetch(`${API_BASE_URL}/api/brokers/disconnect/${brokerType}`, {
    method: 'DELETE',
  });
  if (!response.ok) throw new Error('Failed to disconnect broker');
  return response.json();
}

export async function getBrokerHealth(): Promise<BrokerHealth[]> {
  const response = await fetch(`${API_BASE_URL}/api/brokers/health`);
  if (!response.ok) throw new Error('Failed to get broker health');
  return response.json();
}

export async function getBrokerStatus(): Promise<BrokerStatus> {
  const response = await fetch(`${API_BASE_URL}/api/brokers/status`);
  if (!response.ok) throw new Error('Failed to get broker status');
  return response.json();
}

export function getStatusColor(status: string): string {
  switch (status) {
    case 'healthy': return 'text-green-600';
    case 'degraded': return 'text-yellow-600';
    case 'offline': return 'text-red-600';
    case 'maintenance': return 'text-gray-600';
    default: return 'text-gray-600';
  }
}

export function getStatusBadgeColor(status: string): string {
  switch (status) {
    case 'healthy': return 'bg-green-100 text-green-800';
    case 'degraded': return 'bg-yellow-100 text-yellow-800';
    case 'offline': return 'bg-red-100 text-red-800';
    case 'maintenance': return 'bg-gray-100 text-gray-800';
    default: return 'bg-gray-100 text-gray-800';
  }
}
