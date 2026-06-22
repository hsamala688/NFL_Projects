import type { AnalystResponse } from './types'

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export async function fetchDemoQuestions(apiKey: string): Promise<string[]> {
  const res = await fetch(`${API_BASE}/demo/questions`, {
    headers: { 'X-API-Key': apiKey },
  })
  if (!res.ok) return []
  return res.json() as Promise<string[]>
}

export async function askDemo(question: string): Promise<AnalystResponse> {
  const res = await fetch(`${API_BASE}/demo/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })
  if (!res.ok) {
    const data = await res.json().catch(() => ({})) as { detail?: string }
    throw new Error(data.detail ?? `HTTP ${res.status}`)
  }
  return res.json() as Promise<AnalystResponse>
}

export async function askLive(question: string, apiKey: string): Promise<AnalystResponse> {
  const res = await fetch(`${API_BASE}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-API-Key': apiKey },
    body: JSON.stringify({ question }),
  })
  if (res.status === 403) throw new Error('WRONG_PASSWORD')
  if (res.status === 429) throw new Error('RATE_LIMITED')
  if (res.status === 503) throw new Error('SERVICE_UNAVAILABLE')
  if (!res.ok) {
    const data = await res.json().catch(() => ({})) as { detail?: string }
    throw new Error(data.detail ?? `HTTP ${res.status}`)
  }
  return res.json() as Promise<AnalystResponse>
}
