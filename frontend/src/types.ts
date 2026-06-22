export interface EvidenceRow {
  tool: string
  query: string
  rows: Record<string, unknown>[]
}

export interface ToolCallRecord {
  tool: string
  inputs: Record<string, unknown>
  play_count: number | null
}

export interface AnalystResponse {
  answer: string
  evidence: EvidenceRow[]
  tool_calls: ToolCallRecord[]
  confidence: string
  caveats: string
}
