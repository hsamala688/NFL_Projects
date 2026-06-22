import { useEffect, useState } from 'react'
import type { AnalystResponse, EvidenceRow, ToolCallRecord } from './types'
import { askLive, fetchDemoQuestions } from './api'
import './App.css'

const SESSION_KEY = 'nfl_api_key'

function PasswordGate({ onUnlock }: { onUnlock: (key: string) => void }) {
  const [value, setValue] = useState('')
  const [error, setError] = useState(false)

  function submit(e: React.FormEvent) {
    e.preventDefault()
    if (!value.trim()) return
    onUnlock(value.trim())
    setError(false)
  }

  return (
    <div className="gate">
      <h1>NFL Analyst</h1>
      <p className="subtitle">Enter the access password to continue.</p>
      <form onSubmit={submit} className="gate-form">
        <input
          type="password"
          value={value}
          onChange={(e) => { setValue(e.target.value); setError(false) }}
          placeholder="Password"
          autoFocus
        />
        <button type="submit" disabled={!value.trim()}>Unlock</button>
      </form>
      {error && <p className="gate-error">Wrong password. Try again.</p>}
    </div>
  )
}

export default function App() {
  const [apiKey, setApiKey] = useState<string | null>(
    () => sessionStorage.getItem(SESSION_KEY)
  )
  const [demoQuestions, setDemoQuestions] = useState<string[]>([])
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState<AnalystResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (apiKey) fetchDemoQuestions(apiKey).then(setDemoQuestions).catch(() => {})
  }, [apiKey])

  function unlock(key: string) {
    sessionStorage.setItem(SESSION_KEY, key)
    setApiKey(key)
  }

  function lock() {
    sessionStorage.removeItem(SESSION_KEY)
    setApiKey(null)
    setResult(null)
    setError(null)
  }

  if (!apiKey) return <PasswordGate onUnlock={unlock} />

  async function submit(q: string) {
    if (!q.trim() || loading) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      setResult(await askLive(q, apiKey!))
    } catch (e) {
      if (!(e instanceof Error)) { setError('Unknown error'); return }
      if (e.message === 'WRONG_PASSWORD') { lock(); return }
      if (e.message === 'RATE_LIMITED') {
        setError('The AI service is currently rate limited. Please wait a few minutes and try again.')
        return
      }
      if (e.message === 'SERVICE_UNAVAILABLE') {
        setError('The AI service is temporarily unavailable (503). It should recover shortly -- please try again in a moment.')
        return
      }
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header>
        <div>
          <h1>NFL Analyst</h1>
          <p className="subtitle">Ask about quarterback performance from 1999 to 2025.</p>
        </div>
        <button className="lock-btn" onClick={lock} title="Lock">&#x1F512;</button>
      </header>

      {demoQuestions.length > 0 && (
        <section className="demo-questions">
          <p className="section-label">Demo questions</p>
          <div className="chips">
            {demoQuestions.map((q) => (
              <button
                key={q}
                className="chip"
                onClick={() => { setQuestion(q); submit(q) }}
              >
                {q}
              </button>
            ))}
          </div>
        </section>
      )}

      <section className="input-area">
        <textarea
          value={question}
          maxLength={500}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(question) }
          }}
          placeholder="Ask a question about NFL quarterback statistics..."
          rows={3}
        />
        <div className="input-footer">
          <span className="char-count">{question.length}/500</span>
          <button
            className="ask-btn"
            onClick={() => submit(question)}
            disabled={loading || !question.trim()}
          >
            {loading ? 'Thinking...' : 'Ask'}
          </button>
        </div>
      </section>

      {error && (
        <div className="error" role="alert">{error}</div>
      )}

      {result && <Result result={result} />}
    </div>
  )
}

function cellValue(v: unknown): string {
  if (v === null || v === undefined) return ''
  if (typeof v === 'object') return JSON.stringify(v)
  return String(v)
}

function EvidenceBlock({ ev }: { ev: EvidenceRow }) {
  const cols = ev.rows.length > 0 ? Object.keys(ev.rows[0]) : []
  return (
    <div className="evidence-block">
      <p className="ev-query">
        <span className="badge">{ev.tool}</span> {ev.query}
      </p>
      {cols.length > 0 && (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>{cols.map((c) => <th key={c}>{c}</th>)}</tr>
            </thead>
            <tbody>
              {ev.rows.map((row, ri) => (
                <tr key={ri}>
                  {cols.map((c) => <td key={c}>{cellValue(row[c])}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function describeToolCall(tc: ToolCallRecord): { title: string; detail: string } {
  if (tc.tool === 'run_cpoe') {
    const season = tc.inputs.season ?? 'unknown'
    return {
      title: 'Completion Percentage Over Expected (CPOE)',
      detail: `Computed CPOE for all quarterbacks in the ${season} season${tc.play_count != null ? ` across ${tc.play_count.toLocaleString()} plays` : ''}.`,
    }
  }
  if (tc.tool === 'query_marts') {
    return {
      title: 'NFL Statistics Database Query',
      detail: (tc.inputs.sql as string) ?? '',
    }
  }
  return {
    title: tc.tool,
    detail: JSON.stringify(tc.inputs, null, 2),
  }
}

function ToolCallBlock({ tc }: { tc: ToolCallRecord }) {
  const { title, detail } = describeToolCall(tc)
  const isSql = tc.tool === 'query_marts'
  return (
    <div className="tool-call">
      <p className="tool-title">
        <span className="badge">{tc.tool}</span> {title}
      </p>
      {isSql
        ? <pre className="tool-detail sql">{detail}</pre>
        : <p className="tool-detail">{detail}</p>
      }
    </div>
  )
}

function Result({ result }: { result: AnalystResponse }) {
  return (
    <div className="result">
      <section>
        <h2>Answer</h2>
        <p className="answer-text">{result.answer}</p>
        <div className="meta">
          <span className="meta-item">Confidence: {result.confidence}</span>
          {result.caveats && result.caveats !== 'None' && (
            <span className="meta-item">Caveats: {result.caveats}</span>
          )}
        </div>
      </section>

      {result.tool_calls.length > 0 && (
        <section>
          <h2>Tool calls</h2>
          {result.tool_calls.map((tc, i) => <ToolCallBlock key={i} tc={tc} />)}
        </section>
      )}

      {result.evidence.length > 0 && (
        <section>
          <h2>Evidence</h2>
          {result.evidence.map((ev, i) => <EvidenceBlock key={i} ev={ev} />)}
        </section>
      )}
    </div>
  )
}
