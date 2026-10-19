import { useEffect, useState, type FormEvent } from 'react'
import './App.css'

type PipelineSummary = {
  total_opportunities: number
  won_opportunities: number
  lost_opportunities: number
  open_opportunities: number
  closed_revenue: number
}

type Opportunity = {
  opportunity_id: string
  account: string
  product: string
  sales_agent: string
  deal_stage: string
  close_value: number
  win_probability: number | null
  engage_date: string
}

type ForecastPoint = { month: string; actual: number | null; forecast: number | null }
type RevenueForecast = {
  history: ForecastPoint[]
  forecast: ForecastPoint[]
  weighted_open_pipeline: number
  expected_monthly_revenue: number
  method: string
}
type IntelligenceAnswer = {
  intent: string
  status: string
  answer: string
  data: {
    closed_revenue?: number
    open_opportunities?: number
    total_opportunities?: number
    win_rate?: number
    stages?: Record<string, number>
    results?: Array<{
      opportunity_id: string
      account: string | null
      product: string
      sales_agent: string
      deal_stage: string
      close_value: number
      engage_date: string | null
    }>
  } | null
  sources: Array<{ type: string; source?: string; document_id?: string; title?: string; excerpt?: string }>
  next_action: string | null
}

const fallbackSummary: PipelineSummary = {
  total_opportunities: 8800,
  won_opportunities: 4238,
  lost_opportunities: 2473,
  open_opportunities: 2089,
  closed_revenue: 0,
}

const apiUrl = '/api/v1/pipeline/summary'

function formatNumber(value: number) {
  return new Intl.NumberFormat('en-US').format(value)
}

function formatCurrency(value: number) {
  if (!value) return 'Awaiting revenue data'
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', notation: 'compact' }).format(value)
}

function probabilityLabel(value: number | null) {
  return value === null ? 'Pending' : `${Math.round(value * 100)}%`
}

function forecastHeight(value: number, maximum: number) {
  return `${Math.max(8, Math.round((value / Math.max(maximum, 1)) * 100))}%`
}

function App() {
  const [summary, setSummary] = useState<PipelineSummary>(fallbackSummary)
  const [connected, setConnected] = useState(false)
  const [view, setView] = useState<'overview' | 'pipeline' | 'forecast'>('overview')
  const [opportunities, setOpportunities] = useState<Opportunity[]>([])
  const [pipelineLoading, setPipelineLoading] = useState(false)
  const [stageFilter, setStageFilter] = useState('All stages')
  const [search, setSearch] = useState('')
  const [revenueForecast, setRevenueForecast] = useState<RevenueForecast | null>(null)
  const [forecastLoading, setForecastLoading] = useState(false)
  const [intelligenceQuestion, setIntelligenceQuestion] = useState('')
  const [intelligenceAnswer, setIntelligenceAnswer] = useState<IntelligenceAnswer | null>(null)
  const [intelligenceLoading, setIntelligenceLoading] = useState(false)
  const [intelligenceError, setIntelligenceError] = useState('')

  useEffect(() => {
    fetch(apiUrl)
      .then((response) => {
        if (!response.ok) throw new Error('API unavailable')
        return response.json() as Promise<PipelineSummary>
      })
      .then((data) => {
        setSummary(data)
        setConnected(true)
      })
      .catch(() => setConnected(false))
  }, [])

  useEffect(() => {
    if (view !== 'pipeline') return
    setPipelineLoading(true)
    const params = new URLSearchParams({ limit: '40', stage: stageFilter })
    if (search.trim()) params.set('search', search.trim())
    fetch(`/api/v1/pipeline/opportunities?${params}`)
      .then((response) => response.json())
      .then((data: { items: Opportunity[] }) => setOpportunities(data.items))
      .finally(() => setPipelineLoading(false))
  }, [view, stageFilter, search])

  useEffect(() => {
    if (view !== 'forecast' || revenueForecast) return
    setForecastLoading(true)
    fetch('/api/v1/forecast/revenue')
      .then((response) => response.json())
      .then((data: RevenueForecast) => setRevenueForecast(data))
      .finally(() => setForecastLoading(false))
  }, [view, revenueForecast])

  function submitIntelligenceQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const question = intelligenceQuestion.trim()
    if (!question) return
    setIntelligenceLoading(true)
    setIntelligenceError('')
    fetch('/api/v1/intelligence/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    })
      .then((response) => {
        if (!response.ok) throw new Error('Intelligence service unavailable')
        return response.json() as Promise<IntelligenceAnswer>
      })
      .then((data) => setIntelligenceAnswer(data))
      .catch((error: Error) => setIntelligenceError(error.message))
      .finally(() => setIntelligenceLoading(false))
  }

  const winRate = Math.round((summary.won_opportunities / Math.max(summary.won_opportunities + summary.lost_opportunities, 1)) * 100)

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand"><span className="brand-mark">S</span><span>Sales<span className="brand-accent">Signal</span></span></div>
        <div className="workspace-label">WORKSPACE</div>
        <nav>
          <a className={`nav-item ${view === 'overview' ? 'active' : ''}`} href="#overview" onClick={() => setView('overview')}><span>◈</span> Overview</a>
          <a className={`nav-item ${view === 'pipeline' ? 'active' : ''}`} href="#pipeline-details" onClick={() => setView('pipeline')}><span>⌁</span> Pipeline</a>
          <a className={`nav-item ${view === 'forecast' ? 'active' : ''}`} href="#forecast" onClick={() => setView('forecast')}><span>◌</span> Forecast</a>
          <a className="nav-item" href="#team"><span>◎</span> Team performance</a>
        </nav>
        <div className="sidebar-bottom">
          <div className="model-status"><span className="status-dot" /> Model online<div>Win probability engine</div></div>
          <div className="profile"><span className="avatar">NS</span><span><strong>Najya S.</strong><small>Sales manager</small></span><span className="more">···</span></div>
        </div>
      </aside>

      <section className="content" id="overview">
        <header className="topbar"><div><p className="eyebrow">MONDAY, SEPTEMBER 22, 2026</p><h1>Revenue command center</h1></div><div className="top-actions"><span className={connected ? 'connection live' : 'connection'}><i />{connected ? 'Live data' : 'Demo data'}</span><button className="icon-button" aria-label="Notifications">⌁</button><button className="help-button">?</button></div></header>
        <div className="hero-line"><div><h2>Good morning, Najya.</h2><p>Here is what deserves your attention today.</p></div><button className="primary-button">+ Add opportunity</button></div>
        <section className="intelligence-panel" aria-label="Sales intelligence">
          <div className="intelligence-heading"><div><p className="eyebrow">SALES INTELLIGENCE</p><h2>Ask the company signal</h2><p>Search verified pipeline data and approved company documents.</p></div><span className="intelligence-badge">GROUNDED</span></div>
          <form className="intelligence-form" onSubmit={submitIntelligenceQuestion}><input value={intelligenceQuestion} onChange={(event) => setIntelligenceQuestion(event.target.value)} placeholder="Ask about pipeline, revenue, or a sales playbook..." aria-label="Ask a sales intelligence question" /><button className="primary-button" type="submit" disabled={intelligenceLoading}>{intelligenceLoading ? 'Searching...' : 'Ask signal'}</button></form>
          {intelligenceError && <p className="intelligence-error">{intelligenceError}</p>}
          {intelligenceAnswer && <div className="intelligence-answer"><div><span className="answer-status">{intelligenceAnswer.status.replace('_', ' ')}</span><p>{intelligenceAnswer.answer}</p>{intelligenceAnswer.data && <div className="answer-data">{Object.entries(intelligenceAnswer.data).filter(([key]) => key !== 'results' && key !== 'stages').map(([key, value]) => typeof value === 'number' && <span key={key}><small>{key.replaceAll('_', ' ')}</small><strong>{key.includes('rate') ? `${Math.round(value * 100)}%` : value.toLocaleString()}</strong></span>)}</div>}{intelligenceAnswer.data?.stages && <div className="stage-answer">{Object.entries(intelligenceAnswer.data.stages).map(([stage, count]) => <span key={stage}><small>{stage}</small><strong>{count.toLocaleString()}</strong></span>)}</div>}{intelligenceAnswer.data?.results && <div className="intelligence-results">{intelligenceAnswer.data.results.map((result) => <div className="intelligence-result" key={result.opportunity_id}><strong>{result.opportunity_id}</strong><span>{result.account ?? 'Unassigned account'}</span><span>{result.sales_agent}</span><span className={`table-stage ${result.deal_stage.toLowerCase()}`}>{result.deal_stage}</span><span>{formatCurrency(result.close_value)}</span></div>)}</div>}{intelligenceAnswer.next_action && <div className="next-action"><small>Suggested next action</small><span>{intelligenceAnswer.next_action}</span></div>}</div><div className="answer-sources"><small>Sources</small>{intelligenceAnswer.sources.length ? intelligenceAnswer.sources.map((source, index) => <div className="source-item" key={`${source.type}-${source.source ?? source.document_id}-${index}`}><span>{source.title ?? source.source ?? source.document_id ?? source.type}</span>{source.excerpt && <small>{source.excerpt}</small>}</div>) : <span>No supporting source</span>}</div></div>}
        </section>
        {view === 'pipeline' && <section className="pipeline-details" id="pipeline-details">
          <div className="pipeline-details-heading"><div><p className="eyebrow">PIPELINE WORKBENCH</p><h2>Opportunity details</h2><p>Search every deal and prioritize the next conversation with model-guided signals.</p></div><button className="outline-button" onClick={() => setView('overview')}>← Back to overview</button></div>
          <div className="pipeline-toolbar"><label className="search-field"><span>⌕</span><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search account, product, owner..." /></label><select value={stageFilter} onChange={(event) => setStageFilter(event.target.value)}><option>All stages</option><option>Prospecting</option><option>Engaging</option><option>Won</option><option>Lost</option></select></div>
          <div className="opportunity-table-wrap"><table className="opportunity-table"><thead><tr><th>Opportunity</th><th>Account</th><th>Owner</th><th>Stage</th><th>Value</th><th>Win probability</th><th>Engaged</th></tr></thead><tbody>{pipelineLoading ? <tr><td colSpan={7} className="empty-row">Loading pipeline signals...</td></tr> : opportunities.length === 0 ? <tr><td colSpan={7} className="empty-row">No opportunities match these filters.</td></tr> : opportunities.map((opportunity) => <tr key={opportunity.opportunity_id}><td><strong>{opportunity.opportunity_id}</strong><small>{opportunity.product}</small></td><td>{opportunity.account}</td><td>{opportunity.sales_agent}</td><td><span className={`table-stage ${opportunity.deal_stage.toLowerCase()}`}>{opportunity.deal_stage}</span></td><td>{formatCurrency(opportunity.close_value)}</td><td><span className={`probability ${opportunity.win_probability !== null && opportunity.win_probability >= .7 ? 'high' : ''}`}>{probabilityLabel(opportunity.win_probability)}</span></td><td>{opportunity.engage_date}</td></tr>)}</tbody></table></div>
        </section>}
        {view === 'forecast' && <section className="forecast-view" id="forecast"><div className="pipeline-details-heading"><div><p className="eyebrow">REVENUE PLANNING</p><h2>Expected revenue</h2><p>Historical wins meet probability-weighted pipeline to show what the next quarter could hold.</p></div><button className="outline-button" onClick={() => setView('overview')}>← Back to overview</button></div>{forecastLoading ? <div className="panel forecast-loading">Building forecast from CRM signals...</div> : revenueForecast && <><div className="forecast-metrics"><article className="metric-card highlight"><div className="metric-label">EXPECTED MONTHLY REVENUE</div><strong>{formatCurrency(revenueForecast.expected_monthly_revenue)}</strong><small>Baseline plus weighted open pipeline</small></article><article className="metric-card"><div className="metric-label">WEIGHTED OPEN PIPELINE</div><strong>{formatCurrency(revenueForecast.weighted_open_pipeline)}</strong><small>Deal value x model probability</small></article></div><div className="panel forecast-panel"><div className="panel-heading"><div><p className="eyebrow">REVENUE TRAJECTORY</p><h3>Actuals and outlook</h3></div><span className="forecast-legend"><i className="actual-key" /> Actual <i className="outlook-key" /> Outlook</span></div><div className="forecast-chart">{[...revenueForecast.history, ...revenueForecast.forecast].map((point) => { const value = point.actual ?? point.forecast ?? 0; const max = Math.max(...[...revenueForecast.history, ...revenueForecast.forecast].map((item) => item.actual ?? item.forecast ?? 0)); return <div className="forecast-column" key={point.month}><div className={`forecast-bar ${point.actual === null ? 'outlook' : ''}`} style={{ height: forecastHeight(value, max) }}><span>{formatCurrency(value)}</span></div><small>{point.month}</small></div> })}</div><div className="forecast-note"><span className="signal-icon green">≈</span><span><strong>How to read this</strong><br />{revenueForecast.method}. Use it for planning conversations, not as a guaranteed commitment.</span></div></div></>}</section>}
        <section className={view !== 'overview' ? 'view-hidden' : 'metric-grid'} aria-label="Pipeline metrics">
          <article className="metric-card highlight"><div className="metric-label">TOTAL OPPORTUNITIES <span>↗ 12.4%</span></div><strong>{formatNumber(summary.total_opportunities)}</strong><small>Across every stage</small><div className="sparkline"><b /><b /><b /><b /><b /><b /><b /><b /><b /></div></article>
          <article className="metric-card"><div className="metric-label">CLOSED REVENUE <span className="muted">THIS PERIOD</span></div><strong>{formatCurrency(summary.closed_revenue)}</strong><small>Won opportunities</small><div className="metric-line"><i style={{ width: `${Math.min(winRate + 12, 96)}%` }} /></div></article>
          <article className="metric-card"><div className="metric-label">WIN RATE <span>↗ 4.8%</span></div><strong>{winRate}%</strong><small>Closed opportunities</small><div className="metric-line lime"><i style={{ width: `${winRate}%` }} /></div></article>
          <article className="metric-card"><div className="metric-label">OPEN PIPELINE <span className="muted">NEEDS ACTION</span></div><strong>{formatNumber(summary.open_opportunities)}</strong><small>Prospecting + engaging</small><div className="metric-line amber"><i style={{ width: '64%' }} /></div></article>
        </section>
        <section className={view !== 'overview' ? 'view-hidden' : 'dashboard-grid'}>
          <article className="panel pipeline-panel" id="pipeline"><div className="panel-heading"><div><p className="eyebrow">PIPELINE HEALTH</p><h3>Deals by stage</h3></div><button className="select-button">All time <span>⌄</span></button></div><div className="stage-list"><div className="stage-row"><span className="stage-name"><i className="stage-dot prospecting" /> Prospecting</span><strong>{Math.round(summary.open_opportunities * .24)}</strong><div className="bar"><i style={{ width: '42%' }} /></div><small>24%</small></div><div className="stage-row"><span className="stage-name"><i className="stage-dot engaging" /> Engaging</span><strong>{Math.round(summary.open_opportunities * .76)}</strong><div className="bar"><i style={{ width: '68%' }} /></div><small>38%</small></div><div className="stage-row"><span className="stage-name"><i className="stage-dot won" /> Won</span><strong>{formatNumber(summary.won_opportunities)}</strong><div className="bar"><i style={{ width: '88%' }} /></div><small>48%</small></div><div className="stage-row"><span className="stage-name"><i className="stage-dot lost" /> Lost</span><strong>{formatNumber(summary.lost_opportunities)}</strong><div className="bar"><i style={{ width: '51%' }} /></div><small>28%</small></div></div><div className="panel-foot">Pipeline coverage <strong>3.4x</strong><span>Target 3.0x</span></div></article>
          <article className="panel attention-panel"><div className="panel-heading"><div><p className="eyebrow">AI SIGNALS</p><h3>Worth a closer look</h3></div><span className="signal-count">03</span></div><div className="signal-list"><div className="signal"><span className="signal-icon orange">!</span><div><strong>Open deals are aging</strong><p>12 opportunities have had no activity in 14+ days.</p></div><span className="arrow">→</span></div><div className="signal"><span className="signal-icon green">↗</span><div><strong>Strong closing momentum</strong><p>Win rate is up 4.8% compared with last period.</p></div><span className="arrow">→</span></div><div className="signal"><span className="signal-icon blue">◎</span><div><strong>Team follow-up gap</strong><p>Three high-value deals need an owner update.</p></div><span className="arrow">→</span></div></div><button className="text-button">View all signals <span>→</span></button></article>
        </section>
        <section className={view !== 'overview' ? 'view-hidden' : 'bottom-strip'} id="forecast"><div><p className="eyebrow">NEXT BEST ACTION</p><h3>Focus on momentum, not just volume.</h3><p>Prioritize high-value opportunities with a clear path to close. Your next review is in 2 days.</p></div><button className="outline-button" onClick={() => setView('forecast')}>Open revenue forecast <span>→</span></button></section>
        <footer><span>SALES SIGNAL <b>•</b> Intelligence for decisive teams</span><span>Last synced just now</span></footer>
      </section>
    </main>
  )
}

export default App
