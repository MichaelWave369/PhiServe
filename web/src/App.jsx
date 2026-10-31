import { useEffect, useRef, useState } from 'react';
import {
  DEFAULT_ENDPOINT, friendlyConnectionError, getJSON, normalizeEndpoint, streamChat,
} from './api.js';
import { ECOSYSTEM_APPS, INVENTORY_SCOPE, inventoryCounts } from './ecosystem.js';

const GH = 'https://github.com/MichaelWave369/PhiServe';
const PHISTREAM = 'https://github.com/MichaelWave369/PhiStream';
const VIEWS = [
  { key: 'overview', icon: '◉', label: 'Overview', desc: 'System overview' },
  { key: 'models', icon: '▤', label: 'Model library', desc: 'Registered GGUF' },
  { key: 'ecosystem', icon: '◎', label: 'Ecosystem', desc: 'Applications & bridges' },
  { key: 'playground', icon: '⌁', label: 'Playground', desc: 'Live local chat' },
  { key: 'field', icon: '◇', label: 'Field lab', desc: 'Hardware evidence' },
];
const DEMO_MODELS = [
  { name: 'example-llama-gguf', size: 4127195136, modified_at: null, demo: true },
  { name: 'example-smol-gguf', size: 1875617792, modified_at: null, demo: true },
  { name: 'example-code-gguf', size: 3027951616, modified_at: null, demo: true },
];
const PROMPTS = ['Explain offline-first inference in two sentences.',
  'Write an imaginative name for a tiny sci-fi spaceship.',
  'Describe the difference between GGUF and safetensors.'];
const INITIAL = [{
  id: 'welcome', role: 'assistant',
  content: 'Welcome to the PhiServe Console. This is a clearly labeled interface demo, not an AI response. To run real inference, connect to a local PhiServe daemon on this same computer.',
  demo: true,
}];
const STEPS = [
  {
    num: '01', label: 'Prepare a private token',
    code: 'phiserve auth create --output "$HOME\\.phiserve\\bridge-token.txt"',
    detail: 'Only your local user should be able to read this token file.',
  },
  {
    num: '02', label: 'Start the shared daemon',
    code: 'phiserve daemon --llama-server "C:\\llama.cpp\\build\\bin\\Release\\llama-server.exe" --gpu-layers 20 --browser-origin "https://michaelwave369.github.io" --api-key-file "$HOME\\.phiserve\\bridge-token.txt"',
    detail: 'Requires a trusted local llama-server and at least one registered GGUF. The daemon binds to loopback only.',
  },
  {
    num: '03', label: 'Run a real hardware check',
    code: 'phiserve acceptance field-run YOUR_ALIAS --llama-server "C:\\llama.cpp\\build\\bin\\Release\\llama-server.exe" --gpu-layers 20 --sample-nvidia --output "gpu-field.json"',
    detail: 'Optional process-memory telemetry is a driver observation, not independent proof of CUDA execution.',
  },
];
function PulseIcon() {
  return <svg viewBox="0 0 54 54" fill="none" aria-hidden="true">
    <circle cx="27" cy="27" r="25" stroke="currentColor" strokeOpacity=".32" />
    <circle cx="27" cy="27" r="15" stroke="currentColor" strokeOpacity=".6" />
    <path d="M6 28h10l4-11 7 22 6-16 3 5h12" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
    <circle cx="45" cy="9" r="3.5" fill="#a696fc" />
  </svg>;
}
function SectionHeading({ kicker, title, aside }) {
  return <div className="section-titlebar">
    <div><p className="eyebrow">{kicker}</p><h2>{title}</h2></div>
    {aside ? <span className="section-aside">{aside}</span> : null}
  </div>;
}
function Flag({ children, variant = 'neutral' }) {
  return <span className={'flag ' + variant}><span className="flag-dot" />{children}</span>;
}
function Orbital({ connected, active }) {
  return <div className={'orbital ' + (connected ? 'is-connected' : '')} aria-hidden="true">
    <div className="orbital-grid" /><div className="ring ring-one" />
    <div className="ring ring-two" /><div className="ring ring-three" />
    <span className="orbit-node orbit-node-a">φ</span>
    <span className="orbit-node orbit-node-b">03</span>
    <span className="orbit-node orbit-node-c">09</span>
    <span className="orbit-core"><strong>Φ</strong><small>{connected ? 'LOCAL' : 'DEMO'}</small></span>
    <span className="orbit-label">{active || 'SOVEREIGN RUNTIME'}</span>
  </div>;
}
function ModelCard({ model, resident, selected, onSelect, live }) {
  const bytes = Number(model.size);
  const size = Number.isFinite(bytes) && bytes > 0
    ? (bytes / (1024 ** 3)).toFixed(2) + ' GiB' : 'Size not reported';
  return <article className={'model-card ' + (selected ? 'selected' : '')}>
    <div className="model-top"><span className="model-icon">▥</span>
      <Flag variant={model.demo ? 'demo' : resident ? 'mint' : 'neutral'}>
        {model.demo ? 'SIMULATED' : resident ? 'RESIDENT' : 'REGISTERED'}
      </Flag>
    </div>
    <h3 title={model.name}>{model.name}</h3>
    <p className="muted">{model.demo ? 'Example catalog entry' : 'Local GGUF alias'}</p>
    <div className="model-footer"><span>{size}</span>
      <button className="tiny-button" onClick={onSelect} disabled={!live}
        title={live ? 'Use this model in the playground' : 'Connect a local daemon to select a real model'}>
        {selected ? 'Selected' : 'Use model'} <span aria-hidden="true">↗</span>
      </button>
    </div>
  </article>;
}
function Message({ message }) {
  return <div className={'message ' + (message.role === 'user' ? 'user' : 'assistant')}>
    <div className="avatar">{message.role === 'user' ? 'U' : 'Φ'}</div>
    <div className="message-content">
      <div className="message-meta">{message.role === 'user' ? 'YOU' : 'PHISERVE'}
        {message.demo ? <span className="demo-tag">INTERFACE DEMO</span> : null}
      </div>
      <div className="message-body">{message.content || (message.pending ? <span className="typing">Generating local text<span>...</span></span> : '')}</div>
    </div>
  </div>;
}
function StatCard({ icon, label, value, sub }) {
  return <div className="stat-card">
    <div className="stat-top"><span className="stat-symbol">{icon}</span><span className="tiny-glyph">↗</span></div>
    <div className="stat-value">{value}</div><div className="stat-label">{label}</div><div className="stat-sub">{sub}</div>
  </div>;
}
export default function App() {
  const [view, setView] = useState('overview');
  const [endpoint, setEndpoint] = useState(DEFAULT_ENDPOINT);
  const [token, setToken] = useState('');
  const [phase, setPhase] = useState('demo');
  const [connectionError, setConnectionError] = useState('');
  const [health, setHealth] = useState(null);
  const [models, setModels] = useState([]);
  const [resident, setResident] = useState(null);
  const [selected, setSelected] = useState('');
  const [messages, setMessages] = useState(INITIAL);
  const [draft, setDraft] = useState('');
  const [busy, setBusy] = useState(false);
  const [chatError, setChatError] = useState('');
  const [copied, setCopied] = useState(-1);
  const connectAbort = useRef(null);
  const chatAbort = useRef(null);
  const live = phase === 'connected';
  const displayModels = live ? models : DEMO_MODELS;
  const activeModel = live ? selected : '';
  const selectedView = VIEWS.find((item) => item.key === view) || VIEWS[0];

  useEffect(() => {
    return () => {
      connectAbort.current?.abort();
      chatAbort.current?.abort();
    };
  }, []);

  async function connect() {
    connectAbort.current?.abort();
    chatAbort.current?.abort();
    let base;
    try { base = normalizeEndpoint(endpoint); }
    catch (error) { setConnectionError(friendlyConnectionError(error)); return; }
    if (!token.trim()) { setConnectionError('A local bearer token is required.'); return; }
    const controller = new AbortController();
    connectAbort.current = controller;
    const timeout = setTimeout(() => controller.abort(), 10000);
    setConnectionError('');
    setPhase('connecting');
    try {
      const [h, tags, running] = await Promise.all([
        getJSON(base, '/health', token.trim(), controller.signal),
        getJSON(base, '/api/tags', token.trim(), controller.signal),
        getJSON(base, '/api/ps', token.trim(), controller.signal),
      ]);
      if (h?.status !== 'ok' || !Array.isArray(tags?.models) || !Array.isArray(running?.models)) {
        throw new Error('Unexpected response from local PhiServe. Check daemon version and logs.');
      }
      const entries = tags.models.filter((m) => typeof m?.name === 'string' && m.name.length);
      const runningAlias = running.models[0]?.name || h.active_model || null;
      setHealth(h);
      setModels(entries);
      setResident(runningAlias);
      setSelected(entries.some((m) => m.name === runningAlias) ? runningAlias : (entries[0]?.name || ''));
      setMessages([]);
      setPhase('connected');
      setView('overview');
    } catch (error) {
      if (controller.signal.aborted) setConnectionError('Connection timed out or was cancelled.');
      else setConnectionError(friendlyConnectionError(error));
      setPhase('demo');
      setHealth(null);
      setModels([]);
      setResident(null);
      setSelected('');
    } finally {
      clearTimeout(timeout);
      if (connectAbort.current === controller) connectAbort.current = null;
    }
  }
  function disconnect() {
    connectAbort.current?.abort();
    chatAbort.current?.abort();
    setToken('');
    setHealth(null);
    setModels([]);
    setResident(null);
    setSelected('');
    setMessages(INITIAL);
    setPhase('demo');
    setBusy(false);
    setConnectionError('');
    setChatError('');
  }
  async function refresh() {
    if (!live || !token.trim()) return;
    try {
      const [h, tags, running] = await Promise.all([
        getJSON(endpoint, '/health', token),
        getJSON(endpoint, '/api/tags', token),
        getJSON(endpoint, '/api/ps', token),
      ]);
      if (h.status !== 'ok' || !Array.isArray(tags.models) || !Array.isArray(running.models)) {
        throw new Error('Invalid local daemon response.');
      }
      setHealth(h);
      setModels(tags.models.filter((m) => typeof m?.name === 'string'));
      setResident(running.models[0]?.name || h.active_model || null);
      setChatError('');
    } catch (error) {
      setChatError(friendlyConnectionError(error));
    }
  }
  async function send() {
    const content = draft.trim();
    if (!content || busy) return;
    if (content.length > 4000) { setChatError('Please keep prompts under 4,000 characters.'); return; }
    if (live && !selected) { setChatError('No GGUF models registered. Add a model to your local daemon first.'); return; }
    setChatError('');
    setDraft('');
    const id = Date.now().toString(36) + Math.random().toString(36).slice(2);
    const userMessage = { id: id + 'u', role: 'user', content };
    if (!live) {
      setMessages((previous) => [...previous, userMessage, {
        id: id + 'a', role: 'assistant', demo: true,
        content: 'Interface demo only. No model received this prompt. Connect a local PhiServe daemon using the panel above to generate real text.',
      }]);
      return;
    }
    const previousMessages = messages.filter((m) => !m.demo && !m.pending &&
      (m.role === 'user' || m.role === 'assistant') && m.content.trim()).slice(-10);
    const outbound = [...previousMessages.map((m) => ({ role: m.role, content: m.content })),
      { role: 'user', content }];
    const answerId = id + 'a';
    setMessages((previous) => [...previous, userMessage, {
      id: answerId, role: 'assistant', content: '', pending: true,
    }]);
    const controller = new AbortController();
    chatAbort.current = controller;
    setBusy(true);
    let gotAny = false;
    try {
      for await (const chunk of streamChat({
        endpoint, token, model: selected, messages: outbound, signal: controller.signal,
      })) {
        gotAny = true;
        setMessages((previous) => previous.map((m) => m.id === answerId
          ? { ...m, pending: false, content: (m.content + chunk).slice(0, 100000) } : m));
      }
      if (!gotAny) throw new Error('The local model finished without returning text.');
      setMessages((previous) => previous.map((m) => m.id === answerId ? { ...m, pending: false } : m));
      refresh();
    } catch (error) {
      setChatError(controller.signal.aborted ? 'Generation cancelled.' : friendlyConnectionError(error));
      setMessages((previous) => previous.map((m) => m.id === answerId
        ? { ...m, pending: false, content: m.content || (controller.signal.aborted ? 'Generation cancelled.' : 'Local generation failed. See diagnostics below.') } : m));
    } finally {
      if (chatAbort.current === controller) chatAbort.current = null;
      setBusy(false);
    }
  }
  function chooseModel(alias) {
    if (busy) return;
    setSelected(alias);
    setMessages([]);
    setChatError('');
    setView('playground');
  }
  async function copy(text, index) {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(index);
    } catch {
      setCopied(-2);
    }
  }
  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><span className="brand-mark"><PulseIcon /></span><span className="brand-copy"><strong>Φ PhiServe</strong><small>LOCAL INTELLIGENCE</small></span></div>
      <div className="sidebar-caption">WORKSPACE <span>01 / LOCAL</span></div>
      <nav className="nav-list" aria-label="Main navigation">
        {VIEWS.map((item) => <button key={item.key} onClick={() => setView(item.key)}
            className={'nav-item ' + (view === item.key ? 'active' : '')}>
          <span className="nav-symbol">{item.icon}</span>
          <span><strong>{item.label}</strong><small>{item.desc}</small></span>
          {view === item.key ? <span className="nav-edge" /> : null}
        </button>)}
      </nav>
      <div className="sidebar-bottom">
        <div className="sidebar-info"><span className="small-light" /><div>
          <strong>{live ? 'LOCAL DAEMON' : 'DEMONSTRATION'}</strong>
          <small>{live ? 'Authorized browser session' : 'Nothing connected'}</small></div></div>
        <a href={GH} target="_blank" rel="noopener noreferrer" className="sidebar-link">VIEW SOURCE <span>↗</span></a>
        <div className="sidebar-foot">φ369 LABS / ENTER THE FIELD</div>
      </div>
    </aside>
    <main className="main-area">
      <header className="topbar">
        <div className="breadcrumb">Φ / PHISERVE <span>/</span> {selectedView.label.toUpperCase()}</div>
        <div className="top-right">
          <span className="top-clock">OFFLINE FIRST · NO CLOUD ACCOUNT</span>
          <Flag variant={live ? 'mint' : 'demo'}>{live ? 'LOCAL SESSION' : phase === 'connecting' ? 'CONNECTING' : 'DEMO MODE'}</Flag>
        </div>
      </header>
      <div className="content">
        <div className="headline-row"><div className="headline-title"><p className="eyebrow">YOUR MODELS. YOUR MACHINE. YOUR RULES.</p>
          <h1>{view === 'overview' ? <>Intelligence,<br /><em>under your control.</em></> :
            view === 'models' ? <>Your models.<br /><em>One local gateway.</em></> :
            view === 'playground' ? <>Make the connection.<br /><em>Start a conversation.</em></> :
            view === 'ecosystem' ? <>One local brain.<br /><em>Many independent tools.</em></> :
            <>Evidence before<br /><em>assumptions.</em></>}</h1>
          <p className="headline-sub">{view === 'overview' ?
            'A sovereign control surface for PhiServe. Inspect your local model runtime, explore the model catalog, and run text generation without a cloud intermediary.' :
            view === 'models' ? 'View registered GGUF model aliases. Research-only PhiStream checkpoints do not appear in the chat library.' :
            view === 'playground' ? 'A real streaming text playground when a local daemon is connected. In demo mode, responses are clearly simulated.' :
            view === 'ecosystem' ? 'What the surrounding Φ applications actually provide, how they relate to PhiServe, and which bridges still need real acceptance.' :
            'Real hardware measurements belong on your machine. Inspect the workflow without turning test fixtures into GPU claims.'}</p></div>
          <div className="headline-number">Φ<span>369</span></div>
        </div>
        <section className="hero-panel">
          <div className="hero-content"><Flag variant={live ? 'mint' : 'demo'}>{live ? 'PRIVATE CONNECTION' : 'INTERACTIVE PREVIEW'}</Flag>
            <h2>{live ? 'Connected to your local runtime.' : 'Built to run on your terms.'}</h2>
            <p>{live ? 'Your requests go only to the loopback daemon configured below. The browser token stays in memory for this tab.' :
              'Explore the interface freely. When you want real inference, pair this tab with your own running PhiServe daemon on the same computer.'}</p>
            <div className="hero-actions"><button className="primary-button" onClick={() => setView(live ? 'playground' : 'models')}>
              {live ? 'Open playground' : 'Explore the console'} <span>↗</span></button>
              <a href={GH + '/blob/main/docs/BROWSER_BRIDGE.md'} target="_blank" rel="noopener noreferrer" className="quiet-link">SETUP GUIDE ↗</a></div>
          </div>
          <Orbital connected={live} active={resident} />
        </section>

        <section className="connection-panel">
          <div className="connection-heading"><div><p className="eyebrow">LOCAL RUNTIME BRIDGE</p><h2>Connection control</h2><p>Same-device loopback only. No tokens stored by this site.</p></div>
            <Flag variant={live ? 'mint' : 'neutral'}>{live ? 'CONNECTED' : 'NOT CONNECTED'}</Flag>
          </div>
          <div className="connection-fields">
            <label className="field"><span>LOCAL ENDPOINT</span>
              <input aria-label="Local endpoint" value={endpoint} onChange={(e) => setEndpoint(e.target.value)}
                disabled={live || phase === 'connecting'} placeholder={DEFAULT_ENDPOINT} autoComplete="off" spellCheck="false" /></label>
            <label className="field token-field"><span>LOCAL BEARER TOKEN</span>
              <input aria-label="Local bearer token" type="password" value={token}
                onChange={(e) => setToken(e.target.value)} disabled={live || phase === 'connecting'}
                placeholder={live ? 'Token active in memory' : 'Paste from your private token file'}
                autoComplete="off" spellCheck="false" /></label>
            <button className={live ? 'outline-button connection-button' : 'primary-button connection-button'}
              disabled={phase === 'connecting'} onClick={live ? disconnect : connect}>
              {live ? 'Disconnect' : phase === 'connecting' ? 'Connecting…' : 'Connect locally'} <span>↗</span>
            </button>
          </div>
          {connectionError ? <p className="inline-error" role="alert">{connectionError}</p> : null}
          <p className="connection-note">Your browser needs permission to reach localhost. The daemon must explicitly trust <strong>{window.location.origin}</strong>. This does not enable access from another device.</p>
        </section>

        {view === 'overview' ? <>
          <section className="stats-section">
            <SectionHeading kicker="SYSTEM PULSE" title="Runtime at a glance" aside={live ? 'Live daemon API observations' : 'Demo values, not hardware measurements'} />
            <div className="stats-grid">
              <StatCard icon="◌" label="Runtime state" value={live ? health?.status?.toUpperCase() || 'UNKNOWN' : 'DEMO'} sub={live ? 'Observed from /health' : 'No real daemon connected'} />
              <StatCard icon="▥" label="Registered GGUF" value={live ? String(models.length).padStart(2, '0') : '—'} sub={live ? 'From /api/tags' : 'Example catalog below'} />
              <StatCard icon="◉" label="Resident model" value={live ? (resident ? '01' : '00') : '—'} sub={live ? resident || 'None loaded' : 'Not observed'} />
              <StatCard icon="⌁" label="Inference backend" value={live ? 'LOCAL' : 'NONE'} sub={live ? 'llama.cpp-compatible' : 'No tokens generated'} />
            </div>
          </section>
          <section className="two-column">
            <div className="panel"><SectionHeading kicker="ARCHITECTURE" title="One runtime, many tools" />
              <div className="topology"><div className="topology-apps"><span>PHIOS</span><span>POCKETBRAIN</span><span>GAIA</span></div><div className="connector-line" /><div className="topology-core"><strong>Φ</strong><span>PHISERVE</span><small>LOCAL MODEL GATEWAY</small></div><div className="connector-line" /><div className="topology-end"><span>GGUF / LLAMA.CPP</span><span>ΦSTREAM / RESEARCH ONLY</span></div></div>
              <p className="muted small-copy">Architecture illustration, not a claim that every listed client is currently connected. PhiStream remains experimental.</p>
            </div>
            <div className="panel activity-panel"><SectionHeading kicker="QUICK ACTIONS" title="Where to next?" />
              <button onClick={() => setView('models')} className="action-row"><span className="action-icon">▥</span><span><strong>Explore the model library</strong><small>Registered GGUF models and resident status</small></span><span>↗</span></button>
              <button onClick={() => setView('playground')} className="action-row"><span className="action-icon">⌁</span><span><strong>Open text playground</strong><small>Stream text from a local model</small></span><span>↗</span></button>
              <button onClick={() => setView('field')} className="action-row"><span className="action-icon">◇</span><span><strong>Launch a field experiment</strong><small>Operator-run CPU/GPU evidence commands</small></span><span>↗</span></button>
              {live ? <button className="refresh-button" onClick={refresh}>↻ Refresh local status</button> : null}
            </div>
          </section>
        </> : null}

        {view === 'ecosystem' ? <section className="ecosystem-section">
          <SectionHeading kicker="SOURCE-REVIEWED INTEGRATION MAP" title="The Φ ecosystem"
            aside="Compatibility inventory · not a live connectivity scan" />
          <div className="ecosystem-notice" role="note">
            <strong>CODE INVENTORY, NOT ACTIVE CONNECTIONS</strong>
            <p>These entries document reviewed architecture and adapter status. Connecting PhiServe above does not connect any other app or grant it permission to read memory, run tools or perform actions. This view makes no network calls to the listed projects.</p>
            <code>{INVENTORY_SCOPE}</code>
          </div>
          <div className="ecosystem-metrics">
            <div><strong>{inventoryCounts().listed}</strong><span>PROJECTS MAPPED</span></div>
            <div><strong>{inventoryCounts().wired}</strong><span>ADAPTER IN CODE</span></div>
            <div><strong>{inventoryCounts().inReview}</strong><span>ADAPTER IN REVIEW</span></div>
            <div><strong>{inventoryCounts().research}</strong><span>RESEARCH ENGINE</span></div>
          </div>
          <div className="ecosystem-hub" aria-label="Shared architecture model">
            <div className="ecosystem-hub-center"><span>Φ</span><strong>PHISERVE</strong><small>GGUF / LLAMA.CPP / LOOPBACK</small></div>
            <p>Shared local inference, not shared identities or application permissions. PhiStream remains a separate research engine.</p>
          </div>
          <div className="ecosystem-grid">
            {ECOSYSTEM_APPS.map((item) => <article className="ecosystem-card" key={item.id}>
              <div className="ecosystem-card-top">
                <span className="ecosystem-category">{item.role}</span>
                <span className={'ecosystem-state ' + item.stage}>{item.status}</span>
              </div>
              <h3>{item.name}</h3>
              <p className="ecosystem-direction">{item.direction}</p>
              <p className="ecosystem-description">{item.detail}</p>
              <div className="ecosystem-next"><strong>NEXT VALIDATION</strong><p>{item.next}</p></div>
              <div className="ecosystem-links">
                <a href={item.repo} target="_blank" rel="noopener noreferrer">REPOSITORY ↗</a>
                {item.pr ? <a href={item.pr} target="_blank" rel="noopener noreferrer">ADAPTER PR ↗</a> : null}
              </div>
            </article>)}
          </div>
          <div className="info-strip"><span>◎</span><p>Only registered local GGUF models are currently offered through PhiServe chat APIs. Experimental PhiStream safetensors, browser ONNX weights and client memory are not silently imported.</p>
            <a href={GH + '/blob/main/docs/ECOSYSTEM_BRIDGES.md'} target="_blank" rel="noopener noreferrer">INTEGRATION GUIDE ↗</a></div>
        </section> : null}

        {view === 'models' ? <section className="model-section">
          <SectionHeading kicker="GGUF CATALOG" title={live ? 'Registered models' : 'Preview model shelf'}
            aside={live ? models.length + ' reported by your daemon' : 'Illustrative entries only'} />
          {!live ? <div className="demo-disclaimer">EXAMPLE MODELS ONLY · The cards below are visual samples. Connect to a daemon to fetch your actual registered GGUF aliases.</div> : null}
          {live && models.length === 0 ? <div className="empty-state"><strong>No models registered yet.</strong><p>Use the local PhiServe CLI to register GGUF weights. This page cannot download or modify model files.</p></div> :
            <div className="model-grid">{displayModels.map((model) =>
              <ModelCard key={model.name} model={model} resident={resident === model.name}
                selected={selected === model.name} live={live} onSelect={() => chooseModel(model.name)} />)}</div>}
          <div className="info-strip"><span>◇</span><p>PhiStream safetensors checkpoints remain in a separate research catalog. They are intentionally unavailable to chat requests until the required validation gates pass.</p><a href={PHISTREAM} target="_blank" rel="noopener noreferrer">RESEARCH REPO ↗</a></div>
        </section> : null}

        {view === 'playground' ? <section className="playground-section">
          <SectionHeading kicker="TEXT / INFERENCE" title="Local text playground"
            aside={live ? 'Connected · NDJSON stream' : 'Interface demo · no inference'} />
          <div className="playground-grid"><div className="chat-panel">
            <div className="chat-header"><span><span className="chat-led" />{live ? 'ACTIVE LOCAL SESSION' : 'DEMO SESSION'}</span>
              {live ? <select aria-label="Select model" value={selected}
                  disabled={busy} onChange={(event) => chooseModel(event.target.value)}>
                  {models.map((m) => <option value={m.name} key={m.name}>{m.name}</option>)}
                </select> : <span className="muted">NOT CONNECTED</span>}
            </div>
            <div className="messages" role="log" aria-live="polite">{messages.length ? messages.map((m) => <Message key={m.id} message={m} />) :
              <div className="chat-blank"><span>Φ</span><strong>Ready when you are.</strong><p>{live ? 'Send a message to your selected local model.' : 'Try a sample prompt to see the demo state.'}</p></div>}</div>
            <form className="composer" onSubmit={(e) => { e.preventDefault(); send(); }}>
              <label htmlFor="prompt" className="sr-only">Text prompt</label>
              <textarea id="prompt" rows="2" maxLength={4000} placeholder={live ? 'Message your local model…' : 'Try the interface demo…'}
                value={draft} onChange={(e) => setDraft(e.target.value)} disabled={busy} />
              <div className="composer-actions"><span>{live ? 'LOCAL LOOPBACK · TEXT ONLY' : 'NO MODEL INFERENCE IN DEMO'}</span>
                {busy ? <button type="button" className="outline-button" onClick={() => chatAbort.current?.abort()}>Stop</button> : null}
                <button type="submit" className="primary-button" disabled={!draft.trim() || busy || (live && !selected)}>Send ↗</button>
              </div>
            </form>
            {chatError ? <p className="inline-error chat-error" role="alert">{chatError}</p> : null}
          </div><aside className="prompt-panel"><p className="eyebrow">QUICK PROMPTS</p><h3>Get things moving.</h3><p className="muted">Choose a starting point, or write your own.</p>
            {PROMPTS.map((p, i) => <button key={p} className="prompt-chip" onClick={() => setDraft(p)} disabled={busy}><span>0{i + 1}</span>{p}<span>↗</span></button>)}
            <div className="prompt-note"><strong>SESSION PRIVACY</strong><p>Prompts and token stay in this browser tab's memory. We do not use localStorage, analytics, hosted inference, or cloud fallbacks.</p></div>
          </aside></div>
        </section> : null}

        {view === 'field' ? <section className="field-section">
          <SectionHeading kicker="FIELD VALIDATION" title="Hardware acceptance lab" aside="Commands execute on your own machine" />
          <div className="field-intro"><span className="field-symbol">◇</span><div><h3>Green CI is not proof of GPU execution.</h3><p>PhiServe can run a real GGUF check on your computer and optionally sample NVIDIA's driver-reported process memory. The steps below are operator commands, not actions this hosted React page can perform.</p></div></div>
          <div className="steps">{STEPS.map((step, i) => <article className="step" key={step.num}><span className="step-num">{step.num}</span><div className="step-main"><h3>{step.label}</h3><p>{step.detail}</p><pre><code>{step.code}</code></pre></div><button className="copy-button" onClick={() => copy(step.code, i)}>{copied === i ? 'COPIED ✓' : 'COPY ↗'}</button></article>)}</div>
          {copied === -2 ? <p className="inline-error">Clipboard unavailable. Select and copy the command manually.</p> : null}
          <div className="info-strip"><span>◌</span><p>Research-only PhiStream safetensors validation still needs matching local tokenizers and independently produced reference traces. No automatic promotion to chat.</p><a href={GH + '/blob/main/docs/WINDOWS_FIELD_RUN.md'} target="_blank" rel="noopener noreferrer">READ FIELD GUIDE ↗</a></div>
        </section> : null}
        <footer className="site-footer"><span>PHISERVE <span className="footer-dot">·</span> YOUR INTELLIGENCE, YOUR INFRASTRUCTURE</span><span>MIT CODE / LOCAL MODELS / ZERO MANDATORY CLOUD</span></footer>
      </div>
    </main>
  </div>;
}
