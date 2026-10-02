/* ──────────────────────────────────────────────────────────────────────────
   Weather-Advisory Support Bot — Frontend Application
   ────────────────────────────────────────────────────────────────────────── */

const API_BASE = '';  // same origin

// ── State ────────────────────────────────────────────────────────────────────
let sessionId = null;
let isLoading = false;

// ── DOM refs ─────────────────────────────────────────────────────────────────
const messagesEl   = document.getElementById('chat-messages');
const inputEl      = document.getElementById('message-input');
const sendBtn      = document.getElementById('send-btn');
const resetBtn     = document.getElementById('reset-btn');
const sessionDisp  = document.getElementById('session-id-display');
const statusDot    = document.querySelector('.status-dot');
const statusText   = document.querySelector('.status-text');

// ── Utility ──────────────────────────────────────────────────────────────────

function setStatus(state) {
  statusDot.className = `status-dot ${state}`;
  const labels = { online: 'Ready', loading: 'Thinking…', error: 'Error' };
  statusText.textContent = labels[state] || 'Ready';
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function simpleMarkdown(text) {
  // Bold **text**
  text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  // Horizontal rule
  text = text.replace(/^---$/gm, '<hr/>');
  // Inline code `code`
  text = text.replace(/`([^`]+)`/g, '<code style="background:rgba(255,255,255,0.08);padding:1px 5px;border-radius:3px;font-size:12px">$1</code>');
  // Bullet list lines starting with -
  text = text.replace(/^- (.+)$/gm, '<li>$1</li>');
  text = text.replace(/(<li>.*<\/li>)/s, '<ul style="padding-left:18px;margin:6px 0">$1</ul>');
  return text;
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ── Session management ────────────────────────────────────────────────────────

function initSession() {
  sessionId = null;
  sessionDisp.textContent = '(new)';
}

async function resetSession() {
  if (sessionId) {
    await fetch(`${API_BASE}/session/reset?session_id=${sessionId}`, { method: 'POST' }).catch(() => {});
  }
  sessionId = null;
  sessionDisp.textContent = '(new)';
  messagesEl.innerHTML = `
    <div class="welcome-message">
      <div class="welcome-icon">⛅</div>
      <h2>New Session Started</h2>
      <p>Your conversation history has been cleared. Ask your next outdoor safety question!</p>
    </div>`;
  setStatus('online');
}

// ── Build UI elements ─────────────────────────────────────────────────────────

function buildSOPCard(sop) {
  if (!sop || !sop.id) return '';
  const severityClass = `badge-${sop.severity || 'low'}`;
  return `
    <div class="sop-card">
      <div class="sop-card-header">
        <span class="sop-id">${escapeHtml(sop.id)}</span>
        <span class="sop-title">${escapeHtml(sop.title || '')}</span>
        <span class="badge ${severityClass}">${escapeHtml((sop.severity || '').toUpperCase())}</span>
      </div>
      <div class="sop-meta">
        <span>📂 ${escapeHtml(sop.category || '')}</span>
        <span>📖 ${escapeHtml(sop.source || '')}</span>
      </div>
    </div>`;
}

function buildWeatherCard(weather) {
  if (!weather) return '';
  const loc = weather.location ? `<span style="color:var(--text-secondary);font-size:12px"> — ${escapeHtml(weather.location)}</span>` : '';

  function fmt(val, unit = '') {
    if (val === null || val === undefined) return 'N/A';
    if (typeof val === 'number') return `${val.toFixed(1)}${unit}`;
    return `${val}${unit}`;
  }

  return `
    <div class="weather-card">
      <div class="weather-card-header">🌤️ Live Weather${loc}</div>
      <div class="weather-grid">
        <div class="weather-item">
          <span class="label">Temperature</span>
          <span class="value">${fmt(weather.temperature_2m, ' °C')}</span>
        </div>
        <div class="weather-item">
          <span class="label">Feels Like</span>
          <span class="value">${fmt(weather.apparent_temperature, ' °C')}</span>
        </div>
        <div class="weather-item">
          <span class="label">Precipitation</span>
          <span class="value">${fmt(weather.precipitation, ' mm/h')}</span>
        </div>
        <div class="weather-item">
          <span class="label">Wind Speed</span>
          <span class="value">${fmt(weather.wind_speed_10m, ' km/h')}</span>
        </div>
        <div class="weather-item">
          <span class="label">UV Index</span>
          <span class="value">${fmt(weather.uv_index)}</span>
        </div>
        <div class="weather-item">
          <span class="label">Visibility</span>
          <span class="value">${fmt(weather.visibility, ' m')}</span>
        </div>
        <div class="weather-item">
          <span class="label">Weather Code</span>
          <span class="value">${weather.weather_code ?? 'N/A'}</span>
        </div>
      </div>
    </div>`;
}

function addUserMessage(text) {
  const row = document.createElement('div');
  row.className = 'message-row user';
  row.innerHTML = `
    <div class="message-label">You</div>
    <div class="bubble">${escapeHtml(text)}</div>`;
  messagesEl.appendChild(row);
  scrollToBottom();
}

function addLoadingBubble() {
  const row = document.createElement('div');
  row.className = 'message-row bot';
  row.id = 'loading-row';
  row.innerHTML = `
    <div class="message-label">Advisor</div>
    <div class="loading-bubble">
      <div class="dot-flashing">
        <span></span><span></span><span></span>
      </div>
      Fetching weather &amp; checking policies…
    </div>`;
  messagesEl.appendChild(row);
  scrollToBottom();
}

function removeLoadingBubble() {
  const el = document.getElementById('loading-row');
  if (el) el.remove();
}

function addBotMessage(data) {
  removeLoadingBubble();
  const row = document.createElement('div');
  row.className = 'message-row bot';

  // Strip the structured citation from response text (it's shown in cards instead)
  let cleanResponse = data.response || '';
  const hrIdx = cleanResponse.indexOf('\n\n---\n');
  if (hrIdx !== -1) cleanResponse = cleanResponse.substring(0, hrIdx);

  row.innerHTML = `
    <div class="message-label">Advisor</div>
    <div class="bubble">${simpleMarkdown(escapeHtml(cleanResponse))}</div>
    ${buildSOPCard(data.sop_citation)}
    ${buildWeatherCard(data.weather_facts)}`;

  messagesEl.appendChild(row);
  scrollToBottom();
}

function addErrorMessage(text) {
  removeLoadingBubble();
  const row = document.createElement('div');
  row.className = 'message-row bot';
  row.innerHTML = `
    <div class="message-label">Advisor</div>
    <div class="bubble" style="border-color:var(--danger);background:rgba(224,92,92,0.08)">
      ⚠️ ${escapeHtml(text)}
    </div>`;
  messagesEl.appendChild(row);
  scrollToBottom();
}

// ── Send message ──────────────────────────────────────────────────────────────

async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || isLoading) return;

  // Clear welcome screen on first message
  const welcome = messagesEl.querySelector('.welcome-message');
  if (welcome) welcome.remove();

  addUserMessage(text);
  inputEl.value = '';
  inputEl.style.height = 'auto';

  isLoading = true;
  sendBtn.disabled = true;
  setStatus('loading');
  addLoadingBubble();

  try {
    const body = { message: text };
    if (sessionId) body.session_id = sessionId;

    const resp = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }

    const data = await resp.json();

    // Persist session id
    if (!sessionId && data.session_id) {
      sessionId = data.session_id;
      sessionDisp.textContent = sessionId.substring(0, 12) + '…';
    }

    addBotMessage(data);
    setStatus('online');
  } catch (err) {
    addErrorMessage(err.message || 'Something went wrong. Please try again.');
    setStatus('error');
    setTimeout(() => setStatus('online'), 3000);
  } finally {
    isLoading = false;
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

// ── Event listeners ───────────────────────────────────────────────────────────

sendBtn.addEventListener('click', sendMessage);

inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// Auto-resize textarea
inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + 'px';
});

resetBtn.addEventListener('click', resetSession);

// Suggestion chips
document.querySelectorAll('.suggestion-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    inputEl.value = chip.dataset.q;
    inputEl.dispatchEvent(new Event('input'));
    inputEl.focus();
  });
});

// ── Init ──────────────────────────────────────────────────────────────────────
initSession();
