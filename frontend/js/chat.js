/**
 * chat.js — Panel de chat deslizante: modo normal (agentes individuales) y multi-agente.
 */

import { api } from './api.js';
import { parseMarkdown, extractJSON, badge, signalColor, readSSE, createTypingDots } from './utils.js';
import { renderDebate } from './debate.js';

const AGENT_COLORS = { market: '#6366f1', news: '#f59e0b', reddit: '#ec4899' };

// Estado
let currentAgent = 'market';
let currentMode = 'normal';
const histories = {
  market: '',
  news: '',
  reddit: '',
  'multi-agent': '',
};

// ── Helpers DOM ──────────────────────────────────────────────────────────────

function getMessages() { return document.getElementById('slidingChatMessages'); }
function getInput()    { return document.getElementById('slidingChatInput'); }
function getSendBtn()  { return document.getElementById('sendBtnSliding'); }

function appendMessage(html, isUser = false) {
  const msgs = getMessages();
  if (!msgs) return;
  const div = document.createElement('div');
  div.className = `chat-message ${isUser ? 'user' : 'bot'}`;
  div.innerHTML = html;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
  return div;
}

function saveHistory() {
  const msgs = getMessages();
  if (!msgs) return;
  histories[currentMode === 'multi-agent' ? 'multi-agent' : currentAgent] = msgs.innerHTML;
}

function restoreHistory(key) {
  const msgs = getMessages();
  if (!msgs) return;
  msgs.innerHTML = histories[key] || '';
  msgs.scrollTop = msgs.scrollHeight;
}

function setInputEnabled(enabled) {
  const input = getInput();
  const btn = getSendBtn();
  if (input) input.disabled = !enabled;
  if (btn)   btn.disabled   = !enabled;
}

// ── Modo normal: chat con agente individual ──────────────────────────────────

async function sendMessage() {
  const input = getInput();
  const message = input?.value.trim();
  if (!message) return;

  const agentAtSend = currentAgent;
  const color = AGENT_COLORS[agentAtSend] || '#6366f1';

  appendMessage(`<div class="message-content">${message}</div>`, true);
  input.value = '';

  const typingId = `typing_${Date.now()}`;
  const typingEl = createTypingDots(typingId, `${agentAtSend.toUpperCase()} AGENT`, color);
  getMessages()?.appendChild(typingEl);
  getMessages().scrollTop = getMessages().scrollHeight;

  // Bloque de proceso permanente (se construye en cascada y queda visible)
  const processId = `process_${Date.now()}`;
  let processEl = null;
  let currentToolEl = null;

  const getOrCreateProcess = () => {
    if (processEl) return processEl;
    const wrapper = document.createElement('div');
    wrapper.id = processId;
    wrapper.className = 'chat-message bot';
    wrapper.innerHTML = `<div style="background:#0d0d1a;border:1px solid #2a2a3e;border-radius:10px;padding:10px 14px;font-size:11px;">
      <div style="color:#6b7280;font-size:10px;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">Proceso de análisis</div>
      <div id="${processId}_tools"></div>
    </div>`;
    getMessages()?.appendChild(wrapper);
    getMessages().scrollTop = getMessages().scrollHeight;
    processEl = wrapper;
    return wrapper;
  };

  try {
    const response = await api.chatStream(
      `[Contexto: Ticker activo = NVDA. OBLIGATORIO: llama a tus herramientas ANTES de responder. No uses conocimiento interno.]\n\n${message}`,
      agentAtSend
    );

    await readSSE(response, (msg) => {
      if (msg.type === 'tool_call') {
        // Actualizar indicador de typing
        const statusEl = document.getElementById(`${typingId}_status`);
        if (statusEl) statusEl.textContent = `🔧 ${msg.name}`;

        // Crear bloque de proceso y añadir nueva tool en cascada
        getOrCreateProcess();
        const toolsContainer = document.getElementById(`${processId}_tools`);
        if (toolsContainer) {
          const toolBlock = document.createElement('div');
          toolBlock.style.cssText = 'margin-bottom:6px;';
          toolBlock.innerHTML = `
            <div style="display:flex;align-items:center;gap:6px;color:#a0a0c0;font-weight:600;">
              <span style="color:${color};font-size:10px;">▶</span>
              <span style="font-size:11px;">${msg.name}</span>
            </div>
            <div id="${processId}_steps_${msg.name.replace(/\W/g,'_')}" style="margin-left:16px;margin-top:3px;display:flex;flex-direction:column;gap:2px;"></div>`;
          toolsContainer.appendChild(toolBlock);
          currentToolEl = document.getElementById(`${processId}_steps_${msg.name.replace(/\W/g,'_')}`);
        }

      } else if (msg.type === 'tool_step') {
        // Añadir paso bajo la tool activa
        const target = currentToolEl || (() => { getOrCreateProcess(); return document.getElementById(`${processId}_tools`); })();
        if (target) {
          const isOk   = msg.step.startsWith('✅');
          const isWarn = msg.step.startsWith('⚠️');
          const isErr  = msg.step.startsWith('❌');
          const stepColor = isOk ? '#10b981' : isWarn ? '#f59e0b' : isErr ? '#ef4444' : '#6b7280';
          const line = document.createElement('div');
          line.style.cssText = `font-size:10px;color:${stepColor};padding:1px 0;`;
          line.textContent = msg.step;
          target.appendChild(line);
          getMessages().scrollTop = getMessages().scrollHeight;
        }

      } else if (msg.type === 'text_chunk') {
        let streamDiv = document.getElementById(`${typingId}_stream`);
        if (!streamDiv) {
          document.getElementById(typingId)?.remove();
          const wrapper = document.createElement('div');
          wrapper.className = 'chat-message bot';
          wrapper.innerHTML = `<div class="message-content" id="${typingId}_stream" style="font-size:12px;color:#e0e0e0;"></div>`;
          getMessages()?.appendChild(wrapper);
          streamDiv = document.getElementById(`${typingId}_stream`);
        }
        if (streamDiv) { streamDiv.textContent += msg.text; getMessages().scrollTop = getMessages().scrollHeight; }

      } else if (msg.type === 'response') {
        document.getElementById(`${typingId}_stream`)?.closest('.chat-message')?.remove();
        document.getElementById(typingId)?.remove();
        if (currentAgent === agentAtSend) {
          renderBotMessage(msg.text, agentAtSend);
        } else {
          histories[agentAtSend] += buildBotHTML(msg.text, agentAtSend);
        }

      } else if (msg.type === 'error') {
        document.getElementById(typingId)?.remove();
        if (currentAgent === agentAtSend) appendMessage(`<div class="message-content" style="color:#ef4444;">❌ ${msg.message}</div>`);

      } else if (msg.type === 'done') {
        document.getElementById(typingId)?.remove();
      }
    });
  } catch (e) {
    document.getElementById(typingId)?.remove();
    if (currentAgent === agentAtSend) appendMessage(`<div class="message-content" style="color:#ef4444;">Error: Backend no disponible</div>`);
  }
}

function buildBotHTML(text, agent) {
  const color = AGENT_COLORS[agent] || '#6366f1';
  const data = extractJSON(text);

  // Eliminar bloques JSON del texto markdown
  let md = text
    .replace(/```json[\s\S]*?```/g, '')
    .replace(/```[\s\S]*?```/g, '');
  if (data) md = md.replace(/\{[\s\S]*\}/g, '');
  md = md.trim();

  let html = '';

  // Mostrar texto narrativo siempre que sea sustancial (>50 chars)
  if (md && md.length > 50) {
    html += `<div style="font-size:12px;line-height:1.7;color:#d0d0e0;margin-bottom:${data ? '10px' : '0'};">${parseMarkdown(md)}</div>`;
  }

  // Tarjeta de resumen JSON solo si hay señal/predicción
  if (data && (data.signal || data.sentiment || data.community_sentiment)) {
    const signal = data.signal || data.sentiment || data.community_sentiment || '';
    const prediction = data.prediction || data.price_prediction?.direction || '';
    const confidence = data.confidence != null ? Math.round(data.confidence * 100) : null;
    const reasons = data.reasons || [];
    const risks = data.risks || [];
    const headlines = data.key_headlines || [];
    const reasoning = data.reasoning || data.price_prediction?.reasoning || '';
    const sc = signalColor(signal);

    html += `<div style="background:#0f0f1a;border:1px solid ${color}30;border-radius:8px;padding:10px;font-size:11px;margin-top:${md.length > 50 ? '8px' : '0'};">
      <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:6px;">
        ${badge(signal, sc)}
        ${prediction ? `→ ${badge(prediction, color)}` : ''}
        ${confidence != null ? `<span style="color:#6b7280;">${confidence}% confianza</span>` : ''}
      </div>
      ${headlines.length ? `<div style="margin-bottom:6px;border-top:1px solid #1a1a2e;padding-top:6px;">
        <div style="color:#6b7280;font-size:10px;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.5px;">Titulares</div>
        ${headlines.slice(0, 5).map(h => `<div style="color:#c0c0d0;padding:2px 0;border-bottom:1px solid #1a1a2e;">• ${h}</div>`).join('')}
      </div>` : ''}
      ${reasons.slice(0, 3).map(r => `<div style="color:#a0a0b0;padding:1px 0;">✓ ${r}</div>`).join('')}
      ${risks.slice(0, 2).map(r => `<div style="color:#f87171;padding:1px 0;">⚠ ${r}</div>`).join('')}
      ${reasoning && !md ? `<div style="color:#8080a0;font-size:10px;margin-top:6px;font-style:italic;">${reasoning.slice(0, 200)}${reasoning.length > 200 ? '...' : ''}</div>` : ''}
    </div>`;
  }

  // Si no hay nada parseado, mostrar el texto original
  if (!html) html = `<div style="font-size:12px;line-height:1.7;color:#d0d0e0;">${parseMarkdown(text)}</div>`;

  return `<div class="chat-message bot"><div class="message-content">${html}</div></div>`;
}

function renderBotMessage(text, agent) {
  const msgs = getMessages();
  if (!msgs) return;
  msgs.innerHTML += buildBotHTML(text, agent);
  msgs.scrollTop = msgs.scrollHeight;
}

// ── Selector de agente ───────────────────────────────────────────────────────

function selectAgent(agent) {
  saveHistory();
  currentAgent = agent;

  document.querySelectorAll('.agent-tab-btn').forEach(b => {
    const a = b.getAttribute('data-agent');
    const c = AGENT_COLORS[a] || '#6366f1';
    const active = a === agent;
    b.style.background = active ? `${c}20` : '#2a2a3e';
    b.style.color      = active ? c : '#6b7280';
    b.style.border     = active ? `1px solid ${c}40` : '1px solid #3a3a5e';
  });

  restoreHistory(agent);
}

// ── Modo multi-agente ────────────────────────────────────────────────────────

async function startMultiAgent() {
  saveHistory();
  currentMode = 'multi-agent';
  setInputEnabled(false);

  const msgs = getMessages();
  if (msgs) msgs.innerHTML = '';

  const rounds = parseInt(
    document.querySelector('.rounds-btn.active')?.getAttribute('data-rounds') || '1'
  );

  await renderDebate('NVDA', null, null, rounds, msgs);
  setInputEnabled(false); // debate no usa input
}

function resetToNormal() {
  saveHistory();
  currentMode = 'normal';
  setInputEnabled(true);
  restoreHistory(currentAgent);
}

// ── Init ─────────────────────────────────────────────────────────────────────

export function initChat() {
  // Abrir / cerrar panel
  document.getElementById('floatingChatBtn')?.addEventListener('click', () => {
    document.getElementById('slidingChatPanel')?.classList.add('open');
  });
  document.getElementById('closeChatBtn')?.addEventListener('click', () => {
    document.getElementById('slidingChatPanel')?.classList.remove('open');
  });

  // Enviar mensaje
  getSendBtn()?.addEventListener('click', sendMessage);
  getInput()?.addEventListener('keypress', e => { if (e.key === 'Enter') sendMessage(); });

  // Selector de modo
  document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const mode = btn.getAttribute('data-mode');
      const agentBar  = document.getElementById('agentSelectorBar');
      const roundsBar = document.getElementById('roundsSelectorBar');

      if (mode === 'multi-agent') {
        agentBar?.style.setProperty('display', 'none');
        roundsBar?.style.setProperty('display', 'flex');
        startMultiAgent();
      } else {
        agentBar?.style.setProperty('display', 'flex');
        roundsBar?.style.setProperty('display', 'none');
        resetToNormal();
      }
    });
  });

  // Selector de agente
  document.querySelectorAll('.agent-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => selectAgent(btn.getAttribute('data-agent')));
  });

  // Selector de rondas
  document.querySelectorAll('.rounds-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.rounds-btn').forEach(b => {
        b.style.background = '#2a2a3e'; b.style.color = '#6b7280'; b.style.border = '1px solid #3a3a5e';
      });
      btn.style.background = '#6366f120'; btn.style.color = '#6366f1'; btn.style.border = '1px solid #6366f140';
    });
  });

  // Inicializar historial del agente por defecto
  restoreHistory(currentAgent);
}
