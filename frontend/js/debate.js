/**
 * debate.js — Sistema de debate multi-agente SSE.
 * Usado tanto por el panel deslizante como por cualquier otro contenedor.
 */

import { api } from './api.js';
import { extractJSON, badge, signalColor, confidenceBar, readSSE, createTypingDots } from './utils.js';

const AGENT_DEFS = {
  market: { name: '💹 Market Agent', color: '#6366f1', icon: '💹' },
  news:   { name: '📰 News Agent',   color: '#f59e0b', icon: '📰' },
  reddit: { name: '💬 Reddit Agent', color: '#ec4899', icon: '💬' },
};

/**
 * Ejecuta el debate y renderiza los eventos en `container`.
 * @param {string} ticker
 * @param {string|null} startDate
 * @param {string|null} endDate
 * @param {number} rounds
 * @param {HTMLElement} container
 */
export async function renderDebate(ticker, startDate, endDate, rounds, container) {
  if (!container) return;

  const append = (html) => {
    const div = document.createElement('div');
    div.className = 'chat-message bot';
    div.innerHTML = html;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return div;
  };

  append(`<div style="color:#6b7280;font-size:12px;text-align:center;padding:8px;font-style:italic;">Iniciando debate multi-agente sobre ${ticker}...</div>`);

  try {
    const response = await api.debateStream(ticker, startDate, endDate, rounds);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    await readSSE(response, (msg) => {
      switch (msg.type) {
        case 'status':
          append(`<div style="color:#6b7280;font-size:11px;text-align:center;padding:6px;font-style:italic;">${msg.message}</div>`);
          break;

        case 'agent_init':
          append(renderAgentInitCard(msg));
          break;

        case 'typing': {
          const ag = AGENT_DEFS[msg.agent] || { name: msg.name, color: msg.color };
          const el = createTypingDots(`st_${msg.agent}`, ag.name, ag.color);
          container.appendChild(el);
          container.scrollTop = container.scrollHeight;
          break;
        }

        case 'agent_result':
          document.getElementById(`st_${msg.agent}`)?.remove();
          append(renderAgentResult(msg));
          break;

        case 'verdict':
          append(renderVerdictCard(msg.text));
          break;

        case 'error':
          append(`<div style="color:#ef4444;font-size:12px;">❌ ${msg.message}</div>`);
          break;
      }
    });
  } catch (err) {
    append(`<div style="color:#ef4444;font-size:12px;">❌ Error: ${err.message}</div>`);
  }
}

// ── Renderers ────────────────────────────────────────────────────────────────

function renderAgentInitCard(msg) {
  const tools = (msg.tools || []).map(t => `<span style="background:#2a2a3e;color:#6b7280;border-radius:4px;padding:1px 6px;font-size:10px;">${t}</span>`).join(' ');
  return `<div style="border-left:3px solid ${msg.color};padding:8px 12px;background:#1a1a2e;border-radius:2px 8px 8px 8px;margin:4px 0;">
    <div style="color:${msg.color};font-size:11px;font-weight:700;">✅ ${msg.name} inicializado</div>
    <div style="color:#6b7280;font-size:10px;margin-top:4px;">${msg.description}</div>
    <div style="margin-top:6px;display:flex;flex-wrap:wrap;gap:4px;">${tools}</div>
  </div>`;
}

function renderAgentResult(msg) {
  const ag = AGENT_DEFS[msg.agent] || { color: msg.color };
  const color = ag.color;
  const data = extractJSON(msg.text);
  let content = '';

  if (data) {
    const signal     = data.signal || data.sentiment || data.community_sentiment || data.position || '';
    const prediction = data.prediction || data.price_prediction?.direction || data.adjusted_prediction || '';
    const reasoning  = data.reasoning || data.price_prediction?.reasoning || data.final_reasoning || data.key_argument || '';
    const confidence = data.confidence ?? data.adjusted_confidence;
    const sc = signalColor(signal);

    const metrics = [
      data.trend          && `📈 Tendencia: <strong>${data.trend}</strong>`,
      data.rsi != null    && `RSI: <strong>${data.rsi}</strong>`,
      data.volatility != null && `Volatilidad: <strong>${(data.volatility * 100).toFixed(1)}%</strong>`,
      data.return_period_pct != null && `Retorno: <strong>${data.return_period_pct.toFixed(2)}%</strong>`,
      data.alpaca_score != null && `Alpaca: <strong>${data.alpaca_score}</strong>`,
      data.gdelt_trend    && `GDELT: <strong>${data.gdelt_trend}</strong>`,
      data.post_volume_avg != null && `Posts/día: <strong>${data.post_volume_avg}</strong>`,
      data.agrees_with_market != null && `vs Market: ${data.agrees_with_market ? '✅' : '❌'}`,
      data.agrees_with_news   != null && `vs News: ${data.agrees_with_news   ? '✅' : '❌'}`,
      data.agrees_with_reddit != null && `vs Reddit: ${data.agrees_with_reddit ? '✅' : '❌'}`,
    ].filter(Boolean);

    const reasons = (data.reasons || data.key_headlines || []).slice(0, 3);

    content = `<div style="font-size:12px;line-height:1.7;color:#e0e0e0;">
      <div style="margin-bottom:8px;display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
        ${badge(signal, sc)}
        ${prediction ? `→ ${badge(prediction, '#6366f1')}` : ''}
        ${confidence != null ? `<span style="color:#6b7280;font-size:10px;">${Math.round(confidence * 100)}% confianza</span>` : ''}
      </div>
      ${metrics.length ? `<div style="color:#a0a0b0;font-size:11px;margin-bottom:6px;">${metrics.join(' &nbsp;·&nbsp; ')}</div>` : ''}
      ${reasons.map(r => `<div style="color:#c0c0d0;font-size:11px;padding:2px 0;">• ${r}</div>`).join('')}
      ${reasoning ? `<div style="color:#a0a0b0;font-size:11px;font-style:italic;border-top:1px solid #2a2a3e;padding-top:6px;margin-top:4px;">${reasoning.slice(0, 300)}${reasoning.length > 300 ? '...' : ''}</div>` : ''}
    </div>`;
  } else {
    content = `<div style="font-size:12px;color:#e0e0e0;">${msg.text.replace(/```json[\s\S]*?```/g, '').trim().slice(0, 300)}</div>`;
  }

  const roundLabel = msg.round > 1 ? ` <span style="color:#6b7280;font-weight:400;">(Ronda ${msg.round})</span>` : '';
  return `<div style="border-left:3px solid ${color};padding:10px 14px;background:#1e1e2e;border-radius:2px 12px 12px 12px;">
    <div style="color:${color};font-size:11px;font-weight:700;margin-bottom:6px;">${msg.name}${roundLabel}</div>
    ${content}
  </div>`;
}

function renderVerdictCard(verdictText) {
  const data = extractJSON(verdictText);
  if (!data) return `<div style="color:#e0e0e0;font-size:13px;white-space:pre-wrap;">${verdictText}</div>`;

  const rec = data.investment_recommendation || data.recommendation || '';
  const recColor = rec.includes('COMPRAR') ? '#10b981' : rec.includes('VENDER') ? '#ef4444' : '#f59e0b';
  const worth = data.worth_investing;

  const outlooks = [
    ['CORTO PLAZO', data.short_term_outlook || data.price_outlook?.short_term?.direction],
    ['MEDIO PLAZO', data.medium_term_outlook || data.price_outlook?.medium_term?.direction],
    ['LARGO PLAZO', data.long_term_outlook  || data.price_outlook?.long_term?.direction],
  ].filter(([, v]) => v);

  return `<div style="background:linear-gradient(135deg,#10b98120,#05966920);border:2px solid #10b981;border-radius:12px;padding:16px;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
      <div>
        <div style="color:#10b981;font-weight:700;font-size:14px;">✅ VEREDICTO — ${data.ticker || ticker}</div>
        ${data.analysis_period ? `<div style="color:#6b7280;font-size:11px;">${data.analysis_period.start} → ${data.analysis_period.end}</div>` : ''}
      </div>
      <div style="text-align:right;">
        <div style="background:${recColor}20;color:${recColor};border:1px solid ${recColor};border-radius:8px;padding:4px 12px;font-weight:700;font-size:13px;">${rec}</div>
        ${worth != null ? `<div style="color:${worth ? '#10b981' : '#ef4444'};font-size:11px;margin-top:4px;">${worth ? '✅ Vale la pena' : '❌ No recomendado'}</div>` : ''}
      </div>
    </div>
    ${data.confidence != null ? `<div style="margin-bottom:10px;"><div style="color:#6b7280;font-size:11px;">Confianza global</div>${confidenceBar(data.confidence)}</div>` : ''}
    ${outlooks.length ? `<div style="display:grid;grid-template-columns:repeat(${outlooks.length},1fr);gap:8px;margin-bottom:10px;">
      ${outlooks.map(([label, val]) => `<div style="background:#1a1a2e;border-radius:8px;padding:8px;"><div style="color:#6b7280;font-size:10px;">${label}</div><div style="color:#e0e0e0;font-size:11px;margin-top:4px;">${val}</div></div>`).join('')}
    </div>` : ''}
    ${(data.final_reasoning || data.summary) ? `<div style="color:#a0a0b0;font-size:12px;line-height:1.5;font-style:italic;">${(data.final_reasoning || data.summary).slice(0, 400)}</div>` : ''}
  </div>`;
}
