/**
 * main.js — Entry point. Solo orquesta la inicialización de módulos.
 */

import { api } from './api.js?v=5';
import { initChart, loadChart, loadChartLastTradingDay } from './chart.js';
import { loadNews, initNews } from './news.js';
import { loadPortfolio, initPortfolio } from './portfolio.js';
import { initChat } from './chat.js';
import { parseMarkdown, extractJSON, badge, signalColor } from './utils.js';

// ── Animaciones globales ─────────────────────────────────────────────────────
(function injectGlobalStyles() {
  const s = document.createElement('style');
  s.textContent = `
    @keyframes bounce { 0%,80%,100%{transform:translateY(0)} 40%{transform:translateY(-5px)} }
    @keyframes pulse  { 0%,100%{opacity:1} 50%{opacity:0.4} }
    .typing-dots { display:inline-flex; gap:3px; align-items:center; }
    .typing-dots span {
      display:inline-block; width:6px; height:6px; border-radius:50%;
      background:currentColor; animation:bounce 1.2s infinite;
    }
    .typing-dots span:nth-child(2) { animation-delay:.2s; }
    .typing-dots span:nth-child(3) { animation-delay:.4s; }
    .chart-loading, .chart-error {
      display:flex; align-items:center; justify-content:center;
      height:100%; font-size:13px; color:#6b7280;
    }
    .chart-error { color:#ef4444; }
  `;
  document.head.appendChild(s);
})();

// ── Ensemble Forecast ────────────────────────────────────────────────────────
async function loadEnsembleForecast(horizon = 'daily') {
  const card = document.getElementById('ensembleForecastCard');
  if (!card) return;

  console.log(`[IA Prediction] Cargando predicción ${horizon}...`);
  
  // Mostrar loading
  card.innerHTML = `<div style="color:#6b7280;font-size:12px;text-align:center;padding:20px;">
    <div class="typing-dots" style="justify-content:center;"><span></span><span></span><span></span></div>
    <div style="margin-top:8px;">Cargando predicción ${horizon}...</div>
  </div>`;

  try {
    // Cargar predicción según horizonte
    let data;
    if (horizon === 'daily') {
      data = await api.getEnsembleForecast();
    } else if (horizon === 'weekly') {
      data = await api.getWeeklyForecast();
    } else if (horizon === 'monthly') {
      data = await api.getMonthlyForecast();
    } else {
      card.innerHTML = `<div style="text-align:center;padding:20px;">
        <div style="color:#ef4444;font-size:13px;">❌ Horizonte desconocido: ${horizon}</div>
      </div>`;
      return;
    }
    
    console.log('[IA Prediction] Respuesta:', data);

    if (data.needs_refresh) {
      console.warn('[IA Prediction] Datos desactualizados:', data.message, '| Último dato:', data.last_date);
      card.innerHTML = `
        <div style="text-align:center;padding:16px;">
          <div style="color:#f59e0b;font-size:13px;margin-bottom:8px;">⚠️ ${data.message || 'Datos desactualizados'}</div>
          ${data.last_date ? `<div style="color:#6b7280;font-size:11px;">Último dato: ${data.last_date}</div>` : ''}
          <div style="color:#6b7280;font-size:11px;margin-top:8px;">Pulsa 🔄 Actualizar para recopilar datos recientes.</div>
        </div>`;
      return;
    }

    if (data.error) {
      console.error('[IA Prediction] Error del backend:', data.error, data.trace || '');
      card.innerHTML = `<div style="color:#ef4444;font-size:12px;padding:12px;">❌ ${data.error}</div>`;
      return;
    }

    console.log('[IA Prediction] Predicción:', data.prediction, '| Prob:', data.probability, '| Votos:', data.model_votes);

    const isUp = data.prediction === 'SUBE';
    const color = isUp ? '#10b981' : '#ef4444';
    const confPct = Math.round(data.confidence * 100);
    const probPct = Math.round(data.probability * 100);

    const votes = data.model_votes || {};
    const voteRows = Object.entries(votes).map(([name, prob]) => {
      const v = Math.round(prob * 100);
      const c = prob >= 0.5 ? '#10b981' : '#ef4444';
      return `<div style="display:flex;justify-content:space-between;align-items:center;padding:3px 0;border-bottom:1px solid #1a1a2e;">
        <span style="color:#8080a0;font-size:10px;">${name}</span>
        <span style="color:${c};font-size:10px;font-weight:700;">${v}% ${prob >= 0.5 ? '↑' : '↓'}</span>
      </div>`;
    }).join('');

    card.innerHTML = `
      <div style="text-align:center;margin-bottom:12px;">
        <div style="font-size:28px;font-weight:800;color:${color};">${data.prediction}</div>
        <div style="color:${color};font-size:13px;margin-top:2px;">Probabilidad: ${probPct}%</div>
        <div style="margin:8px auto;max-width:160px;">
          <div style="background:#2a2a3e;border-radius:4px;height:6px;">
            <div style="width:${probPct}%;background:${color};border-radius:4px;height:6px;transition:width 0.5s;"></div>
          </div>
        </div>
        <div style="color:#6b7280;font-size:11px;">Confianza: ${confPct}%</div>
      </div>
      <div style="border-top:1px solid #2a2a3e;padding-top:10px;">
        <div style="color:#6b7280;font-size:10px;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">Votos del ensemble</div>
        ${voteRows}
      </div>
      <div style="margin-top:8px;padding-top:8px;border-top:1px solid #2a2a3e;">
        <div style="color:#6b7280;font-size:10px;display:flex;justify-content:space-between;">
          <span>📅 Predicción basada en:</span>
          <span style="color:#8080a0;font-weight:600;">${data.prediction_date || data.last_news_date || '—'}</span>
        </div>
      </div>`;

  } catch (e) {
    console.error('[IA Prediction] Excepción:', e);
    card.innerHTML = `<div style="color:#ef4444;font-size:12px;padding:12px;">❌ Error: ${e.message}</div>`;
  }
}

// ── Dashboard ────────────────────────────────────────────────────────────────
async function initDashboard() {
  try {
    const stock = await api.getStockCurrent();
    document.querySelector('.price').textContent = stock.price.toFixed(2);
    const changeEl = document.querySelector('.price-change');
    changeEl.textContent = `${stock.change >= 0 ? '▲' : '▼'} ${Math.abs(stock.change_percent).toFixed(2)}% Since yesterday`;
    changeEl.className = `price-change ${stock.change >= 0 ? 'positive' : 'negative'}`;
    const stats = document.querySelectorAll('.stat-value');
    if (stats[0]) stats[0].textContent = `${(stock.volume_24h / 1e9).toFixed(2)}B`;
    if (stats[1]) stats[1].textContent = `${stock.high_52w.toFixed(2)}`;
    if (stats[2]) stats[2].textContent = `${stock.low_52w.toFixed(2)}`;
  } catch (e) { console.warn('stock/current error:', e); }

  // Forecast eliminado del dashboard

  await loadEnsembleForecast();

  // Botón de refresh global
  document.getElementById('globalRefreshBtn')?.addEventListener('click', async () => {
    const btn   = document.getElementById('globalRefreshBtn');
    const label = document.getElementById('globalRefreshLabel');
    btn.disabled = true;
    label.textContent = 'Actualizando...';
    try {
      const result = await api.refreshAllData();
      const fin = result.results?.financial_data;
      const ens = result.results?.ensemble_news;
      const added = fin?.added ?? 0;
      label.textContent = added > 0 ? `✅ +${added} días` : '✅ Al día';
      // Refrescar ensemble y recargar gráfica con fecha de predicción
      await loadEnsembleForecast();
      // Recargar la gráfica - initChart recargará con la fecha de predicción
      const { initChart: reloadChart } = await import('./chart.js');
      reloadChart();
    } catch (e) {
      label.textContent = '❌ Error';
    }
    setTimeout(() => { label.textContent = 'Actualizar'; btn.disabled = false; }, 3000);
  });

  initChart();
  
  // Event listener para switch de tipo de gráfico
  const chartSwitch = document.getElementById('chartTypeSwitch');
  
  if (chartSwitch) {
    chartSwitch.addEventListener('change', (e) => {
      const type = e.target.checked ? 'candle' : 'line';
      import('./chart.js').then(module => module.switchChartType(type));
    });
  }
  
  // Event listeners para selector de horizonte de predicción (switch de 3 posiciones)
  const horizonBtns = document.querySelectorAll('.horizon-switch-btn');
  
  horizonBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      // Actualizar estado activo
      horizonBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      
      // Cargar predicción del horizonte seleccionado
      const horizon = btn.getAttribute('data-horizon');
      loadEnsembleForecast(horizon);
    });
  });
}

// ── Navegación ───────────────────────────────────────────────────────────────
function initNav() {
  const views = {
    dashboard: document.getElementById('dashboardView'),
    news:      document.getElementById('newsView'),
    portfolio: document.getElementById('portfolioView'),
  };

  document.querySelectorAll('.top-nav a').forEach(link => {
    link.addEventListener('click', async (e) => {
      e.preventDefault();
      document.querySelectorAll('.top-nav a').forEach(l => l.classList.remove('active'));
      link.classList.add('active');

      const view = link.getAttribute('data-view');
      Object.values(views).forEach(v => { if (v) v.style.display = 'none'; });

      if (view === 'dashboard') {
        views.dashboard.style.display = 'grid';
      } else if (view === 'news') {
        views.news.style.display = 'block';
        await loadNews();
      } else if (view === 'portfolio') {
        views.portfolio.style.display = 'block';
        await loadPortfolio();
      }
    });
  });
}

// ── Chatbot Dashboard Principal (Solo Market Agent) ─────────────────────────
function initDashboardChat() {
  const input = document.getElementById('chatInput');
  const sendBtn = document.getElementById('sendBtn');
  const messages = document.getElementById('chatMessages');

  if (!input || !sendBtn || !messages) return;

  async function sendMessage() {
    const message = input.value.trim();
    if (!message) return;

    // Añadir mensaje del usuario
    const userDiv = document.createElement('div');
    userDiv.className = 'chat-message user';
    userDiv.innerHTML = `<div class="message-content">${message}</div>`;
    messages.appendChild(userDiv);
    messages.scrollTop = messages.scrollHeight;

    input.value = '';
    input.disabled = true;
    sendBtn.disabled = true;

    // Añadir indicador de typing con animación
    const typingId = `typing_${Date.now()}`;
    const typingDiv = document.createElement('div');
    typingDiv.className = 'chat-message bot';
    typingDiv.id = typingId;
    typingDiv.innerHTML = `<div class="message-content" style="color:#6b7280;">
      💹 Analizando<span class="typing-dots"><span>.</span><span>.</span><span>.</span></span>
    </div>`;
    messages.appendChild(typingDiv);
    messages.scrollTop = messages.scrollHeight;

    try {
      const response = await api.chatStream(message, 'market');
      let fullResponse = '';

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.type === 'text_chunk') {
                fullResponse += data.text;
                // Actualizar mensaje en tiempo real
                const streamDiv = document.getElementById(`${typingId}_stream`);
                if (!streamDiv) {
                  document.getElementById(typingId).innerHTML = `<div class="message-content" id="${typingId}_stream"></div>`;
                }
                document.getElementById(`${typingId}_stream`).textContent = fullResponse;
                messages.scrollTop = messages.scrollHeight;
              } else if (data.type === 'response') {
                fullResponse = data.text;
              }
            } catch (e) {
              // Ignorar errores de parsing
            }
          }
        }
      }

      // Mostrar respuesta final parseada (mismo enfoque que Wallet)
      document.getElementById(typingId)?.remove();
      const botDiv = document.createElement('div');
      botDiv.className = 'chat-message bot';
      
      const text = fullResponse || 'Sin respuesta';
      const data = extractJSON(text);
      
      // Eliminar bloques JSON del texto markdown
      let md = text
        .replace(/```json[\s\S]*?```/g, '')
        .replace(/```[\s\S]*?```/g, '');
      if (data) md = md.replace(/\{[\s\S]*\}/g, '');
      md = md.trim();
      
      let html = '';
      
      // Mostrar texto narrativo si es sustancial
      if (md && md.length > 50) {
        html += `<div style="font-size:12px;line-height:1.7;color:#d0d0e0;margin-bottom:${data ? '10px' : '0'};">${parseMarkdown(md)}</div>`;
      }
      
      // Tarjeta de resumen JSON si hay señal/predicción
      if (data && (data.signal || data.sentiment)) {
        const signal = data.signal || data.sentiment || '';
        const prediction = data.prediction || data.price_prediction?.direction || '';
        const confidence = data.confidence != null ? Math.round(data.confidence * 100) : null;
        const reasons = data.reasons || [];
        const risks = data.risks || [];
        const reasoning = data.reasoning || data.price_prediction?.reasoning || '';
        const sc = signalColor(signal);
        const color = '#6366f1'; // Market agent color
        
        html += `<div style="background:#0f0f1a;border:1px solid ${color}30;border-radius:8px;padding:10px;font-size:11px;margin-top:${md.length > 50 ? '8px' : '0'};">
          <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:6px;">
            ${badge(signal, sc)}
            ${prediction ? `→ ${badge(prediction, color)}` : ''}
            ${confidence != null ? `<span style="color:#6b7280;">${confidence}% confianza</span>` : ''}
          </div>
          ${reasons.slice(0, 3).map(r => `<div style="color:#a0a0b0;padding:1px 0;">✓ ${r}</div>`).join('')}
          ${risks.slice(0, 2).map(r => `<div style="color:#f87171;padding:1px 0;">⚠ ${r}</div>`).join('')}
          ${reasoning && !md ? `<div style="color:#8080a0;font-size:10px;margin-top:6px;font-style:italic;">${reasoning.slice(0, 200)}${reasoning.length > 200 ? '...' : ''}</div>` : ''}
        </div>`;
      }
      
      // Si no hay nada parseado, mostrar el texto original
      if (!html) html = `<div style="font-size:12px;line-height:1.7;color:#d0d0e0;">${parseMarkdown(text)}</div>`;
      
      botDiv.innerHTML = `<div class="message-content">${html}</div>`;
      messages.appendChild(botDiv);
      messages.scrollTop = messages.scrollHeight;

    } catch (e) {
      document.getElementById(typingId)?.remove();
      const errorDiv = document.createElement('div');
      errorDiv.className = 'chat-message bot';
      errorDiv.innerHTML = `<div class="message-content" style="color:#ef4444;">❌ Error: Backend no disponible</div>`;
      messages.appendChild(errorDiv);
      messages.scrollTop = messages.scrollHeight;
    }

    input.disabled = false;
    sendBtn.disabled = false;
    input.focus();
  }

  sendBtn.addEventListener('click', sendMessage);
  input.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
  });
}

// ── Boot ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initNav();
  initNews();
  initPortfolio();
  initChat(); // Chat completo para sección Wallet
  initDashboardChat(); // Chat simple para dashboard principal
  initDashboard();
});
