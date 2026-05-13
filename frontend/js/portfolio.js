/**
 * portfolio.js — Vista wallet: estadísticas, gráfica y historial de trades.
 */

import { api } from './api.js';

const PER_PAGE = 10;
let allTrades = [];
let currentPage = 1;

export async function loadPortfolio() {
  try {
    const data = await api.getUserTrades();
    if (data.error) { showError(data.error); return; }
    updateStats(data.stats);
    renderChart(data.stats);
    allTrades = data.trades || [];
    currentPage = 1;
    renderTrades();
  } catch (e) {
    showError('Error cargando portfolio.');
  }
}

function updateStats(s) {
  setText('walletBalance', fmt(s.total_portfolio_value));
  setText('investedCash', fmt(s.total_invested));
  setText('availableCash', fmt(s.available_cash));

  const profit = s.total_profit;
  const isPos = profit >= 0;
  const changeEl = document.getElementById('walletChange');
  if (changeEl) {
    changeEl.innerHTML = `
      <span class="change-value">${isPos ? '+' : ''}${fmt(Math.abs(profit))}</span>
      <span class="change-percent ${isPos ? 'positive' : 'negative'}">${isPos ? '▲' : '▼'} ${Math.abs(s.return_percent).toFixed(2)}%</span>`;
  }
}

function renderChart(s) {
  const el = document.getElementById('walletChart');
  if (!el) return;

  const initial = s.total_invested + s.available_cash;
  const final = s.total_portfolio_value;
  const points = 50;
  const values = Array.from({ length: points + 1 }, (_, i) => {
    const p = i / points;
    return initial + (final - initial) * p + Math.sin(p * Math.PI * 4) * (final - initial) * 0.08;
  });

  const min = Math.min(...values);
  const max = Math.max(...values);
  const rng = max - min || 1;
  const W = 100, H = 100, PAD = 10;

  const pts = values.map((v, i) => {
    const x = (i / points) * W;
    const y = H - ((v - min) / rng) * (H - PAD * 2) - PAD;
    return `${x},${y}`;
  }).join(' ');

  const color = final >= initial ? '#10b981' : '#ef4444';
  el.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
      <defs>
        <linearGradient id="wGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="${color}" stop-opacity="0.3"/>
          <stop offset="100%" stop-color="${color}" stop-opacity="0"/>
        </linearGradient>
      </defs>
      <polygon points="${pts} ${W},${H} 0,${H}" fill="url(#wGrad)"/>
      <polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2" vector-effect="non-scaling-stroke"/>
    </svg>`;
}

function renderTrades() {
  const list = document.getElementById('tradesList');
  const countEl = document.getElementById('tradesCount');
  if (!list) return;

  if (countEl) countEl.textContent = `${allTrades.length} operaciones`;
  list.innerHTML = '';

  const start = (currentPage - 1) * PER_PAGE;
  const slice = allTrades.slice(start, start + PER_PAGE);

  if (!slice.length) { list.innerHTML = '<div class="no-trades-sidebar"><div class="no-trades-sidebar-text">Sin movimientos.</div></div>'; return; }

  slice.forEach(t => list.appendChild(createTradeItem(t)));
  updatePagination();
}

function createTradeItem(trade) {
  const item = document.createElement('div');
  item.className = 'trade-sidebar-item';
  const type = trade.type.toLowerCase();
  const label = type === 'buy' ? 'COMPRA' : 'VENTA';
  const date = new Date(trade.date).toLocaleDateString('es-ES', { day: '2-digit', month: 'short', year: 'numeric' });

  item.innerHTML = `
    <div class="trade-sidebar-header">
      <span class="trade-sidebar-date">${date}</span>
      <span class="trade-sidebar-type ${type}">${label}</span>
    </div>
    <div class="trade-sidebar-price">${parseFloat(trade.price).toFixed(2)}</div>`;
  return item;
}

function updatePagination() {
  const total = Math.ceil(allTrades.length / PER_PAGE);
  const info = document.getElementById('tradesPageInfo');
  if (info) info.textContent = `${currentPage}/${total}`;
  const prev = document.getElementById('tradesPrevBtn');
  const next = document.getElementById('tradesNextBtn');
  if (prev) prev.disabled = currentPage === 1;
  if (next) next.disabled = currentPage === total;
}

function showError(msg) {
  const el = document.getElementById('walletChart');
  if (el) el.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#6b7280;font-size:13px;">${msg}</div>`;
}

function fmt(n) {
  return `$${Number(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

export function initPortfolio() {
  document.getElementById('tradesPrevBtn')?.addEventListener('click', () => {
    if (currentPage > 1) { currentPage--; renderTrades(); }
  });
  document.getElementById('tradesNextBtn')?.addEventListener('click', () => {
    const total = Math.ceil(allTrades.length / PER_PAGE);
    if (currentPage < total) { currentPage++; renderTrades(); }
  });
}
