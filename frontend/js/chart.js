/**
 * chart.js — Renderizado de la gráfica de precios SVG.
 */

import { api } from './api.js';

const API_BASE = 'http://localhost:5000/api';
const TIME_RANGES = { '1D': 1, '1W': 5, '1M': 22, '6M': 126, '1Y': 252 };

let PREDICTION_DATE = null; // Variable global para almacenar la fecha de predicción

export function initChart() {
  document.querySelectorAll('.time-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.time-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const label = btn.textContent.trim();
      if (label === '1D') {
        loadChartLastTradingDayWithPrediction();
      } else {
        loadChartWithRange(TIME_RANGES[label] ?? 22);
      }
    });
  });

  loadChartWithPredictionDate(); // Cargar con fecha de predicción por defecto
}

async function loadChartWithPredictionDate() {
  try {
    // Obtener fecha de predicción del ensemble
    const res = await fetch(`${API_BASE}/forecast/ensemble`);
    const data = await res.json();
    PREDICTION_DATE = data.prediction_date || '2025-12-31';
    
    // Cargar gráfica hasta esa fecha (1 mes de datos)
    const histRes = await fetch(`${API_BASE}/stock/history?date=${PREDICTION_DATE}&days=22`);
    const candles = await histRes.json();
    
    if (!Array.isArray(candles) || candles.length === 0) {
      throw new Error('Sin datos para ' + PREDICTION_DATE);
    }
    
    renderChart(candles);
  } catch (e) {
    console.error('[Chart] Error cargando con fecha de predicción:', e);
    loadChart(22); // Fallback a carga normal
  }
}

async function loadChartWithRange(days) {
  const container = document.querySelector('.chart-container');
  if (!container) return;
  container.innerHTML = `<div class="chart-loading">Cargando datos...</div>`;

  try {
    // Si tenemos fecha de predicción, usarla; si no, obtenerla
    if (!PREDICTION_DATE) {
      const res = await fetch(`${API_BASE}/forecast/ensemble`);
      const data = await res.json();
      PREDICTION_DATE = data.prediction_date || '2025-12-31';
    }
    
    const histRes = await fetch(`${API_BASE}/stock/history?date=${PREDICTION_DATE}&days=${days}`);
    const candles = await histRes.json();
    
    if (!Array.isArray(candles) || candles.length === 0) throw new Error('Sin datos');
    renderLineChart(container, candles);
  } catch (e) {
    console.error('[Chart] Error:', e);
    container.innerHTML = `<div class="chart-error">Error cargando datos</div>`;
  }
}

async function loadChartLastTradingDayWithPrediction() {
  const container = document.querySelector('.chart-container');
  if (!container) return;
  container.innerHTML = `<div class="chart-loading">Cargando último día...</div>`;

  try {
    // Si tenemos fecha de predicción, usarla; si no, obtenerla
    if (!PREDICTION_DATE) {
      const res = await fetch(`${API_BASE}/forecast/ensemble`);
      const data = await res.json();
      PREDICTION_DATE = data.prediction_date || '2025-12-31';
    }

    // Para 1D pedimos los últimos 2 días hasta la fecha de predicción
    const histRes = await fetch(`${API_BASE}/stock/history?date=${PREDICTION_DATE}&days=2`);
    const candles = await histRes.json();

    if (!Array.isArray(candles) || candles.length === 0) throw new Error('Sin datos');

    // Si solo hay 1 punto, duplicarlo para poder dibujar
    const data = candles.length === 1
      ? [{ ...candles[0], x: candles[0].x + ' open' }, candles[0]]
      : candles;

    renderLineChart(container, data);
  } catch (e) {
    console.error('[1D] Error:', e);
    container.innerHTML = `<div class="chart-error">Error: ${e.message}</div>`;
  }
}

export async function loadChartLastTradingDay() {
  const container = document.querySelector('.chart-container');
  if (!container) return;
  container.innerHTML = `<div class="chart-loading">Cargando último día de trading...</div>`;

  try {
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    const yStr = yesterday.toISOString().slice(0, 10);

    console.log('[1D] Consultando calendario para:', yStr);
    const cal = await api.getMarketCalendar(yStr);
    console.log('[1D] Respuesta calendario:', cal);

    const targetDate = cal.is_trading_day ? yStr : (cal.previous_trading_day || yStr);
    console.log('[1D] Fecha objetivo:', targetDate);

    // Para 1D pedimos los últimos 2 días para poder dibujar la línea
    const res = await fetch(`${API_BASE}/stock/history?date=${targetDate}&days=2`);
    const candles = await res.json();
    console.log('[1D] Candles recibidos:', candles);

    if (!Array.isArray(candles) || candles.length === 0) throw new Error('Sin datos para ' + targetDate);

    // Si solo hay 1 punto, duplicarlo para poder dibujar
    const data = candles.length === 1
      ? [{ ...candles[0], x: candles[0].x + ' open' }, candles[0]]
      : candles;

    renderLineChart(container, data);
  } catch (e) {
    console.error('[1D] Error:', e);
    container.innerHTML = `<div class="chart-error">Error: ${e.message}</div>`;
  }
}

export async function loadChart(days) {
  const container = document.querySelector('.chart-container');
  if (!container) return;

  container.innerHTML = `<div class="chart-loading">Cargando datos...</div>`;

  try {
    const candles = await api.getStockHistory(days);
    if (!Array.isArray(candles) || candles.length === 0) throw new Error('Sin datos');
    renderLineChart(container, candles);
  } catch (e) {
    container.innerHTML = `<div class="chart-error">Error cargando datos</div>`;
  }
}

function renderChart(candles) {
  const container = document.querySelector('.chart-container');
  if (!container) return;
  renderLineChart(container, candles);
}

// Variable global para almacenar el tipo de gráfico actual
let currentChartType = 'line';
let currentCandles = [];

function renderLineChart(container, candles) {
  currentCandles = candles; // Guardar para poder cambiar tipo
  
  const W = container.clientWidth || 700;
  const H = 280;
  const PAD = { top: 20, right: 52, bottom: 30, left: 10 };
  const cW = W - PAD.left - PAD.right;
  const cH = H - PAD.top - PAD.bottom;

  const closes = candles.map(c => c.c);
  const minP = Math.min(...closes);
  const maxP = Math.max(...closes);
  const range = maxP - minP || 1;

  const toX = i => PAD.left + (i / (candles.length - 1 || 1)) * cW;
  const toY = v => PAD.top + cH - ((v - minP) / range) * cH;

  const pts = candles.map((c, i) => `${toX(i).toFixed(1)},${toY(c.c).toFixed(1)}`).join(' ');
  const area = `${PAD.left},${PAD.top + cH} ${pts} ${toX(candles.length - 1)},${PAD.top + cH}`;
  const color = closes.at(-1) >= closes[0] ? '#10b981' : '#ef4444';

  const yLabels = Array.from({ length: 5 }, (_, i) => {
    const val = minP + (range * i) / 4;
    const y = toY(val).toFixed(1);
    return `<text x="${W - PAD.right + 4}" y="${y}" fill="#6b7280" font-size="10" dominant-baseline="middle">$${val.toFixed(0)}</text>
            <line x1="${PAD.left}" y1="${y}" x2="${W - PAD.right}" y2="${y}" stroke="#1f1f2e" stroke-width="1"/>`;
  }).join('');

  const xCount = Math.min(6, candles.length);
  const xLabels = Array.from({ length: xCount }, (_, i) => {
    const idx = Math.round(i * (candles.length - 1) / (xCount - 1 || 1));
    return `<text x="${toX(idx).toFixed(1)}" y="${H - 4}" fill="#6b7280" font-size="10" text-anchor="middle">${candles[idx].x.slice(5)}</text>`;
  }).join('');

  container.innerHTML = `
    <svg width="100%" height="${H}" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
      <defs>
        <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="${color}" stop-opacity="0.25"/>
          <stop offset="100%" stop-color="${color}" stop-opacity="0"/>
        </linearGradient>
      </defs>
      ${yLabels}${xLabels}
      <polygon points="${area}" fill="url(#areaGrad)"/>
      <polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round"/>
    </svg>`;
}

function renderCandleChart(container, candles) {
  currentCandles = candles; // Guardar para poder cambiar tipo
  
  const W = container.clientWidth || 700;
  const H = 280;
  const PAD = { top: 20, right: 52, bottom: 30, left: 10 };
  const cW = W - PAD.left - PAD.right;
  const cH = H - PAD.top - PAD.bottom;

  // Calcular rango de precios usando high/low
  const highs = candles.map(c => c.h);
  const lows = candles.map(c => c.l);
  const minP = Math.min(...lows);
  const maxP = Math.max(...highs);
  const range = maxP - minP || 1;

  const candleWidth = Math.max(2, Math.min(12, cW / candles.length * 0.7));
  const toX = i => PAD.left + (i + 0.5) * (cW / candles.length);
  const toY = v => PAD.top + cH - ((v - minP) / range) * cH;

  // Grid y labels Y
  const yLabels = Array.from({ length: 5 }, (_, i) => {
    const val = minP + (range * i) / 4;
    const y = toY(val).toFixed(1);
    return `<text x="${W - PAD.right + 4}" y="${y}" fill="#6b7280" font-size="10" dominant-baseline="middle">$${val.toFixed(0)}</text>
            <line x1="${PAD.left}" y1="${y}" x2="${W - PAD.right}" y2="${y}" stroke="#1f1f2e" stroke-width="1"/>`;
  }).join('');

  // Labels X
  const xCount = Math.min(6, candles.length);
  const xLabels = Array.from({ length: xCount }, (_, i) => {
    const idx = Math.round(i * (candles.length - 1) / (xCount - 1 || 1));
    return `<text x="${toX(idx).toFixed(1)}" y="${H - 4}" fill="#6b7280" font-size="10" text-anchor="middle">${candles[idx].x.slice(5)}</text>`;
  }).join('');

  // Renderizar velas
  const candlesHTML = candles.map((c, i) => {
    const x = toX(i);
    const yHigh = toY(c.h);
    const yLow = toY(c.l);
    const yOpen = toY(c.o);
    const yClose = toY(c.c);
    
    const isGreen = c.c >= c.o;
    const color = isGreen ? '#10b981' : '#ef4444';
    
    const bodyTop = Math.min(yOpen, yClose);
    const bodyHeight = Math.abs(yClose - yOpen) || 1;
    
    return `
      <line x1="${x}" y1="${yHigh}" x2="${x}" y2="${yLow}" stroke="${color}" stroke-width="1.5"/>
      <rect x="${x - candleWidth/2}" y="${bodyTop}" width="${candleWidth}" height="${bodyHeight}" 
            fill="${isGreen ? color : 'none'}" stroke="${color}" stroke-width="1.5" rx="1"/>
    `;
  }).join('');

  container.innerHTML = `
    <svg width="100%" height="${H}" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
      ${yLabels}${xLabels}
      ${candlesHTML}
    </svg>`;
}

export function switchChartType(type) {
  currentChartType = type;
  const container = document.querySelector('.chart-container');
  if (!container || currentCandles.length === 0) return;
  
  if (type === 'line') {
    renderLineChart(container, currentCandles);
  } else if (type === 'candle') {
    renderCandleChart(container, currentCandles);
  }
}
