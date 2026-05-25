/**
 * api.js — Capa de acceso a datos. Todas las llamadas al backend aquí.
 */

const API_BASE = 'http://localhost:5000/api';

export const api = {
  async getStockCurrent() {
    const r = await fetch(`${API_BASE}/stock/current`);
    return r.json();
  },

  async refreshAllData() {
    const r = await fetch(`${API_BASE}/refresh`, { method: 'POST' });
    return r.json();
  },

  async getMarketCalendar(date) {
    const r = await fetch(`${API_BASE}/market/calendar?date=${date}`);
    return r.json();
  },

  async getStockHistory(days) {
    const r = await fetch(`${API_BASE}/stock/history?days=${days}`);
    return r.json();
  },

  async getForecast() {
    const r = await fetch(`${API_BASE}/forecast`);
    return r.json();
  },

  async getEnsembleForecast() {
    const r = await fetch(`${API_BASE}/forecast/ensemble`);
    return r.json();
  },

  async getWeeklyForecast() {
    const r = await fetch(`${API_BASE}/forecast/weekly`);
    return r.json();
  },

  async getMonthlyForecast() {
    const r = await fetch(`${API_BASE}/forecast/monthly`);
    return r.json();
  },

  async refreshEnsembleData() {
    const r = await fetch(`${API_BASE}/forecast/ensemble/refresh`, { method: 'POST' });
    return r.json();
  },

  async getNews(limit = 200) {
    const r = await fetch(`${API_BASE}/news?limit=${limit}`);
    return r.json();
  },

  async refreshNews() {
    const r = await fetch(`${API_BASE}/news/refresh`, { method: 'POST' });
    return r.json();
  },

  async getUserTrades() {
    const r = await fetch(`${API_BASE}/user/trades`);
    return r.json();
  },

  async chatStream(message, agent) {
    return fetch(`${API_BASE}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, agent }),
    });
  },

  async debateStream(ticker, startDate, endDate, rounds) {
    return fetch(`${API_BASE}/debate/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ticker, start_date: startDate || null, end_date: endDate || null, rounds }),
    });
  },
};
