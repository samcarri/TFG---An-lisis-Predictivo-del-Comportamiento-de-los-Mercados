/**
 * news.js — Vista de noticias con paginación.
 */

import { api } from './api.js';

const PER_PAGE = 10;
let allNews = [];
let currentPage = 1;

export async function loadNews() {
  try {
    allNews = await api.getNews(200);
    currentPage = 1;
    renderPage();
  } catch (e) {
    document.getElementById('newsGrid').innerHTML =
      '<p style="color:#6b7280;text-align:center;padding:40px;">Error cargando noticias.</p>';
  }
}

function renderPage() {
  const grid = document.getElementById('newsGrid');
  grid.innerHTML = '';

  const start = (currentPage - 1) * PER_PAGE;
  const slice = allNews.slice(start, start + PER_PAGE);

  if (!slice.length) {
    grid.innerHTML = '<p style="color:#6b7280;text-align:center;padding:40px;">Sin noticias disponibles.</p>';
    return;
  }

  slice.forEach(item => grid.appendChild(createCard(item)));
  updatePagination();
}

function createCard(item) {
  const card = document.createElement('div');
  card.className = 'news-card';

  const sentiment = item.sentiment_score > 0.1 ? 'positive'
    : item.sentiment_score < -0.1 ? 'negative' : 'neutral';

  const date = new Date(item.date);
  card.innerHTML = `
    <div class="news-date-badge">
      <div class="date-day">${date.getDate()}</div>
      <div class="date-month">${date.toLocaleDateString('en-US', { month: 'long' })}</div>
    </div>
    <img class="news-image" src="https://picsum.photos/seed/${encodeURIComponent(item.headline.slice(0, 10))}/180/120" alt="News">
    <div class="news-content">
      <div class="news-meta">
        <span class="news-source">${item.source || 'NVDA'}</span>
        <span class="news-sentiment ${sentiment}">${sentiment}</span>
      </div>
      <h3 class="news-title">${item.headline}</h3>
    </div>`;

  if (item.url) card.addEventListener('click', () => window.open(item.url, '_blank'));
  return card;
}

function updatePagination() {
  const total = Math.ceil(allNews.length / PER_PAGE);
  document.getElementById('pageInfo').textContent = `Page ${currentPage} of ${total}`;
  document.getElementById('prevBtn').disabled = currentPage === 1;
  document.getElementById('nextBtn').disabled = currentPage === total;
}

export function initNews() {
  document.getElementById('prevBtn')?.addEventListener('click', () => {
    if (currentPage > 1) { currentPage--; renderPage(); }
  });
  document.getElementById('nextBtn')?.addEventListener('click', () => {
    const total = Math.ceil(allNews.length / PER_PAGE);
    if (currentPage < total) { currentPage++; renderPage(); }
  });
}
