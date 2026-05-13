// API Configuration
const API_BASE = 'http://localhost:5000/api';

// Inyectar CSS de animaciones globales al cargar
(function() {
    const s = document.createElement('style');
    s.id = 'globalAnimStyle';
    s.textContent = `
        @keyframes bounce {
            0%, 80%, 100% { transform: translateY(0); }
            40% { transform: translateY(-5px); }
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        .typing-dots { display: inline-flex; gap: 3px; align-items: center; }
        .typing-dots span {
            display: inline-block;
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: currentColor;
            animation: bounce 1.2s infinite;
        }
        .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
        .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
    `;
    document.head.appendChild(s);
})();

// Update stock price display
const updateStockPrice = async () => {
    try {
        const response = await fetch(`${API_BASE}/stock/current`);
        const data = await response.json();
        
        document.querySelector('.price').textContent = `$${data.price.toFixed(2)}`;
        
        const changeEl = document.querySelector('.price-change');
        const changeText = `${data.change >= 0 ? '▲' : '▼'} ${Math.abs(data.change_percent).toFixed(2)}% Since yesterday`;
        changeEl.textContent = changeText;
        changeEl.className = `price-change ${data.change >= 0 ? 'positive' : 'negative'}`;
        
        const stats = document.querySelectorAll('.stat-value');
        stats[0].textContent = `$${(data.volume_24h / 1e9).toFixed(2)}B`;
        stats[1].textContent = `$${data.high_52w.toFixed(2)}`;
        stats[2].textContent = `$${data.low_52w.toFixed(2)}`;
    } catch (error) {
        console.error('Error updating stock price:', error);
    }
};

// Chart rendering
const TIME_RANGES = { '1D': 1, '1W': 5, '1M': 22, '6M': 126, '1Y': 252 };
let currentDays = 22;

const loadChart = async (days) => {
    currentDays = days;
    const container = document.querySelector('.chart-container');
    if (!container) return;

    // Loading state
    container.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#6b7280;font-size:13px;">Cargando datos...</div>`;

    try {
        const res = await fetch(`${API_BASE}/stock/history?days=${days}`);
        const candles = await res.json();
        if (!candles.length || candles.error) {
            container.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#ef4444;font-size:13px;">Error cargando datos</div>`;
            return;
        }
        renderLineChart(container, candles);
    } catch (e) {
        container.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#ef4444;font-size:13px;">Backend no disponible</div>`;
    }
};

const renderLineChart = (container, candles) => {
    const W = container.clientWidth || 700;
    const H = 280;
    const PAD = { top: 20, right: 50, bottom: 30, left: 10 };
    const chartW = W - PAD.left - PAD.right;
    const chartH = H - PAD.top - PAD.bottom;

    const closes = candles.map(c => c.c);
    const minP = Math.min(...closes);
    const maxP = Math.max(...closes);
    const range = maxP - minP || 1;

    const xStep = chartW / (candles.length - 1 || 1);
    const toY = v => PAD.top + chartH - ((v - minP) / range) * chartH;
    const toX = i => PAD.left + i * xStep;

    // Build path
    const pts = candles.map((c, i) => `${toX(i).toFixed(1)},${toY(c.c).toFixed(1)}`).join(' ');
    const areaClose = `${PAD.left},${PAD.top + chartH} ` + pts + ` ${toX(candles.length - 1)},${PAD.top + chartH}`;

    const isUp = closes[closes.length - 1] >= closes[0];
    const color = isUp ? '#10b981' : '#ef4444';

    // Y-axis labels (5 levels)
    const yLabels = Array.from({ length: 5 }, (_, i) => {
        const val = minP + (range * i) / 4;
        const y = toY(val);
        return `<text x="${W - PAD.right + 4}" y="${y.toFixed(1)}" fill="#6b7280" font-size="10" dominant-baseline="middle">$${val.toFixed(0)}</text>
                <line x1="${PAD.left}" y1="${y.toFixed(1)}" x2="${W - PAD.right}" y2="${y.toFixed(1)}" stroke="#1f1f2e" stroke-width="1"/>`;
    }).join('');

    // X-axis labels (up to 6 evenly spaced)
    const xCount = Math.min(6, candles.length);
    const xLabels = Array.from({ length: xCount }, (_, i) => {
        const idx = Math.round(i * (candles.length - 1) / (xCount - 1 || 1));
        const x = toX(idx);
        const label = candles[idx].x.slice(5); // MM-DD
        return `<text x="${x.toFixed(1)}" y="${H - 4}" fill="#6b7280" font-size="10" text-anchor="middle">${label}</text>`;
    }).join('');

    container.innerHTML = `
        <svg width="100%" height="${H}" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
            <defs>
                <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="${color}" stop-opacity="0.25"/>
                    <stop offset="100%" stop-color="${color}" stop-opacity="0"/>
                </linearGradient>
            </defs>
            ${yLabels}
            ${xLabels}
            <polygon points="${areaClose}" fill="url(#areaGrad)"/>
            <polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round"/>
        </svg>`;
};
const updateForecast = async () => {
    try {
        const response = await fetch(`${API_BASE}/forecast`);
        const data = await response.json();
        
        const forecastValue = document.querySelector('.forecast-value');
        forecastValue.textContent = `${data.forecast_change_percent >= 0 ? '+' : ''}${data.forecast_change_percent.toFixed(1)}%`;
        forecastValue.className = `forecast-value ${data.direction === 'up' ? 'positive' : 'negative'}`;
        
        document.querySelector('.forecast-price').textContent = `$${data.forecast_price.toFixed(2)}`;
        document.querySelector('.forecast-time').textContent = data.timestamp;
    } catch (error) {
        console.error('Error updating forecast:', error);
    }
};

// Chat functionality
const addMessage = (text, isUser = false) => {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${isUser ? 'user' : 'bot'}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = text;
    
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
};

const sendMessage = async () => {
    const chatInput = document.getElementById('chatInput');
    const message = chatInput.value.trim();
    if (!message) return;
    
    addMessage(message, true);
    chatInput.value = '';
    
    // Mostrar indicador de "escribiendo..."
    const typingIndicator = addTypingIndicator();
    
    try {
        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });
        
        const data = await response.json();
        
        // Remover indicador de "escribiendo..."
        typingIndicator.remove();
        
        addMessage(data.response, false);
    } catch (error) {
        console.error('Error sending message:', error);
        
        // Remover indicador de "escribiendo..."
        typingIndicator.remove();
        
        addMessage('Error: Backend not available', false);
    }
};

const addTypingIndicator = () => {
    const chatMessages = document.getElementById('chatMessages');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'chat-message bot typing-indicator';
    typingDiv.id = 'typingIndicator';
    
    typingDiv.innerHTML = `
        <div class="message-content">
            <div class="typing-dots" style="color:#6b7280;">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return typingDiv;
};

// News functionality
let allNews = [];
let currentPage = 1;
const newsPerPage = 10;

const loadNews = async () => {
    try {
        const response = await fetch(`${API_BASE}/news?limit=200`);
        allNews = await response.json();
        renderNews();
    } catch (error) {
        console.error('Error loading news:', error);
        document.getElementById('newsGrid').innerHTML = '<p style="color: #6b7280; text-align: center; padding: 40px;">Error loading news. Make sure the backend is running.</p>';
    }
};

const renderNews = () => {
    const newsGrid = document.getElementById('newsGrid');
    newsGrid.innerHTML = '';
    
    // Calculate pagination
    const startIndex = (currentPage - 1) * newsPerPage;
    const endIndex = startIndex + newsPerPage;
    const paginatedNews = allNews.slice(startIndex, endIndex);
    
    if (paginatedNews.length === 0) {
        newsGrid.innerHTML = '<p style="color: #6b7280; text-align: center; padding: 40px;">No news available.</p>';
        return;
    }
    
    paginatedNews.forEach(item => {
        const card = createNewsCard(item);
        newsGrid.appendChild(card);
    });
    
    // Update pagination controls
    updatePagination();
};

const updatePagination = () => {
    const totalPages = Math.ceil(allNews.length / newsPerPage);
    document.getElementById('pageInfo').textContent = `Page ${currentPage} of ${totalPages}`;
    
    document.getElementById('prevBtn').disabled = currentPage === 1;
    document.getElementById('nextBtn').disabled = currentPage === totalPages;
};

const createNewsCard = (newsItem) => {
    const card = document.createElement('div');
    card.className = 'news-card';
    
    const sentiment = newsItem.sentiment_score > 0.1 ? 'positive' : 
                     newsItem.sentiment_score < -0.1 ? 'negative' : 'neutral';
    
    const date = new Date(newsItem.date);
    const day = date.getDate();
    const month = date.toLocaleDateString('en-US', { month: 'long' });
    
    card.innerHTML = `
        <div class="news-date-badge">
            <div class="date-day">${day}</div>
            <div class="date-month">${month}</div>
        </div>
        <img class="news-image" src="https://picsum.photos/seed/${newsItem.headline.substring(0, 10)}/180/120" alt="News">
        <div class="news-content">
            <div class="news-meta">
                <span class="news-source">${newsItem.source || 'NVDA'}</span>
                <span class="news-sentiment ${sentiment}">${sentiment}</span>
            </div>
            <h3 class="news-title">${newsItem.headline}</h3>
        </div>
    `;
    
    if (newsItem.url) {
        card.addEventListener('click', () => window.open(newsItem.url, '_blank'));
    }
    
    return card;
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Initialize dashboard
    updateStockPrice();
    updateForecast();
    loadChart(22); // 1M por defecto
    
    // Setup navigation
    document.querySelectorAll('.top-nav a').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            
            document.querySelectorAll('.top-nav a').forEach(l => l.classList.remove('active'));
            link.classList.add('active');
            
            const view = link.getAttribute('data-view');
            const dashboardView = document.getElementById('dashboardView');
            const newsView = document.getElementById('newsView');
            const portfolioView = document.getElementById('portfolioView');

            // Hide all views first
            dashboardView.style.display = 'none';
            newsView.style.display = 'none';
            portfolioView.style.display = 'none';

            // Show the selected view
            if (view === 'dashboard') {
                dashboardView.style.display = 'grid';
            } else if (view === 'news') {
                newsView.style.display = 'block';
                loadNews();
            } else if (view === 'portfolio') {
                portfolioView.style.display = 'block';
                loadPortfolio();
            }
        });
    });
    
    // Setup time range buttons
    document.querySelectorAll('.time-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.time-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const label = btn.textContent.trim();
            loadChart(TIME_RANGES[label] || 22);
        });
    });
    
    // Setup chat
    const sendBtn = document.getElementById('sendBtn');
    const chatInput = document.getElementById('chatInput');
    
    if (sendBtn && chatInput) {
        sendBtn.addEventListener('click', sendMessage);
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
    }
    
    // Setup pagination
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    
    if (prevBtn) {
        prevBtn.addEventListener('click', () => {
            if (currentPage > 1) {
                currentPage--;
                renderNews();
                document.getElementById('newsGrid').scrollIntoView({ behavior: 'smooth' });
            }
        });
    }
    
    if (nextBtn) {
        nextBtn.addEventListener('click', () => {
            const totalPages = Math.ceil(allNews.length / newsPerPage);
            if (currentPage < totalPages) {
                currentPage++;
                renderNews();
                document.getElementById('newsGrid').scrollIntoView({ behavior: 'smooth' });
            }
        });
    }
    
    // Setup trades pagination
    const tradesPrevBtn = document.getElementById('tradesPrevBtn');
    const tradesNextBtn = document.getElementById('tradesNextBtn');
    
    if (tradesPrevBtn) {
        tradesPrevBtn.addEventListener('click', () => {
            if (currentTradesPage > 1) {
                currentTradesPage--;
                renderTrades();
                document.getElementById('tradesList').scrollIntoView({ behavior: 'smooth' });
            }
        });
    }
    
    if (tradesNextBtn) {
        tradesNextBtn.addEventListener('click', () => {
            const totalPages = Math.ceil(allTrades.length / tradesPerPage);
            if (currentTradesPage < totalPages) {
                currentTradesPage++;
                renderTrades();
                document.getElementById('tradesList').scrollIntoView({ behavior: 'smooth' });
            }
        });
    }

    // Setup sliding chat

    const floatingChatBtn = document.getElementById('floatingChatBtn');
    const slidingChatPanel = document.getElementById('slidingChatPanel');
    const closeChatBtn = document.getElementById('closeChatBtn');
    const slidingChatInput = document.getElementById('slidingChatInput');
    const sendBtnSliding = document.getElementById('sendBtnSliding');
    
    if (floatingChatBtn) {
        floatingChatBtn.addEventListener('click', () => {
            slidingChatPanel.classList.add('open');
        });
    }
    
    if (closeChatBtn) {
        closeChatBtn.addEventListener('click', () => {
            slidingChatPanel.classList.remove('open');
        });
    }
    
    if (sendBtnSliding && slidingChatInput) {
        sendBtnSliding.addEventListener('click', () => sendSlidingMessage());
        slidingChatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendSlidingMessage();
        });
    }
    
    // Setup mode selector
    const modeButtons = document.querySelectorAll('.mode-btn');
    modeButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            modeButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const mode = btn.getAttribute('data-mode');
            const agentBar = document.getElementById('agentSelectorBar');
            const roundsBar = document.getElementById('roundsSelectorBar');

            if (mode === 'multi-agent') {
                if (agentBar) agentBar.style.display = 'none';
                if (roundsBar) roundsBar.style.display = 'flex';
                startMultiAgentConversation();
            } else {
                if (agentBar) agentBar.style.display = 'flex';
                if (roundsBar) roundsBar.style.display = 'none';
                resetToNormalMode();
            }
        });
    });

    // Setup selector de agente (modo normal)
    document.querySelectorAll('.agent-tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            // Guardar historial del agente actual
            saveCurrentHistory();

            currentSlidingAgent = btn.getAttribute('data-agent');
            document.querySelectorAll('.agent-tab-btn').forEach(b => {
                b.style.background = '#2a2a3e';
                b.style.color = '#6b7280';
                b.style.border = '1px solid #3a3a5e';
            });
            const colors = { market: '#6366f1', news: '#f59e0b', reddit: '#ec4899' };
            const c = colors[currentSlidingAgent] || '#6366f1';
            btn.style.background = `${c}20`;
            btn.style.color = c;
            btn.style.border = `1px solid ${c}40`;

            // Restaurar historial del nuevo agente
            loadHistory(currentSlidingAgent);
        });
    });

    // Setup selector de rondas (modo multi-agente)
    document.querySelectorAll('.rounds-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.rounds-btn').forEach(b => {
                b.style.background = '#2a2a3e';
                b.style.color = '#6b7280';
                b.style.border = '1px solid #3a3a5e';
            });
            btn.style.background = '#6366f120';
            btn.style.color = '#6366f1';
            btn.style.border = '1px solid #6366f140';
        });
    });
});

// Chat mode state
let currentChatMode = 'normal';
let currentSlidingAgent = 'market'; // agente activo en modo normal

// Historiales independientes por chat
const chatHistories = {
    market: `<div class="chat-message bot"><div class="message-content">¡Hola! Soy el <strong>Market Agent</strong>. Pregúntame sobre análisis técnico de NVDA.</div></div>`,
    news: `<div class="chat-message bot"><div class="message-content">¡Hola! Soy el <strong>News Agent</strong>. Pregúntame sobre noticias y sentimiento de NVDA.</div></div>`,
    reddit: `<div class="chat-message bot"><div class="message-content">¡Hola! Soy el <strong>Reddit Agent</strong>. Pregúntame sobre el sentimiento de la comunidad sobre NVDA.</div></div>`,
    'multi-agent': '',
};

const saveCurrentHistory = () => {
    const chatMessages = document.getElementById('slidingChatMessages');
    if (!chatMessages) return;
    if (currentChatMode === 'multi-agent') {
        chatHistories['multi-agent'] = chatMessages.innerHTML;
    } else {
        chatHistories[currentSlidingAgent] = chatMessages.innerHTML;
    }
};

const loadHistory = (key) => {
    const chatMessages = document.getElementById('slidingChatMessages');
    if (!chatMessages) return;
    chatMessages.innerHTML = chatHistories[key] || '';
    chatMessages.scrollTop = chatMessages.scrollHeight;
};

const resetToNormalMode = () => {
    // Guardar historial del modo actual antes de cambiar
    saveCurrentHistory();
    currentChatMode = 'normal';

    const agentBar = document.getElementById('agentSelectorBar');
    const roundsBar = document.getElementById('roundsSelectorBar');
    if (agentBar) agentBar.style.display = 'flex';
    if (roundsBar) roundsBar.style.display = 'none';

    // Restaurar historial del agente activo
    loadHistory(currentSlidingAgent);

    const slidingChatInput = document.getElementById('slidingChatInput');
    const sendBtnSliding = document.getElementById('sendBtnSliding');
    if (slidingChatInput) slidingChatInput.disabled = false;
    if (sendBtnSliding) sendBtnSliding.disabled = false;
};

const startMultiAgentConversation = async () => {
    currentChatMode = 'multi-agent';
    const chatMessages = document.getElementById('slidingChatMessages');
    const slidingChatInput = document.getElementById('slidingChatInput');
    const sendBtnSliding = document.getElementById('sendBtnSliding');
    if (!chatMessages) return;
    if (slidingChatInput) slidingChatInput.disabled = true;
    if (sendBtnSliding) sendBtnSliding.disabled = true;

    // Limpiar completamente el historial anterior
    chatMessages.innerHTML = '';

    // Helpers locales
    const appendMsg = (html) => {
        const div = document.createElement('div');
        div.className = 'chat-message bot';
        div.innerHTML = html;
        chatMessages.appendChild(div);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    };

    const addTypingIndicatorLocal = (id, name, color) => {
        const div = document.createElement('div');
        div.className = 'chat-message bot';
        div.id = id;
        div.innerHTML = `
            <div style="display:flex;align-items:center;gap:8px;padding:10px 14px;background:#1e1e2e;border-radius:12px;border-left:3px solid ${color};">
                <span style="color:${color};font-size:12px;font-weight:700;">${name}</span>
                <span id="${id}_status" style="color:#6b7280;font-size:11px;">está analizando</span>
                <span class="typing-dots" style="color:${color};display:inline-flex;gap:3px;margin-left:4px;">
                    <span style="width:5px;height:5px;border-radius:50%;background:${color};animation:bounce 1.2s infinite 0s;"></span>
                    <span style="width:5px;height:5px;border-radius:50%;background:${color};animation:bounce 1.2s infinite 0.2s;"></span>
                    <span style="width:5px;height:5px;border-radius:50%;background:${color};animation:bounce 1.2s infinite 0.4s;"></span>
                </span>
            </div>`;
        chatMessages.appendChild(div);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    };

    // Inyectar CSS de animación si no existe (ya está en globalAnimStyle, pero por si acaso)
    if (!document.getElementById('bounceStyle')) {
        const s = document.createElement('style');
        s.id = 'bounceStyle';
        s.textContent = `@keyframes bounce{0%,80%,100%{transform:translateY(0)}40%{transform:translateY(-5px)}}`;
        document.head.appendChild(s);
    }

    // Mensaje de inicio
    appendMsg(`<div style="color:#6b7280;font-size:12px;text-align:center;padding:8px;font-style:italic;"> Iniciando debate multi-agente sobre NVDA...</div>`);

    try {
        const response = await fetch(`${API_BASE}/debate/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker: 'NVDA', rounds: parseInt(document.querySelector('.rounds-btn.active, [data-rounds="1"]')?.getAttribute('data-rounds') || '1') })
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const raw = line.slice(6).trim();
                if (!raw) continue;
                try {
                    const msg = JSON.parse(raw);

                    if (msg.type === 'status') {
                        appendMsg(`<div style="color:#6b7280;font-size:11px;text-align:center;padding:6px;font-style:italic;">${msg.message}</div>`);

                    } else if (msg.type === 'typing') {
                        addTypingIndicatorLocal(`st_${msg.agent}`, msg.name, msg.color);

                    } else if (msg.type === 'agent_result') {
                        document.getElementById(`st_${msg.agent}`)?.remove();
                        // Parsear JSON del agente para mostrar como chat natural
                        const data = extractJSON(msg.text);
                        let content = '';
                        if (data) {
                            const signal = data.signal || data.sentiment || data.community_sentiment || data.position || '';
                            const prediction = data.prediction || data.price_prediction?.direction || data.adjusted_prediction || '';
                            const reasoning = data.reasoning || data.price_prediction?.reasoning || data.final_reasoning || data.key_argument || '';
                            const confidence = data.confidence != null ? Math.round(data.confidence * 100)
                                             : data.adjusted_confidence != null ? Math.round(data.adjusted_confidence * 100) : null;
                            const trend = data.trend || '';
                            const rsi = data.rsi != null ? data.rsi : null;
                            const volatility = data.volatility != null ? (data.volatility * 100).toFixed(1) + '%' : null;
                            const returnPct = data.return_period_pct != null ? data.return_period_pct.toFixed(2) + '%' : null;
                            const reasons = data.reasons || data.key_headlines || [];
                            const alpacaScore = data.alpaca_score != null ? data.alpaca_score : null;
                            const gdeltTrend = data.gdelt_trend || '';
                            const sentimentScore = data.sentiment_score != null ? data.sentiment_score : null;
                            const postVolume = data.post_volume_avg != null ? data.post_volume_avg : null;
                            const reliability = data.reliability || '';
                            // Campos de debate ronda 2
                            const agreesMarket = data.agrees_with_market;
                            const agreesNews = data.agrees_with_news;
                            const agreesReddit = data.agrees_with_reddit;

                            const signalColor = signal.includes('BULL') || signal.includes('POSIT') || signal.includes('SUBE') ? '#10b981'
                                             : signal.includes('BEAR') || signal.includes('NEGAT') || signal.includes('BAJA') ? '#ef4444'
                                             : '#9ca3af';

                            const badge = (val, color) => val ? `<span style="background:${color}20;color:${color};border:1px solid ${color}40;border-radius:4px;padding:1px 7px;font-size:10px;font-weight:700;">${val}</span>` : '';

                            const metrics = [];
                            if (trend) metrics.push(`📈 Tendencia: <strong>${trend}</strong>`);
                            if (rsi != null) metrics.push(`RSI: <strong>${rsi}</strong>`);
                            if (volatility) metrics.push(`Volatilidad: <strong>${volatility}</strong>`);
                            if (returnPct) metrics.push(`Retorno período: <strong>${returnPct}</strong>`);
                            if (alpacaScore != null) metrics.push(`Alpaca score: <strong>${alpacaScore}</strong>`);
                            if (gdeltTrend) metrics.push(`GDELT: <strong>${gdeltTrend}</strong>`);
                            if (sentimentScore != null) metrics.push(`Sentiment: <strong>${sentimentScore}</strong>`);
                            if (postVolume != null) metrics.push(`Posts/día: <strong>${postVolume}</strong>`);
                            if (reliability) metrics.push(`Fiabilidad: ${badge(reliability, '#6366f1')}`);
                            // Acuerdos del debate
                            if (agreesMarket != null) metrics.push(`vs Market: ${agreesMarket ? '✅' : '❌'}`);
                            if (agreesNews != null) metrics.push(`vs News: ${agreesNews ? '✅' : '❌'}`);
                            if (agreesReddit != null) metrics.push(`vs Reddit: ${agreesReddit ? '✅' : '❌'}`);

                            content = `<div style="font-size:12px;line-height:1.7;color:#e0e0e0;">
                                <div style="margin-bottom:8px;display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
                                    ${signal ? badge(signal, signalColor) : ''}
                                    ${prediction ? `→ ${badge(prediction, '#6366f1')}` : ''}
                                    ${confidence != null ? `<span style="color:#6b7280;font-size:10px;">${confidence}% confianza</span>` : ''}
                                </div>
                                ${metrics.length ? `<div style="color:#a0a0b0;font-size:11px;margin-bottom:6px;">${metrics.join(' &nbsp;·&nbsp; ')}</div>` : ''}
                                ${reasons.length ? `<div style="margin-bottom:6px;">${reasons.slice(0,3).map(r => `<div style="color:#c0c0d0;font-size:11px;padding:2px 0;">• ${r}</div>`).join('')}</div>` : ''}
                                ${reasoning ? `<div style="color:#a0a0b0;font-size:11px;font-style:italic;border-top:1px solid #2a2a3e;padding-top:6px;margin-top:4px;">${reasoning.substring(0, 300)}${reasoning.length > 300 ? '...' : ''}</div>` : ''}
                            </div>`;
                        } else {
                            content = `<div style="font-size:12px;color:#e0e0e0;">${msg.text.replace(/```json[\s\S]*?```/g, '').trim().substring(0, 300)}</div>`;
                        }
                        const agentDiv = document.createElement('div');
                        agentDiv.className = 'chat-message bot';
                        agentDiv.innerHTML = `
                            <div style="border-left:3px solid ${msg.color};padding:10px 14px;background:#1e1e2e;border-radius:2px 12px 12px 12px;">
                                <div style="color:${msg.color};font-size:11px;font-weight:700;margin-bottom:6px;">${msg.name}${msg.round > 1 ? ` <span style="color:#6b7280;font-weight:400;">(Ronda ${msg.round})</span>` : ''}</div>
                                ${content}
                            </div>`;
                        chatMessages.appendChild(agentDiv);
                        chatMessages.scrollTop = chatMessages.scrollHeight;

                    } else if (msg.type === 'verdict') {
                        chatMessages.appendChild(Object.assign(document.createElement('div'), {
                            className: 'chat-message bot',
                            innerHTML: renderVerdictCard(msg.text)
                        }));
                        chatMessages.scrollTop = chatMessages.scrollHeight;

                    } else if (msg.type === 'error') {
                        appendMsg(`<div style="color:#ef4444;font-size:12px;">❌ ${msg.message}</div>`);
                    }
                } catch (e) { console.error('SSE parse error:', e, 'line:', line); }
            }
        }
    } catch (err) {
        appendMsg(`<div style="color:#ef4444;font-size:12px;">❌ Error: ${err.message}. Asegúrate de que el backend está corriendo en localhost:5000.</div>`);
    }

    if (slidingChatInput) slidingChatInput.disabled = false;
    if (sendBtnSliding) sendBtnSliding.disabled = false;
};
const sendSlidingMessage = async () => {
    const slidingChatInput = document.getElementById('slidingChatInput');
    const message = slidingChatInput.value.trim();
    if (!message) return;

    const agentAtSend = currentSlidingAgent;
    const chatMessages = document.getElementById('slidingChatMessages');

    addSlidingMessage(message, true);
    slidingChatInput.value = '';

    const agentColors = { market: '#6366f1', news: '#f59e0b', reddit: '#ec4899' };
    const color = agentColors[agentAtSend] || '#6366f1';

    // Indicador de escritura
    const typingId = `typing_simple_${Date.now()}`;
    const typingDiv = document.createElement('div');
    typingDiv.className = 'chat-message bot';
    typingDiv.id = typingId;
    typingDiv.innerHTML = `<div style="display:flex;align-items:center;gap:8px;padding:8px 12px;background:#1e1e2e;border-radius:12px;border-left:3px solid ${color};">
        <span style="color:${color};font-size:11px;font-weight:700;">${agentAtSend.toUpperCase()} AGENT</span>
        <span id="${typingId}_status" style="color:#6b7280;font-size:11px;">pensando</span>
        <span class="typing-dots" style="color:${color};display:inline-flex;gap:3px;">
            <span style="width:5px;height:5px;border-radius:50%;background:${color};animation:bounce 1.2s infinite 0s;"></span>
            <span style="width:5px;height:5px;border-radius:50%;background:${color};animation:bounce 1.2s infinite 0.2s;"></span>
            <span style="width:5px;height:5px;border-radius:50%;background:${color};animation:bounce 1.2s infinite 0.4s;"></span>
        </span>
    </div>`;
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        const response = await fetch(`${API_BASE}/chat/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, agent: agentAtSend })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                try {
                    const msg = JSON.parse(line.slice(6));

                    if (msg.type === 'tool_call') {
                        const statusEl = document.getElementById(`${typingId}_status`);
                        if (statusEl) statusEl.textContent = `🔧 ${msg.name}`;

                    } else if (msg.type === 'tool_step') {
                        const statusEl = document.getElementById(`${typingId}_status`);
                        if (statusEl) statusEl.textContent = msg.step;

                    } else if (msg.type === 'text_chunk') {
                        // Streaming de texto token a token
                        let streamDiv = document.getElementById(`${typingId}_stream`);
                        if (!streamDiv) {
                            document.getElementById(typingId)?.remove();
                            const newDiv = document.createElement('div');
                            newDiv.className = 'chat-message bot';
                            newDiv.innerHTML = `<div class="message-content" id="${typingId}_stream" style="font-size:12px;color:#e0e0e0;"></div>`;
                            chatMessages.appendChild(newDiv);
                            streamDiv = document.getElementById(`${typingId}_stream`);
                        }
                        if (streamDiv) {
                            streamDiv.textContent += msg.text;
                            chatMessages.scrollTop = chatMessages.scrollHeight;
                        }

                    } else if (msg.type === 'response') {
                        // Respuesta final — reemplazar el stream con la versión parseada
                        const streamEl = document.getElementById(`${typingId}_stream`);
                        if (streamEl) streamEl.closest('.chat-message')?.remove();
                        document.getElementById(typingId)?.remove();
                        if (currentSlidingAgent === agentAtSend) {
                            addSlidingMessage(msg.text, false);
                        } else {
                            chatHistories[agentAtSend] += `<div class="chat-message bot"><div class="message-content">${msg.text}</div></div>`;
                        }

                    } else if (msg.type === 'error') {
                        document.getElementById(typingId)?.remove();
                        if (currentSlidingAgent === agentAtSend) {
                            addSlidingMessage(`❌ Error: ${msg.message}`, false);
                        }

                    } else if (msg.type === 'done') {
                        document.getElementById(typingId)?.remove();
                    }
                } catch (e) {}
            }
        }
    } catch (error) {
        document.getElementById(typingId)?.remove();
        if (currentSlidingAgent === agentAtSend) {
            addSlidingMessage('Error: Backend no disponible', false);
        }
    }
};

const addSlidingMessage = (text, isUser = false) => {
    const chatMessages = document.getElementById('slidingChatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${isUser ? 'user' : 'bot'}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    if (!isUser) {
        const agentColors = { market: '#6366f1', news: '#f59e0b', reddit: '#ec4899' };
        const color = agentColors[currentSlidingAgent] || '#6366f1';

        // Extraer JSON si existe al final
        const data = extractJSON(text);

        // Renderizar el texto markdown (sin el bloque JSON)
        const markdownText = text.replace(/```json[\s\S]*?```/g, '').replace(/```[\s\S]*?```/g, '').trim();
        let html = '';

        if (markdownText) {
            html += `<div style="font-size:12px;line-height:1.7;color:#d0d0e0;margin-bottom:${data ? '10px' : '0'};">${parseMarkdown(markdownText)}</div>`;
        }

        // Si hay JSON, añadir tarjeta de resumen
        if (data) {
            const signal = data.signal || data.sentiment || data.community_sentiment || '';
            const prediction = data.prediction || data.price_prediction?.direction || '';
            const confidence = data.confidence != null ? Math.round(data.confidence * 100) : null;
            const reasons = data.reasons || [];
            const risks = data.risks || [];

            const signalColor = signal.includes('BULL') || signal.includes('POSIT') || signal.includes('SUBE') ? '#10b981'
                             : signal.includes('BEAR') || signal.includes('NEGAT') || signal.includes('BAJA') ? '#ef4444'
                             : '#9ca3af';

            const badge = (val, c) => val ? `<span style="background:${c}20;color:${c};border:1px solid ${c}40;border-radius:4px;padding:1px 7px;font-size:10px;font-weight:700;">${val}</span>` : '';

            html += `<div style="background:#0f0f1a;border:1px solid ${color}30;border-radius:8px;padding:10px;font-size:11px;">
                <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:6px;">
                    ${signal ? badge(signal, signalColor) : ''}
                    ${prediction ? `→ ${badge(prediction, color)}` : ''}
                    ${confidence != null ? `<span style="color:#6b7280;">${confidence}% confianza</span>` : ''}
                </div>
                ${reasons.slice(0,3).map(r => `<div style="color:#a0a0b0;padding:1px 0;">✓ ${r}</div>`).join('')}
                ${risks.slice(0,2).map(r => `<div style="color:#f87171;padding:1px 0;">⚠ ${r}</div>`).join('')}
            </div>`;
        }

        contentDiv.innerHTML = html || text;
    } else {
        contentDiv.textContent = text;
    }

    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
};

const addSlidingTypingIndicator = () => {
    const chatMessages = document.getElementById('slidingChatMessages');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'chat-message bot typing-indicator';
    typingDiv.id = 'slidingTypingIndicator';
    
    typingDiv.innerHTML = `
        <div class="message-content">
            <div class="typing-dots" style="color:#6b7280;">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return typingDiv;
};

// Portfolio functionality
let allTrades = [];
let currentTradesPage = 1;
const tradesPerPage = 10;

const loadPortfolio = async () => {
    try {
        // Load user trades from CSV
        const userTradesResponse = await fetch(`${API_BASE}/user/trades`);
        const userTradesData = await userTradesResponse.json();
        
        if (userTradesData.error) {
            showNoPortfolio(userTradesData.error);
            return;
        }
        
        // Update wallet display with user data
        updateWalletDisplayFromUserTrades(userTradesData.stats);
        
        // Load and display user trades
        allTrades = userTradesData.trades || [];
        currentTradesPage = 1;
        renderTrades();
        
    } catch (error) {
        console.error('Error loading portfolio:', error);
        showNoPortfolio('Error loading user trades. Make sure the backend is running.');
    }
};

const updateWalletDisplayFromUserTrades = (stats) => {
    // Update balance
    const balanceEl = document.getElementById('walletBalance');
    const changeEl = document.getElementById('walletChange');
    const investedCashEl = document.getElementById('investedCash');
    const availableCashEl = document.getElementById('availableCash');
    
    if (balanceEl) {
        balanceEl.textContent = `$${stats.total_portfolio_value.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
    }
    
    if (changeEl) {
        const profit = stats.total_profit;
        const isPositive = profit >= 0;
        
        changeEl.innerHTML = `
            <span class="change-value">${isPositive ? '+' : ''}$${Math.abs(profit).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}</span>
            <span class="change-percent ${isPositive ? 'positive' : 'negative'}">${isPositive ? '▲' : '▼'} ${Math.abs(stats.return_percent).toFixed(2)}%</span>
        `;
    }
    
    // Update invested and available cash
    if (investedCashEl) {
        investedCashEl.textContent = `$${stats.total_invested.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
    }
    
    if (availableCashEl) {
        availableCashEl.textContent = `$${stats.available_cash.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
    }
    
    // Render chart with portfolio value
    renderWalletChartFromStats(stats);
};

const renderWalletChartFromStats = (stats) => {
    const chartContainer = document.getElementById('walletChart');
    if (!chartContainer) return;
    
    const initialCash = 10000;
    const finalValue = stats.total_portfolio_value;
    const totalReturn = stats.return_percent;
    
    // Create a smooth curve from initial to final value
    const points = 50;
    const values = [];
    
    for (let i = 0; i <= points; i++) {
        const progress = i / points;
        // Add some volatility to make it look realistic
        const volatility = Math.sin(progress * Math.PI * 4) * (finalValue - initialCash) * 0.1;
        const value = initialCash + (finalValue - initialCash) * progress + volatility;
        values.push(value);
    }
    
    // Find min and max for scaling
    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);
    const range = maxValue - minValue;
    
    // Create SVG path
    const width = 100;
    const height = 100;
    const padding = 10;
    
    const pathPoints = values.map((value, index) => {
        const x = (index / points) * width;
        const y = height - ((value - minValue) / range) * (height - padding * 2) - padding;
        return `${x},${y}`;
    }).join(' ');
    
    const isPositive = totalReturn >= 0;
    const color = isPositive ? '#10b981' : '#ef4444';
    
    chartContainer.innerHTML = `
        <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
            <defs>
                <linearGradient id="chartGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" style="stop-color:${color};stop-opacity:0.3" />
                    <stop offset="100%" style="stop-color:${color};stop-opacity:0" />
                </linearGradient>
            </defs>
            <polyline 
                points="${pathPoints} ${width},${height} 0,${height}" 
                fill="url(#chartGradient)" 
                stroke="none"
            />
            <polyline 
                points="${pathPoints}" 
                fill="none" 
                stroke="${color}" 
                stroke-width="2"
                vector-effect="non-scaling-stroke"
            />
        </svg>
    `;
};

const loadTrades = async (strategyName) => {
    try {
        const response = await fetch(`${API_BASE}/backtest/trades/${strategyName}`);
        const data = await response.json();
        
        if (data.error) {
            showNoTrades(data.error);
            return;
        }
        
        allTrades = data.trades || [];
        currentTradesPage = 1;
        renderTrades();
        
    } catch (error) {
        console.error('Error loading trades:', error);
        showNoTrades('Error loading trades.');
    }
};

const renderTrades = () => {
    const tradesList = document.getElementById('tradesList');
    const tradesCount = document.getElementById('tradesCount');
    
    if (!tradesList) return;
    
    tradesList.innerHTML = '';
    
    // Update trades count
    if (tradesCount) {
        tradesCount.textContent = `${allTrades.length} operaciones`;
    }
    
    // Calculate pagination
    const startIndex = (currentTradesPage - 1) * tradesPerPage;
    const endIndex = startIndex + tradesPerPage;
    const paginatedTrades = allTrades.slice(startIndex, endIndex);
    
    if (paginatedTrades.length === 0) {
        showNoTrades('No hay movimientos disponibles.');
        return;
    }
    
    paginatedTrades.forEach(trade => {
        const tradeItem = createTradeSidebarItem(trade);
        tradesList.appendChild(tradeItem);
    });
    
    // Update pagination
    updateTradesPagination();
};

const createTradeSidebarItem = (trade) => {
    const item = document.createElement('div');
    item.className = 'trade-sidebar-item';
    
    const date = new Date(trade.date);
    const formattedDate = date.toLocaleDateString('es-ES', { 
        day: '2-digit',
        month: 'short',
        year: 'numeric'
    });
    
    const type = trade.type.toLowerCase();
    const typeClass = type === 'buy' ? 'buy' : 'sell';
    const typeLabel = type === 'buy' ? 'COMPRA' : 'VENTA';
    
    // Only show PnL for SELL trades
    let pnlHTML = '';
    
    if (type === 'sell' && trade.pnl_net !== undefined) {
        const pnl = parseFloat(trade.pnl_net);
        const pnlPercent = parseFloat(trade.pnl_pct);
        const pnlClass = pnl >= 0 ? 'positive' : 'negative';
        const pnlSign = pnl >= 0 ? '+' : '';
        const percentSign = pnlPercent >= 0 ? '▲' : '▼';
        
        pnlHTML = `
            <div class="trade-sidebar-pnl">
                <span class="trade-sidebar-pnl-amount ${pnlClass}">${pnlSign}$${Math.abs(pnl).toFixed(2)}</span>
                <span class="trade-sidebar-pnl-percent ${pnlClass}">${percentSign} ${Math.abs(pnlPercent).toFixed(2)}%</span>
            </div>
        `;
    }
    
    item.innerHTML = `
        <div class="trade-sidebar-header">
            <span class="trade-sidebar-date">${formattedDate}</span>
            <span class="trade-sidebar-type ${typeClass}">${typeLabel}</span>
        </div>
        <div class="trade-sidebar-price">$${parseFloat(trade.price).toFixed(2)}</div>
        ${pnlHTML}
    `;
    
    return item;
};

const updateTradesPagination = () => {
    const totalPages = Math.ceil(allTrades.length / tradesPerPage);
    const pageInfo = document.getElementById('tradesPageInfo');
    const prevBtn = document.getElementById('tradesPrevBtn');
    const nextBtn = document.getElementById('tradesNextBtn');
    
    if (pageInfo) {
        pageInfo.textContent = `${currentTradesPage}/${totalPages}`;
    }
    
    if (prevBtn) {
        prevBtn.disabled = currentTradesPage === 1;
    }
    
    if (nextBtn) {
        nextBtn.disabled = currentTradesPage === totalPages;
    }
};

const showNoTrades = (message) => {
    const tradesList = document.getElementById('tradesList');
    if (!tradesList) return;
    
    tradesList.innerHTML = `
        <div class="no-trades-sidebar">
            <div class="no-trades-sidebar-text">${message}</div>
        </div>
    `;
};

const updateWalletDisplay = (strategy) => {
    // Update balance
    const balanceEl = document.getElementById('walletBalance');
    const changeEl = document.getElementById('walletChange');
    const availableCashEl = document.getElementById('availableCash');
    
    if (balanceEl) {
        balanceEl.textContent = `$${strategy.final_value.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
    }
    
    if (changeEl) {
        const profit = strategy.final_value - strategy.initial_cash;
        const isPositive = strategy.total_return >= 0;
        
        changeEl.innerHTML = `
            <span class="change-value">${isPositive ? '+' : ''}$${Math.abs(profit).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}</span>
            <span class="change-percent ${isPositive ? 'positive' : 'negative'}">${isPositive ? '▲' : '▼'} ${Math.abs(strategy.total_return).toFixed(2)}%</span>
        `;
    }
    
    // Update available cash (final value of the strategy)
    if (availableCashEl) {
        availableCashEl.textContent = `$${strategy.final_value.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
    }
};

const renderWalletChart = (strategy) => {
    const chartContainer = document.getElementById('walletChart');
    if (!chartContainer) return;
    
    // Simulate portfolio value over time based on trades
    // Starting with initial cash and applying returns
    const initialCash = strategy.initial_cash;
    const finalValue = strategy.final_value;
    const totalReturn = strategy.total_return;
    
    // Create a smooth curve from initial to final value
    const points = 50;
    const values = [];
    
    for (let i = 0; i <= points; i++) {
        const progress = i / points;
        // Add some volatility to make it look realistic
        const volatility = Math.sin(progress * Math.PI * 4) * (finalValue - initialCash) * 0.1;
        const value = initialCash + (finalValue - initialCash) * progress + volatility;
        values.push(value);
    }
    
    // Find min and max for scaling
    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);
    const range = maxValue - minValue;
    
    // Create SVG path
    const width = 100;
    const height = 100;
    const padding = 10;
    
    const pathPoints = values.map((value, index) => {
        const x = (index / points) * width;
        const y = height - ((value - minValue) / range) * (height - padding * 2) - padding;
        return `${x},${y}`;
    }).join(' ');
    
    const isPositive = totalReturn >= 0;
    const color = isPositive ? '#10b981' : '#ef4444';
    
    chartContainer.innerHTML = `
        <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
            <defs>
                <linearGradient id="chartGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" style="stop-color:${color};stop-opacity:0.3" />
                    <stop offset="100%" style="stop-color:${color};stop-opacity:0" />
                </linearGradient>
            </defs>
            <polyline 
                points="${pathPoints} ${width},${height} 0,${height}" 
                fill="url(#chartGradient)" 
                stroke="none"
            />
            <polyline 
                points="${pathPoints}" 
                fill="none" 
                stroke="${color}" 
                stroke-width="2"
                vector-effect="non-scaling-stroke"
            />
        </svg>
    `;
};

const showNoPortfolio = (message) => {
    const balanceEl = document.getElementById('walletBalance');
    const changeEl = document.getElementById('walletChange');
    const chartContainer = document.getElementById('walletChart');
    
    if (balanceEl) balanceEl.textContent = '$10,000';
    if (changeEl) {
        changeEl.innerHTML = `
            <span class="change-value">+$0.00</span>
            <span class="change-percent positive">▲ 0.00%</span>
        `;
    }
    if (chartContainer) {
        chartContainer.innerHTML = `
            <div style="display: flex; align-items: center; justify-content: center; height: 100%; color: rgba(255,255,255,0.5); font-size: 14px;">
                ${message}
            </div>
        `;
    }
};

const updatePortfolioSummary = (strategies) => {
    // Total strategies
    document.getElementById('totalStrategies').textContent = strategies.length;
    
    // Best performer
    const bestStrategy = strategies.reduce((best, current) => 
        current.total_return > best.total_return ? current : best
    );
    document.getElementById('bestStrategy').textContent = bestStrategy.name.replace('_', ' ').toUpperCase();
    
    // Average return
    const avgReturn = strategies.reduce((sum, s) => sum + s.total_return, 0) / strategies.length;
    const avgReturnEl = document.getElementById('avgReturn');
    avgReturnEl.textContent = `${avgReturn >= 0 ? '+' : ''}${avgReturn.toFixed(2)}%`;
    avgReturnEl.className = `summary-value ${avgReturn >= 0 ? 'positive' : 'negative'}`;
    
    // Total trades
    const totalTrades = strategies.reduce((sum, s) => sum + s.total_trades, 0);
    document.getElementById('totalTrades').textContent = totalTrades;
};

const renderStrategies = (strategies) => {
    const grid = document.getElementById('strategiesGrid');
    grid.innerHTML = '';
    
    strategies.forEach(strategy => {
        const card = createStrategyCard(strategy);
        grid.appendChild(card);
    });
};

const createStrategyCard = (strategy) => {
    const card = document.createElement('div');
    card.className = 'strategy-card';
    
    const returnClass = strategy.total_return >= 0 ? 'positive' : 'negative';
    const strategyDisplayName = strategy.name.replace(/_/g, ' ').toUpperCase();
    
    // Format parameters
    let paramsHTML = '';
    if (strategy.params && Object.keys(strategy.params).length > 0) {
        const paramsList = Object.entries(strategy.params)
            .map(([key, value]) => `<span class="param-badge">${key}: ${value}</span>`)
            .join('');
        
        paramsHTML = `
            <div class="strategy-params">
                <div class="params-title">Parameters</div>
                <div class="params-list">${paramsList}</div>
            </div>
        `;
    }
    
    card.innerHTML = `
        <div class="strategy-header">
            <div class="strategy-name">${strategyDisplayName}</div>
            <div class="strategy-return ${returnClass}">
                ${strategy.total_return >= 0 ? '+' : ''}${strategy.total_return.toFixed(2)}%
            </div>
        </div>
        
        <div class="strategy-metrics">
            <div class="metric-item">
                <div class="metric-label">Initial Cash</div>
                <div class="metric-value">$${strategy.initial_cash.toLocaleString()}</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Final Value</div>
                <div class="metric-value">$${strategy.final_value.toLocaleString()}</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value">${strategy.sharpe_ratio.toFixed(4)}</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value">${strategy.max_drawdown.toFixed(2)}%</div>
            </div>
        </div>
        
        <div class="strategy-stats">
            <div class="stat-item">
                <div class="stat-item-label">Trades</div>
                <div class="stat-item-value">${strategy.total_trades}</div>
            </div>
            <div class="stat-item">
                <div class="stat-item-label">Won</div>
                <div class="stat-item-value positive">${strategy.won_trades}</div>
            </div>
            <div class="stat-item">
                <div class="stat-item-label">Lost</div>
                <div class="stat-item-value negative">${strategy.lost_trades}</div>
            </div>
            <div class="stat-item">
                <div class="stat-item-label">Win Rate</div>
                <div class="stat-item-value">${strategy.win_rate.toFixed(1)}%</div>
            </div>
        </div>
        
        ${paramsHTML}
    `;
    
    return card;
};

const showNoStrategies = (message) => {
    const grid = document.getElementById('strategiesGrid');
    grid.innerHTML = `
        <div class="no-strategies">
            <div class="no-strategies-icon">📊</div>
            <div class="no-strategies-text">No backtest results available</div>
            <div class="no-strategies-hint">${message}</div>
        </div>
    `;
    
    // Clear summary
    document.getElementById('totalStrategies').textContent = '0';
    document.getElementById('bestStrategy').textContent = '-';
    document.getElementById('avgReturn').textContent = '-';
    document.getElementById('totalTrades').textContent = '0';
};
