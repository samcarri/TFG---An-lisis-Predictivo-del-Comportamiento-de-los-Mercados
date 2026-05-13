/**
 * utils.js — Helpers reutilizables: markdown, JSON, badges, SSE reader.
 */

/** Parsea markdown básico a HTML */
export function parseMarkdown(text) {
  if (!text) return '';
  return text
    .replace(/```json[\s\S]*?```/g, '')
    .replace(/```[\s\S]*?```/g, '')
    .replace(/^#### (.+)$/gm, '<h4 style="color:#a0a0c0;font-size:11px;margin:8px 0 4px;text-transform:uppercase;">$1</h4>')
    .replace(/^### (.+)$/gm, '<h3 style="color:#c0c0e0;font-size:12px;font-weight:700;margin:10px 0 4px;">$1</h3>')
    .replace(/^## (.+)$/gm, '<h2 style="color:#d0d0f0;font-size:13px;font-weight:700;margin:10px 0 4px;">$1</h2>')
    .replace(/\*\*(.+?)\*\*/g, '<strong style="color:#e0e0ff;">$1</strong>')
    .replace(/^- (.+)$/gm, '<div style="padding:2px 0 2px 12px;color:#c0c0d0;font-size:12px;">• $1</div>')
    .replace(/^\d+\. (.+)$/gm, '<div style="padding:2px 0 2px 12px;color:#c0c0d0;font-size:12px;">$1</div>')
    .replace(/\n\n/g, '<br>')
    .replace(/\n/g, '<br>')
    .trim();
}

/** Extrae el primer bloque JSON de un texto */
export function extractJSON(text) {
  if (!text) return null;
  try {
    const match = text.match(/```json\s*([\s\S]*?)```/);
    if (match) return JSON.parse(match[1]);
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (jsonMatch) return JSON.parse(jsonMatch[0]);
  } catch (_) {}
  return null;
}

/** Badge de color para señales */
export function badge(val, color) {
  if (!val) return '';
  return `<span style="background:${color}20;color:${color};border:1px solid ${color}40;border-radius:4px;padding:1px 7px;font-size:10px;font-weight:700;">${val}</span>`;
}

/** Barra de confianza */
export function confidenceBar(val) {
  const pct = Math.round((val || 0) * 100);
  const color = pct >= 70 ? '#10b981' : pct >= 50 ? '#f59e0b' : '#ef4444';
  return `<div style="display:flex;align-items:center;gap:8px;margin-top:4px;">
    <div style="flex:1;background:#2a2a3e;border-radius:4px;height:6px;">
      <div style="width:${pct}%;background:${color};border-radius:4px;height:6px;"></div>
    </div>
    <span style="color:${color};font-size:11px;font-weight:700;">${pct}%</span>
  </div>`;
}

/** Color de señal según texto */
export function signalColor(signal = '') {
  const s = signal.toUpperCase();
  if (s.includes('BULL') || s.includes('POSIT') || s.includes('SUBE')) return '#10b981';
  if (s.includes('BEAR') || s.includes('NEGAT') || s.includes('BAJA')) return '#ef4444';
  return '#9ca3af';
}

/** Lee un stream SSE y llama a onMessage(parsedObj) por cada evento */
export async function readSSE(response, onMessage) {
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
        onMessage(JSON.parse(raw));
      } catch (e) {
        console.warn('SSE parse error:', e, raw);
      }
    }
  }
}

/** Crea un indicador de typing con dots animados */
export function createTypingDots(id, label, color) {
  const div = document.createElement('div');
  div.className = 'chat-message bot';
  if (id) div.id = id;
  div.innerHTML = `
    <div style="display:flex;align-items:center;gap:8px;padding:8px 12px;background:#1e1e2e;border-radius:12px;border-left:3px solid ${color};">
      <span style="color:${color};font-size:11px;font-weight:700;">${label}</span>
      <span id="${id}_status" style="color:#6b7280;font-size:11px;">pensando</span>
      <span style="display:inline-flex;gap:3px;">
        <span style="width:5px;height:5px;border-radius:50%;background:${color};animation:bounce 1.2s infinite 0s;"></span>
        <span style="width:5px;height:5px;border-radius:50%;background:${color};animation:bounce 1.2s infinite 0.2s;"></span>
        <span style="width:5px;height:5px;border-radius:50%;background:${color};animation:bounce 1.2s infinite 0.4s;"></span>
      </span>
    </div>`;
  return div;
}
