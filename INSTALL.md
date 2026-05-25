# Manual de Instalación — NVDA Trade Dashboard

## Requisitos Previos

| Requisito | Versión mínima | Notas |
|-----------|---------------|-------|
| Python | 3.9+ | Recomendado 3.10 o 3.11 |
| pip | 21.0+ | Incluido con Python |
| Git | 2.30+ | Para clonar el repositorio |
| Ollama | 0.1.0+ | Para ejecutar el LLM local |
| RAM | 8 GB mínimo | 16 GB recomendado (FinBERT + Ollama) |
| Disco | ~10 GB libres | Modelo qwen2.5:7b ocupa ~4.7 GB |

---

## Instalación en Windows

### 1. Instalar Python

1. Descargar Python 3.11 desde [python.org/downloads](https://www.python.org/downloads/)
2. **Marcar la casilla "Add Python to PATH"** durante la instalación
3. Verificar en PowerShell:

```powershell
python --version
pip --version
```

### 2. Clonar el repositorio

```powershell
git clone https://github.com/samcarri/TFG---An-lisis-Predictivo-del-Comportamiento-de-los-Mercados.git
cd TFG---An-lisis-Predictivo-del-Comportamiento-de-los-Mercados
```

### 3. Instalar dependencias Python

```powershell
pip install -r requirements.txt
```

> Si da error con PyTorch en Windows con GPU NVIDIA:
> ```powershell
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```
> Sin GPU (CPU only): se instala automáticamente con el requirements.txt.

> Si da error con TensorFlow: requiere Windows 10 o superior. Alternativa:
> ```powershell
> pip install tensorflow-cpu
> ```

### 4. Instalar Ollama

1. Descargar desde [ollama.com/download/windows](https://ollama.com/download/windows) e instalar
2. Ollama arranca automáticamente como servicio en segundo plano tras la instalación
3. Verificar que está corriendo:

```powershell
ollama list
```

4. Descargar el modelo (4.7 GB, puede tardar varios minutos):

```powershell
ollama pull qwen2.5:7b
```

5. Verificar que el modelo está disponible:

```powershell
ollama list
# Debe mostrar: qwen2.5:7b
```

> Si Ollama no arranca automáticamente, ejecutar en una terminal aparte:
> ```powershell
> ollama serve
> ```

### 5. Configurar variables de entorno

Crear el archivo `.env` en la raíz del proyecto con este contenido exacto:

```env
ALPACA_API_KEY=tu_alpaca_api_key
ALPACA_API_SECRET=tu_alpaca_api_secret
REDDIT_USER_AGENT=TFG-NVDA-Collector/1.0 (educational)
```

Desde PowerShell:
```powershell
@"
ALPACA_API_KEY=tu_alpaca_api_key
ALPACA_API_SECRET=tu_alpaca_api_secret
REDDIT_USER_AGENT=TFG-NVDA-Collector/1.0 (educational)
"@ | Out-File -Encoding utf8 .env
```

> `ALPACA_API_KEY` y `ALPACA_API_SECRET` son obligatorias para el agente de noticias. Sin ellas, el sistema usa GDELT como fallback automático y el resto de la app funciona con normalidad.
>
> `REDDIT_USER_AGENT` puede dejarse tal cual — solo identifica las peticiones al JSON público de Reddit.

Ver sección "Obtención de API Keys" al final de este documento.

### 6. Iniciar la aplicación

Abrir **dos terminales** en la raíz del proyecto:

**Terminal 1 — Backend API (puerto 5000):**
```powershell
cd frontend
python backend\api.py
```

**Terminal 2 — Frontend estático (puerto 8080):**
```powershell
cd frontend
python server.py
```

### 7. Abrir en el navegador

```
http://localhost:8080
```

> El puerto correcto es **8080**, no 8000.

### 8. Cargar datos iniciales

La primera vez que se abre la app, los datos financieros de NVDA no existen todavía. Hay dos formas de generarlos:

**Opción A — desde el navegador:** hacer clic en el botón "Actualizar datos" del dashboard.

**Opción B — desde terminal:**
```powershell
curl -X POST http://localhost:5000/api/refresh
```

Esto descarga automáticamente el histórico de NVDA del último año desde Yahoo Finance y lo guarda en `data/financial_data.csv`. No requiere ninguna API key.

---

## Instalación en macOS

### 1. Instalar Python (si no lo tienes)

```bash
# Con Homebrew (recomendado)
brew install python@3.11

# Verificar
python3 --version
```

### 2. Clonar el repositorio

```bash
git clone https://github.com/samcarri/TFG---An-lisis-Predictivo-del-Comportamiento-de-los-Mercados.git
cd TFG---An-lisis-Predictivo-del-Comportamiento-de-los-Mercados
```

### 3. Crear entorno virtual (recomendado)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> En Mac con Apple Silicon (M1/M2/M3): PyTorch se instala con soporte MPS automáticamente.

> Si TensorFlow falla en Apple Silicon:
> ```bash
> pip install tensorflow-macos tensorflow-metal
> ```

### 5. Instalar Ollama

```bash
# Descargar desde https://ollama.com/download/mac
# O con Homebrew:
brew install ollama

# Iniciar el servicio
ollama serve &

# Descargar el modelo (4.7 GB)
ollama pull qwen2.5:7b
```

### 6. Configurar variables de entorno

Crear el archivo `.env` en la raíz del proyecto con este contenido exacto:

```env
ALPACA_API_KEY=tu_alpaca_api_key
ALPACA_API_SECRET=tu_alpaca_api_secret
REDDIT_USER_AGENT=TFG-NVDA-Collector/1.0 (educational)
```

Desde terminal:
```bash
cat > .env << 'EOF'
ALPACA_API_KEY=tu_alpaca_api_key
ALPACA_API_SECRET=tu_alpaca_api_secret
REDDIT_USER_AGENT=TFG-NVDA-Collector/1.0 (educational)
EOF
```

> `ALPACA_API_KEY` y `ALPACA_API_SECRET` son obligatorias para el agente de noticias. Sin ellas, el sistema usa GDELT como fallback automático y el resto de la app funciona con normalidad.
>
> `REDDIT_USER_AGENT` puede dejarse tal cual — solo identifica las peticiones al JSON público de Reddit.

### 7. Iniciar la aplicación

```bash
# Terminal 1 — Backend API
cd frontend
python3 backend/api.py

# Terminal 2 — Frontend estático
cd frontend
python3 server.py
```

Abrir en el navegador: **http://localhost:8080**

### 8. Cargar datos iniciales

```bash
curl -X POST http://localhost:5000/api/refresh
```

---

## Estructura de Puertos

| Servicio | Puerto | Descripción |
|----------|--------|-------------|
| Frontend (SPA) | 8080 | Archivos estáticos HTML/JS/CSS |
| Backend (Flask API) | 5000 | Endpoints REST + SSE streaming |
| Ollama | 11434 | Servidor LLM local |

---

## Obtención de API Keys

### Alpaca Markets (noticias financieras — obligatoria)
1. Crear cuenta gratuita en [alpaca.markets](https://alpaca.markets/)
2. Ir a Dashboard → API Keys → Generate New Key
3. Copiar `API Key ID` y `Secret Key` al `.env`

> Sin esta key, el endpoint `/api/news` devuelve `"News data not found"`. El resto de la app (gráfica, predicciones, chat de mercado) funciona sin ella.

### GDELT (cobertura mediática global — sin API key)
GDELT es completamente público. El colector descarga dumps directamente desde `data.gdeltproject.org` sin autenticación. No requiere registro ni credenciales.

### FinBERT (análisis de sentimiento — sin API key)
El modelo `ProsusAI/finbert` se descarga automáticamente desde Hugging Face la primera vez que se ejecuta. No requiere cuenta ni key. Solo necesita `transformers` y `torch`, que ya están en `requirements.txt`. Se usa para analizar el sentimiento de posts de Reddit.

### Reddit (opcional — funciona sin auth)
El scraper usa el JSON público de Reddit sin autenticación OAuth. El `REDDIT_USER_AGENT` es solo para identificar las peticiones y evitar rate limiting.

---

## Solución de Problemas Comunes

### El navegador muestra "Method Not Allowed"
Estás accediendo directamente a un endpoint de API (como `/api/chat`) desde la barra de direcciones del navegador. Esos endpoints solo aceptan POST y deben usarse desde el frontend. Abre `http://localhost:8080` en su lugar.

### Error: `ModuleNotFoundError: No module named 'strands'`
```bash
pip install strands-agents strands-agents-tools
```

### El chat responde "All connection attempts failed"
Ollama no está corriendo. Verificar:
```bash
ollama list
```
Si no responde, iniciar el servicio:
```bash
# Windows: buscar "Ollama" en el menú inicio y ejecutarlo
# macOS/Linux:
ollama serve
```
Y asegurarse de que el modelo está descargado:
```bash
ollama pull qwen2.5:7b
```

### Error: `financial_data.csv not found`
El directorio `data/` no existe o está vacío. Ejecutar:
```bash
curl -X POST http://localhost:5000/api/refresh
```

### Error: `ALPACA_API_KEY not configured`
El archivo `.env` no existe o las keys no son válidas. Crear el `.env` en la raíz del proyecto con las keys de [alpaca.markets](https://alpaca.markets/).

### El frontend no carga datos (pantalla en blanco o errores en consola)
1. Verificar que el backend corre en el puerto 5000: `curl http://localhost:5000/api/stock/current`
2. Verificar que el frontend corre en el puerto 8080
3. Abrir la consola del navegador (F12) para ver errores de red concretos

### `torch` no se instala en Mac M1/M2
```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
```

### El ensemble forecast no devuelve predicción
Los modelos `.joblib` están incluidos en el repositorio en `entrenamiento_agentes/trained_models/`. Si faltan, ejecutar los scripts de entrenamiento correspondientes desde `entrenamiento_agentes/scripts/`.
