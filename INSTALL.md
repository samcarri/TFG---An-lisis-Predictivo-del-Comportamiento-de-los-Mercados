# Manual de Instalación — NVDA Trade Dashboard

## Requisitos Previos

| Requisito | Versión mínima | Notas |
|-----------|---------------|-------|
| Python | 3.9+ | Recomendado 3.10 o 3.11 |
| pip | 21.0+ | Incluido con Python |
| Git | 2.30+ | Para clonar el repositorio |
| Ollama | 0.1.0+ | Para ejecutar el LLM local |
| RAM | 8 GB mínimo | 16 GB recomendado (FinBERT + Ollama) |
| Disco | ~5 GB libres | Modelos + datos + dependencias |

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
git clone https://github.com/tu-usuario/multiagente.git
cd multiagente
```

### 3. Crear entorno virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Nota sobre PyTorch en Mac con Apple Silicon (M1/M2/M3):**
> Si tienes un Mac con chip Apple Silicon, PyTorch se instalará con soporte MPS (Metal Performance Shaders) automáticamente. No necesitas CUDA.

### 5. Instalar Ollama

```bash
# Descargar desde la web oficial
# https://ollama.com/download/mac

# O con Homebrew
brew install ollama

# Iniciar el servicio
ollama serve

# En otra terminal, descargar el modelo
ollama pull qwen2.5:7b
```

### 6. Configurar variables de entorno

```bash
cp .env.example .env
# Editar .env con tus API keys:
# ALPACA_API_KEY=tu_key
# ALPACA_API_SECRET=tu_secret
```

Si no tienes `.env.example`, crea `.env` manualmente:

```bash
cat > .env << 'EOF'
ALPACA_API_KEY=tu_alpaca_api_key
ALPACA_API_SECRET=tu_alpaca_api_secret
REDDIT_USER_AGENT=TFG-NVDA-Collector/1.0 (educational)
EOF
```

### 7. Verificar la instalación

```bash
# Verificar que los imports funcionan
python3 -c "import pandas, torch, transformers, flask, strands; print('OK')"

# Verificar Ollama
curl http://localhost:11434/api/tags
```

### 8. Iniciar la aplicación

```bash
# Opción A: Script automático
cd frontend
chmod +x start.sh
./start.sh

# Opción B: Manual (dos terminales)
# Terminal 1 — Backend API
cd frontend
python3 backend/api.py

# Terminal 2 — Frontend estático
cd frontend
python3 server.py
```

Abrir en el navegador: **http://localhost:8080**

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
git clone https://github.com/tu-usuario/multiagente.git
cd multiagente
```

### 3. Crear entorno virtual

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> Si da error de permisos en PowerShell:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 4. Instalar dependencias

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

> **Nota sobre PyTorch en Windows:**
> - Con GPU NVIDIA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
> - Sin GPU (CPU only): se instala automáticamente con el requirements.txt

> **Nota sobre TensorFlow en Windows:**
> TensorFlow 2.15+ requiere Windows 10 o superior. Si tienes problemas, usa `pip install tensorflow-cpu`.

### 5. Instalar Ollama

1. Descargar desde [ollama.com/download/windows](https://ollama.com/download/windows)
2. Ejecutar el instalador
3. Abrir una terminal y ejecutar:

```powershell
ollama serve
```

4. En otra terminal:

```powershell
ollama pull qwen2.5:7b
```

### 6. Configurar variables de entorno

Crear archivo `.env` en la raíz del proyecto:

```powershell
@"
ALPACA_API_KEY=tu_alpaca_api_key
ALPACA_API_SECRET=tu_alpaca_api_secret
REDDIT_USER_AGENT=TFG-NVDA-Collector/1.0 (educational)
"@ | Out-File -Encoding utf8 .env
```

### 7. Verificar la instalación

```powershell
python -c "import pandas, torch, transformers, flask, strands; print('OK')"
curl http://localhost:11434/api/tags
```

### 8. Iniciar la aplicación

```powershell
# Terminal 1 — Backend API
cd frontend
python backend\api.py

# Terminal 2 — Frontend estático
cd frontend
python server.py
```

Abrir en el navegador: **http://localhost:8080**

---

## Solución de Problemas Comunes

### Error: `ModuleNotFoundError: No module named 'strands'`

```bash
pip install strands-agents strands-agents-tools
```

### Error: `Ollama no disponible` en el chat

Verificar que Ollama está corriendo:
```bash
curl http://localhost:11434/api/tags
```

Si no responde, iniciar el servicio:
```bash
ollama serve
```

### Error: `torch` no se instala correctamente en Mac M1/M2

```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
```

### Error: `tensorflow` falla en Mac Apple Silicon

```bash
pip install tensorflow-macos tensorflow-metal
```

### Error: `ALPACA_API_KEY not configured`

Asegúrate de que el archivo `.env` existe en la raíz del proyecto y contiene las keys válidas. Puedes obtener keys gratuitas en [alpaca.markets](https://alpaca.markets/).

### El frontend no carga datos

1. Verificar que el backend está corriendo en puerto 5000
2. Verificar que `data/financial_data.csv` existe (se crea automáticamente al primer request)
3. Comprobar la consola del navegador (F12) para errores de red

### Walk-forward de modelos ML tarda mucho

Es normal. El walk-forward reentrena el modelo cada 10 pasos. Tiempos típicos:
- Random Forest / XGBoost: 2-5 minutos
- LSTM / GRU: 15-30 minutos
- Transformers: 20-40 minutos

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

### GDELT (cobertura mediática global — sin API key)
GDELT es completamente público. El colector descarga dumps directamente desde `data.gdeltproject.org` sin autenticación. No requiere registro ni credenciales.

### FinBERT (análisis de sentimiento — sin API key)
El modelo `ProsusAI/finbert` se descarga automáticamente desde Hugging Face la primera vez que se ejecuta. No requiere cuenta ni key. Solo necesita `transformers` y `torch`, que ya están en `requirements.txt`. Se usa para analizar el sentimiento de posts de Reddit.

### Reddit (opcional — funciona sin auth)
El scraper usa el JSON público de Reddit sin autenticación OAuth. El `REDDIT_USER_AGENT` es solo para identificar las peticiones y evitar rate limiting.
