#!/bin/bash
# Mata procesos en puertos 5000 y 8080, activa venv y arranca la app

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/venv/bin/activate"

if [ ! -f "$VENV" ]; then
  echo "❌ No se encontró el venv en $VENV"
  exit 1
fi

echo "🔪 Matando procesos en puertos 5000 y 8080..."
lsof -ti:5000,8080 | xargs kill -9 2>/dev/null
sleep 1

echo "🐍 Activando entorno virtual..."
source "$VENV"

echo "🚀 Arrancando backend (puerto 5000)..."
python3 "$SCRIPT_DIR/frontend/backend/api.py" > /tmp/backend.log 2>&1 &
BACKEND_PID=$!
echo "   PID backend: $BACKEND_PID"
sleep 3

# Verificar que el backend arrancó
if ! lsof -ti:5000 > /dev/null 2>&1; then
  echo "❌ El backend no arrancó. Log:"
  cat /tmp/backend.log
  exit 1
fi
echo "✅ Backend corriendo en puerto 5000"

echo "🌐 Arrancando frontend (puerto 8080)..."
python3 "$SCRIPT_DIR/frontend/server.py"
