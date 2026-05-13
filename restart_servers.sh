#!/bin/bash

echo "🔄 Reiniciando servidores..."

# Detener servidores existentes
echo "⏹️  Deteniendo servidores..."
pkill -f "python3 api.py" 2>/dev/null
pkill -f "python3 -m http.server 8000" 2>/dev/null

# Esperar a que los procesos terminen
sleep 2

# Activar venv
source venv/bin/activate

# Iniciar backend en segundo plano
echo "🚀 Iniciando backend en http://localhost:5000..."
cd frontend/backend
python3 api.py > /dev/null 2>&1 &
BACKEND_PID=$!
cd ../..

# Esperar a que el backend esté listo
sleep 2

# Iniciar frontend en segundo plano
echo "🚀 Iniciando frontend en http://localhost:8000..."
cd frontend
python3 -m http.server 8000 > /dev/null 2>&1 &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ Servidores iniciados:"
echo "   Backend:  http://localhost:5000 (PID: $BACKEND_PID)"
echo "   Frontend: http://localhost:8000 (PID: $FRONTEND_PID)"
echo ""
echo "📝 Para detener los servidores:"
echo "   pkill -f 'python3 api.py'"
echo "   pkill -f 'python3 -m http.server 8000'"
echo ""
echo "🌐 Abriendo navegador..."
sleep 1
open http://localhost:8000

echo "✅ Listo!"