#!/bin/bash

echo "======================================"
echo "🚀 Iniciando Dashboard NVDA"
echo "======================================"
echo ""

# Verificar dependencias
echo "📦 Verificando dependencias..."
if ! python3 -c "import flask" 2>/dev/null; then
    echo "⚠️  Flask no está instalado. Instalando..."
    pip3 install flask flask-cors pandas
fi

echo ""
echo "✅ Dependencias verificadas"
echo ""

# Iniciar backend en background
echo "🔧 Iniciando backend API en puerto 5000..."
python3 backend/api.py &
BACKEND_PID=$!

# Esperar a que el backend esté listo
sleep 3

# Iniciar frontend
echo "🌐 Iniciando frontend en puerto 8080..."
python3 server.py &
FRONTEND_PID=$!

echo ""
echo "======================================"
echo "✅ Dashboard iniciado correctamente"
echo "======================================"
echo ""
echo "📊 Frontend: http://localhost:8080"
echo "🔌 Backend:  http://localhost:5000"
echo ""
echo "Presiona Ctrl+C para detener ambos servidores"
echo ""

# Función para limpiar procesos al salir
cleanup() {
    echo ""
    echo "🛑 Deteniendo servidores..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ Servidores detenidos"
    exit 0
}

# Capturar Ctrl+C
trap cleanup INT

# Mantener el script corriendo
wait
