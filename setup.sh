#!/bin/bash

echo "=================================================="
echo "Setup del Sistema Multi-Agente Financiero"
echo "=================================================="

# Colores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Función para imprimir con color
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 1. Verificar Python
echo ""
echo "1️⃣  Verificando Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    print_success "Python encontrado: $PYTHON_VERSION"
else
    print_error "Python 3 no encontrado. Por favor instala Python 3.8 o superior."
    exit 1
fi

# 2. Crear entorno virtual
echo ""
echo "2️⃣  Creando entorno virtual..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Entorno virtual creado"
else
    print_warning "Entorno virtual ya existe"
fi

# 3. Activar entorno virtual
echo ""
echo "3️⃣  Activando entorno virtual..."
source venv/bin/activate
print_success "Entorno virtual activado"

# 4. Actualizar pip
echo ""
echo "4️⃣  Actualizando pip..."
pip install --upgrade pip > /dev/null 2>&1
print_success "pip actualizado"

# 5. Instalar dependencias
echo ""
echo "5️⃣  Instalando dependencias..."
pip install -r requirements.txt
if [ $? -eq 0 ]; then
    print_success "Dependencias instaladas correctamente"
else
    print_error "Error instalando dependencias"
    exit 1
fi

# 6. Verificar Ollama
echo ""
echo "6️⃣  Verificando Ollama..."
if command -v ollama &> /dev/null; then
    print_success "Ollama encontrado"
    
    # Verificar si Qwen 2.5 está instalado
    echo ""
    echo "7️⃣  Verificando modelo Qwen 2.5..."
    if ollama list | grep -q "qwen2.5:7b"; then
        print_success "Qwen 2.5:7b ya está instalado"
    else
        print_warning "Qwen 2.5:7b no encontrado. Descargando..."
        echo "   Esto puede tardar varios minutos..."
        ollama pull qwen2.5:7b
        if [ $? -eq 0 ]; then
            print_success "Qwen 2.5:7b descargado correctamente"
        else
            print_error "Error descargando Qwen 2.5:7b"
            exit 1
        fi
    fi
else
    print_warning "Ollama no encontrado"
    echo ""
    echo "Para instalar Ollama:"
    echo "  macOS:   brew install ollama"
    echo "  Linux:   curl -fsSL https://ollama.com/install.sh | sh"
    echo "  Windows: Descarga desde https://ollama.com/download"
    echo ""
    echo "Después ejecuta: ollama pull qwen2.5:7b"
fi

# 7. Verificar archivo .env
echo ""
echo "8️⃣  Verificando configuración..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    print_success "Archivo .env creado desde .env.example"
else
    print_warning "Archivo .env ya existe"
fi

# 8. Crear directorios necesarios
echo ""
echo "9️⃣  Creando directorios..."
mkdir -p cache
mkdir -p logs
print_success "Directorios creados"

# Resumen final
echo ""
echo "=================================================="
echo "✅ Setup completado exitosamente!"
echo "=================================================="
echo ""
echo "📋 Próximos pasos:"
echo ""
echo "1. Activar el entorno virtual:"
echo "   source venv/bin/activate"
echo ""
echo "2. Verificar que Ollama esté corriendo:"
echo "   ollama serve"
echo ""
echo "3. Ejecutar pruebas:"
echo "   python test_market_agent.py"
echo ""
echo "4. Ejecutar el agente:"
echo "   python agents/market_agent.py"
echo ""
echo "=================================================="