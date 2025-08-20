#!/bin/bash
# 🚀 PatchCore Educational Dashboard - Setup & Launch Script

echo "🔬 PatchCore Educational Platform"
echo "================================="
echo ""

# Controlla se Python è installato
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 non trovato. Installa Python 3.8+ per continuare."
    exit 1
fi

echo "✅ Python 3 trovato: $(python3 --version)"

# Crea virtual environment se non esiste
if [ ! -d ".venv" ]; then
    echo "📦 Creazione virtual environment..."
    python3 -m venv .venv
    echo "✅ Virtual environment creato"
fi

# Attiva virtual environment
echo "🔌 Attivazione virtual environment..."
source .venv/bin/activate

# Controlla se pip è installato nel venv
if ! command -v pip &> /dev/null; then
    echo "❌ pip non trovato nel virtual environment."
    exit 1
fi

echo "✅ Virtual environment attivo"

# Installa dipendenze se non esistono
echo ""
echo "📦 Verifica dipendenze nel virtual environment..."
pip show dash plotly numpy Pillow > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "📥 Installazione dipendenze..."
    pip install -r requirements.txt
else
    echo "✅ Tutte le dipendenze core sono già installate"
fi

echo ""
echo "🚀 Avvio PatchCore Dashboard..."
echo "🌐 Dashboard disponibile su: http://localhost:8051"
echo "🛑 Per fermare: Ctrl+C"
echo ""

# Avvia dashboard nel virtual environment
python patchcore_dashboard.py
