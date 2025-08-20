#!/bin/bash
# 🔌 Virtual Environment Activation Script

echo "🔌 Attivazione virtual environment PatchCore..."

if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment non trovato."
    echo "💡 Esegui ./setup_and_run.sh per creare l'ambiente"
    exit 1
fi

source .venv/bin/activate
echo "✅ Virtual environment attivato!"
echo ""
echo "📌 Comandi disponibili:"
echo "  • python patchcore_dashboard.py  (dashboard)"  
echo "  • jupyter notebook               (tutorial)"
echo "  • deactivate                     (disattiva venv)"
echo ""

# Mantieni la shell attiva
exec "$SHELL"
