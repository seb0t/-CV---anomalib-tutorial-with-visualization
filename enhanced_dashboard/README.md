# Enhanced PatchCore Dashboard

Questa cartella contiene una versione "enhanced" della dashboard PatchCore con UI migliorata basata su Dash + Bootstrap.

Obiettivi:
- Interfaccia moderna e reattiva con `dash-bootstrap-components`
- Stessa funzionalità di `patchcore_dashboard.py` (visualizzazione immagini, heatmap, score)
- Layout a pannelli, thumbnails cliccabili, pannello laterale con controlli avanzati

Files:
- `app.py`: entrypoint della nuova dashboard

Note:
- Esegue `pip install -r ../requirements.txt` per installare `dash-bootstrap-components` e le altre dipendenze.
- Avvia con `python app.py` dalla cartella `enhanced_dashboard`.
