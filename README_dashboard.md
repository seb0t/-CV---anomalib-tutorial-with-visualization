# 🔬 PatchCore Dashboard Standalone

Dashboard web interattiva per l'analisi delle anomalie usando l'algoritmo PatchCore.

## 📋 Descrizione

Questa dashboard standalone permette di esplorare interattivamente come funziona l'algoritmo PatchCore per la detection di anomalie:

- **Analisi patch-based**: Visualizza come PatchCore analizza le immagini suddividendole in patch
- **Parametri interattivi**: Modifica patch_size e stride in tempo reale  
- **Visualizzazioni multiple**: Heatmap, istogrammi, statistiche
- **Tema scuro professionale**: Interface ottimizzata per l'analisi

## 🚀 Installazione e Uso

### Prerequisiti
```bash
pip install dash plotly numpy
```

### Avvio
```bash
python patchcore_dashboard.py
```

### Accesso
Apri il browser su: **http://localhost:8050**

## 🎛️ Controlli

- **📏 Patch Size**: Dimensione delle patch quadrate (2-10 pixel)
- **🔄 Stride**: Passo di scorrimento delle patch (1-10 pixel)

## 📊 Grafici

1. **🖼️ Immagine Test**: Mostra l'immagine con anomalia
2. **🔥 Anomaly Heatmap**: Mappa delle anomalie rilevate (scala Blues)
3. **📊 Distribuzione Scores**: Istogramma degli score di anomalia
4. **📈 Statistiche**: Min, Media, Max scores e numero di patch

## 🔧 Funzionalità

### Hover Interattivo
- Passa il mouse sopra le heatmap per vedere valori precisi
- Coordinate e score dettagliati per ogni pixel

### Aggiornamenti Real-time
- Modifica i parametri e vedi gli effetti istantaneamente
- Statistiche aggiornate automaticamente

### Algoritmo PatchCore Semplificato
- Usa immagini smiley 20x20 con anomalie simulate
- Calcola distanze euclidee tra patch di test e patch normali
- Score alto = anomalia probabile

## 📁 Struttura Progetto

```
test_notebook/
├── patchcore_dashboard.py          # Dashboard standalone
├── patchcore_anomalib_tutorial.ipynb  # Notebook educativo completo
└── README_dashboard.md             # Questa guida
```

## 🎯 Origini

Questa dashboard è stata estratta dal notebook Jupyter `patchcore_anomalib_tutorial.ipynb` per:

- ✅ **Portabilità**: Funziona senza Jupyter
- ✅ **Distribuzione**: Facile da condividere
- ✅ **Performance**: Ottimizzata per uso standalone
- ✅ **Manutenibilità**: Codice separato e pulito

## 🛠️ Sviluppo

### Personalizzazione
- Modifica `create_sample_data()` per usare i tuoi dati
- Cambia `calculate_anomaly_map()` per algoritmi diversi  
- Personalizza layout e stili in `app.layout`

### Estensioni Possibili
- Caricamento immagini custom
- Salvataggio risultati
- Export dei grafici
- Parametri algoritmo avanzati

## 📝 Note Tecniche

- **Framework**: Dash (Flask + React)
- **Grafici**: Plotly
- **Calcoli**: NumPy
- **Porta default**: 8050
- **Tema**: Dark mode professionale

## 🤝 Contributi

Il codice è nato dal tutorial educativo PatchCore e può essere esteso per:
- Dataset reali
- Algoritmi più complessi  
- Interfacce avanzate
- Integrazione con anomalib completo

---

*Creato dal notebook PatchCore Tutorial - Versione standalone per facilità d'uso* 🚀
