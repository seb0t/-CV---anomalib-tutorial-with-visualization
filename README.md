# Guida all'installazione di PyTorch (CPU/GPU)

Per garantire la massima compatibilità su Windows (con o senza GPU) e su Mac, il file `requirements.txt` specifica solo la versione minima di torch/torchvision. Tuttavia, per sfruttare la GPU su Windows/Linux, è necessario installare manualmente la versione corretta di PyTorch.

## 1. Installazione base (CPU-only, compatibile ovunque)

```bash
pip install -r requirements.txt
```

## 2. Installazione PyTorch con supporto GPU (CUDA) su Windows/Linux

Se hai una scheda NVIDIA e vuoi usare la GPU:

1. Vai su https://pytorch.org/get-started/locally/ e scegli la configurazione adatta alla tua versione di CUDA e Python.
2. Oppure usa direttamente (per CUDA 12.x):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

3. Dopo l'installazione, verifica che la GPU sia visibile:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

## 3. Mac (CPU o Apple Silicon/MPS)

Su Mac Intel: la versione CPU-only va bene.

Su Mac M1/M2: puoi usare la versione base oppure seguire le istruzioni ufficiali per abilitare il backend MPS:
https://pytorch.org/docs/stable/notes/mps.html

## 4. Note
- Se aggiorni la versione di torch, assicurati che sia compatibile con anomalib e torchvision.
- Se usi Jupyter/Notebook, riavvia il kernel dopo aver cambiato la versione di torch.

---
# 🔬 PatchCore Educational Platform
### *Comprensione pratica degli algoritmi di Anomaly Detection*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In_Development-yellow.svg)](README.md)

---

## 📖 **Panoramica del Progetto**

Questa piattaforma educativa fornisce una **comprensione approfondita e pratica** degli algoritmi di anomaly detection, con focus particolare su **PatchCore** e altri metodi state-of-the-art. Il progetto combina **teoria, implementazione pratica e visualizzazioni interattive** per rendere accessibili concetti complessi del machine learning.

### 🎯 **Obiettivi Educativi**

- **Comprensione teorica**: Spiegare il funzionamento interno di PatchCore e algoritmi correlati
- **Implementazione pratica**: Notebook didattici step-by-step con esempi semplici
- **Visualizzazione interattiva**: Dashboard web per esplorare l'effetto dei parametri
- **Preparazione avanzata**: Base per algoritmi più complessi di anomaly detection

---

## 📁 **Struttura del Progetto**

```
patchcore-educational/
│
├── 📓 patchcore_anomalib_tutorial.ipynb    # Tutorial completo con anomalib
├── 🖥️  patchcore_dashboard.py              # Dashboard interattiva standalone  
├── 📋 requirements.txt                     # Dipendenze del progetto
├── 🚀 run_dashboard.sh                     # Script di avvio rapido
├── 📖 README.md                            # Documentazione principale
└── 📖 README_dashboard.md                  # Documentazione dashboard specifica
```

---

## 🚀 **Quick Start**

### **Metodo 1: Setup Automatico (Raccomandato)**
```bash
# Clona il repository
git clone <repository-url>
cd patchcore-educational

# Avvio automatico con virtual environment
./setup_and_run.sh
```
🌐 **Dashboard disponibile su**: http://localhost:8051

### **Metodo 2: Setup Manuale**
```bash
# Crea virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Installa dipendenze
pip install -r requirements.txt

# Avvia dashboard
python patchcore_dashboard.py
```

### **Metodo 3: Jupyter Notebook**
```bash
# Attiva virtual environment (se creato)
source .venv/bin/activate

# Avvia Jupyter
jupyter notebook patchcore_anomalib_tutorial.ipynb
```

---

## 🎛️ **Componenti del Progetto**

### 📊 **Dashboard Interattiva** (`patchcore_dashboard.py`)

**Caratteristiche principali**:
- ✅ **Visualizzazione real-time** delle anomaly maps
- ✅ **Controlli interattivi** per patch size e stride  
- ✅ **Heatmap con scala fissa** (0.0 - 8.0) per confronti
- ✅ **Immagini cliccabili** per selezione test cases
- ✅ **Metriche dinamiche** con statistiche dettagliate
- ✅ **Tema scuro professionale** ottimizzato per l'analisi

**Funzionalità educative**:
- **Patch-based analysis**: Comprendi come PatchCore suddivide le immagini
- **Parameter tuning**: Vedi l'effetto di diversi parametri in tempo reale
- **Anomaly scoring**: Esplora come vengono calcolati gli score di anomalia
- **Visual comparison**: Confronta diverse immagini con scala consistente

### 📓 **Tutorial Jupyter** (`patchcore_anomalib_tutorial.ipynb`)

**Contenuto didattico**:
- 🎓 **Introduzione teorica** a PatchCore e anomaly detection
- 🔬 **Implementazione step-by-step** con immagini semplici (10x10 pixel)
- 📊 **Visualizzazioni dettagliate** di ogni fase dell'algoritmo
- 🛠️ **Esempi pratici** con dataset di test
- 📈 **Analisi dei risultati** e interpretazione delle metriche

---

## 🧠 **Concetti Teorici Coperti**

### **PatchCore Algorithm**
- **Memory Bank**: Come PatchCore costruisce la memoria delle features
- **Patch Embedding**: Estrazione di feature da patch sovrapposte
- **Anomaly Scoring**: Calcolo delle distanze nel feature space
- **Threshold Selection**: Determinazione automatica delle soglie

### **Visualizzazioni Educative**
- **Anomaly Heatmaps**: Mappe di calore delle anomalie rilevate
- **Score Distributions**: Distribuzione degli score di normalità/anomalia
- **Feature Spaces**: Rappresentazione delle embeddings nel spazio multidimensionale
- **Patch Analysis**: Analisi dettagliata delle patch più anomale

---

## 📦 **Dipendenze**

### **Core Requirements**
```python
dash>=3.2.0          # Dashboard web interattiva
plotly>=5.17.0        # Visualizzazioni avanzate
numpy>=1.24.0         # Calcoli numerici
Pillow>=10.0.0        # Elaborazione immagini
```

### **Jupyter & Analysis**
```python
jupyter>=1.0.0        # Notebook interattivi
matplotlib>=3.7.0     # Grafici statici
seaborn>=0.12.0       # Visualizzazioni statistiche
```

### **Machine Learning**
```python
torch>=2.0.0          # Framework deep learning
torchvision>=0.15.0   # Visione computazionale
scikit-learn>=1.3.0   # Algoritmi ML classici
```

### **Advanced (Optional)**
```python
anomalib[full]>=1.0.0    # Libreria Intel per anomaly detection
opencv-python>=4.8.0     # Computer vision avanzata
albumentations>=1.3.0    # Data augmentation
```

---

## 🎯 **Roadmap Educativa**

### **✅ Fase 1: Fondamenti (Completata)**
- [x] Dashboard PatchCore interattiva
- [x] Tutorial base con anomalib
- [x] Visualizzazioni core (heatmaps, distributions)
- [x] Sistema di patch analysis

### **🔄 Fase 2: Approfondimenti (In Sviluppo)**
- [ ] Algoritmi di comparison (VAE, AutoEncoder)
- [ ] Advanced feature extraction methods
- [ ] Real-world dataset integration
- [ ] Performance benchmarking tools

### **📅 Fase 3: Produzione (Pianificata)**
- [ ] Multi-algorithm comparison dashboard
- [ ] Custom dataset support
- [ ] Advanced visualization techniques
- [ ] Export/Import functionality

---

## 🛠️ **Utilizzo Avanzato**

### **Personalizzazione Dashboard**
```python
# Modifica parametri globali in patchcore_dashboard.py
ANOMALY_SCALE_MIN = 0.0    # Scala minima colormap
ANOMALY_SCALE_MAX = 8.0    # Scala massima colormap
DEFAULT_PATCH_SIZE = 3     # Dimensione patch di default
DEFAULT_STRIDE = 1         # Stride di default
```

### **Estensione Algoritmi**
Il progetto è strutturato per facilitare l'aggiunta di nuovi algoritmi:
1. Implementa la funzione di scoring in `calculate_anomaly_map()`
2. Aggiungi controlli UI per i nuovi parametri
3. Estendi le visualizzazioni per le nuove metriche

---

## 🎓 **Materiale Didattico**

### **Per Studenti**
- Tutorial step-by-step con spiegazioni dettagliate
- Esercizi interattivi con feedback immediato
- Visualizzazioni intuitive dei concetti complessi

### **Per Educatori**
- Materiale pronto per lezioni universitarie
- Esempi progressivi da base ad avanzato
- Strumenti di valutazione integrati

### **Per Ricercatori**
- Implementazioni reference di algoritmi
- Benchmarking tools e metriche
- Estensibilità per nuovi metodi

---

## 📞 **Supporto e Contributi**

### **Come Contribuire**
1. 🍴 Fork del repository
2. 🌟 Crea feature branch (`git checkout -b feature/AmazingFeature`)
3. ✅ Commit delle modifiche (`git commit -m 'Add AmazingFeature'`)
4. 📤 Push al branch (`git push origin feature/AmazingFeature`)
5. 🔀 Apri Pull Request

### **Segnalazione Bug**
- Usa GitHub Issues per segnalare problemi
- Includi dettagli sull'ambiente (OS, Python version, dependencies)
- Fornisci steps per riprodurre il problema

---

## 📜 **Licenza**

Questo progetto è distribuito sotto licenza MIT. Vedi `LICENSE` per maggiori dettagli.

---

## 🙏 **Ringraziamenti**

- **Intel Labs** per la libreria [anomalib](https://github.com/openvinotoolkit/anomalib)
- **Plotly Team** per le incredibili visualizzazioni interattive
- **Community PyTorch** per il framework di deep learning

---

<div align="center">

**⭐ Se questo progetto ti è stato utile, lascia una stella! ⭐**

*Costruito con ❤️ per rendere l'anomaly detection accessibile a tutti*

</div>
