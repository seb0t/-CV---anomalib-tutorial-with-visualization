# Tutti gli import necessari per il notebook
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import subprocess
import os
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
import faiss

# Qui andranno inserite le funzioni estratte dal notebook centrale

def diagnose_gpu_cuda():
    """
    Esegue controlli diagnostici dettagliati per GPU e CUDA.
    """
    print("=== DIAGNOSI GPU/CUDA ===")
    print(f"Versione Python: {sys.version}")
    print(f"Versione PyTorch: {torch.__version__}")

    # 1. Verifica CUDA disponibilità
    print(f"\ntorch.cuda.is_available(): {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    else:
        print("CUDA non disponibile")

    # 2. Verifica versione CUDA compilata in PyTorch
    print(f"\nCUDA version compilata in PyTorch: {torch.version.cuda}")

    # 3. Verifica se PyTorch è stato compilato con supporto CUDA
    print(f"PyTorch compilato con CUDA: {torch.cuda.is_available()}")

    # 4. Verifica driver NVIDIA (se su Windows)
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            print(f"\nnvidia-smi output:")
            print(result.stdout)
        else:
            print(f"\nnvidia-smi non trovato o errore: {result.stderr}")
    except Exception as e:
        print(f"\nErrore nell'eseguire nvidia-smi: {e}")

    # 5. Verifica variabili d'ambiente CUDA
    cuda_path = os.environ.get('CUDA_PATH', 'Non trovato')
    cuda_home = os.environ.get('CUDA_HOME', 'Non trovato')
    print(f"\nCUDA_PATH: {cuda_path}")
    print(f"CUDA_HOME: {cuda_home}")

    print("\n=== FINE DIAGNOSI ===")

# AUTOMATIZZAZIONE: Reinstallazione PyTorch con supporto CUDA

def reinstall_pytorch_with_cuda():
    """
    Funzione per reinstallare PyTorch con supporto CUDA.
    """
    try:
        print("🔄 Disinstallazione di PyTorch CPU-only...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', 'torch', 'torchvision', 'torchaudio', '-y'])
        
        print("📦 Installazione di PyTorch con supporto CUDA 12.1...")
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', 
            'torch', 'torchvision', 'torchaudio',
            '--index-url', 'https://download.pytorch.org/whl/cu121'
        ])
        
        print("✅ Installazione completata!")
        print("⚠️  IMPORTANTE: Riavvia il kernel del notebook per applicare le modifiche!")
        return True
    except Exception as e:
        print(f"❌ Errore durante l'installazione: {e}")
        return False
    
# VERIFICA FINALE: GPU ora dovrebbe funzionare!

def check_cuda_availability():
    """
    Verifica la disponibilità di CUDA e stampa informazioni dettagliate sulla GPU.
    """
    print("🔍 VERIFICA FINALE DOPO REINSTALLAZIONE:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"✅ SUCCESS! GPU rilevata: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Test rapido di allocazione GPU
        device = torch.device("cuda")
        test_tensor = torch.randn(1000, 1000).to(device)
        print(f"✅ Test allocazione GPU: OK ({test_tensor.device})")
        del test_tensor
        torch.cuda.empty_cache()
    else:
        print("❌ CUDA ancora non disponibile. Assicurati di aver riavviato il kernel!")

def get_device():
    """
    Determina il device corretto per l'esecuzione: CUDA, MPS o CPU.

    Returns:
        torch.device: Il device selezionato.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: {device} (Apple Silicon MPS)")
    else:
        device = torch.device("cpu")
        print("Using device: cpu")
    return device