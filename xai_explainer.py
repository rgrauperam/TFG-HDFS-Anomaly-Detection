import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from model import LogTransformer
from data_loader import get_hdfs_data

# Configuración
MODEL_DIR = 'Resultados_HDFS_Prova22'
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pth')
CSV_PATH = 'data/processed/HDFS.log_structured.csv'
LABEL_PATH = 'data/raw/anomaly_label.csv'
THRESHOLD = 0.83 # El umbral que obtuvimos en la prueba 21

def main():
    print("🧠 Iniciando sistema XAI (Explainable AI)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Cargar Dataset
    dataset = get_hdfs_data(CSV_PATH, LABEL_PATH)
    
    # Calcular vocab_size igual que en train.py
    vocab_size = 0
    for seq in dataset.data:
        if len(seq) > 0:
            m = np.max(seq)
            if m > vocab_size: vocab_size = m
    vocab_size = int(vocab_size + 2)
    
    # 2. Cargar Modelo
    model = LogTransformer(
        vocab_size=vocab_size, 
        embedding_dim=128, 
        num_heads=4, 
        num_layers=3, 
        dropout=0.2
    ).to(device)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ Modelo cargado correctamente.")

    # 3. Buscar 3 ejemplos de Anomalías en el dataset
    anomaly_indices = [i for i, label in enumerate(dataset.targets) if label == 1.0]
    
    # Cogemos 3 ejemplos aleatorios o específicos
    sample_indices = anomaly_indices[:3] 
    
    os.makedirs(os.path.join(MODEL_DIR, 'XAI_Plots'), exist_ok=True)
    
    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            seq_raw = dataset.data[idx]
            
            # Convertimos a tensor
            seq_tensor = torch.tensor(seq_raw, dtype=torch.long).unsqueeze(0).to(device)
            
            # Pasamos por el modelo PIDIENDO LA ATENCIÓN
            logits, attn_probs = model(seq_tensor, return_attention=True)
            prob_anomaly = torch.sigmoid(logits).item()
            
            # Preparamos datos para la gráfica
            attn_weights = attn_probs.squeeze().cpu().numpy()
            
            # Quitamos los eventos de padding (0) si los hubiera
            valid_length = len(seq_raw)
            attn_weights = attn_weights[:valid_length]
            
            # Nombres para el eje X (E22, E5...) -> Restamos 1 porque sumaste 1 en el DataLoader
            x_labels = [f"E{event - 1}" for event in seq_raw]
            
            # 4. Dibujar la Gráfica
            plt.figure(figsize=(14, 6))
            sns.barplot(x=list(range(len(attn_weights))), y=attn_weights, color='salmon')
            plt.xticks(ticks=list(range(len(attn_weights))), labels=x_labels, rotation=90)
            
            plt.title(f"XAI: Explicación de Anomalía #{i+1} (Probabilidad: {prob_anomaly*100:.1f}%)", fontsize=16)
            plt.xlabel("Secuencia de Eventos (Logs)", fontsize=12)
            plt.ylabel("Peso de Atención (Importancia)", fontsize=12)
            plt.tight_layout()
            
            plot_path = os.path.join(MODEL_DIR, 'XAI_Plots', f'anomaly_explanation_{i+1}.png')
            plt.savefig(plot_path)
            plt.close()
            
            print(f"📊 Gráfica XAI generada: {plot_path}")
            
    print("🎉 Proceso XAI completado. Revisa la carpeta Resultados_HDFS_Prova21/XAI_Plots/")

if __name__ == '__main__':
    main()