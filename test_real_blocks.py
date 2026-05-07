import torch
import pandas as pd
import random
import os
from model import LogTransformer

# ================= CONFIGURACIÓN =================
UMBRAL = 0.64  # El umbral perfecto que sacaste en la Prova 22
MODEL_PATH = 'Resultados_HDFS_Prova23/best_model.pth' 
VOCAB_SIZE = 62825  # El que nos dio el train.py
# =================================================

def event_to_int(e):
    """Convierte E5 a número igual que en el entrenamiento"""
    try:
        if isinstance(e, str) and e.startswith('E'):
            return int(e[1:]) + 1
        return int(e) + 1
    except:
        return 1

def main():
    print("🔍 Cargando datos reales para el examen sorpresa al modelo...")
    
    # 1. Cargar las etiquetas que me acabas de pasar
    labels_df = pd.read_csv('data/raw/anomaly_label.csv')
    
    # Elegir 2 bloques anómalos y 2 normales (fijamos la semilla para que salgan unos chulos)
    anomalous_blocks = labels_df[labels_df['Label'] == 'Anomaly']['BlockId'].sample(5, random_state=42).tolist()
    normal_blocks = labels_df[labels_df['Label'] == 'Normal']['BlockId'].sample(5, random_state=42).tolist()
    test_blocks = anomalous_blocks + normal_blocks

    # 2. Buscar sus logs en el archivo estructurado
    print("📂 Buscando los eventos de estos bloques en HDFS.log_structured.csv...")
    logs_df = pd.read_csv('data/processed/HDFS.log_structured.csv')
    test_logs = logs_df[logs_df['BlockId'].isin(test_blocks)]
    secuencias = test_logs.groupby('BlockId')['EventId'].apply(list).to_dict()

    # 3. Cargar tu Modelo Ganador
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LogTransformer(
        vocab_size=VOCAB_SIZE, 
        embedding_dim=128, 
        num_heads=4, 
        num_layers=3, 
        dropout=0.2
    ).to(device)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print("\n🚀 ¡INICIANDO PRUEBA CON LOGS REALES!\n" + "="*60)

    for block_id in test_blocks:
        if block_id not in secuencias:
            print(f"⚠️ No se encontraron logs para {block_id} (quizás se cortaron por el límite de RAM)")
            continue
            
        eventos_crudos = secuencias[block_id]
        numeros = [event_to_int(e) for e in eventos_crudos]
        
        # ¿Cuál es la verdad absoluta?
        real_label = "🚨 ANOMALÍA" if block_id in anomalous_blocks else "✅ NORMAL"
        
        # ¿Qué opina la Inteligencia Artificial?
        tensor = torch.tensor([numeros], dtype=torch.long).to(device)
        with torch.no_grad():
            output = model(tensor)
            prob = torch.sigmoid(output).item()
            
        pred_label = "🚨 ANOMALÍA" if prob >= UMBRAL else "✅ NORMAL"
        
        # Mostrar el resultado
        print(f"📦 Bloque ID: {block_id}")
        
        # Mostramos los primeros 10 eventos y los últimos 5 para no llenar toda la pantalla
        if len(eventos_crudos) > 15:
            seq_str = f"{eventos_crudos[:10]} ... {eventos_crudos[-5:]}"
        else:
            seq_str = str(eventos_crudos)
            
        print(f"📋 Secuencia: {seq_str} (Total: {len(eventos_crudos)} eventos)")
        print(f"🏷️ Realidad:      {real_label}")
        print(f"🤖 Predicción IA: {pred_label} (Confianza: {prob*100:.2f}%)")
        
        if real_label == pred_label:
            print("✨ ¡EL MODELO HA ACERTADO!")
        else:
            print("❌ El modelo ha fallado.")
        print("-" * 60)

if __name__ == '__main__':
    main()