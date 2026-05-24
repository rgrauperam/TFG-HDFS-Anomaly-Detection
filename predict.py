import torch
import pandas as pd
from model import LogTransformer

# ================= CONFIGURACIÓN =================
UMBRAL = 0.64
# Asegúrate de que el nombre del archivo coincida con el que tienes en tu carpeta
MODEL_PATH = 'Resultados_HDFS_Prova23/best_model.pth' 
VOCAB_SIZE = 62825
# =================================================

def event_to_int(e):
    """Convierte E5 a número (token) igual que en el entrenamiento"""
    try:
        if isinstance(e, str) and e.startswith('E'):
            return int(e[1:]) + 1
        return int(e) + 1
    except:
        return 1 # <UNK> Token desconocido

print("Cargando el modelo...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inicializar el modelo con los parámetros exactos de tu Prova 22
model = LogTransformer(
    vocab_size=VOCAB_SIZE, 
    embedding_dim=128, 
    num_heads=4, 
    num_layers=3, 
    dropout=0.2
).to(device)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ Modelo cargado correctamente.\n")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    print("Revisa que la ruta MODEL_PATH sea correcta (best_model.pth o mejor_modelo.pth).")
    exit()

def predecir_secuencia(secuencia_eventos):
    # Convertir las etiquetas E1, E5... a los números que entiende la IA
    numeros = [event_to_int(e) for e in secuencia_eventos]
    
    # Crear un tensor (Batch de tamaño 1)
    tensor = torch.tensor([numeros], dtype=torch.long).to(device) 
    
    with torch.no_grad():
        output = model(tensor)
        probabilidad = torch.sigmoid(output).item()
        
    estado = "🚨 ANOMALÍA" if probabilidad >= UMBRAL else "✅ NORMAL"
    
    print("-" * 50)
    print(f"Secuencia: {secuencia_eventos}")
    print(f"Probabilidad de fallo: {probabilidad*100:.2f}% -> {estado}")


# Pruebas manuales con secuencias de eventos

print("Iniciando predicciones manuales...\n")

# Prueba 1: Una secuencia típica normal
predecir_secuencia(['E1', 'E5', 'E5', 'E5', 'E22'])

# Prueba 2: Error fatal (E427)
predecir_secuencia(['E1', 'E5', 'E427', 'E5'])

# Prueba 3: ¡Borrador inesperado!
predecir_secuencia(['E1', 'E1', 'E452'])

# Prueba 4: Datanode muerto
predecir_secuencia(['E1', '1262'])

# Prueba 5: ¡Replica fallida!
predecir_secuencia(['E1', 'E1', 'E10'])

# Prueba 6: ¡Anomalia difícil de detectar!
predecir_secuencia(['E1', 'E3', 'E5', 'E22'])