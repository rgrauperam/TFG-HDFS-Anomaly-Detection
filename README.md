# TFG - HDFS Anomaly Detection

Aquest repositori conté la implementació principal del sistema de detecció d'anomalies en logs desenvolupat en el Treball de Fi de Grau.

El sistema utilitza un model basat en arquitectura Transformer per analitzar seqüències de logs del dataset HDFS. L'objectiu és classificar cada seqüència com a normal o anòmala a partir dels esdeveniments associats a cada `BlockId`.

---

## Descripció general

El projecte implementa un pipeline complet de detecció d'anomalies amb les fases següents:

1. Lectura dels logs originals del dataset HDFS.
2. Preprocessament dels logs i extracció dels `BlockId`.
3. Conversió dels esdeveniments en identificadors numèrics.
4. Construcció de seqüències de logs.
5. Entrenament d'un model Transformer.
6. Validació del model i selecció del millor llindar.
7. Avaluació final sobre el conjunt de test.
8. Generació de gràfiques, mètriques i matriu de confusió.

El model final combina:

- Transformer Encoder
- Attention Pooling
- Mean Pooling
- Classificador binari

Aquesta arquitectura permet analitzar tant els esdeveniments més rellevants de la seqüència com el comportament global del bloc.

---

## Dataset necessari

Per executar el projecte és necessari descarregar prèviament el dataset HDFS.

El dataset original no s'inclou en aquest repositori perquè el fitxer de logs és massa gran. Per aquest motiu, cada usuari ha de descarregar-lo manualment des del repositori públic Loghub. Cal descarregar la versió **HDFS_v1**, ja que és la que inclou el fitxer de logs i el fitxer d'etiquetes necessari per entrenar el model.

```text
https://github.com/logpai/loghub
```

Un cop descarregat, cal col·locar els fitxers dins de la carpeta `data/raw/` amb aquesta estructura:

```text
data/
├── raw/
│   ├── HDFS.log
│   └── anomaly_label.csv
│
└── processed/
```

És important que els fitxers tinguin exactament aquests noms:

```text
data/raw/HDFS.log
data/raw/anomaly_label.csv
```

---

## Execució del projecte

Per executar correctament el projecte, és important seguir l'ordre següent:

1. Primer s'ha d'executar `preparar_dades.py`.
2. Després s'ha d'executar `train.py`.
3. Opcionalment, es pot executar `predict.py` per provar seqüències manualment.

Aquest ordre és necessari perquè el model no pot entrenar-se directament amb el fitxer original `HDFS.log`. Primer cal transformar els logs originals en un fitxer estructurat que pugui ser llegit pel model.

### 1. Preparar les dades

```bash
python preparar_dades.py
```

Aquest script llegeix:

```text
data/raw/HDFS.log
```

i genera automàticament:

```text
data/processed/HDFS.log_structured.csv
```

### 2. Entrenar el model

```bash
python train.py
```

Aquest script utilitza el fitxer processat i les etiquetes:

```text
data/processed/HDFS.log_structured.csv
data/raw/anomaly_label.csv
```

Durant l'entrenament es genera una carpeta de resultats amb el model entrenat, les gràfiques i la matriu de confusió.

### 3. Fer prediccions

Després d'entrenar el model, es pot executar:

```bash
python predict.py
```

Aquest fitxer permet introduir seqüències pròpies d'esdeveniments i comprovar si el model les classifica com a normals o anòmales.

L'ordre resumit és:

```bash
python preparar_dades.py
python train.py
python predict.py
```

---

## Estructura del repositori

L'estructura principal del projecte és la següent:

```text
TFG-HDFS-Anomaly-Detection/
│
├── data/
│   ├── raw/
│   │   └── anomaly_label.csv
│   │
│   └── processed/
│
├── data_loader.py
├── model.py
├── preparar_dades.py
├── predict.py
├── train.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Instal·lació de dependències

Les llibreries necessàries es troben al fitxer `requirements.txt`.

Per instal·lar-les:

```bash
pip install -r requirements.txt
```
