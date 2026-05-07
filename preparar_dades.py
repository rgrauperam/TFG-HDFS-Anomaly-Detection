import csv
import re
import os

# ================= CONFIGURACIÓ RAM SEGURA =================
RAW_LOG_FILE = 'data/raw/HDFS.log'
OUTPUT_FILE = 'data/processed/HDFS.log_structured.csv'

# ⚠️ LIMIT: Només llegirem els primers 2 milions de línies.
# Això ocupa molt poca RAM i és suficient per entrenar.
# Si en vols més o menys, canvia aquest número.
MAX_LINES = 15_000_000 
# ===========================================================

def clean_log(log_line):
    """ Neteja la línia per extreure BlockId i Template """
    # 1. Extreure BlockId
    blk_match = re.search(r'(blk_[-0-9]+)', log_line)
    block_id = blk_match.group(1) if blk_match else "unknown"
    
    # 2. Netejar per fer el Template
    text = re.sub(r'blk_[-0-9]+', '<BLK>', log_line)
    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?', '<IP>', text)
    text = re.sub(r'(?<= )\d+(?= )', '<NUM>', text)
    
    # Treure capçaleres típiques HDFS
    split_parts = text.split(':', 1)
    content = split_parts[1].strip() if len(split_parts) > 1 else text.strip()
        
    return block_id, content

def main():
    print(f"🚀 INICIANT MODE BAIX CONSUM DE RAM")
    print(f"📂 Llegint: {RAW_LOG_FILE}")
    print(f"🛑 Límit: {MAX_LINES} línies")

    if not os.path.exists(RAW_LOG_FILE):
        print(f"❌ ERROR: No trobo {RAW_LOG_FILE}")
        return

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    templates_map = {} # Diccionari petit per guardar IDs d'events (ocupa molt poc)
    
    # Obrim els fitxers en mode flux (stream)
    with open(RAW_LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f_in, \
         open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f_out:
        
        writer = csv.writer(f_out)
        # Escrivim la capçalera manualment
        writer.writerow(['LineId', 'BlockId', 'Content', 'EventId', 'EventTemplate'])
        
        count = 0
        for line in f_in:
            line = line.strip()
            if not line: continue
            
            # Processar
            block_id, template_text = clean_log(line)
            
            # Assignar EventId
            if template_text not in templates_map:
                templates_map[template_text] = f"E{len(templates_map) + 1}"
            event_id = templates_map[template_text]
            
            # Escriure directament al disc (NO es guarda a la RAM)
            writer.writerow([count + 1, block_id, line, event_id, template_text])
            
            count += 1
            if count % 100000 == 0:
                print(f"   ... processades {count} línies ...", end='\r')
                
            if count >= MAX_LINES:
                print(f"\n⚠️ Límit de {MAX_LINES} línies assolit. Parant.")
                break

    print(f"\n✅ FET! Arxiu guardat a: {OUTPUT_FILE}")
    print(f"📊 Events únics detectats: {len(templates_map)}")
    print("👉 El teu ordinador hauria d'estar fresc com una rosa 🌹")

if __name__ == '__main__':
    main()