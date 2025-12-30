import pdfplumber
import re
import os
import shutil
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- AYARLAR ---
DB_PATH = "./finance_db"
COLLECTION_NAME = "finance_knowledge"
EMBED_MODEL = "all-minilm"
BATCH_SIZE = 25

CHUNK_SIZE = 400
CHUNK_OVERLAP = 150

# Kitabının adı
PDF_FILES = ["finance1.pdf"] 

def extract_text_excluding_tables(page):
    """
    Sayfadaki tabloları bulur ve o alanlar dışındaki metni çeker.
    """
    # 1. Tabloları tespit et
    tables = page.find_tables()
    
    # Eğer tablo yoksa direkt tüm metni ver
    if not tables:
        return page.extract_text() or ""
    
    # 2. Tabloların koordinatlarını (Bounding Box) al
    # bbox formatı: (x0, top, x1, bottom)
    table_bboxes = [table.bbox for table in tables]
    
    # 3. Filtreleme Fonksiyonu: Bir nesne tablonun içindeyse FALSE döner (yani atılır)
    def not_inside_tables(obj):
        # pdfplumber her harf/kelime için bu fonksiyonu çalıştırır
        obj_x0 = obj["x0"]
        obj_top = obj["top"]
        obj_x1 = obj["x1"]
        obj_bottom = obj["bottom"]
        
        for (tx0, ttop, tx1, tbottom) in table_bboxes:
            # Eğer obje tablonun sınırları içindeyse alma
            if (obj_x0 >= tx0 and obj_x1 <= tx1 and 
                obj_top >= ttop and obj_bottom <= tbottom):
                return False
        return True

    # 4. Sayfayı filtrele ve sadece tablo dışı metni al
    try:
        filtered_page = page.filter(not_inside_tables)
        return filtered_page.extract_text() or ""
    except Exception as e:
        # Filtreleme hatası olursa (nadir) normal metni döndür
        print(f"⚠️ Tablo filtreleme hatası: {e}")
        return page.extract_text() or ""

def repair_reversed_text(text):
    """
    Ters (ayna) metinleri düzeltir.
    Örn: 'emocnI' -> 'Income'
    """
    if not text: return ""
    
    lines = text.split('\n')
    repaired_lines = []
    
    for line in lines:
        # Kontrol kelimeleri (ters halleri)
        # eht=the, dna=and, fo=of, si=is
        if re.search(r"\b(eht|dna|fo|si)\b", line):
            repaired_lines.append(line[::-1]) # Satırı ters çevir
        else:
            repaired_lines.append(line)
            
    return "\n".join(repaired_lines)

def clean_text(text):
    if not text: return ""
    
    # 1. Önce ters metin kontrolü yap
    text = repair_reversed_text(text)

    # 2. Gereksiz sayfaları atla
    if re.search(r"(?i)table of contents|copyright|index|acknowledg", text):
        return ""
    if len(text.split()) < 30: 
        return ""
    
    # 3. Temizlik
    text = re.sub(r"-+\s*PAGE\s+\d+\s*-+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def process_pdfs(pdf_paths):
    documents = []
    print(f"📚 İşleniyor: {pdf_paths}")

    for path in pdf_paths:
        if not os.path.exists(path):
            print(f"❌ PDF YOK: {path}")
            continue

        print(f"📖 Okunuyor (Tablolar Hariç): {path}")
        with pdfplumber.open(path) as pdf:
            for page_id, page in enumerate(pdf.pages):
                # ÖZEL FONKSİYONUMUZU BURADA ÇAĞIRIYORUZ
                raw = extract_text_excluding_tables(page)
                
                cleaned = clean_text(raw)
                
                if cleaned:
                    documents.append({
                        "text": cleaned,
                        "source": os.path.basename(path),
                        "page": page_id + 1
                    })
    print(f"📄 Toplam {len(documents)} sayfa metin çıkarıldı.")
    return documents

def main():
    # 1. TEMİZLİK
    if os.path.exists(DB_PATH):
        try:
            shutil.rmtree(DB_PATH)
            print("🗑️  Eski veritabanı silindi.")
        except: pass

    # 2. BAĞLANTI
    client = chromadb.PersistentClient(path=DB_PATH)
    print(f"🔌 Embedding Modeli: {EMBED_MODEL}")
    
    embed_fn = embedding_functions.OllamaEmbeddingFunction(
        model_name=EMBED_MODEL,
        url="http://localhost:11434"
    )

    # 3. İŞLEME
    raw_docs = process_pdfs(PDF_FILES)
    if not raw_docs:
        print("⚠️ Veri yok.")
        return

    # 4. PARÇALAMA
    print(f"✂️  Parçalanıyor (Chunk Size: {CHUNK_SIZE})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks, metadatas, ids = [], [], []
    counter = 0

    for doc in raw_docs:
        doc_chunks = splitter.split_text(doc["text"])
        for chunk in doc_chunks:
            chunks.append(chunk)
            metadatas.append({
                "source": doc["source"],
                "page": doc["page"]
            })
            ids.append(f"id_{counter}")
            counter += 1

    print(f"📦 Toplam {len(chunks)} parça.")

    # 5. KAYIT
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn
    )

    print("💾 Kaydediliyor...")
    total_batches = (len(chunks) // BATCH_SIZE) + 1
    
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[i:i+BATCH_SIZE]
        batch_metas = metadatas[i:i+BATCH_SIZE]
        batch_ids = ids[i:i+BATCH_SIZE]
        collection.add(documents=batch_chunks, metadatas=batch_metas, ids=batch_ids)
        print(f"   ⏳ Batch {i//BATCH_SIZE + 1}/{total_batches} bitti.")

    print("\n✅ Veritabanı (Tablosuz) HAZIR!")

if __name__ == "__main__":
    main()