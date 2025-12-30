import chromadb
from chromadb.utils import embedding_functions

DB_PATH = "./finance_db"
COLLECTION_NAME = "finance_knowledge"
EMBED_MODEL = "all-minilm"

class KnowledgeBase:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=DB_PATH)

        self.embed_fn = embedding_functions.OllamaEmbeddingFunction(
            model_name=EMBED_MODEL,
            url="http://localhost:11434"
        )

        try:
            self.collection = self.client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embed_fn
            )
            print(f"KnowledgeBase Bağlandı: {COLLECTION_NAME}")
        except:
            print("HATA: Veritabanı bulunamadı. Önce create_vectorDB.py çalıştır!")
            self.collection = None

    def retrieve(self, query, top_k=4):
        """
        Semantic Arama İşlemi Burada Yapılır.
        Sorguyu vektöre çevirir -> En yakın chunk'ları bulur.
        """
        if not self.collection:
            return "Veritabanı yok."

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )

            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]

            if not docs:
                return "Dökümanlarda ilgili bilgi bulunamadı."

            output = []
            for i, (doc, meta) in enumerate(zip(docs, metas)):
                src = meta.get("source", "?")
                page = meta.get("page", "?")
                output.append(f"--- KANIT {i+1} (Kaynak: {src}, Sayfa: {page}) ---\n{doc}")

            return "\n\n".join(output)

        except Exception as e:
            return f"Arama hatası: {str(e)}"

# --- TEST ---
if __name__ == "__main__":
    kb = KnowledgeBase()
    # Semantic Arama Testi
    print(kb.retrieve("What should a defensive investor do?"))