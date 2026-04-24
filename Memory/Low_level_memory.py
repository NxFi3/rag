from Utils.logger import get_logger
import faiss
import numpy as np
import uuid
import os
import pickle

logger = get_logger("LOWMEM")
logger.info("LOW_LEVEL Memory Started")


DIM = 384  
INDEX_FILE = "memory_index.faiss"
DATA_FILE = "memory_data.pkl"

try:
    if os.path.exists(INDEX_FILE) and os.path.exists(DATA_FILE):
        
        logger.info("Loading existing index...")
        index = faiss.read_index(INDEX_FILE)
        with open(DATA_FILE, "rb") as f:
            memory_data = pickle.load(f)
        logger.info(f"Loaded {index.ntotal} memories")
    else:
        
        logger.info("Creating new memory index...")
        index = faiss.IndexFlatIP(DIM)  # Inner Product (Cosine Similarity)
        memory_data = {}  
        logger.info(f"New index created with dimension {DIM}")
        
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise



def add_to_memory(data: np.ndarray, memory_id: uuid.UUID, metadata: dict = None):
    """
    Save To FAISS index
    
    Args:
        data: embedding shape (1,dim)
        memory_id:UUID
        metadata: Information Non Embedding
    """
    try:
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        
        id_int = memory_id.int & 0x7FFFFFFFFFFFFFFF  
        
        
        index.add_with_ids(data, np.array([id_int], dtype=np.int64))
        
        
        if metadata:
            memory_data[id_int] = metadata
        
        logger.debug(f"Memory saved with ID: {memory_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving memory: {e}")
        return False



def search_memory(query_embedding: np.ndarray, k: int = 5):
    try:
        if index.ntotal == 0:
            logger.warning("Index is empty")
            return [], []
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        scores, indices = index.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx in memory_data:
                results.append({
                    'id': idx,
                    'score': float(scores[0][i]),
                    'metadata': memory_data[idx]
                })
        
        return results
        
    except Exception as e:
        logger.error(f"Error searching memory: {e}")
        return [], []



def persist_memory():
    """
    save on disk
    """
    try:
        faiss.write_index(index, INDEX_FILE)
        with open(DATA_FILE, "wb") as f:
            pickle.dump(memory_data, f)
        logger.info(f"Memory persisted: {index.ntotal} items saved")
        return True
    except Exception as e:
        logger.error(f"Error persisting memory: {e}")
        return False


def load_persisted_memory():
    """
    load from disk
    """
    global index, memory_data
    
    try:
        if os.path.exists(INDEX_FILE) and os.path.exists(DATA_FILE):
            index = faiss.read_index(INDEX_FILE)
            with open(DATA_FILE, "rb") as f:
                memory_data = pickle.load(f)
            logger.info(f"Loaded {index.ntotal} memories from disk")
            return True
        else:
            logger.warning("No persisted memory found")
            return False
    except Exception as e:
        logger.error(f"Error loading persisted memory: {e}")
        return False


# ========== مثال استفاده ==========
if __name__ == "__main__":
    test_emb = np.random.random((1, DIM)).astype('float32')
    faiss.normalize_L2(test_emb)
    
    metadata = {
        "text": "Python is a high-level language",
        "importance": 0.8,
        "timestamp": time.time()
    }
    
    memory_id = uuid.uuid4()
    add_to_memory(test_emb, memory_id, metadata)
    
    # ذخیره دائمی
    persist_memory()
    
    # جستجو
    results, scores = search_memory(test_emb, k=3)
    for r in results:
        print(f"Score: {r['score']:.4f} - {r['metadata']['text']}")
    
load_persisted_memory()