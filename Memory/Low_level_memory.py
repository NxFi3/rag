from Utils.logger import get_logger
from core.generator import GeneratorManager
import faiss
import numpy as np
import uuid
import os
import pickle
logger = get_logger("LOWMEM")
logger.info("LOW_LEVEL Memory Started")
INDEX_FILE = "memory_index.faiss"
DATA_FILE = "memory_data.pkl"


DIM = 768
index = None
memory_data = {}

def EmbeddingDIMLOW(dim: int = 768):
    global DIM, index, memory_data
    DIM = dim
    
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
            index = faiss.IndexFlatIP(DIM)  
            memory_data = {}
            logger.info(f"New index created with dimension {DIM}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
def Save_search(Query:np.ndarray):
    try:
       
        if Query is None :
            return False  
        
        emb_value = Query.reshape(1, -1).astype(np.float32)
        D, I = index.search(emb_value, k=1)
        
       
        if I[0][0] != -1 and D[0][0] >= 0.85:  
            logger.info(f"Duplicate detected for (score: {D[0][0]:.3f})")
            return False  
        else:
            return True  
            
    except Exception as e:
        logger.error(f"Error in Save_Search: {e}")
        return True 

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
        
        if Save_search(data):
            id_int = memory_id.int & 0x7FFFFFFFFFFFFFFF  
            
            
            index.add_with_ids(data, np.array([id_int], dtype=np.int64))
            
            
            if metadata:
                memory_data[id_int] = metadata
            
            logger.debug(f"Memory saved with ID: {memory_id}")
            return True
        else:
            return False  
    except Exception as e:
            logger.error(f"Error saving memory: {e}")
            return False



def search_memory(Query: str,gen, k: int = 5):
    try:
        if index.ntotal == 0:
            logger.warning("Index is empty")
            return [], []
        query_embedding = gen.Encode(Query)
        if query_embedding is None:
            logger.error("Encoding failed")
            return [], []
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
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

