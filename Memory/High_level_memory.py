from Utils.logger import get_logger
import faiss
from Memory.MemoryItem import MemoryItem
from core.generator import GeneratorManager
from core.Parser import ParserManager
import numpy as np
import pickle
import os
from core.prompts import Preprocessing_prompt_fixed

logger = get_logger("HIGH_level Memory started")

DIM = 384 #default 

LTM_index = {
    'identity': faiss.IndexIDMap(faiss.IndexFlatIP(DIM)),
    "semantic": faiss.IndexIDMap(faiss.IndexFlatIP(DIM)),
    "episodic": faiss.IndexIDMap(faiss.IndexFlatIP(DIM)),
    "procedural": faiss.IndexIDMap(faiss.IndexFlatIP(DIM)),
    "emotional": faiss.IndexIDMap(faiss.IndexFlatIP(DIM)),
    "code": faiss.IndexIDMap(faiss.IndexFlatIP(DIM))
}
LTM_text = {}

generator = GeneratorManager('llama3.1:8B', "D:/SentenceTransformer Model")
parser = ParserManager()

def save_memory_to_disk():
    """Save On Disk """
    try:
        for mem_type, index in LTM_index.items():
            if index.ntotal > 0:
                faiss.write_index(index, f"{mem_type}_index.faiss")
        with open("LTM_text.pkl", "wb") as f:
            pickle.dump(LTM_text, f)
        logger.info(f"Saved {len(LTM_text)} memories")
        return True
    except Exception as e:
        logger.error(f"Save failed: {e}")
        return False

def load_memory_from_disk():
    """Load memory from the disk files name most be LTM_text.pkl"""
    global LTM_index, LTM_text
    try:
  
        if os.path.exists("LTM_text.pkl"):
            with open("LTM_text.pkl", "rb") as f:
                LTM_text = pickle.load(f)
            logger.info(f"Loaded {len(LTM_text)} memories")
        
        
        for mem_type in LTM_index.keys():
            index_file = f"{mem_type}_index.faiss"
            if os.path.exists(index_file):
                LTM_index[mem_type] = faiss.read_index(index_file)
                logger.info(f"Loaded {mem_type} index")
    except Exception as e:
        logger.error(f"Load failed: {e}")


def Search_memory(query):
    search_result = []
    
    if len(query) >= 4:
        res = generator.generator(Preprocessing_prompt_fixed(query))
        extracted_query = parser.OutputManage(res)
        
        if extracted_query.get('type') == 'success':
            logger.info("Processing Memory")
            memory_types = extracted_query.get('Memory', [])
            query_values = extracted_query.get('Query', [])
            
            if memory_types and query_values:
                for mem_type in memory_types:
                    if mem_type not in LTM_index:
                        continue
                    
                    for value in query_values:
                        try:
                            emb_value = generator.Encode(value)
                            if emb_value is None:
                                continue
                            emb_value = emb_value.reshape(1, -1).astype(np.float32)
                            faiss.normalize_L2(emb_value)
                            
                            D, I = LTM_index[mem_type].search(emb_value, k=3)
                            
                            for idx, ids in enumerate(I[0]):
                                if ids != -1 and ids in LTM_text:
                                    item = LTM_text[ids]
                                    search_result.append({
                                        'ID': item.ID,
                                        'text': item.Value,
                                        'score': float(D[0][idx]) if len(D[0]) > idx else 0
                                    })
                        except Exception as e:
                            logger.error(f"Error encoding: {e}")
                            continue
    
    seen = set()
    unique = []
    for r in search_result:
        if r['ID'] not in seen:
            seen.add(r['ID'])
            unique.append(r)
    
    sorted_results = sorted(unique, key=lambda x: x['score'], reverse=True)
    
    if sorted_results:
        best = sorted_results[0]
        LTM_text[best['ID']].update()
        return [best['text']]
    else:
        return []

def Save_memory(STM, User_inputs, Model_outputs=None):
    STM_context = f"User:{User_inputs}\nAssistant:{Model_outputs}"
    STM.append(STM_context)
    
    if len(User_inputs) >= 6:
        if Model_outputs is not None:
            query = User_inputs + '\n' + Model_outputs
        else:
            query = User_inputs
        res = generator.generator(Preprocessing_prompt_fixed(query))
        Extracted = parser.OutputManage(res)
        
        logger.info("saving Memory")
        if Extracted.get('type') == 'success':
            Memory_types = Extracted.get("Memory", [])
            Values = Extracted.get("Query", [])
            
            if Memory_types and Values:
                for Mem_type in Memory_types:
                    if Mem_type not in LTM_index:
                        continue
                    for Value in Values:
                        if Value and len(Value.strip()) > 2:
                            try:
                                emb = generator.Encode(Value)
                                if emb is None:
                                    continue
                                emb = emb.reshape(1, -1).astype(np.float32)
                                faiss.normalize_L2(emb)
                                importance = 0.5
                                
                                item = MemoryItem([Mem_type], Value, emb[0], importance)
                                LTM_text[item.ID] = item
                                
                                LTM_index[Mem_type].add_with_ids(emb, np.array([item.ID], dtype=np.int64))
                                print(f"saved {Mem_type}: {Value[:50]}...")
                            except Exception as e:
                                logger.error(f"Error saving: {e}")
                                continue
    
    return STM

load_memory_from_disk()
logger.info("HIGH_level Memory ready")