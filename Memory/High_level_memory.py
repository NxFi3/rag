from Utils.logger import get_logger
import faiss
from Memory.MemoryItem import MemoryItem
from core.Parser import ParserManager
import numpy as np
import pickle
import os
from core.prompts import Save_memory_prompt , Search_memory_prompt

logger = get_logger("HIGH_level")

DIM = 768

LTM_index = {}
LTM_text = {}

def EmbeddingDIMHIGH(dim: int = 768):
    global DIM, LTM_index
    DIM = dim
    

    LTM_index.clear()  
    LTM_index.update({
        'identity': faiss.IndexIDMap(faiss.IndexFlatIP(DIM)),
        "semantic": faiss.IndexIDMap(faiss.IndexFlatIP(DIM)),
        "episodic": faiss.IndexIDMap(faiss.IndexFlatIP(DIM)),
        "procedural": faiss.IndexIDMap(faiss.IndexFlatIP(DIM)),
        "emotional": faiss.IndexIDMap(faiss.IndexFlatIP(DIM)),
        "code": faiss.IndexIDMap(faiss.IndexFlatIP(DIM))
    })
    logger.info(f"✅ HIGH-LEVEL Memory initialized with DIM={DIM}")


parser = ParserManager()

def save_memory_to_disk():
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

def Save_Search(gen, mem_type, query):
    try:
        emb = gen.Encode(query)
        if emb is None:
            return False  
      
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        
        emb_value = emb.reshape(1, -1).astype(np.float32)
        D, I = LTM_index[mem_type].search(emb_value, k=1)
        
 
        if D[0][0] < -1e10:
            logger.warning(f"Weird score detected: {D[0][0]}, saving anyway")
            return True
        
        threshold = 0.88 if mem_type == "identity" else 0.98
        if I[0][0] != -1 and D[0][0] >= threshold:  
            logger.info(f"Duplicate detected for '{query[:50]}...' (score: {D[0][0]:.3f})")
            return False  
        else:
            return True  
            
    except Exception as e:
        logger.error(f"Error in Save_Search: {e}")
        return True 

def Search_memory(query, gen, previous_results=None):
    """
    search with support for multiple different answers
    previous_results: list of already returned answers to avoid duplicates
    """
    search_result = []
    
    if len(query) >= 4:
        prompt_text = Search_memory_prompt(query)
        res = gen.generator(prompt_text)  
        extracted_query = parser.OutputManage(res)
        
        if extracted_query.get('type') == 'success':
            logger.info("Processing Memory")
            memory_types = extracted_query.get('Memory', [])
            query_values = extracted_query.get('Query', [])
            
            if memory_types and query_values:
                for mem_type, value in zip(memory_types, query_values):
                    if mem_type not in LTM_index:
                        continue
                    
                    try:
                        emb_value = gen.Encode(value)
                        if emb_value is None:
                            continue
                        
             
                        emb_value = emb_value.reshape(1, -1).astype(np.float32)
                        
                        D, I = LTM_index[mem_type].search(emb_value, k=10)
                        
                        for idx, ids in enumerate(I[0]):
                            if ids != -1 and ids in LTM_text:
                                item = LTM_text[ids]
                                score = float(D[0][idx])
                                
                          
                                if previous_results and item.Value in previous_results:
                                    score = score * 0.3
                                
                                search_result.append({
                                    'ID': item.ID,
                                    'text': item.Value,
                                    'score': score
                                })
                    except Exception as e:
                        logger.error(f"Error encoding: {e}")
                        continue
    

    seen_text = set()
    unique = []
    for r in search_result:
        if r['text'] not in seen_text:
            seen_text.add(r['text'])
            unique.append(r)
    
    sorted_results = sorted(unique, key=lambda x: x['score'], reverse=True)
    
    if sorted_results:

        if previous_results:
            filtered = [x for x in sorted_results if x['text'] not in previous_results]
            if filtered:
                sorted_results = filtered
        

        for i in range(min(3, len(sorted_results))):
            LTM_text[sorted_results[i]['ID']].update()
        
        return [x['text'] for x in sorted_results[:5]]
    else:
        return []
    
def Save_memory(STM, User_inputs, gen, Model_outputs=None):
    user_input_clean = User_inputs.strip()
    

    if user_input_clean.endswith('?') or user_input_clean.startswith('?'):
        logger.info("Question detected, skipping save to LTM")
        STM.append(f"User: {User_inputs}")
        return STM
    
    if len(user_input_clean) < 5:
        logger.info("Input too short, skipping save to LTM")
        STM.append(f"User: {User_inputs}")
        return STM
    
    logger.info("Input passed filters, processing for LTM storage")

    if Model_outputs:
        STM.append(f"User: {User_inputs}\nAI: {Model_outputs}")
    else:
        STM.append(f"User: {User_inputs}")

    if Model_outputs:
        query = f"{User_inputs}\n{Model_outputs}"
    else:
        query = User_inputs
    
    prompt_text = Save_memory_prompt(query)
    res = gen.generator(prompt_text)  
    logger.info(f"🔴 LLM OUTPUT: {res}") 
    Extracted = parser.OutputManage(res)
    
    if Extracted.get('type') != 'success':
        logger.info("No meaningful memory to extract")
        return STM
    
    Memory_types = Extracted.get("Memory", [])
    Values = Extracted.get("Query", [])
    
    if not Memory_types or not Values:
        logger.info("No memory type or value extracted")
        return STM
    
    saved_count = 0
    
    for Mem_type, Value in zip(Memory_types, Values):
        if Mem_type not in LTM_index:
            logger.warning(f"Unknown memory type: {Mem_type}")
            continue
        
        if not Value or len(Value.strip()) < 3:
            continue
        
        if not Save_Search(gen, Mem_type, Value):
            logger.info(f"Skipping duplicate: {Value[:50]}...")
            continue
        
        try:
            emb = gen.Encode(Value)
            if emb is None:
                continue

            emb = emb.reshape(1, -1).astype(np.float32)
            
            importance = 0.6
            item = MemoryItem([Mem_type], Value, emb[0], importance)
            LTM_text[item.ID] = item
            LTM_index[Mem_type].add_with_ids(emb, np.array([item.ID], dtype=np.int64))
            
            print(f"💾 saved {Mem_type}: {Value[:50]}...")
            saved_count += 1
            
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
            continue

    if saved_count > 0:
        logger.info(f"Saved {saved_count} memories to LTM")
    else:
        logger.info("No memories were saved")
    
    return STM

def GetEmotional():
    emotional_memories = []
    for mem_id, item in LTM_text.items():
        if 'emotional' in item.Memory_type:
            emotional_memories.append(item.Value)
    return emotional_memories