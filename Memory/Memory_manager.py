from Memory.High_level_memory import Save_memory, Search_memory, save_memory_to_disk, load_memory_from_disk
from Memory.Low_level_memory import search_memory as LowSearch, persist_memory, load_persisted_memory, add_to_memory
from Utils.logger import get_logger
from collections import deque
from core.generator import GeneratorManager
import uuid
import time 

logger = get_logger("MEM")
logger.info("MemoryManager started")

generator = GeneratorManager()

class MemoryManager:
    def __init__(self, STM_SIZE: int = 15):  
        self.stm = deque(maxlen=STM_SIZE)
        self.stm_size = STM_SIZE
    
    def add_interaction(self, user_input: str, assistant_output: str = ''):
        
        
        if len(user_input) >= 500:
            encoded = generator.Encode(user_input + ' ' + assistant_output)
            if encoded is not None:
                metadata = {
                    "text": f"{user_input} {assistant_output}",
                    "importance": 0.8,  # SOLID
                    "timestamp": time.time()
                }
                add_to_memory(encoded, uuid.uuid4(), metadata)
        
    
        stm_list = list(self.stm)
        new_stm_list = Save_memory(stm_list, user_input, assistant_output)
        
        self.stm = deque(new_stm_list, maxlen=self.stm_size)
        logger.info(f"Added: {user_input[:50]}...")

    def search(self, query: str, efficient: bool = True):  
        if efficient:
            return LowSearch(query)
        else:
            if len(query) >= 500:
                return LowSearch(query)
            else:
                return Search_memory(query)

    def get_stm_context(self, last_n: int = 0):
        if last_n:
            return "\n".join(list(self.stm)[-last_n:])
        return "\n".join(self.stm)

    def get_relevant_memory(self, query: str):
        return {
            'stm': self.get_stm_context(last_n=5),
            'ltm': self.search(query, efficient=False)  
        }

    def save_all(self):
        save_memory_to_disk()
        persist_memory()
        logger.info("All saved")

    def load_all(self):
        load_memory_from_disk()
        load_persisted_memory()
        logger.info("All loaded")

    def clear_stm(self):
        self.stm.clear()
        logger.info("STM cleared")

    def get_stats(self):
        return {
            'stm_size': len(self.stm),
            'stm_max': self.stm_size
        }

logger.info("MemoryManager ready")