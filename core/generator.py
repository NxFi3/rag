from Utils.logger import get_logger
import requests
import json
from sentence_transformers import SentenceTransformer
import numpy as np

logger = get_logger("GNR")
logger.info("GENERATOR started")

class GeneratorManager:
    def __init__(self, Model_Name: str = 'llama3.1:8B', Encoder_name: str = 'all-MiniLM-L6-v2'):
        """SentenceTransformers ONLY for Encoder"""
        self.model_name = Model_Name
        try:
            logger.info("loading ENCODER")
            self.encoder = SentenceTransformer(Encoder_name)
            logger.info(f"Loading Model : {self.model_name}")
        except Exception as e:
            logger.error(f"unexpected Error : {e}")
            raise  

    def generator(self, prompt: str, temperature: float = 0.7, max_new_tokens: int = 200, do_sample: bool = False):
        try:
            logger.info('sending Generation Request to ollama')
            res = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature if do_sample else 1.0,
                        "num_predict": max_new_tokens,
                        "do_sample": do_sample,
                        "repetition_penalty": 1.0
                    }
                },
                timeout=60
            )
            
            if res.status_code != 200:
                logger.error(f"Ollama returned error: {res.status_code}")
                return ''
                
            text = res.json()["response"].strip()
            return text
            
        except requests.exceptions.ConnectionError:
            logger.error("Could not connect to Ollama. Is it running?")
            return ''
        except Exception as e:
            logger.error(f"Generation Failed: {e}")
            return ''

    def Encode(self, Prompt):
        logger.info(f"Encoding: {Prompt[:400]}")
        try:
            encoded = self.encoder.encode(Prompt, normalize_embeddings=True)
            return encoded
        except Exception as e:
            logger.error(f"unexpected Error: {e}")
            
            return np.zeros(384)  