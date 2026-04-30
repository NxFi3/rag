from Utils.logger import get_logger
import requests
import ollama
from sentence_transformers import SentenceTransformer
import numpy as np
from core.prompts import ImageProcessPrompt
logger = get_logger("GNR")
logger.info("GENERATOR started")
DIM = 768
class GeneratorManager:
    def __init__(self, Model_Name: str = 'llama3.1:8B', Encoder_name: str = "E:\Multilingual-e5-base"): # iran the land of dedifficulty
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
            res = ollama.generate(model= self.model_name,prompt = prompt,stream=False,
                    options= {
                        "temperature": temperature if do_sample else 1.0,
                        "num_predict": max_new_tokens,
                        "do_sample": do_sample,
                        "repetition_penalty": 1.0} )
            text = res["response"].strip()
            return text
            
        except requests.exceptions.ConnectionError:
            logger.error("Could not connect to Ollama. Is it running?")
            return ''
        except Exception as e:
            logger.error(f"Generation Failed: {e}")
            return ''
    def ImageProcess(self,IMG:bytes):
        logger.info("VISION in process")
        try:
            messages = [  
                {
                    'role': 'user',
                    'content': ImageProcessPrompt(),
                    'images': [IMG]
                }
            ]
            res = ollama.chat(
                model=self.model_name,
                messages=messages  
            )
            return res['message']['content'].strip()
        except Exception as e:
            logger.error(f"unexpected Error : {e}")
            logger.error("VISION GEBERATION ERROR")
            return "You Are Blind."

    def Encode(self, Prompt):
        logger.info(f"Encoding: {Prompt[:400]}")
        try:
            encoded = self.encoder.encode(Prompt, normalize_embeddings=True)
            return encoded
        except Exception as e:
            logger.error(f"unexpected Error: {e}")
            
            return np.zeros(DIM)  