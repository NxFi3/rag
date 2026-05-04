from Utils.logger import get_logger
from piper import PiperVoice
from sentence_transformers import SentenceTransformer
import requests
import ollama
import sounddevice as sd
import numpy as np
import json
import whisper
logger = get_logger("GNR")
logger.info("GENERATOR started")

logger.info("loading Config ...")





class GeneratorManager:
    def __init__(self,Config:dict,MultiModal:bool=False): 
        """SentenceTransformers ONLY for Encoder
           TTS Piper-tts ONLY
           FOR TTS and STT need MultiModal=True
        """
        self._readConfig(Config)
        self._loadModels(MultiModal)
    def _readConfig(self, Config:dict):
        try:

            with open(Config, 'r', encoding='utf-8') as f:
                Config = json.load(f)
        except Exception as e:
            logger.error(f"Error Invalid Config {e}")
            exit()

        self.LLMName = Config['LLM']
        self.encoderName = Config['encoder']
        self.ttsName = Config['TTS']
        self.whisperName = Config['Whisper']
        
    def _loadModels(self,multimodal:bool):
        try: 
            logger.info("LLM Requested.")
            logger.info(f"loading Encoder {self.encoderName}")
            self.encoder = SentenceTransformer(self.encoderName)
            self.DIM = self.encoder.get_embedding_dimension()
            if multimodal:
                logger.info("MultiModal Triggered.")

                logger.info("Loading TTS model")
                self.tts = PiperVoice.load(self.ttsName['model'],self.ttsName['config'])
                logger.info(f"loading WHISPER {self.whisperName}")
                self.whisper = whisper.load_model(self.whisperName)
                logger.info("ALL MODEL LOADED.")
        except Exception as e:
            logger.error(f"Something Went Wrong {e}")
            exit()


    def generator(self, prompt: str, temperature: float = 0.7, max_new_tokens: int = 200, do_sample: bool = False):
        try:
            logger.info('sending Generation Request to ollama')
            res = ollama.generate(model= self.LLMName,prompt = prompt,stream=False,
                    options= {
                        "temperature": temperature if do_sample else 1.0,
                        "num_predict": max_new_tokens,
                        "do_sample": do_sample,
                        "repetition_penalty": 1.0} )
            text = res["response"].strip()
            return text
            
        except requests.exceptions.ConnectionError:
            logger.error("Could not connect to Ollama. Is it running?")
            return 'ollama Problem'
        except Exception as e:
            logger.error(f"Generation Failed: {e}")
            return 'Generation Failed'
    def Encode(self, Prompt):
        logger.info(f"Encoding: {Prompt[:400]}")
        try:
            encoded = self.encoder.encode(Prompt, normalize_embeddings=True)
            return encoded
        except Exception as e:
            logger.error(f"unexpected Error: {e}")
            
            return np.zeros(self.DIM) 

    def speech(self,text:str,Block:bool=True):
        """ONLY work when MultiModal is True"""
        logger.info(f"generating Speech For Query: {text}  ||||")
        if not text or not text.strip():
            logger.info("SPEECH GENERATION STOPPED BAD QUERY.")
            return False
        audio_chunks = []
        try:
            for chunk in self.tts.synthesize(text):
                audio_chunks.append(chunk.audio_float_array)
            if not audio_chunks:
                return False
            AudioData = np.concatenate(audio_chunks)
            SampleRate = self.tts.config.sample_rate
            if Block:
                sd.play(AudioData,SampleRate)
                sd.wait()
            else:
                sd.play(AudioData,SampleRate,blocking=False)

            return True
        except Exception as e:
            logger.error(f"SPEECH ... Error While Generation or Speaking {e}")
            return False
        
    def STOPSPEECH(self):
        """ONLY work when MultiModal is True"""
        try:
            sd.stop()
            return True
        except Exception as e:
            logger.error(f"SPEECH STOP ERROR {e}")
            return False
                
    def Listening(self):
        pass