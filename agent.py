# agent.py
from Memory.Memory_manager import MemoryManager
from core.generator import GeneratorManager

class Agent:
    def __init__(self):
        self.memory = MemoryManager(STM_SIZE=15)
        self.generator = GeneratorManager()
        self.memory.load_all()
    
    def chat(self, user_input: str) -> str:
    
        context = self.memory.get_relevant_memory(user_input)
        
    
        prompt = f"""
You Are a helpful AI assistant.
Previous conversation:
{context['stm']}

Relevant memories:
{chr(10).join([f"- {m}" for m in context['ltm']])}

User: {user_input}
Assistant:"""
        
    
        response = self.generator.generator(prompt)
        

        self.memory.add_interaction(user_input, response)
        
        return "AI: "+response
    
    def save(self):
        self.memory.save_all()

agent = Agent()
while True:
    inputs = input("You: ")
    print(agent.chat(inputs))
    agent.save()