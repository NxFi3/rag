import asyncio
from concurrent.futures import ThreadPoolExecutor
from Memory.Memory_manager import MemoryManager
from core.generator import GeneratorManager

class AsyncAgent:
    def __init__(self, gen):
        self.gen = gen
        self.memory = MemoryManager(STM_SIZE=15, gen=self.gen)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.memory.load_all()
    
    async def chat_async(self, user_input: str) -> str:
        loop = asyncio.get_running_loop()
        
        context = await loop.run_in_executor(
            self.executor, 
            self.memory.get_relevant_memory, 
            user_input
        )
        
        prompt = f"""You Are a helpful AI assistant.
Previous conversation:
{context['stm']}

Relevant memories:
{chr(10).join([f"- {m}" for m in context['ltm']])}

User: {user_input}
Assistant:"""
        
        response = await loop.run_in_executor(
            self.executor,
            self.gen.generator,
            prompt
        )
        
    
        loop.run_in_executor(
            self.executor,
            self.memory.add_interaction,
            user_input,
            response
        )
        
        return "AI: " + response
    
    def chat(self, user_input: str) -> str:
    
        return asyncio.run(self.chat_async(user_input))

async def main():
    agent = AsyncAgent(GeneratorManager())
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            agent.memory.save_all()
            break
        response = await agent.chat_async(user_input)
        print(response)


asyncio.run(main())