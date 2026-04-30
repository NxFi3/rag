def Search_memory_prompt(query):
    prompt = f"""
You are a query understanding system for memory search. Convert user question to search queries for different memory types.

CRITICAL RULES:
1. Extract search keywords for EACH memory type that might be relevant
2. Each value MUST be a COMPLETE phrase (minimum 2 words, maximum 10 words)
3. Keep the original meaning
4. If user asks about themselves → identity Memory
5. If user asks about definitions/concepts → semantic Memory  
6. If user asks about feelings/preferences → emotional Memory
7. If user asks about past events → episodic Memory
8. If user asks about how to do something → procedural Memory

Memory TYPE DETECTION:
- identity: questions about "me", "my", "name", "age", "job", "live", "born", "I am"
- semantic: questions about "what is", "definition", "concept", "meaning of"
- procedural: questions about "how to", "steps", "instructions", "guide"
- emotional: questions about "like", "love", "hate", "feel", "prefer"
- episodic: questions about "yesterday", "before", "last time", "earlier", "remember when"

OUTPUT FORMAT:
{{"Memory": ["identity", "semantic", "emotional"], "value": ["search for identity", "search for semantic"]}}

EXAMPLES:

Query: "what is my name?"
Output: {{"Memory": ["identity"], "value": ["my name"]}}

Query: "what is RAG?"
Output: {{"Memory": ["semantic"], "value": ["RAG definition"]}}

Query: "do you remember what I like?"
Output: {{"Memory": ["identity", "emotional"], "value": ["what I like", "my preferences"]}}

Query: "what did we talk about yesterday?"
Output: {{"Memory": ["episodic"], "value": ["yesterday conversation", "previous discussion"]}}

Query: "how do I install PyTorch?"
Output: {{"Memory": ["procedural"], "value": ["install PyTorch", "how to install PyTorch"]}}

Query: "tell me about my interests and what I love"
Output: {{"Memory": ["identity", "emotional"], "value": ["my interests", "what I love"]}}

Query: "what is my name and how old am I?"
Output: {{"Memory": ["identity"], "value": ["my name", "my age"]}}

Query: "do you remember my job and where I live?"
Output: {{"Memory": ["identity"], "value": ["my job", "where I live"]}}

IMPORTANT REMINDERS:
- Return MULTIPLE types if the question asks about different things
- Each Query value should be a SEARCHABLE phrase
- Keep queries short and meaningful

Now process:
Query: {query}

Output (JSON only, no extra text):
"""
    return prompt

def Save_memory_prompt(query):
    prompt = f"""
You are a memory extraction system. Extract ONLY factual information.

ABSOLUTE RULES (NEVER BREAK):
1. NEVER extract questions (anything with "?", "what", "how", "why", "when", "who", "where")
2. NEVER extract commands or requests ("help me", "can you", "tell me", "do this")
3. NEVER extract meta comments about memory or testing
4. ONLY extract when user shares NEW information about themselves or the world

REQUIRED PATTERNS (only extract these):
- "my name is X"
- "i am X years old"  
- "i love/hate X"
- "i am building/working on X"
- "X stands for Y"
- "yesterday/last week i did X"
- "your name is X"

FORBIDDEN PATTERNS (NEVER extract):
- Any sentence with "?" at the end
- Starting with: what, how, why, when, where, who, can, could, would, should, do, does, is, are
- Containing: test, remember, memory, help, assist, can you, tell me, show me

Output format: {{"Memory": ["type"], "value": ["exact phrase"]}}
If nothing to extract: {{"Memory": [], "value": []}}

EXAMPLES:
✅ "my name is alireza" → {{"Memory": ["identity"], "value": ["my name is alireza"]}}
✅ "i hate math exams" → {{"Memory": ["emotional"], "value": ["i hate math exams"]}}
✅ "i am building a RAG system" → {{"Memory": ["procedural"], "value": ["i am building a RAG system"]}}
✅ "RAG stands for retrieval generation" → {{"Memory": ["semantic"], "value": ["RAG stands for retrieval generation"]}}
✅ "yesterday i worked 5 hours" → {{"Memory": ["episodic"], "value": ["yesterday i worked 5 hours"]}}

❌ "what is your name?" → {{"Memory": [], "value": []}}
❌ "can you help me?" → {{"Memory": [], "value": []}}
❌ "i want to test memory" → {{"Memory": [], "value": []}}
❌ "do you remember me?" → {{"Memory": [], "value": []}}
❌ "tell me about X" → {{"Memory": [], "value": []}}

NOW PROCESS: {query}

REMEMBER: If it's a question, request, or test, return EMPTY. Only extract REAL information.
"""
    return prompt

def ImageProcessPrompt():
    prompt = "Explain What you see in Image completely with all details."
    return prompt