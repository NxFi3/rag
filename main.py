import streamlit as st
import asyncio
from agent import Agent
from core.generator import GeneratorManager
import threading
import time

st.set_page_config(
    page_title="Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stChatMessage {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .css-1v3fvcr {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .memory-card {
        background: #f0f2f5;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        border-left: 3px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)


col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.title("RAG system")
    st.caption("Your AI remembers you!")

st.divider()

with st.sidebar:
    st.header("📊 System Status")
    

    if 'agent' in st.session_state:
        st.success("✅ Agent is ready")
    else:
        st.warning("⏳ Loading Agent...")
    
    st.divider()
    

    st.header("💾 Memory Stats")
    memory_stats = st.empty()
    
    st.divider()
    

    st.header("🔍 Search Memory")
    search_query = st.text_input("Search for specific memories:", key="search_input")
    if search_query:
        if st.button("Search", type="primary"):
            with st.spinner("Searching..."):
                results = st.session_state.agent.memory.search(search_query, efficient=False)
                if results:
                    st.subheader(f"📝 Found {len(results)} memories:")
                    for r in results[:5]:  
                        st.markdown(f"<div class='memory-card'>📌 {r}</div>", unsafe_allow_html=True)
                else:
                    st.info("No memories found")
    
    st.divider()
    

    st.header("⚙️ Controls")
    
    if st.button("🗑️ Clear Conversation", type="secondary"):
        st.session_state.messages = []
        st.success("Conversation cleared!")
        st.rerun()
    
    if st.button("💾 Save Memory Now", type="primary"):
        with st.spinner("Saving..."):
            st.session_state.agent.memory.save_all()
        st.success("Memory saved to disk!")
    
    st.divider()
    

    with st.expander("ℹ️ How to use"):
        st.markdown("""
        **Features:**
        - 🧠 **Long-term memory**: Remembers facts about you
        - 💬 **Short-term memory**: Remembers conversation context  
        - 🔍 **Search memories**: Find past information
        - 🚀 **Persistent storage**: Memory survives restarts
        
        **Examples:**
        - "My name is X"
        - "I'm allergic to penicillin"
        - "I love Python programming"
        - "What's my name?"
        - "What am I allergic to?"
        """)
    
    st.divider()

    st.caption(f"🔧 Version 1.0 | Made with Streamlit")


class AsyncCallback:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
    
    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
    
    def run_async(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()


@st.cache_resource
def init_agent():
    with st.spinner("🧠 Loading AI model (this takes 15-30 seconds first time)..."):
        gen = GeneratorManager()
        agent = Agent(gen)
        return agent

try:
    if 'agent' not in st.session_state:
        st.session_state.agent = init_agent()
        st.session_state.messages = []
        
        welcome_msg = "Hello! 👋 I'm your AI assistant. I remember facts about you across conversations!\n\nAsk me anything or tell me about yourself. For example:\n- *My name is X*\n- *I'm allergic to penicillin*\n- *I love programming*"
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})

    with memory_stats:
        stats = st.session_state.agent.memory.get_stats()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("STM Size", f"{stats['stm_size']}/{stats['stm_max']}")
        with col2:
            st.metric("STM Max", stats['stm_max'])
    
except Exception as e:
    st.error(f"❌ Error loading Agent: {e}")
    st.stop()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Type your message here..."):

    with st.chat_message("user"):
        st.markdown(prompt)
    

    st.session_state.messages.append({"role": "user", "content": prompt})
    

    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking and searching memory..."):
            try:

                async_callback = AsyncCallback()
                response = async_callback.run_async(
                    st.session_state.agent.chat_async(prompt)
                )
                st.markdown(response)
                

                st.session_state.messages.append({"role": "assistant", "content": response})
                

                memories_used = st.session_state.agent.memory.search(prompt, efficient=False)
                if memories_used:
                    with st.expander("📚 Memories used for this response"):
                        for mem in memories_used[:3]:
                            st.info(f"• {mem}")
                
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Sorry, an error occurred: {e}"})
    
   
    st.rerun()

# فوتر
st.divider()
st.caption("💡 Tip: The AI remembers facts you share. Try saying 'My name is X' and then later ask 'What's my name?'")