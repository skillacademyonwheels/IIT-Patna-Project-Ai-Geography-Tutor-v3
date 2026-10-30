import streamlit as st
from backend import chatbot, get_all_threads
from langchain_core.messages import HumanMessage, AIMessage

import uuid


# ********************** UTILITY FUNCTIONS ********************** #
## Generate a unique thread ID for each chat session
def generate_thread_id():
    thread_id = uuid.uuid4()
    return str(thread_id) 

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', [])

def delete_thread(thread_id):
    pass  # Implement the logic to delete a thread from the checkpointer and session state

# ********************** SESSION SET UP  ********************** #
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = get_all_threads()

add_thread(st.session_state['thread_id'])

# ********************** SIDEBAR UI ********************** #
st.sidebar.title("UTILITIES")
st.sidebar.text("NEW CHAT")

if st.sidebar.button("Start New Chat"):
     reset_chat()
st.sidebar.header("My Conversations")


for thread_id in st.session_state['chat_threads'][::-1]:
    messages = load_conversation(thread_id)
    msg_usr = "No messages yet"

    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            msg_usr = msg.content
            break

    if st.sidebar.button(f" *-{str(thread_id)[-4:]} - {msg_usr[:20]}"):

        st.session_state['message_history'] = []
        st.session_state['thread_id'] = thread_id

        temp_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                role='user'
            else:
                role='assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages

        if st.sidebar.button(f"🗑️ {str(thread_id)[-4:]} "):
            delete_thread(thread_id)
           






# ********************** Main UI ****st.set_page_config(
st.set_page_config(
    page_title="Geography Tutor",
    page_icon="🗺️"
)

st.title("Geography Tutor")
st.markdown(
    """
    <style>
    .geo-intro {
        margin: 0.5rem 0 1.5rem;
    }
    .geo-intro p {
        margin: 0.35rem 0 0.65rem;
    }
    .geo-eyebrow {
        color: #2e7d32;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0 0 0.35rem;
    }
    .geo-books {
        margin: 0 0 1.1rem;
        padding-left: 1.25rem;
    }
    .geo-books li {
        margin: 0.3rem 0;
    }
    .geo-credit {
        display: flex;
        flex-wrap: wrap;
        align-items: baseline;
        gap: 0.35rem;
        margin-top: 0.8rem;
        font-size: 0.9rem;
    }
    .geo-credit-label {
        color: #2e7d32;
        font-weight: 700;
        animation: developer-blink 1.4s ease-in-out infinite;
    }
    @keyframes developer-blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.45; }
    }
    @media (prefers-reduced-motion: reduce) {
        .geo-credit-label { animation: none; }
    }
    </style>
    <section class="geo-intro">
        <h2 class="geo-eyebrow">RAG based AI tutor</h2>
        <p>100% NCERT - ZERO Hallucination:</p>
        <ul class="geo-books">
            <li><strong>Book:</strong> Fundamentals of Physical Geography</li>
            <li><strong>Class:</strong> 11</li>
            <li><strong>Contents:</strong> 6 units and 14 chapters</li>
            <li><strong>Source:</strong> <a href="https://ncert.nic.in/">NCERT</a>. All credits and attribution belong to NCERT.</li>
        </ul>
        <div class="geo-credit">
            <span class="geo-credit-label">Developed by</span>
            <strong>Anubhav Lal</strong>
            <span>· Mob: 9560058067 · Email: advelestica.design@gmail.com</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

for message in st.session_state["message_history"]:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(message["content"])
            else:
                st.text(message["content"])


user_input = st.chat_input("Type your message here...")
    
if user_input:
    with st.chat_message("user"):
        st.text(user_input)
        st.session_state["message_history"].append({"role": "user", "content": user_input})

    CONFIG = {
        'configurable': {
            'thread_id': st.session_state['thread_id'],
        },
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }
    
    # result = chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config=CONFIG)
   
        
    with st.chat_message("assistant"):
        aimessage = st.write_stream(
           message_chunk.content for message_chunk,metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode = 'messages'
           )
         )
        st.session_state["message_history"].append({"role": "assistant", "content": aimessage})
    
    
        
