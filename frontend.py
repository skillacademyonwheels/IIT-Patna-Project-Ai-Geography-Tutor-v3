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
        padding: 1.25rem 1.5rem;
        margin: 0.5rem 0 1.5rem;
        border: 1px solid rgba(46, 125, 50, 0.22);
        border-radius: 1rem;
        background: linear-gradient(135deg, rgba(46, 125, 50, 0.10), rgba(33, 150, 243, 0.06));
    }
    .geo-intro p {
        margin: 0.35rem 0 1rem;
    }
    .geo-eyebrow {
        color: #2e7d32;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .geo-books {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 0 0 1.1rem;
    }
    .geo-book {
        padding: 0.35rem 0.65rem;
        border: 1px solid rgba(46, 125, 50, 0.2);
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.55);
        font-size: 0.85rem;
    }
    .geo-credit {
        display: flex;
        flex-wrap: wrap;
        align-items: baseline;
        gap: 0.35rem;
        padding-top: 0.8rem;
        border-top: 1px solid rgba(46, 125, 50, 0.18);
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
        <div class="geo-eyebrow">Explore geography with your AI tutor</div>
        <p>Get answers grounded in these NCERT books:</p>
        <div class="geo-books">
            <span class="geo-book">Fundamentals of Human Geography</span>
            <span class="geo-book">India People and Economy</span>
            <span class="geo-book">India Physical Environment</span>
            <span class="geo-book">Practical Work in Geography</span>
        </div>
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
    
    result = chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config=CONFIG)
   
        
    with st.chat_message("assistant"):
        aimessage = st.write_stream(
           message_chunk.content for message_chunk,metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode = 'messages'
           )
         )
        st.session_state["message_history"].append({"role": "assistant", "content": aimessage})
    
    
        
