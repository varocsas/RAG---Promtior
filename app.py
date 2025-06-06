# RAG Code Challenge
# Valeria Rocha
# valeria.rocha.ferreira@gmail.com

import torch
import os

# Fix Streamlit/torch interop
try:
    torch.classes.__path__ = []
except Exception:
    pass

# Set fake user agent
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"

import streamlit as st
from rag import RAG  # Your custom RAG class

# --- Sidebar UI ---
st.sidebar.title("RAG coding challenge")
st.sidebar.caption("Leveraging LangChain, RailGuards, Pinecone, Groq, Llama3, LangSmith, and Streamlit")
st.sidebar.write ("-- Valeria Rocha")
st.sidebar.divider()
# --- Initialize session state ---
if "rag_pipe" not in st.session_state:
    
    with st.spinner("Initializing..."):
        st.session_state.rag_pipe = RAG()    
    st.success("Initialized!")
    
if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False

if "history" not in st.session_state:
    st.session_state.history = []

if "last_url" not in st.session_state:
    st.session_state.last_url = ""

if "challenge_mode" not in st.session_state:
    st.session_state.challenge_mode = False

# --- Challenge button ---
if st.sidebar.button("Run Challenge"):
    with st.spinner("Creating new RAG pipeline..."):
        st.session_state.rag_pipe.setup_challenge()
        st.session_state.rag_ready = True
        st.session_state.challenge_mode = True  # signal to do QA next time
    st.rerun()  # trigger re-render
    
    
# Challenge Q&A — only runs after rerun
if st.session_state.get("challenge_mode"):
    st.session_state.challenge_mode = False  # reset flag

    question_1 = "What services does Promtior offer?"
    question_2 = "When was it founded?"

    with st.spinner("Thinking..."):
        res_1 = st.session_state.rag_pipe.qa(question_1)         
        st.session_state.history.append((question_1, res_1))
        res_2 = st.session_state.rag_pipe.qa(question_2)
        st.session_state.history.append((question_2, res_2))

    st.subheader(question_1)
    st.write(res_1)
    st.subheader(question_2)
    st.write(res_2)

# --- User input fields ---
st.sidebar.write(" ")
question = st.sidebar.text_input("Question:", key = "question")
url = st.sidebar.text_input("URL (web or PDF)", key = "url")
st.sidebar.caption("Entering a new URL builds a new RAG context.")

# --- URL submission ---
if url:
    if (not st.session_state.rag_ready) or (url != st.session_state.last_url):
        with st.spinner("Creating new RAG pipeline..."):
            st.session_state.rag_pipe.setup(url)
            st.session_state.rag_ready = True
            st.session_state.last_url = url
            
        st.success("RAG pipeline created!")
        st.rerun()  # Trigger rerun to show prompt box immediately

# --- Reset option ---
if st.session_state.rag_ready:
    if st.sidebar.button("Delete pipeline"):
        st.session_state.rag_pipe.reset_pipeline()
        st.session_state.rag_ready = False
        st.success("RAG pipeline deleted!")

# --- Question submission ---
if question:
    if st.session_state.rag_ready:
        with st.spinner("Thinking..."):
            answer = st.session_state.rag_pipe.qa(question)
            st.session_state.history.append((question, answer))
            st.subheader("Answer")
            st.write(answer)
    else:
        st.error("Please create a RAG pipeline before asking questions.", icon="🚨")

# --- Display chat history ---
if st.session_state.history:
    st.markdown("---")
    st.subheader("Conversation History")
    for q, a in reversed(st.session_state.history):
        st.caption(f"**Q:** {q}")
        st.caption(f"**A:** {a}")
        st.caption("---")