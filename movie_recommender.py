import streamlit as st
import ollama

from langchain.agents.agent_types import AgentType

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

import pandas as pd
import re
import ast
import numpy as np




from langchain_community.llms import Ollama


llm = Ollama(model='llama3')





import os
import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import ollama
import time
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.embeddings import OllamaEmbeddings
import re

embeddings = OllamaEmbeddings(model='nomic-embed-text')

vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
llm = Ollama(model="llama3", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Act like a movie recommender with the information you have. 
whenever you recommend a movie give a little description about it as well.
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

from langchain.chains import RetrievalQA


qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )


# Title of the app
st.title("Movie Recommender Chatbot")

# Initialize session state for conversation history
if 'history' not in st.session_state:
    st.session_state.history = []

# Input for user query
query = st.text_input("Ask for your movie recommendation:")

# Display response if query is entered
if query:
    response = qa_chain({"query": query})
    # Update conversation history
    st.session_state.history.append({'role': 'user', 'content': query})
    st.session_state.history.append({'role': 'bot', 'content': response['result']})

# Display the conversation history
for message in st.session_state.history:
    if message['role'] == 'user':
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Bot:** {message['content']}")