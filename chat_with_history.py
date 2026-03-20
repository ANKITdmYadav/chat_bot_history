import streamlit as st
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
import os

from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("Conversational RAG (LCEL Version)")
st.write("Upload PDFs and chat using pure LCEL logic")

api_key = st.text_input("Enter your Groq API key:", type="password")

def inspect(state):
    print("\n--- CURRENT CHAIN STATE ---")
    print(state)
    print("---------------------------\n")
    return state 

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")

    session_id = st.text_input("Session ID", value="default_session")

    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        documents =[]
        for uploaded_file in uploaded_files:
            temppdf="./temp.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
            loader = PyPDFLoader(temppdf)
            documents.extend(loader.load())

        text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
        splits=text_splitter.split_documents(documents)
        vectorstore=Chroma.from_documents(documents=splits,embedding=embeddings)
        retriever=vectorstore.as_retriever()

        # --- LCEL REORGANIZATION START ---

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question, "
            "formulate a standalone question. Do NOT answer it."
        )
        contextualize_q_prompt=ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        contextualize_chain=contextualize_q_prompt | llm | StrOutputParser()

        qa_system_prompt=(
            "You are an assistant for question-answering tasks. Use the context below to answer. "
            "If you don't know, say you don't know. Max 3 sentences.\n\n{context}"
        )
        qa_prompt=ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        rag_chain=(
            RunnablePassthrough.assign(
                context=lambda x: contextualize_chain | retriever | format_docs
            )
            | RunnableLambda(inspect)  
            | qa_prompt
            | llm
            | StrOutputParser()
        )
        # --- LCEL REORGANIZATION END ---

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        user_input=st.text_input("Your question:")
        if user_input:
            session_history=get_session_history(session_id)
            response_text=conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )

            st.write("Assistant:", response_text)
            
            with st.expander("View Message History"):
                for msg in session_history.messages:
                    st.write(f"{msg.type}: {msg.content}")
else:
    st.warning("Please enter the Groq API Key")