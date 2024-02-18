# print("hi")

import os
import openai
import shutil
import hashlib
from glob import glob
from datetime import datetime

#streamlit is used for creating the web app. 
#The langchain library is used for processing documents and building the chatbot functionality.
import streamlit as st
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

st.title("Hi, I'm the help ChatBot for MSc ACES at FAU Erlangen!")

openai.api_key = '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'
os.environ['OPENAI_API_KEY'] = openai.api_key
tmp_directory = 'tmp'

UPLOAD_NEW = "Upload new one"
ALREADY_UPLOADED = "Already Uploaded"

chatbot_started = False

def load_docs(directory: str):
    """
    Load documents from the given directory.
    """
    loader = DirectoryLoader(directory)
    documents = loader.load()

    return documents

#split_docs splits these documents into manageable chunks for processing. This is useful for handling large documents
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    """
    Split the documents into chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)

    return docs

#It loads and processes documents, and generates embeddings.
@st.cache_resource
def startup_event(last_update: str):
    """
    Load all the necessary models and data once the server starts.
    """
    print(f"{last_update}")
    directory = 'tmp/'
    documents = load_docs(directory)
    docs = split_docs(documents)

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    persist_directory = "chroma_db"
#initializes the chatbot model and question-answering chain.
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()

    model_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=model_name)

    db = Chroma.from_documents(docs, embeddings)
    chain = load_qa_chain(llm, chain_type="stuff", verbose=True)

    return db, chain

#The get_answer function searches the document database for documents similar to the user's query
def get_answer(query: str, db, chain):
    """
    Queries the model with a given question and returns the answer.
    """
    matching_docs_score = db.similarity_search_with_score(query)

    matching_docs = [doc for doc, score in matching_docs_score]
    answer = chain.run(input_documents=matching_docs, question=query)

    # Prepare the sources
    sources = [{
        "content": doc.page_content,
        "metadata": doc.metadata,
        "score": score
    } for doc, score in matching_docs_score]

    return {"answer": answer, "sources": sources}

#The start_chatbot function manages the chatbot's state and interaction with the user. 
#It records the conversation history and uses the get_answer function to respond to user queries.
def start_chatbot():
    db, chain = startup_event(last_db_updated)
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
    
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = get_answer(st.session_state.messages[-1]["content"], db, chain)
            answer = full_response['answer']
            message_placeholder.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

#Users can choose between using an existing knowledge base or uploading new documents. 
content_type = st.sidebar.radio("Which Knowledge base you want to use?",
                                [ALREADY_UPLOADED, UPLOAD_NEW])


if content_type == UPLOAD_NEW:
    uploaded_files = st.sidebar.file_uploader("Choose a txt file", accept_multiple_files=True)

    uploaded_file_names = [file.name for file in uploaded_files]
##If new documents are uploaded, the application processes and stores them in a temporary directory.
    if uploaded_files is not None and len(uploaded_files):
        if os.path.exists(tmp_directory):
            shutil.rmtree(tmp_directory)

        os.makedirs(tmp_directory)

        if len(uploaded_files):
            last_db_updated = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        for file in uploaded_files:
            with open(f"{tmp_directory}/{file.name}", 'wb') as temp:
                temp.write(file.getvalue())
                temp.seek(0)
                

curr_dir = [path.split(os.path.sep)[-1] for path in glob(tmp_directory + '/*')]

if content_type == ALREADY_UPLOADED:
    st.sidebar.write("Current Knowledge Base")
    if len(curr_dir):
        st.sidebar.write(curr_dir)
    else:
        st.sidebar.write('**No KB Uploaded**')

last_db_updated = hashlib.md5(','.join(curr_dir).encode()).hexdigest()

if curr_dir and len(curr_dir):
    start_chatbot()
else:
    st.header('No KB Loaded, use the left menu to start')

    

