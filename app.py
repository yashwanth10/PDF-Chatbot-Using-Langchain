
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
import time

# Set page configuration as the first Streamlit command
st.set_page_config(page_title="PDF Document Chatbot", page_icon="📄", layout="wide")

load_dotenv()

# Function to initialize API keys
def initialize_keys():
    groq_api_key = st.text_input("Enter your GROQ API Key:", type="password")
    os.environ["GROQ_API_KEY"] = groq_api_key
    os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

    if groq_api_key:
        return groq_api_key
    else:
        st.warning("Please enter your GROQ API Key.")
        return None

groq_api_key = initialize_keys()
if not groq_api_key:
    st.stop()  # Stop execution if the API key is not provided

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama-3.1-70b-Versatile")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    Question: {input}
    """
)


# Using a relative path or an environment variable for universal paths
save_directory = os.getenv("SAVE_DIRECTORY", "research_papers")

#UI 
st.title("📚 PDF Document Chatbot")
st.markdown("Upload your PDFs and interact with them using natural language.")

uploaded_files = st.file_uploader("Choose PDF files to upload", type="pdf", accept_multiple_files=True)

# Process uploaded PDFs
if uploaded_files:
    st.write("Uploading files...")
    for uploaded_file in uploaded_files:
        save_path = os.path.join(save_directory, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())
    st.success("Files uploaded successfully!")

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        documents = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(save_directory, uploaded_file.name)
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
        
        st.session_state['text_splitter'] = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state['final_documents'] = st.session_state['text_splitter'].split_documents(documents)
        st.session_state['vectors'] = FAISS.from_documents(st.session_state['final_documents'], st.session_state['embeddings'])

st.subheader("Ask a Question")
user_prompt = st.text_area("Enter your query:")

col1, col2 = st.columns([1, 2])

with col1:
    if st.button("Let's chat"):
        create_vector_embedding()
        st.write("Document Embedding Done")

with col2:
    if user_prompt:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state['vectors'].as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        response_time = time.process_time() - start

        st.session_state.chat_history.insert(0, {"user": user_prompt, "bot": response['answer']})

        st.write(f"Response Time: {response_time:.2f} seconds")

        st.write("### Chat History")
        for chat in st.session_state.chat_history:
            st.write(f"**You:** {chat['user']}")
            st.write(f"**Bot:** {chat['bot']}")
            st.write("---")

        with st.expander("Document similarity search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write('--------------------------')



