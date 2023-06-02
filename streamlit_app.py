import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

if 'temp' not in st.session_state:
    st.session_state['temp'] = ''
def clear_text():
    st.session_state.temp = st.session_state.text
    st.session_state.text = ''

def generate_response(uploaded_file, openai_api_key, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        documents = [uploaded_file.read().decode()]
    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.create_documents(documents)
    # Select embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # Create vectorstore index
    db = Chroma.from_documents(texts, embeddings)
    # Create retriever interface
    retriever = db.as_retriever()
    # Create QA chain
    qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
    response = qa.run(query_text)
    return response

st.set_page_config(page_title='🦜🔗 Ask the Doc App')

st.title('🦜🔗 Ask the Doc App')
#openai_api_key = st.sidebar.text_input('OpenAI API Key')
openai_api_key = st.sidebar.text_input('OpenAI API Key', key='text')

# File upload
uploaded_file = st.file_uploader('Upload an article', type='txt')

# Form input and query
result = []
with st.form('myform'):
    query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled = not uploaded_file)
    #submitted = st.form_submit_button('Submit', disabled = not (uploaded_file and openai_api_key))
    submitted = st.form_submit_button('Submit', disabled = not (uploaded_file and openai_api_key), on_click=clear_text)
    if openai_api_key.startswith('sk-'):
        st.success('API key provided!', icon='✅')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='❗️')
    if submitted and openai_api_key.startswith('sk-'):
        #response = generate_response(uploaded_file, openai_api_key, query_text)
        response = generate_response(uploaded_file, st.session_state.temp, query_text)
        result.append(response)
        st.info(response)
# Use template in place of the openai_api_key in the IF conditions above
