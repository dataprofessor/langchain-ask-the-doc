import streamlit as st
from langchain.llms import OpenAI
from langchain import PromptTemplate

st.title('🦜🔗 Ask the Doc App')
st.set_page_config(page_title="🦜🔗 Ask the Doc App")
openai_api_key = st.sidebar.text_input('OpenAI API Key')

uploaded_file = st.file_uploader('Upload an article', type='txt')
question = st.text_input('Ask something about the article', placeholder = 'Can you give me a short summary?', disabled = not uploaded_file)

def generate_response(topic):
  llm = OpenAI(model_name='text-davinci-003', openai_api_key=openai_api_key)
  # Prompt
  template = 'As an experienced data scientist and technical writer, generate an outline for a blog about {topic}.'
  prompt = PromptTemplate(input_variables = ['topic'], template = template)
  prompt_query = prompt.format(topic=topic)
  # Run LLM model
  response = llm(prompt_query)
  return st.info(response)

with st.form('myform'):
  topic_text = st.text_input('Enter prompt:', '')
  submitted = st.form_submit_button('Submit')
  if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
  if submitted and openai_api_key.startswith('sk-'):
    generate_response(topic_text)
