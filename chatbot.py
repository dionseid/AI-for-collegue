import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS  # Langchain has lots of other vector stores that it can use
from langchain.chains.question_answering import load_qa_chain  # package created specifically for Q&A
from langchain_community.chat_models import ChatOpenAI


OPENAI_API_KEY = ''

# Upload PDF files
st.header('colleague chatbot')

# sidebar interface
with st.sidebar:
    st.title('documents')
    file = st.file_uploader('Upload a PDF file', type='pdf')


# Extract the text
if file:
    text = ""
    for page in PdfReader(file).pages:
        text += page.extract_text()
        # st.write(text)

    # Replace newline characters with spaces, as the entry may have had complex formatting rules that include line
    # breaks to position the text correctly on the page
    text = text.replace('\n', ' ')

    # Break the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[' '],
        chunk_size=500,
        chunk_overlap=150,
        length_function=len
    )  # You may keep playing with this parameters to test accuracy of the outputs
    chunks = text_splitter.split_text(text)
    # st.write(chunks)

    # Generating embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Creating the vector store – FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Get the user request
    user_request = st.text_input("Type your request here")

    # Do similarity search
    if user_request:
        matches = vector_store.similarity_search(user_request)  # We get multiple chunks that contain a match to our
        # session

        # Define the LLM
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0.7,  # from 0 (very specific) to 1 (more random), or even 2. The LLMs like ChatGPT, Google
            # Bard, or Facebook LLaMA behind the scene are generating content on their own, where it can be a bit of
            # randomness
            max_tokens=4000,  # 1000 tokens translates to something like 750 words
            model_name='gpt-4-vision-preview'  # I would like to use gpt-4-vision-preview
        )  # All the parameters here are open to optimization

        # Output results in a sequence of events: Take the request, get relevant documents, pass to the LLM, and
        # generate the output
        chain = load_qa_chain(llm, chain_type='stuff')  # stuff is something like stuffing everything into a bucket,
        # before passing it to the LLM
        response = chain.run(input_documents=matches, question=user_request)
        st.write(response)

