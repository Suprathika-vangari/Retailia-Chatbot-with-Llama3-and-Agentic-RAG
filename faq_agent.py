import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings

load_dotenv()

groq_api_key=os.environ["GROQ_API_KEY"]

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    # embeddings = SentenceTransformerEmbeddings(model ='all-MiniLM-L6-v2' )
    # embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not available in the provided context, respond with "The answer is not available in the context." Do not attempt to provide an answer if the information is not present, and avoid including incomplete or potentially misleading information.

    Note: If the question is a greeting (e.g., "hi," "hello"), respond with a friendly greeting message.
    
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
"""


    model = ChatGroq(groq_api_key=groq_api_key,model_name=os.environ["MODEL_NAME"])
    from langchain.chains.combine_documents import create_stuff_documents_chain
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = create_stuff_documents_chain(model, prompt)
    # chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def get_faq_response(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain.invoke(
        {"context": docs, "question": user_question}
        , return_only_outputs=True)
    print("RES : ", response)
    res = response.split(":")
    if len(res)>1:
        return ''.join(res[1::])
    return response

print(get_faq_response("WHat should i do if my payment is declined?"))
print(get_faq_response("Who is Sharukh Khan?"))
