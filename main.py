from typing import Literal, List
from langchain.schema import Document
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from faq_agent import get_faq_response, get_pdf_text, get_text_chunks, get_vector_store
from sql_agent import agent
from langgraph.graph import END, StateGraph, START
import streamlit as st
from transformers import pipeline
import sqlite3



# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["database", "vectorstore"] = Field(
        ...,
        description="Given a user question, choose to route it to the database or vectorstore",
    )

# Context manager class
class SessionContext:
    def __init__(self):
        self.context = {}

    def update_context(self, user_id, new_data):
        if user_id not in self.context:
            self.context[user_id] = new_data
        else:
            self.context[user_id].update(new_data)

    def get_context(self, user_id):
        return self.context.get(user_id, {})

context_manager = SessionContext()

# Function to generate response considering the sentiment
def sentiment_adjusted_response(query):
    sentiment = sentiment_analyzer(query)
    tone = 'positive' if sentiment[0]['label'] == 'POSITIVE' else 'neutral'
    response = response_generator.generate(query, tone=tone)
    return response.replace('from our database', '').replace('according to our records', 'I found that')

# Function to handle queries and maintain context
def handle_query(user_id, query):
    context = context_manager.get_context(user_id)
    response = sentiment_adjusted_response(query)
    context_manager.update_context(user_id, {'last_query': query, 'last_response': response})
    return response


# LLM with function call
from langchain_groq import ChatGroq
import os
groq_api_key=os.environ["GROQ_API_KEY"]
# os.environ["GROQ_API_KEY"]="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"
llm=ChatGroq(groq_api_key=groq_api_key,model_name=os.environ["MODEL_NAME"])
structured_llm_router = llm.with_structured_output(RouteQuery)

# Prompt
system = """You are a supervisor tasked with managing a conversation between the following workers: [database, vectorstore]. Given the user request, respond with the worker to act next. 
If the question specifies a need for an SQL query to retrieve data, direct it to the database worker, who will write and execute the query. database contains data related to products, users, cart, orders, and ratings.
For all other inquiries, respond as the vector store worker, which contains FAQs and enquiries. Each worker will perform a task and respond with their results and status."""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router

class SessionContext:
    def __init__(self):
        self.context = {}

    def update_context(self, user_id, new_data):
        if user_id not in self.context:
            self.context[user_id] = new_data
        else:
            self.context[user_id].update(new_data)

    def get_context(self, user_id):
        return self.context.get(user_id, {})

context_manager = SessionContext()

## Graph
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    response = get_faq_response(question)
    return {"documents": response, "question": question}

def db_search(state):
    """
    wiki search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    QUERY = """
       You are a friendly and helpful customer service representative named Retailia at our online store. A customer with the ID {user_id} has asked you the following question: '{question}'. 
    Utilise your knowledge of retail operations, products, and common customer inquiries to formulate a response. 

    When responding:
    Prioritise clear, concise answers in natural, conversational language. Avoid technical jargon and SQL-specific terms. 
    Personalise your response by referencing the customer's previous interactions and order history (if applicable). However, never reveal sensitive information like the customer's full name, address, or payment details.
    If the customer's question requires accessing their specific data, construct a syntactically correct SQL query to retrieve the necessary information. The database schema includes tables for products, users, cart, orders, and ratings.
    If the customer expresses frustration or concern, empathise with their feelings and offer reassurance. 
    If you cannot answer their question or need more information, politely ask clarifying questions.
    If you are not able to answer their question at all, offer helpful alternatives.
    Always end your response with a helpful suggestion or a follow-up question to continue the conversation. 
    For instance, if a customer asks "What is the status of my order?", you could respond with:

    "Let me check that for you. Could you please provide your order number?"
    Once the customer provides their order number, you can query the database and provide a detailed update on their order status.      

Remember, your goal is to create a positive customer service experience that leaves the customer feeling valued and satisfied. 
    user_id : {user_id}

    Question : 
    {question}
"""
    question = state["question"]
    print(question)

    # Wiki search
    response = agent.run(QUERY.format(user_id=st.session_state['user_id'], question=question))
    if response=="I don't know":
        response = get_faq_response(question)

    return {"documents": response, "question": question}

### Edges ###


def route_question(state):
    """
    Route question to wiki search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---", state)
    question = state["question"]
    source = question_router.invoke({"question": question})
    print(source,question)
    if source.datasource == "database":
        print("---ROUTE QUESTION TO SQL AGENT---")
        return "database"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO FAQ AGENT---")
        return "vectorstore"


workflow = StateGraph(GraphState)
# Define the nodes
workflow.add_node("database", db_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve

# Build graph
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "database": "database",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge( "retrieve", END)
workflow.add_edge( "database", END)
# Compile
app = workflow.compile()




import streamlit as st
import sqlite3

st.set_page_config(page_title="Retailia", page_icon="🤖", layout="centered", initial_sidebar_state='collapsed')

st.title("🤖 Retailia")

# Function to connect to the SQLite database
def get_db_connection():
    conn = sqlite3.connect('users.db')
    return conn

# Function to verify user credentials
def check_credentials(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT user_id FROM users WHERE username = ? AND password = ?', (username, password))
    user = cursor.fetchone()
    conn.close()
    return user

# User login form
if "logged_in" not in st.session_state:
    placeholder = st.empty()

    with placeholder.form("login"):
        st.markdown("#### Login")
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", placeholder="Enter your password", type="password")
        login_button = st.form_submit_button("Login")

        if login_button:
            user_id = check_credentials(username, password)
            if user_id:
                st.success("Login successful!")
                st.session_state['logged_in'] = True
                st.session_state['user_id'] = user_id
                st.session_state['username'] = username
                placeholder.empty()
            else:
                st.error("Invalid username or password.")

if "logged_in" in st.session_state and st.session_state['logged_in']:
    st.header(f"Welcome!")

    # Collapsible sidebar
    with st.sidebar.expander("Upload your PDF Files (FAQs)", expanded=False):
        pdf_docs = st.file_uploader("PDF Uploader", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing PDF..."):
                raw_text = get_pdf_text(pdf_docs)
            with st.spinner("Creating chunks for PDF..."):
                text_chunks = get_text_chunks(raw_text)
            with st.spinner("Creating Vector DB..."):
                get_vector_store(text_chunks)
                st.success("Done")

    st.header("Conversation Window")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Hello, I am Retailia. How can I assist you?"}]

 
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_query = st.chat_input(placeholder="Please type in your query")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)
        with st.chat_message("assistant"):
            response_container = st.container()
            with response_container:
                with st.spinner("Typing..."):
                    response = app.invoke({"question": user_query})
                response_container.markdown(response["documents"])
                print("RESPONSE : ", response)

            st.session_state.messages.append({"role": "assistant", "content": response["documents"]})
