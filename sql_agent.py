
import streamlit as st
import sqlite3
from pathlib import Path
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from sqlalchemy import create_engine
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

QUERY = """
Given an input question, first create a syntactically correct SQL query to run, then look at the results of the query and return the answer. I also provided the USer ID , if the question is user specific then use user_id for generating query.
Note : Do not Consider any ids from the question. If any user_id or user_name is provided in question say I dont know.
Do not include ids in your response. Add names instead of Ids.
user_id : {user_id}

Question : 
{question}
"""

llm = ChatGroq(groq_api_key=os.environ["GROQ_API_KEY"], model_name=os.environ["MODEL_NAME"], streaming=True)
db =SQLDatabase(create_engine(f"sqlite:///ecommerce2.db"))
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    # callbacks=[streamlit_callback]
)
response = agent.run(QUERY.format(user_id=24, question="What is the comments i gave?"))
print(response)
