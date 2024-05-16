import os
import google.generativeai as genai

import streamlit as st
from langchain.vectorstores import Chroma
import tempfile
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders.csv_loader import CSVLoader
from IPython.display import Markdown
from langchain_experimental.agents import create_csv_agent
from typing import TextIO
import time

os.environ['GOOGLE_API_KEY'] = "<GoogleAPT token>"
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])


def get_gemini_response(question):
    llm_model = genai.GenerativeModel('gemini-pro')
    response = llm_model.generate_content(question)
    return response.text


def get_without_index_response(file: TextIO, query: str) -> str:
    """
    Returns the answer to the given query by querying a CSV file.

    Args:
    - file (str): the file path to the CSV file to query.
    - query (str): the question to ask the agent.

    Returns:
    - answer (str): the answer to the query from the CSV file.
    """

    agent = create_csv_agent(ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.9), file, verbose=False)

    response = agent.invoke(query)
    if not response.get('output', ''):
        return "Sorry, I couldn't find an answer to that question."
    return response.get('output', '')


def display_response(text, frame_title):
    with st.container():
        st.markdown(f"## {frame_title}")
        st.write(text)


def main():
    st.title("Chat with CSV using Gemini Pro")

    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        csv_loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
            'delimiter': ','
        })
        # load data into csv loader
        data = csv_loader.load()
        embedding_function = SentenceTransformerEmbeddings(
            model_name="sentence-transformers/bert-large-nli-max-tokens")
        # load vector db file
        persist_directory = '/content/VectorStore'
        if len(os.listdir(persist_directory)) == 0:
            db = Chroma.from_documents(data, embedding_function, persist_directory=persist_directory)
            db.persist()
        vectorDB = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.9)

        chain = (
                {"context": vectorDB.as_retriever(), "question": RunnablePassthrough()}
                | prompt
                | model
                | StrOutputParser()
        )

        # initialize chat interface
        user_input = st.text_input("User message:")
        print(user_input)

        if user_input:
            start_time = time.time()
            response_a = chain.invoke(user_input)
            end_time = time.time()
            response_a_time = end_time - start_time
            start_time = time.time()
            response_b = get_without_index_response(tmp_file_path, user_input)
            end_time = time.time()
            response_b_time = end_time - start_time
            display_response(response_a, "Index based response:")
            display_response(response_b, "Without-index based response:")
            with st.container():
                st.subheader("Response Times")
                st.line_chart({
                    'Index-based': response_a_time,
                    'Without-index': response_b_time
                })



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# sample questions
# can you tell me the type of lung cancers fron the report
# which patients diagnoised on left kidney
# the type of cancer available in lymph node
# colon cancer was found in which type of specimens
# Renal cell carcinoma (cancer) was identified in which part of kidney ?
