from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import os
import openai
import sqlite3
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from difflib import SequenceMatcher


# Initialize the database
def init_db():
    conn = sqlite3.connect('rag_database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS qa_pairs
                 (question TEXT, answer TEXT, timestamp DATETIME)''')
    conn.commit()
    conn.close()


def save_qa_pair(question, answer):
    conn = sqlite3.connect('rag_database.db')
    c = conn.cursor()
    timestamp = datetime.now()
    c.execute("INSERT INTO qa_pairs (question, answer, timestamp) VALUES (?, ?, ?)", (question, answer, timestamp))
    conn.commit()
    conn.close()


def get_answer(question):
    conn = sqlite3.connect('rag_database.db')
    c = conn.cursor()
    c.execute("SELECT answer FROM qa_pairs WHERE question = ?", (question,))
    answer = c.fetchone()
    conn.close()
    return answer[0] if answer else None


def extract_text_from_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def split_text(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_text(text)


def semantic_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def evaluate_interaction(doc_chunks, predicted_answer):
    predicted_answer_normalized = predicted_answer.strip().lower()

    similarities = [semantic_similarity(chunk.strip().lower(), predicted_answer_normalized) for chunk in doc_chunks]
    max_similarity = max(similarities)

    exact_match = 1 if max_similarity == 1.0 else 0
    semantic_match = 1 if max_similarity > 0.1 else 0  # Fine-tune threshold based on evaluation

    combined_match = max(exact_match, semantic_match)
    return combined_match, exact_match, semantic_match


def count_tokens(text):
    # Approximate token count for simplicity (actual token count may vary)
    return len(text.split())


def main():
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = 'cc'
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")

    # Initialize the database
    init_db()

    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if "questions" not in st.session_state:
        st.session_state["questions"] = []
    if "answers" not in st.session_state:
        st.session_state["answers"] = []
    if "evaluation" not in st.session_state:
        st.session_state["evaluation"] = []

    # extract the text
    if pdf is not None:
        text = extract_text_from_pdf(pdf)

        # split into chunks
        chunks = split_text(text)

        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Ensure embeddings and chunks are created correctly
        #print(f"Number of chunks: {len(chunks)}")
        #print(f"First chunk: {chunks[0]}")

        # Initialize conversation memory
        memory = ConversationBufferMemory()

        # show user input
        user_question = st.text_input("Ask a question about your PDF:")

        if user_question:
            # Check if the question has already been asked
            existing_answer = get_answer(user_question)
            if existing_answer:
                response = existing_answer
                st.write(response)  # Display the existing answer
            else:
                docs = knowledge_base.similarity_search(user_question)

                # Ensure the combined tokens in the prompt are within the model's limit
                context_length = 4097 - 256  # Reserve 256 tokens for the completion
                selected_docs = []
                current_length = count_tokens(user_question)

                for doc in docs:
                    doc_length = count_tokens(doc.page_content)  # Use page_content for token count
                    if current_length + doc_length <= context_length:
                        selected_docs.append(doc)
                        current_length += doc_length
                    else:
                        break

                llm = OpenAI()
                chain = load_qa_chain(llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=selected_docs, question=user_question)
                    print(cb)
                save_qa_pair(user_question, response)  # Save the new Q&A pair
                st.write(response)

                # Update conversation memory
                memory.save_context({"input": user_question}, {"output": response})

            st.session_state["questions"].append(user_question)
            st.session_state["answers"].append(response)

            # Evaluate the interaction
            combined_match, exact_match, semantic_match = evaluate_interaction(chunks, response)
            st.session_state["evaluation"].append((combined_match, exact_match, semantic_match))

            accuracy = np.mean([eval[0] for eval in st.session_state["evaluation"]])
            f1 = f1_score([1] * len(st.session_state["evaluation"]),
                          [eval[0] for eval in st.session_state["evaluation"]])

            st.write(f"Model Accuracy: {accuracy:.4f}")
            st.write(f"Model F1 Score: {f1:.4f}")


if __name__ == '__main__':
    main()
