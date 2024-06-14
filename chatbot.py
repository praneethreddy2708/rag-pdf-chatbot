from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from langchain_community.callbacks.manager import get_openai_callback
import os
import pandas as pd
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
import numpy as np
from difflib import SequenceMatcher
from langchain import hub
from langchain_openai import ChatOpenAI

# Initialize the LangSmith evaluator
grade_prompt_answer_accuracy = hub.pull("langchain-ai/rag-answer-vs-reference")


def answer_evaluator(run, example) -> dict:
    input_question = example["input_question"]
    reference = example["output_answer"]
    prediction = run["answer"]

    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
    answer_grader = grade_prompt_answer_accuracy | llm

    score = answer_grader.invoke({
        "question": input_question,
        "correct_answer": reference,
        "student_answer": prediction
    })

    score = score["Score"]

    return {"key": "answer_v_reference_score", "score": score}


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
    semantic_match = 1 if max_similarity > 0.1 else 0

    combined_match = max(exact_match, semantic_match)
    return combined_match, exact_match, semantic_match


def count_tokens(text):
    return len(text.split())


def main():
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = 'hh'
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")

    # Load the CSV database for evaluation
    try:
        csv_data = pd.read_csv('csv')
        st.write("CSV file loaded successfully.")
        st.write(csv_data.head())  # Display the first few rows of the CSV file for debugging
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return

    # Ensure the CSV has the correct columns
    if 'question' not in csv_data.columns or 'answer' not in csv_data.columns:
        st.error("CSV file must contain 'question' and 'answer' columns.")
        return

    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if "questions" not in st.session_state:
        st.session_state["questions"] = []
    if "answers" not in st.session_state:
        st.session_state["answers"] = []
    if "evaluation" not in st.session_state:
        st.session_state["evaluation"] = []

    if pdf is not None:
        text = extract_text_from_pdf(pdf)
        chunks = split_text(text)

        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        memory = ConversationBufferMemory()
        user_question = st.text_input("Ask a question about your PDF:")

        if user_question:
            # Generate response from the document
            docs = knowledge_base.similarity_search(user_question)
            if len(docs) == 0:
                response = "Ask a question related to the document."
                st.write(response)
            else:
                context_length = 4097 - 256
                selected_docs = []
                current_length = count_tokens(user_question)

                for doc in docs:
                    doc_length = count_tokens(doc.page_content)
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

                st.write(response)

                memory.save_context({"input": user_question}, {"output": response})

            st.session_state["questions"].append(user_question)
            st.session_state["answers"].append(response)

            # Evaluate the generated response against the chunks of the document
            combined_match, exact_match, semantic_match = evaluate_interaction(chunks, response)
            st.session_state["evaluation"].append((combined_match, exact_match, semantic_match))

            # Metrics lists for aggregated evaluation
            y_true = [1] * len(st.session_state["evaluation"])
            y_pred = [eval[0] for eval in st.session_state["evaluation"]]

            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)

            st.write(f"Model Accuracy: {accuracy:.4f}")
            st.write(f"Model F1 Score: {f1:.4f}")
            st.write(f"Model Precision: {precision:.4f}")
            st.write(f"Model Recall: {recall:.4f}")

            # Check if the question is present in the sample database and evaluate accordingly
            sample_data = csv_data[csv_data['question'] == user_question]
            if not sample_data.empty:
                correct_answer = sample_data.iloc[0]['answer']

                eval_example = {
                    "input_question": user_question,
                    "output_answer": correct_answer
                }
                eval_run = {"answer": response}
                eval_result = answer_evaluator(eval_run, eval_example)
                st.write(f"Evaluation Result for your question: {eval_result['score']}")

                # Evaluate against the user's original document and the retrieved documents
                user_doc_eval_example = {
                    "input_question": user_question,
                    "output_answer": text
                }
                user_doc_eval_result = answer_evaluator({"answer": response}, user_doc_eval_example)
                st.write(f"Evaluation Result against user document: {user_doc_eval_result['score']}")

                retrieved_docs_text = " ".join([doc.page_content for doc in selected_docs])
                retrieved_doc_eval_example = {
                    "input_question": user_question,
                    "output_answer": retrieved_docs_text
                }
                retrieved_doc_eval_result = answer_evaluator({"answer": response}, retrieved_doc_eval_example)
                st.write(f"Evaluation Result against retrieved documents: {retrieved_doc_eval_result['score']}")

            else:
                st.write("Question not found in the sample database.")

                # Evaluate against the user's original document and the retrieved documents
                user_doc_eval_example = {
                    "input_question": user_question,
                    "output_answer": text
                }
                user_doc_eval_result = answer_evaluator({"answer": response}, user_doc_eval_example)
                st.write(f"Evaluation Result against user document: {user_doc_eval_result['score']}")

                retrieved_docs_text = " ".join([doc.page_content for doc in selected_docs])
                retrieved_doc_eval_example = {
                    "input_question": user_question,
                    "output_answer": retrieved_docs_text
                }
                retrieved_doc_eval_result = answer_evaluator({"answer": response}, retrieved_doc_eval_example)
                st.write(f"Evaluation Result against retrieved documents: {retrieved_doc_eval_result['score']}")


if __name__ == '__main__':
    main()
