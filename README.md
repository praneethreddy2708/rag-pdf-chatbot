# rag-pdf-chatbot
rag pdf chatbot


This Python application enables you to load a PDF and ask questions about its content using natural language. It leverages a Large Language Model (LLM) to generate responses based on your PDF. The LLM will only answer questions related to the document.

# How It Works
The application processes the PDF by dividing the text into smaller chunks suitable for the LLM. It utilizes OpenAI embeddings to create vector representations of these chunks. When you ask a question, the application identifies chunks semantically similar to your query and feeds them to the LLM to generate an appropriate response.

Streamlit is used to create the graphical user interface (GUI), and Langchain handles the interaction with the LLM.

# Usage
To use the application, execute the main.py file with the Streamlit CLI (ensure Streamlit is installed):

```bash
streamlit run app.py

