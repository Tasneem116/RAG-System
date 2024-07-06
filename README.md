# Retrieval-Augmented Generation (RAG) with OpenAI and Elasticsearch

This project demonstrates a Retrieval-Augmented Generation (RAG) approach for question-answering tasks using OpenAI's GPT-4 and Elasticsearch. 

## Features

- **Data Loading:** Loads documents from a text file for processing.
- **Document Splitting:** Splits loaded documents into smaller chunks for efficient storage and retrieval.
- **Embeddings:** Creates embeddings for document chunks using OpenAI.
- **Elasticsearch Integration:** Stores document chunks and embeddings in Elasticsearch for fast retrieval.
- **Question-Answering Pipeline:** Uses a RAG pipeline to retrieve relevant document chunks, format the query, and generate answers using GPT-4.
- **Customizable Templates:** Supports customizable question-answering templates for flexible query handling.
