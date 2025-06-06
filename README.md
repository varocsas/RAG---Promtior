# Solution deployed using Render

https://rag-promtior.onrender.com




## Features

- **Data retrieval, embedding, and storage**: Scrapes, processes and embeds contextual information from multiple potential formats-
- **Retrieval-Augmented Generation (RAG)**: Uses a hybrid approach for question answering with tokenizing, embedding, and chunking.
- **Streamlit Interface**: User-friendly web interface for inputting URLs and questions.

## Tools

- **Orchestration (chaining different APIs and providers)**: LangChain
- **Guardrails (security)**: NeMo-Guardrails
- **Monitoring (to migrate from dev to prod)**: Langsmith
- **Retrieval (querying)**: Hugging Face
- **Vector Database (storage)**: Pinecone
- **Generation (LLM)**: Groq, Llama3
- **Deployment (may run locally or be deployed to Render)**
- **

## Workflow Architecture

1. **Data Loading**: Accepts various data formats such as PDF, CSV, or URL links.
2. **Chunking**: Uses text splitters to divide content into manageable chunks.
3. **Embedding**: Generates embeddings from text using Hugging Face models.
4. **Vector Store**: Stores embeddings in Pinecone for efficient retrieval.
5. **User Query (Input)**: Receives the user's question.
6. **Input Check**: Uses Guardrails (NeMo-Guardrails) to ensure the question is within the scope of the dataset.
7. **Retriever**: Retrieves relevant chunks from Pinecone based on the user’s query.
8. **Generator**: Groq or Llama3 generates responses based on retrieved information.
9. **Monitoring**: Langsmith monitors responses to ensure output quality.
10. **Output**: The final answer is presented to the user.

# To run the solution locally

```sh

git clone git@github.com:varocsas/RAG--Promtior.git
python -m venv RAG--Promtior-venv
source RAG--Promtior-venv/bin/activate
pip install -r requirements.txt
streamlit run app.py

```

## Usage
1. Press RunChallenge to answer the challenge questions using Promtior website and the provided PDF file
2. The RAG pipeline retrieves relevant information and generates the answers.
 --
3. To create a new context based on anoter URL or PDF file just type de information in the URL textBox. Previous context will be deleted and a new one will replace it.
4. Ask Questions based on the website content.
5. The RAG pipeline retrieves relevant information and generates answers.
