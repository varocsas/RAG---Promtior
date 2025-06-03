# RAG — Promtior 🚀
*A Retrieval-Augmented Generation (RAG) pipeline for answering questions from website content.*

<p align="center">
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue"></a>
  <a href="https://streamlit.io/"><img alt="Streamlit" src="https://img.shields.io/badge/streamlit-app-brightgreen"></a>
  <img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey">
</p>

---

## Table of Contents
1. [Features](#features)
2. [Tooling](#tooling)
3. [Architecture](#architecture)
4. [Getting Started](#getting-started)
5. [Usage](#usage)
6. [Project Structure](#project-structure)
7. [Contributing](#contributing)
8. [License](#license)

---

## Features
- **Website Content Parsing** – Extracts and indexes text from a given URL (HTML, PDF, CSV).
- **Retrieval-Augmented Generation** – Combines similarity search with Llama 3 / Groq for accurate Q-A.
- **Guardrails** – Validates user queries and filters unsafe or out-of-scope requests.
- **Streamlit UI** – Clean two-column interface for URL input and interactive chat.

---

## Tooling
| Purpose            | Library / Service |
|--------------------|-------------------|
| Orchestration      | **LangChain**     |
| Safety Guardrails  | **NeMo-Guardrails** |
| Monitoring         | **LangSmith**     |
| Embeddings         | **Hugging Face**  |
| Vector DB          | **Pinecone**      |
| Generation         | **Groq**, **Llama 3** |
| Deployment         | **Render**        |

---

## Architecture

```mermaid
flowchart TD
    subgraph Data Ingestion
        A[Data Loading<br/>(URL / PDF / CSV)] --> B[Chunking]
        B --> C[Embeddings]
        C --> D[(Pinecone DB)]
    end

    subgraph Question Answering
        E[User Query] --> F[NeMo Guardrails]
        F --> G[Retriever<br/>(Pinecone)]
        G --> H[Generator<br/>(Llama 3 / Groq)]
        H --> I[Answer]
    end

    classDef store fill:#f9f,stroke:#333,stroke-width:1px;
    D:::store
    style I fill:#d0ffd0,stroke:#333,stroke-width:1px
