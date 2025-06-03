

The objective of this project was to implement a **Retrieval-Augmented Generation (RAG)** chatbot that can answer user questions based on the content of the Promtior website. The solution leverages LangChain's modular architecture for orchestrating the pipeline, using web scraping, text chunking, semantic search via vector databases, and generative models to return accurate answers. The deployment was made accessible via a Streamlit interface and monitored with LangSmith.

**Key challenges addressed:**

- **Real-time web parsing**: Ensured accurate extraction of dynamic website content using BeautifulSoup.
    
- **Semantic coherence**: Introduced chunk overlap in text splitting to maintain contextual integrity.
    
- **Integration and orchestration**: Used LangChain to manage retrieval and generation, with additional guardrails to ensure prompt quality.
    
- **Deployment**: While LangServe was suggested, deployment was executed via Streamlit for rapid prototyping and will be migrated to LangServe if required.
    



|Component|Technology / Tool|Purpose|
|---|---|---|

## 0.1:  Architecture and Implementation

### 0.1.1:  Components and Technologies

|Component|Technology / Tool|Purpose|
|---|---|---|
|Scraper|`requests`, `BeautifulSoup`|Extract and clean HTML content from URLs|
|Chunking|`LangChain TextSplitter`|Divide text into overlapping semantic units|
|Embeddings|`sentence-transformers/all-MiniLM`|Generate dense vector representations of text|
|Vector Store|Pinecone|Perform fast vector similarity retrieval|
|Generation|Groq-hosted LLaMA3 / HuggingFace|Generate answers from retrieved context|
|Orchestration|LangChain|Manage full RAG pipeline execution|
|Monitoring|LangSmith|Log and inspect each processing step|
|Guardrails|NeMo-Guardrails|Validate and constrain user inputs|
|Interface|Streamlit|Interactive frontend for URL and query input|
|Deployment|Render (prototype), LangServe (planned)|Cloud deployment and scaling|

## 0.2:  RAG Pipeline Summary

1. **Website URL Submission**: The user initiates the process by entering a website URL through the Streamlit interface.
    
2. **Content Extraction**: The system fetches and parses the HTML, extracting only meaningful textual content while removing scripts, styles, and navigation elements.
    
3. **Text Segmentation**: The extracted text is divided into overlapping chunks using LangChain’s recursive splitter to ensure semantic continuity.
    
4. **Vector Embedding**: Each chunk is converted into a dense vector using a sentence-transformer model, enabling semantic similarity comparisons.
    
5. **Vector Indexing**: The embeddings are stored in a Pinecone vector database, allowing for efficient and scalable retrieval.
    
6. **User Query Submission**: The user enters a natural language question via the interface.
    
7. **Input Validation**: NeMo-Guardrails checks the query for appropriateness, scope, and coherence.
    
8. **Semantic Retrieval**: Using the query’s embedding, the system retrieves the most relevant document chunks from Pinecone.
    
9. **Response Generation**: A language model (e.g., LLaMA3 hosted by Groq) generates an answer based on the retrieved content.
    
10. **Pipeline Monitoring**: LangSmith logs every step of the process, supporting transparency, observability, and debugging.
    
11. **Answer Delivery**: The generated response is presented back to the user through the Streamlit UI.
    
12. **URL Input**: User provides a website URL via Streamlit.
    
13. **Content Extraction**: HTML is parsed, and relevant text is extracted.
    
14. **Text Chunking**: Text is split into overlapping segments.
    
15. **Embedding Generation**: Segments are embedded into vectors.
    
16. **Vector Indexing**: Embeddings are stored in Pinecone for retrieval.
    
17. **Question Input**: User enters a natural language question.
    
18. **Input Validation**: Guardrails verify question validity and scope.
    
19. **Similarity Search**: Relevant chunks are retrieved based on cosine similarity.
    
20. **Answer Generation**: LLM generates a coherent response from retrieved content.
    
21. **Monitoring**: LangSmith captures detailed logs of each pipeline step.
    
22. **Answer Output**: The final response is displayed in the UI.
    

---

  **Diagram**

The system architecture may be visually represented as follows:

- **Frontend**: Streamlit UI
    
- **Processing Chain**:
    
    - User URL → Scraper → Text Splitter → Embedder → Pinecone
        
    - User Question → NeMo Guardrails → Retriever → Generator → Answer
        
- **Oversight Layer**: LangSmith monitors and logs the complete execution flow
    

The diagram can be created using tools like Draw.io or Lucidchart and saved to `/doc/component_diagram.png`.

---

### 0.2.1:  Repository Structure

```
/
├── app.py                     # Streamlit interface
├── rag.py                     # Core RAG logic
├── requirements.txt           # Python dependencies
├── README.md                  # Project usage and overview
└── /doc
    ├── technical_report.md    # This document
    └── component_diagram.png  # Architecture diagram (to be added)
```

---

### 0.2.2:  Sample Questions

**Q: What services does Promtior offer?**  
_A: Promtior provides consulting services in digital transformation and applied AI, specializing in generative architectures such as Retrieval-Augmented Generation (RAG) to help organizations remain competitive in a rapidly evolving technological landscape._

**Q: When was the company founded?**  
_A: Promtior was established in May 2023._

---

### 0.2.3:  Final Considerations

This project demonstrates the feasibility and effectiveness of an adaptable RAG-based chatbot capable of synthesizing knowledge from arbitrary web content. Its modular structure enables future extensions, including domain adaptation, multilingual support, and integration with enterprise APIs.

The prototype uses **Streamlit** to support rapid development and user interaction testing. For production, the same pipeline is compatible with **LangServe**, which would enable scalable, API-driven deployment. The inclusion of **LangSmith** and **NeMo-Guardrails** highlights a commitment to transparency, safety, and user trust—principles essential for real-world AI systems.

**Future enhancements may include:**

- Multi-site content ingestion and indexing
    
- LangServe deployment with OpenAPI and Swagger documentation
    
- Customizable guardrails and compliance policies
    
- Continuous learning via logged user feedback for model fine-tuning---

- LangServe deployment was initially considered but deprioritized due to rapid prototyping constraints and the need for live iteration; Streamlit offered a quicker testing loop.
    
- LangSmith allowed us to monitor and analyze generated responses in a production-like setup.
    
- The architecture is modular and ready to be adapted to LangServe or serverless cloud environments.


