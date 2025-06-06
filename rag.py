import asyncio
import threading
import os
import time
import requests
from dotenv import load_dotenv, find_dotenv
from io import BytesIO
from PyPDF2 import PdfReader
import bs4
from bs4 import BeautifulSoup
from langchain import hub
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
# from urllib.parse import urlparse

        
# Set up async loop for Streamlit
if threading.current_thread().name == "ScriptRunner.scriptThread":
    try:
        asyncio.get_running_loop()

    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

load_dotenv(find_dotenv())  # .env config file






# TODO temp variable while we have config data hardcoded, 
# not needed when using env variables and/or config files
local_deploy = True  

# TODO move all these to a secret and create a shell script that defins the environment variables
# locally 

if local_deploy:
    os.environ["DATABASE_URL"] = "https://valet-p5mv8g8.svc.aped-4627-b74a.pinecone.io"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = "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"
    os.environ['LANGCHAIN_PROJECT'] = "default"
    os.environ['GROQ_API_KEY'] = "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" #groq_rag_key
    os.environ['PINECONE_API_KEY'] = "pcsk_2NHuPE_7q9BuQRTTn9ST8t3cYADhHsd7MZ5qzyhWWyebRoQcXiZb4rk5DGDb5bEthVDkVE"
    os.environ['LANGSMITH_API_KEY'] = "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"


# *********************
class RAG:
    
    def __init__(self):
                
        # TODO move all configuration parameters to config files or environment variables
        
        # no RAG context yet
        self.RAG_ready = False
        # database_url = os.environ.get("DATABASE_URL")
        
        # RAG and embeddings configuration
        self.vectorstore_index_name = "valet"   
        
        # Ideally, the chosen chunk_size allows each chunk capture exactly one concept. 
        # self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=768, chunk_overlap=128)
        self.model_name = "BAAI/bge-small-en-v1.5"  # Model for embeddings
        self.model_kwargs = {"device": "cpu"}       # CPU when running locally, GPU when remote
        self.encode_kwargs = {"normalize_embeddings": True}

        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs
            # query_instruction="Represent this sentence for searching relevant passages:"            
        )

        # Guardrails configuration
        from nemoguardrails import LLMRails, RailsConfig
        from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

        # We use llama3-70b-8192 running on Groq as LLM in Guardrails, 
        # but GuardRails is LLM-agnostic.
        self.groq_llm = ChatGroq(
            api_key=os.environ.get("GROQ_API_KEY"),
            model="llama3-70b-8192",
            temperature=0
        )  
        
        # Guardrails uses an LLM to evaluate texts that are part of a task within a flow
        # against some policy. 
        # Tasks and flows are defined in general config file (./config/config.yaml)
        # LLM prompts for each task are specified in ./config/prompts.yaml 
                  
        rails_config = RailsConfig.from_path("./config") 
        
        self.guardrails = RunnableRails(
            config = rails_config,
            llm = self.groq_llm 
        )  

        # RAG injection in user prompts setup  
        # TODO move to configuration file?
        # We use a standard prompt template stored in LangChain community (rlm/rag-prompt)
        # with injections in two fields, {context} and {question}
        self.rag_prompt = hub.pull(
            "rlm/rag-prompt",              
            api_key=os.environ.get("LANGSMITH_API_KEY")
        )

        # Pinecone setup (vector database server)
        self.pinecone_api_key = os.environ.get("PINECONE_API_KEY")        
        self.pc = Pinecone(os.environ.get("PINECONE_API_KEY"))  # create Pinecone object
        self.spec = ServerlessSpec(cloud="aws", region="us-east-1")
        self.index_name = self.vectorstore_index_name # TODO default index name move to the config file
        
        
    # Index for the RAG context in Pinecone
    def update_pinecone_index(self, index_name = None):
        
        if index_name is None:
            index_name = self.index_name    
          
        # if an index with this name already exists we delete it (we don't store previous RAG context)
        if index_name in self.pc.list_indexes().names():
            self.pc.delete_index(index_name)

        # create the new, empty index
        self.pc.create_index(
            name = index_name,
            dimension = 384, # must match the vertex dimension defined in Pinecone, TODO move to config file
            metric = "dotproduct",
            spec = self.spec 
        )

        # Wait until the index is ready
        while not self.pc.describe_index(index_name).status["ready"]:
            time.sleep(1)


    def warmup_query(self):
        # Executing a "warmup" query to solve synchronisation issues
        # due to the wrapper wrapper object that lazily deferring 
        # connecting to Pinecone until the first query.
        # The warmup does two important things:
	    #   1.	Forces the retriever to connect to the index and test its search interface, which ensures the vector store is usable and fully indexed.
	    #   2.	Triggers any internal state setup that LangChain or Pinecone client needs, avoiding lazy-load bugs.   
        docs = self.retriever.get_relevant_documents("What is Promtior?")
        print(f"Retrieved {len(docs)} docs")
        '''for i, doc in enumerate(docs):
            print(f"Doc {i}: {doc.page_content[:200]}")
        '''      

    # Create a new RAG context with the URL and PDF file of the coding challenge
    def setup_challenge (self):
        challenge_url = "https://www.promtior.ai/"
        pdf_filename = "./data/AI_Engineer-1.pdf"
        
        self.web_url = challenge_url
        
        self.update_pinecone_index()
        self.create_vectorstore() # new PineconeVectorStore        
        
        self.retriever = self.vectorstore.as_retriever() # a lightweight wrapper 
               
                
        # scrape the challenge URL
        content_url = self.load_web_into_string(challenge_url)        
               
        # add the challenge PDF
        content_pdf = self.load_pdf_into_string (pdf_filename)
        
        # update vectorstore
        self.add_string_to_vectorstore(content_pdf + content_url)
        
        # create a new LangChain
        self.create_retrieval_chain()   
        
        
        self.warmup_query()     
       
        
                      
        
        self.RAG_ready = True
        
        
        
        
    # Create a new RAG context from an URL pointing to a web or to a PDF file online
    def setup(self, web_url):
        self.web_url = web_url      # URL with the information for the RAG context
        
        self.update_pinecone_index()
        self.create_vectorstore()  # Create an new empty PineconeVectorStore
        self.retriever = self.vectorstore.as_retriever() # a lightweight wrapper 
                   

        # Now vectorstore is ready to receive .add_texts() or .add_documents()
        # we use add_texts for both html and pdfs, but get ahold of the actual data 
        # in slightly different ways, therefore we have two fuctions that return
        # a string with the info
        
        if web_url.strip().lower().endswith(".pdf"):  # if the url points to a pdf
            content = self.load_pdf_url_into_string(web_url)
            
        else:  # if it points to something else, we assume it's an html
            content = self.load_web_into_string(web_url)        
                
        # update vectorstore         
        self.add_string_to_vectorstore(content)
        
        # create a new LangChain
        self.create_retrieval_chain()
        
        
        
        self.warmup_query()
                
        self.RAG_ready = True
        
    
    # adds the info in content to self.vectorstore
    def add_string_to_vectorstore(self, content):
        chunks = self.text_splitter.split_text(content) # split text into chunks        
        embeddings = self.embeddings.embed_documents(chunks)  # embed the chunks
                
        # store chunks and embeddings (plus an arbitrary, sequential id) in the vectorstore
        self.vectorstore.add_texts(
            texts = chunks,
            embeddings = embeddings,
            ids=[f"{i}" for i in range(len(chunks))]
        )    
        
        print(f"Added {len(chunks)} chunks to vectorstore.")
        
        
        # print("First chunk:", chunks[0][:200])
        
        # Wait for Pinecone to make writes available
        print("Waiting for Pinecone to sync, just in case...")
        time.sleep(7) 
        print ("done.")
                
    # can I be prompted?
    def ready(self):
        return self.RAG_ready

    # can't be prompted until a new context is built
    def reset_pipeline (self):
        self.RAG_ready = False

    # returns all visible text in a webpage with BeautifulSoup
    def load_web_into_string(self, url):        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/112.0.0.0"
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        # prune DOM (i.e. remove TAGs that are not rendered or irrelevant)
        for tag in soup(["script", "style", "footer", "nav", "meta", "noscript"]):
            tag.decompose()
            
        raw_text = soup.get_text(separator="\n", strip=True) # everything in a string
        return raw_text

    
    # returns all the text from a local pdf with PdfReader
    def load_pdf_into_string(self, filename):                       
        # Load PDF
        reader = PdfReader(filename)

        #  Extract all text from each pages and join them into one single string
        raw_text = "\n".join(
            page.extract_text() for page in reader.pages if page.extract_text()
        )            
        
        return raw_text
    

   
    #returns all the text from a remote pdf with PdfReader
    def load_pdf_url_into_string(self, url):          
        # Download PDF
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/112.0.0.0"
        }
        response = requests.get(url, headers=headers)                    
        reader = PdfReader(BytesIO(response.content))

        #  Extract all text from each pages and join them into one single string
        raw_text = "\n".join(
            page.extract_text() for page in reader.pages if page.extract_text()
        )            
        return raw_text


    # Join the content of a set of LangChain docs in a single string object
    def format_docs(self, docs):
        docs_text = "\n\n".join(doc.page_content for doc in docs)
        # return f"Source URL: {self.web_url}\n\n{docs_text}"
        return (docs_text)


    # Create a new PineconeVectorStore
    def create_vectorstore(self):
        self.vectorstore = PineconeVectorStore(
            index_name = self.vectorstore_index_name,                 # from init
            embedding = self.embeddings,                              # from setup
            pinecone_api_key = self.pinecone_api_key                  # from init            
        )


    # Create a LangChain retrieval chain and store it as self.rag_chain
    def create_retrieval_chain(self):
        # update the retriever
        self.retriever = self.vectorstore.as_retriever()
     
        # define the actual chain
        self.rag_chain = (
            {
                "context": self.retriever | self.format_docs,  # the RAG context 
                "question": RunnablePassthrough()  # the users prompt to augment
            }
            | self.rag_prompt
            | self.groq_llm
            | StrOutputParser()
        )

        # Use guardrails to check the chain 
        self.rag_chain = self.guardrails | self.rag_chain 


    
    # Query (prompting) function
    # It creates the RAG chain if necessary
    # Precondition, RAG is ready
    def qa(self, query):                    
        return self.rag_chain.invoke(query)