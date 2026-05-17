from langchain_core.documents import Document
from langchain_community.document_loaders import(
    TextLoader,
    PyMuPDFLoader,
    DirectoryLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

#loaded the document

dir_loader = DirectoryLoader(
    "data/pdf",
    glob ="**/*.pdf",
    loader_cls=PyMuPDFLoader,
    show_progress=True
)
pdf_load = dir_loader.load()

#for doc in pdf_load:
    #print(doc.metadata)

#creating the text Spitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50,
    separators = ["\n","\n\n"," ",""]
)

chunks = text_splitter.split_documents(pdf_load)

#print(chunks[0])
print(len(chunks))

print(chunks[1].metadata)

class EmbeddingManager:
    """ Handles document  Embedding generation  using the Sentence transformers"""
    def __init__(self, model_name:str="all-MiniLm-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the Sentnce transformer model"""
        try:
            print(f"Loading embedding model:{self.model_name}")
            self.model= SentenceTransformer(self.model_name)
            print(f"Model Loaded Successfully :Embedding Dimension :{self.model.get_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embedding(self, texts:List[str]):
        """Generate Embeddings for a list of text

        Args: texts List of text substring to embed

        Return: numpy array of embeddings with shape (len(text),embedding_dim)
        """
        if not self.model:
            raise ValueError("Model not loaded")
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts,show_progress_bar=True)
        print(f"Generated embddings with shape: {embeddings.shape}")
        return embeddings

embedding_manager = EmbeddingManager()
# texts = [doc.page_content for doc in pdf_load]
# embedding_manager.generate_embedding(texts)

class VectorStore:
    """Manages vector embedding in the chromadb"""
    def __init__(self, collection_name:str="pdf_documents", persist_directory:str="data/vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()
    
    def _initialize_store(self):
        """Initialize chromadb client and collection"""
        try:
            #create a persistent chromadb client
            os.makedirs(self.persist_directory,exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            self.collection = self.client.get_or_create_collection(
                name = self.collection_name,
                metadata = {"description":"Metadata for RAG Embedding"}
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documnets in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise
    def add_documents(self,documents:List[Any],embeddings:np.array):
        """Add documents and their embedding tot he vector store 
        Args:
                documents: List of Langchain documents
                embeddings: Corresponding embeddings for the documents
        """
        if len(documents)!=len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        print(f"Adding {len(documents)} documents to vector store...")

        #prepare data for chromadb
        ids=[]
        metadatas=[] 
        documents_texts=[]
        embeddings_list=[]
        
        for i , (doc,embeddings) in enumerate(zip(documents,embeddings)):
            #Generae unique id
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            #prepare metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content']= len(doc.page_content)
            metadatas.append(metadata)

            documents_texts.append(doc.page_content)
            embeddings_list.append(embeddings.tolist())

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas = metadatas,
                documents = documents_texts
            )
            print(f"Successfully added {len(documents)} documents tp vector store")
            print(f"Total documents in collection:{self.collection.count()}")

        except Exception as e:
            print(f"Error adding to vector store: {e}")
            raise
vectorStore = VectorStore()
vectorStore


## convert the text into embeddings

texts=[doc.page_content for doc in chunks]

##Generae the embeddings

embeddings=embedding_manager.generate_embedding(texts)

##store in vectordb
vectorStore.add_documents(chunks,embeddings)


class RAGretriever:
    """Hanndles query based retrival from vector store"""
    def __init__(self, vector_store:VectorStore , embedding_manager:EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self,query:str, top_k:int=5,score_threshold:float=0.0)->List[Dict[str,Any]]:

        """ Retrieve relevant for a query
        query: The search query
        top_k: Number of top k results(chunks) to return after the similarity search
        score_threshold: Minimum similarity score threshold
        Returns List of dictionaries containing retrieved document and metadata"""


        print(f"Retrieveing document for searh query: {query}")
        print(f"Top_K: {top_k} , Threshold Score:{score_threshold}")

        #generate query embedding
        query_embedding = self.embedding_manager.generate_embedding([query])[0]

        #Search in vector Store
        try:
            results = self.vector_store.collection.query(
                query_embeddings = [query_embedding.tolist()],
                n_results = top_k
            )
            #process results
            retrieved_docs=[]

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata,distance) in enumerate(zip(ids,documents,metadatas,distances)):
                    similarity_score = 1 - distance
                    if similarity_score>=score_threshold:
                        retrieved_docs.append({
                            'id':doc_id,
                            'content':document,
                            'metadata':metadata,
                            'similarity_score':similarity_score,
                            'distance':distance,
                            'rank':i+1
                        })
                print(f"Retrieved {len(retrieved_docs)} documents after filtering")
            else:
                print("No document found")
            return retrieved_docs
        
        except Exception as e:
            print(f"error during retrieval {e}")
            return []
groq_Api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=groq_Api_key,
    model="llama-3.1-8b-instant",
    temperature=0.1,
    max_tokens=1024
)
def rag_qa(query: str , retriever:RAGretriever, llm , top_k=3):
    results = retriever.retrieve(query,top_k=top_k)
    context = "\n\n".join([doc['content'] for doc in results]) if results else ""
    if not context:
        return ""
    
    ## generate the answer using the groq LLM
    prompt ="""Use the folllowing context to answer the question concisely.
    {context}
    Question: {query}
     Answer:
    """
    response = llm.invoke([prompt.format(context=context,query=query)])
    return response.content
retriever = RAGretriever(vectorStore, embedding_manager)

answer = rag_qa(
    "Who wrote attention is all you need?",
    retriever,
    llm
)

print(answer)