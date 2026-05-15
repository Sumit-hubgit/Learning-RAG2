from langchain_community.document_loaders import (
    DirectoryLoader,
    PyMuPDFLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
directory_load  = DirectoryLoader(
    "data/pdf",
    glob ="**/*.pdf",
    loader_cls=PyMuPDFLoader,
    show_progress=True
)
doc = directory_load.load()  #  it returns list of documents where each document contains the metadata and the page_content
# print(type(doc))
# for content in doc:
#     print(content.metadata)

text_Splitter  = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 100,
    separators = ["\n","\n\n"," ",""]
)
#split_documents() is used to split LangChain Document objects into smaller chunks. It takes a list of documents and returns a new list of smaller Document chunks.
chunks = text_Splitter.split_documents(doc)

print(f"Chunk metadata: {chunks[0].metadata} \n Chunks content {chunks[0].page_content}")

model  = SentenceTransformer("all-MiniLM-L6-v2")
print(model.get_embedding_dimension())

texts = [ doc.page_content for doc in chunks]
encodings = model.encode(texts)
print(encodings)
