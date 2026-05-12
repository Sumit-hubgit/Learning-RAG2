from langchain_core.documents import Document
from langchain_community.document_loaders import(
    TextLoader,
    PyMuPDFLoader,
    DirectoryLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter


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
    chunk_overlap = 50
)

chunks = text_splitter.split_documents(pdf_load)

#print(chunks[0])
print(len(chunks))

#print(chunks[1].metadata)
