from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyMuPDFLoader,
    DirectoryLoader,
)

import os
# -----------------------------------
# Create a LangChain Document Object
# -----------------------------------
doc = Document(
    page_content="This is the story of a boy who lived in Bengaluru and explored as much as he can.",
    metadata={
        "source": "life.txt",
        "author": "sumit",
        "pages": "1",
        "date_created": "2026",
    },
)

print(doc)

# -----------------------------------
# Create folders
# -----------------------------------
os.makedirs("data/text_files", exist_ok=True)
os.makedirs("data/pdf", exist_ok=True)

# -----------------------------------
# Create sample text file
# -----------------------------------
sample_texts = {
    "data/text_files/python_intro.txt": """
Python is a high-level, interpreted, and beginner-friendly programming language.
"""
}

# -----------------------------------
# Write text files
# -----------------------------------
for filepath, content in sample_texts.items():
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    print("Sample text file created!")

# -----------------------------------
# Load text file
# -----------------------------------
loader = TextLoader(
    "data/text_files/python_intro.txt",
    encoding="utf-8"
)

document = loader.load()

print("\nLoaded Text Documents:\n")
print(document)

# -----------------------------------
# Load PDFs from directory
# -----------------------------------
dir_loader = DirectoryLoader(
    "data/pdf",
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader,
    show_progress=True
)

pdf_load = dir_loader.load()


print("\nLoaded PDF Documents:\n")
for doc in pdf_load:
    print(doc.metadata)
print(len(pdf_load))
