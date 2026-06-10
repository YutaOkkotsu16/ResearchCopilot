"""parser.py — PDF → raw text.

First step of the ingestion pipeline. Turns an uploaded PDF into text while
preserving page numbers, so downstream chunks can carry page-level citations
(required by the answer generator in the QA graph).

This module only *extracts*; cleaning and chunking happen in later steps.
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

load_dotenv()

#Loading the PDF document using PyPDFLoader


def load_pdf():
    """Loads the PDF document and returns a list of documents with page content and metadata."""
    docs = []
    for pdfs in os.listdir(os.getenv("PDF_PATH")):
        if pdfs.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(os.getenv("PDF_PATH"), pdfs))
            docs.extend(loader.load())
    return docs


#Chunking the text using RecursiveCharacterTextSplitter

def split_text(docs):
    """Splits the text into chunks while preserving page numbers and source.

    Each chunk carries both its page number and its source filename, so that
    with multiple documents indexed we can tell papers apart, cite them
    correctly, and filter retrieval to a specific document.
    """
    # Create a list to hold the chunks
    chunks = []

    # One splitter for all docs (no need to rebuild it per document).
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Iterate through the documents and split them into chunks
    for doc in docs:
            text = doc.page_content
            page_number = doc.metadata['page']
            # PyPDFLoader stores the full path in 'source'; keep just the filename.
            source = os.path.basename(doc.metadata.get('source', 'unknown'))

            # Split the text into chunks
            doc_chunks = text_splitter.split_text(text)

            # Add page + source metadata to each chunk and append to the list
            for chunk in doc_chunks:
                chunks.append({
                    'text': chunk,
                    'metadata': {
                        'page': page_number,
                        'source': source,
                    }
                })

    return chunks


def main():
    """Main function to execute the parsing and chunking."""
    # Load the PDF document
    docs = load_pdf()
    
    # Split the text into chunks
    chunks = split_text(docs)
    
    # Print the chunks with their metadata
    for chunk in chunks:
        print(f"Page: {chunk['metadata']['page']}, Text: {chunk['text'][:100]}...")  # Print first 100 characters of each chunk

if __name__ == "__main__":
    main()