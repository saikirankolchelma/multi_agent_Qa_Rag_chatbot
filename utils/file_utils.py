import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPowerPointLoader, CSVLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import config

def get_file_loader(file_path):
    """Returns the appropriate LangChain document loader for a given file extension."""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    loaders = {
        '.pdf': PyPDFLoader, '.pptx': UnstructuredPowerPointLoader, '.csv': CSVLoader,
        '.docx': UnstructuredWordDocumentLoader, '.txt': TextLoader, '.md': TextLoader
    }
    return loaders.get(ext)

def load_and_split_document(file_path):
    """Loads a document using the correct loader and splits it into chunks."""
    loader_class = get_file_loader(file_path)
    if not loader_class:
        return []

    loader = loader_class(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    return text_splitter.split_documents(docs)