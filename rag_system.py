#!/usr/bin/env python3
"""
Complete RAG System with Document Processing and Query Functionality
"""

import os
import sys
import uuid
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

# LangChain imports
from langchain_community.document_loaders import (
    TextLoader, 
    DirectoryLoader, 
    PyPDFLoader, 
    PyMuPDFLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ML and Vector DB imports
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

class RAGSystem:
    """Complete RAG system with document processing and retrieval"""
    
    def __init__(self, 
                 collection_name: str = "rag_documents",
                 persist_directory: str = "data/vector_store",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        
        # Initialize components
        self.embedding_model = None
        self.client = None
        self.collection = None
        self.text_splitter = None
        
        # Setup system
        self._setup_system()
    
    def _setup_system(self):
        """Initialize all system components"""
        try:
            self._load_embedding_model()
            self._initialize_vector_store()
            self._setup_text_splitter()
            print("✅ RAG system initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing RAG system: {e}")
            raise
    
    def _load_embedding_model(self):
        """Load the sentence transformer model"""
        try:
            print(f"🔄 Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            print(f"✅ Model loaded. Dimension: {self.embedding_model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            raise
    
    def _initialize_vector_store(self):
        """Initialize ChromaDB vector store"""
        try:
            print("🔄 Initializing vector store...")
            
            # Create directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Create persistent client
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "RAG document collection"}
            )
            
            doc_count = self.collection.count()
            print(f"✅ Vector store ready. Documents: {doc_count}")
            
        except Exception as e:
            print(f"❌ Error initializing vector store: {e}")
            raise
    
    def _setup_text_splitter(self):
        """Setup text splitter for document chunking"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        print("✅ Text splitter configured")
    
    def load_text_documents(self, directory_path: str) -> List[Document]:
        """Load text documents from directory"""
        try:
            print(f"🔄 Loading text documents from: {directory_path}")
            
            loader = DirectoryLoader(
                directory_path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'},
                show_progress=True
            )
            
            documents = loader.load()
            print(f"✅ Loaded {len(documents)} text documents")
            return documents
            
        except Exception as e:
            print(f"❌ Error loading text documents: {e}")
            return []
    
    def load_pdf_documents_robust(self, directory_path: str) -> List[Document]:
        """Load PDF documents with robust error handling"""
        try:
            print(f"🔄 Loading PDF documents from: {directory_path}")
            
            all_documents = []
            pdf_dir = Path(directory_path)
            
            # Find all PDFs
            pdf_files = list(pdf_dir.glob("**/*.pdf"))
            
            if not pdf_files:
                print("⚠️  No PDF files found")
                return []
            
            print(f"Found {len(pdf_files)} PDF files")
            
            for pdf_file in pdf_files:
                print(f"Processing: {pdf_file.name}")
                
                # Try PyMuPDFLoader first (more robust)
                try:
                    loader = PyMuPDFLoader(str(pdf_file))
                    documents = loader.load()
                    
                    # Add metadata
                    for doc in documents:
                        doc.metadata['source_file'] = pdf_file.name
                        doc.metadata['file_type'] = 'pdf'
                    
                    all_documents.extend(documents)
                    print(f"✅ Successfully loaded {len(documents)} pages")
                    
                except Exception as e1:
                    print(f"⚠️  PyMuPDFLoader failed: {e1}")
                    
                    # Fallback to PyPDFLoader
                    try:
                        loader = PyPDFLoader(str(pdf_file))
                        documents = loader.load()
                        
                        # Add metadata
                        for doc in documents:
                            doc.metadata['source_file'] = pdf_file.name
                            doc.metadata['file_type'] = 'pdf'
                        
                        all_documents.extend(documents)
                        print(f"✅ PyPDFLoader loaded {len(documents)} pages")
                        
                    except Exception as e2:
                        print(f"❌ Both loaders failed for {pdf_file.name}: {e2}")
                        continue
            
            print(f"✅ Total PDF documents loaded: {len(all_documents)}")
            return all_documents
            
        except Exception as e:
            print(f"❌ Error loading PDF documents: {e}")
            return []
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        try:
            print(f"🔄 Processing {len(documents)} documents...")
            
            if not documents:
                print("⚠️  No documents to process")
                return []
            
            chunks = self.text_splitter.split_documents(documents)
            print(f"✅ Split into {len(chunks)} chunks")
            
            # Show example
            if chunks:
                print(f"Example chunk preview: {chunks[0].page_content[:150]}...")
            
            return chunks
            
        except Exception as e:
            print(f"❌ Error processing documents: {e}")
            return []
    
    def add_documents_to_vector_store(self, documents: List[Document]) -> bool:
        """Add documents to vector store"""
        try:
            if not documents:
                print("⚠️  No documents to add")
                return False
            
            print(f"🔄 Adding {len(documents)} documents to vector store...")
            
            # Generate embeddings
            texts = [doc.page_content for doc in documents]
            print("🔄 Generating embeddings...")
            embeddings = self.embedding_model.encode(
                texts, 
                show_progress_bar=True, 
                convert_to_numpy=True
            )
            
            # Prepare data for ChromaDB
            ids = []
            metadatas = []
            documents_text = []
            embeddings_list = []
            
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                # Generate unique ID
                doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
                ids.append(doc_id)
                
                # Metadata
                metadata = dict(doc.metadata)
                metadata["doc_index"] = i
                metadata["content_length"] = len(doc.page_content)
                metadatas.append(metadata)
                
                # Document text and embedding
                documents_text.append(doc.page_content)
                embeddings_list.append(embedding.tolist())
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            
            total_docs = self.collection.count()
            print(f"✅ Successfully added documents. Total in store: {total_docs}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error adding documents to vector store: {e}")
            return False
    
    def query(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """Query the vector store"""
        try:
            print(f"🔄 Querying: '{query_text}'")
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query_text, convert_to_numpy=True)
            
            # Search vector store
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                })
            
            print(f"✅ Found {len(formatted_results)} results")
            return {
                'query': query_text,
                'results': formatted_results,
                'count': len(formatted_results)
            }
            
        except Exception as e:
            print(f"❌ Error during query: {e}")
            return {'query': query_text, 'results': [], 'count': 0, 'error': str(e)}
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            'embedding_model': self.embedding_model_name,
            'embedding_dimension': self.embedding_model.get_sentence_embedding_dimension() if self.embedding_model else None,
            'collection_name': self.collection_name,
            'document_count': self.collection.count() if self.collection else 0,
            'persist_directory': self.persist_directory
        }

def main():
    """Main function for testing the RAG system"""
    print("🚀 RAG System Test")
    print("=" * 50)
    
    # Initialize RAG system
    rag = RAGSystem()
    
    # Show system info
    info = rag.get_system_info()
    print(f"📊 System Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 50)
    
    # Test queries
    test_queries = [
        "What is Python programming?",
        "Machine learning algorithms",
        "Data types in Python",
        "Financial analysis with Python"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        result = rag.query(query, n_results=3)
        
        if result['results']:
            for i, res in enumerate(result['results'][:2], 1):
                print(f"  Result {i}: {res['content'][:100]}...")
                print(f"  Source: {res['metadata'].get('source_file', 'Unknown')}")
                print(f"  Distance: {res['distance']:.4f}")
        else:
            print("  No results found")
        
        print("-" * 30)

if __name__ == "__main__":
    main()
