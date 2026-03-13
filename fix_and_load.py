#!/usr/bin/env python3
"""
Fix RAG system issues and load documents properly
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from rag_system import RAGSystem

def main():
    print("🔧 RAG System Fix and Load")
    print("=" * 50)
    
    # Initialize RAG system with the original collection
    rag = RAGSystem(collection_name="pdf_documents")  # Use existing collection
    
    # Check current status
    info = rag.get_system_info()
    print(f"📊 Current Status:")
    print(f"  Documents in store: {info['document_count']}")
    
    if info['document_count'] > 0:
        print("✅ Documents already exist in vector store")
        
        # Test retrieval
        print("\n🔍 Testing retrieval...")
        result = rag.query("Python programming", n_results=3)
        
        if result['results']:
            print("✅ Retrieval is working!")
            for i, res in enumerate(result['results'][:2], 1):
                print(f"  Result {i}: {res['content'][:100]}...")
                print(f"  Source: {res['metadata'].get('source_file', 'Unknown')}")
        else:
            print("❌ Retrieval failed - no results found")
    
    else:
        print("⚠️  No documents found. Loading documents...")
        
        # Load text documents
        text_docs = rag.load_text_documents("data/text_files")
        print(f"Loaded {len(text_docs)} text documents")
        
        # Load PDF documents with robust method
        pdf_docs = rag.load_pdf_documents_robust("data/pdf_files")
        print(f"Loaded {len(pdf_docs)} PDF documents")
        
        # Combine all documents
        all_docs = text_docs + pdf_docs
        
        if all_docs:
            # Process documents
            chunks = rag.process_documents(all_docs)
            
            if chunks:
                # Add to vector store
                success = rag.add_documents_to_vector_store(chunks)
                
                if success:
                    print("✅ Documents successfully loaded and indexed!")
                    
                    # Test retrieval
                    print("\n🔍 Testing retrieval...")
                    result = rag.query("Python programming", n_results=3)
                    
                    if result['results']:
                        print("✅ Retrieval is working!")
                        for i, res in enumerate(result['results'][:2], 1):
                            print(f"  Result {i}: {res['content'][:100]}...")
                            print(f"  Source: {res['metadata'].get('source_file', 'Unknown')}")
                    else:
                        print("❌ Retrieval failed")
                else:
                    print("❌ Failed to add documents to vector store")
            else:
                print("❌ No chunks created from documents")
        else:
            print("❌ No documents found to load")

if __name__ == "__main__":
    main()
