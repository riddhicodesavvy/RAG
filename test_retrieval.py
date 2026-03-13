#!/usr/bin/env python3
"""
Test script for RAG system retrieval functionality
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add notebook directory to path for imports
sys.path.append(str(Path(__file__).parent / "notebook"))

try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.config import Settings
    print("✅ All required libraries imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed")
    sys.exit(1)

class RetrievalTester:
    def __init__(self):
        self.embedding_model = None
        self.vector_store = None
        self.collection = None
        
    def setup_embedding_model(self):
        """Initialize the embedding model"""
        try:
            print("🔄 Loading embedding model...")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            print(f"✅ Model loaded successfully. Dimension: {self.embedding_model.get_sentence_embedding_dimension()}")
            return True
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            return False
    
    def setup_vector_store(self):
        """Connect to the existing vector store"""
        try:
            print("🔄 Connecting to vector store...")
            persist_directory = "data/vector_store"
            
            if not os.path.exists(persist_directory):
                print(f"❌ Vector store directory not found: {persist_directory}")
                return False
                
            client = chromadb.PersistentClient(path=persist_directory)
            self.collection = client.get_collection("pdf_documents")
            
            doc_count = self.collection.count()
            print(f"✅ Connected to vector store. Found {doc_count} documents")
            
            if doc_count == 0:
                print("⚠️  No documents found in vector store")
                return False
                
            return True
        except Exception as e:
            print(f"❌ Error connecting to vector store: {e}")
            return False
    
    def test_embedding_generation(self, query_text):
        """Test embedding generation for a query"""
        try:
            print(f"🔄 Generating embedding for query: '{query_text}'")
            query_embedding = self.embedding_model.encode(query_text, convert_to_numpy=True)
            print(f"✅ Embedding generated successfully. Shape: {query_embedding.shape}")
            return query_embedding
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            return None
    
    def test_retrieval(self, query_text, n_results=5):
        """Test document retrieval from vector store"""
        try:
            print(f"🔄 Searching for documents with query: '{query_text}'")
            
            # Generate query embedding
            query_embedding = self.test_embedding_generation(query_text)
            if query_embedding is None:
                return False
            
            # Search in vector store
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results
            )
            
            print(f"✅ Retrieved {len(results['documents'][0])} documents")
            
            # Display results
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0], 
                results['metadatas'][0], 
                results['distances'][0]
            )):
                print(f"\n--- Result {i+1} ---")
                print(f"Distance: {distance:.4f}")
                print(f"Source: {metadata.get('source_file', 'Unknown')}")
                print(f"Content preview: {doc[:200]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during retrieval: {e}")
            return False
    
    def run_comprehensive_test(self):
        """Run comprehensive retrieval tests"""
        print("🚀 Starting RAG Retrieval Test Suite")
        print("=" * 50)
        
        # Test 1: Setup
        if not self.setup_embedding_model():
            return False
            
        if not self.setup_vector_store():
            return False
        
        print("\n" + "=" * 50)
        
        # Test 2: Various query types
        test_queries = [
            "Python programming",
            "machine learning algorithms", 
            "data types in Python",
            "supervised learning",
            "artificial intelligence"
        ]
        
        success_count = 0
        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            if self.test_retrieval(query, n_results=3):
                success_count += 1
            print("-" * 30)
        
        # Summary
        print(f"\n📊 Test Results:")
        print(f"Successful queries: {success_count}/{len(test_queries)}")
        
        if success_count == len(test_queries):
            print("✅ All retrieval tests passed!")
            return True
        else:
            print("⚠️  Some tests failed")
            return False

def main():
    """Main function to run the test"""
    tester = RetrievalTester()
    
    if tester.run_comprehensive_test():
        print("\n🎉 RAG retrieval system is working correctly!")
        return 0
    else:
        print("\n❌ RAG retrieval system has issues that need to be fixed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
