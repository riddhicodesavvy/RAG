#!/usr/bin/env python3
"""
Interactive RAG Application for Querying Documents
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from rag_system import RAGSystem

class InteractiveRAG:
    """Interactive RAG application"""
    
    def __init__(self):
        self.rag = RAGSystem(collection_name="pdf_documents")
        self.running = True
    
    def display_welcome(self):
        """Display welcome message"""
        print("\n" + "="*60)
        print("🤖 INTERACTIVE RAG SYSTEM")
        print("="*60)
        print("Ask questions about your documents!")
        print("Type 'help' for commands, 'quit' to exit")
        print("="*60)
    
    def display_help(self):
        """Display help information"""
        print("\n📚 Available Commands:")
        print("  help     - Show this help message")
        print("  info     - Show system information")
        print("  stats    - Show document statistics")
        print("  quit     - Exit the application")
        print("\n💡 Query Examples:")
        print("  What is Python programming?")
        print("  Explain machine learning algorithms")
        print("  What are Python data types?")
        print("  How is Python used in finance?")
    
    def display_info(self):
        """Display system information"""
        info = self.rag.get_system_info()
        print("\n📊 System Information:")
        for key, value in info.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    def display_stats(self):
        """Display document statistics"""
        try:
            # Get a sample of documents to analyze
            result = self.rag.query("", n_results=10)  # Empty query to get sample
            
            if result['results']:
                sources = {}
                content_lengths = []
                
                for res in result['results']:
                    source = res['metadata'].get('source_file', 'Unknown')
                    sources[source] = sources.get(source, 0) + 1
                    content_lengths.append(len(res['content']))
                
                print("\n📈 Document Statistics:")
                print(f"  Total documents: {self.rag.collection.count()}")
                print(f"  Sample sources: {len(sources)}")
                
                print("\n📚 Source Distribution:")
                for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"    {source}: {count} chunks")
                
                if content_lengths:
                    avg_length = sum(content_lengths) / len(content_lengths)
                    print(f"\n📏 Average chunk length: {avg_length:.0f} characters")
            else:
                print("\n⚠️  No statistics available")
                
        except Exception as e:
            print(f"\n❌ Error getting statistics: {e}")
    
    def format_results(self, result):
        """Format query results for display"""
        if not result['results']:
            print("❌ No relevant documents found.")
            return
        
        print(f"\n🎯 Found {result['count']} relevant documents:")
        print("-" * 60)
        
        for i, res in enumerate(result['results'], 1):
            print(f"\n📄 Result {i}:")
            print(f"📂 Source: {res['metadata'].get('source_file', 'Unknown')}")
            print(f"📏 Relevance Score: {res['distance']:.4f}")
            print(f"📝 Content:")
            
            # Truncate content for display
            content = res['content']
            if len(content) > 300:
                content = content[:300] + "..."
            
            print(f"   {content}")
            print("-" * 40)
    
    def process_query(self, query):
        """Process user query"""
        if not query.strip():
            print("⚠️  Please enter a valid query.")
            return
        
        print(f"\n🔍 Searching for: '{query}'")
        print("⏳ Processing...")
        
        result = self.rag.query(query, n_results=5)
        self.format_results(result)
    
    def run(self):
        """Run the interactive application"""
        self.display_welcome()
        
        while self.running:
            try:
                # Get user input
                user_input = input("\n💬 Enter your query or command: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    self.running = False
                
                elif user_input.lower() == 'help':
                    self.display_help()
                
                elif user_input.lower() == 'info':
                    self.display_info()
                
                elif user_input.lower() == 'stats':
                    self.display_stats()
                
                else:
                    # Process as query
                    self.process_query(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                self.running = False
            
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again.")

def main():
    """Main function"""
    try:
        app = InteractiveRAG()
        app.run()
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
