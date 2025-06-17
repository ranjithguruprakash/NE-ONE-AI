import os
import fitz  # PyMuPDF
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PIL import Image
import io
import re

# --- Configuration ---
PDF_FILE_PATH = r"C:\Users\ranjith.guruprakash\OneDrive - Calnex Solutions\Desktop\ne-one-ai\OperatorManual.pdf"
CHROMA_DB_PATH = "./chroma_db_multimodal_improved"
EMBEDDING_MODEL_NAME = "models/text-embedding-004"

# --- Environment Variable Setup ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    exit()

try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"Error configuring Google Generative AI: {e}")
    exit()


class ImprovedDocumentProcessor:
    """
    Improved document processor with better text chunking, 
    enhanced image descriptions, and optimized embeddings.
    """

    def __init__(self, db_path: str, embedding_model_name: str):
        self.db_path = db_path
        self.embedding_model_name = embedding_model_name
        self.multimodal_model = genai.GenerativeModel('gemini-1.5-pro')  # Use latest model
        
        # Use different task types for documents vs queries
        self.doc_embedding_model = GoogleGenerativeAIEmbeddings(
            model=self.embedding_model_name, 
            task_type="retrieval_document"
        )
        self.query_embedding_model = GoogleGenerativeAIEmbeddings(
            model=self.embedding_model_name, 
            task_type="retrieval_query"
        )
        
        # Text splitter for better chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.vectordb = self._load_or_create_vectordb()

    def _load_or_create_vectordb(self):
        """Loads existing ChromaDB or creates new one."""
        if os.path.exists(self.db_path) and os.listdir(self.db_path):
            print(f"Loading existing ChromaDB from: {self.db_path}")
            return Chroma(persist_directory=self.db_path, embedding_function=self.doc_embedding_model)
        else:
            print(f"Creating new ChromaDB at: {self.db_path}")
            os.makedirs(self.db_path, exist_ok=True)
            return Chroma(persist_directory=self.db_path, embedding_function=self.doc_embedding_model)

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might interfere
        text = re.sub(r'[^\w\s\.\,\!\?\-\(\)\[\]\{\}\'\"\/\:]', ' ', text)
        # Strip and return
        return text.strip()

    def _get_enhanced_image_description(self, image_bytes: bytes, page_context: str = "") -> str:
        """
        Generate enhanced image description with context.
        """
        try:
            image = Image.open(io.BytesIO(image_bytes))
            
            # Enhanced prompt for better descriptions
            prompt = f"""
            Analyze this image in detail and provide a comprehensive description.
            
            Context from the page: {page_context[:200]}...
            
            Please describe:
            1. What type of image this is (diagram, photo, chart, table, etc.)
            2. Main visual elements and their purpose
            3. Any text, labels, or numbers visible
            4. How this image relates to the document content
            5. Key information or data this image conveys
            
            Be specific and detailed in your description.
            """
            
            response = self.multimodal_model.generate_content([prompt, image])
            return response.text if response.text else "No description generated"
            
        except Exception as e:
            print(f"Error generating image description: {e}")
            return f"Error processing image: {str(e)}"

    def _extract_content_from_pdf_page(self, page):
        """Extract and process content from PDF page."""
        # Extract text
        text = page.get_text()
        cleaned_text = self._clean_text(text)
        
        # Extract images
        images_info = []
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Filter out very small images (likely decorative)
                if len(image_bytes) > 1000:  # Skip tiny images
                    images_info.append({
                        'bytes': image_bytes,
                        'index': img_index,
                        'size': len(image_bytes)
                    })
            except Exception as e:
                print(f"Error extracting image {img_index}: {e}")
                continue
                
        return cleaned_text, images_info

    def generate_embeddings_from_pdf(self, pdf_path: str):
        """
        Process PDF with improved chunking and embedding strategy.
        """
        if not os.path.exists(pdf_path):
            print(f"Error: PDF file not found at '{pdf_path}'")
            return

        print(f"\nProcessing PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        all_documents = []

        for page_num, page in enumerate(doc):
            print(f"Processing page {page_num + 1}/{len(doc)}")
            
            text, images_info = self._extract_content_from_pdf_page(page)
            
            # Process text content
            if text and len(text.strip()) > 50:  # Only process substantial text
                # Split text into chunks
                text_chunks = self.text_splitter.split_text(text)
                
                for chunk_idx, chunk in enumerate(text_chunks):
                    if len(chunk.strip()) > 20:  # Skip very short chunks
                        text_doc = Document(
                            page_content=chunk,
                            metadata={
                                "source": os.path.basename(pdf_path),
                                "page_number": page_num + 1,
                                "content_type": "text",
                                "chunk_index": chunk_idx,
                                "total_chunks": len(text_chunks)
                            }
                        )
                        all_documents.append(text_doc)

            # Process images with context
            for img_info in images_info:
                print(f"  Processing image {img_info['index'] + 1} on page {page_num + 1}")
                
                description = self._get_enhanced_image_description(
                    img_info['bytes'], 
                    text[:500] if text else ""
                )
                
                if description and len(description) > 50:
                    # Create separate document for image
                    image_doc = Document(
                        page_content=f"Image Description: {description}",
                        metadata={
                            "source": os.path.basename(pdf_path),
                            "page_number": page_num + 1,
                            "content_type": "image",
                            "image_index": img_info['index'],
                            "image_size": img_info['size']
                        }
                    )
                    all_documents.append(image_doc)

        # Add documents to vector store in batches
        if all_documents:
            print(f"\nAdding {len(all_documents)} documents to ChromaDB...")
            
            # Process in batches to avoid memory issues
            batch_size = 50
            for i in range(0, len(all_documents), batch_size):
                batch = all_documents[i:i + batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(all_documents) + batch_size - 1)//batch_size}")
                self.vectordb.add_documents(batch)
            
            print("Successfully added all documents to ChromaDB.")
        else:
            print("No documents were generated from the PDF.")

        doc.close()

    def search_embeddings(self, query: str, k: int = 5, filter_dict: dict = None):
        """
        Enhanced search with filtering options.
        """
        print(f"\nSearching for: '{query}'")
        
        try:
            # Use the query embedding model for searches
            search_kwargs = {"k": k}
            if filter_dict:
                search_kwargs["filter"] = filter_dict
            
            docs = self.vectordb.similarity_search(query, **search_kwargs)
            
            if docs:
                print(f"Found {len(docs)} relevant documents:")
                for i, doc in enumerate(docs):
                    print(f"\n--- Result {i+1} ---")
                    print(f"Content Type: {doc.metadata.get('content_type', 'unknown')}")
                    print(f"Page: {doc.metadata.get('page_number', 'unknown')}")
                    print(f"Source: {doc.metadata.get('source', 'unknown')}")
                    
                    # Show content preview
                    content = doc.page_content
                    if len(content) > 300:
                        content = content[:300] + "..."
                    print(f"Content: {content}")
                    print("-" * 50)
            else:
                print("No relevant documents found.")
                
            return docs
            
        except Exception as e:
            print(f"Error during search: {e}")
            return []

    def search_by_content_type(self, query: str, content_type: str = None, k: int = 5):
        """Search with content type filtering."""
        filter_dict = None
        if content_type:
            filter_dict = {"content_type": content_type}
        
        return self.search_embeddings(query, k=k, filter_dict=filter_dict)


if __name__ == "__main__":
    # Initialize the improved processor
    processor = ImprovedDocumentProcessor(
        db_path=CHROMA_DB_PATH,
        embedding_model_name=EMBEDDING_MODEL_NAME
    )

    # Process the PDF (comment out if already processed)
    processor.generate_embeddings_from_pdf(PDF_FILE_PATH)

    # Example searches
    print("\n" + "="*60)
    print("                RUNNING IMPROVED QUERIES")
    print("="*60)

    # Search for text content only
    print("\n1. Searching for text content about introduction of technical publication:")
    processor.search_by_content_type("introduction document overview", content_type="text", k=3)

    # Search for image content only
    print("\n2. Searching for first image descriptions:")
    processor.search_by_content_type("diagram chart image", content_type="image", k=3)

    # General search
    print("\n3. General search about  time configuration:")
    processor.search_embeddings("safety instructions precautions", k=5)

    # Search for specific topics
    print("\n4. Search for specific technical content for CHANGING THE DEFAULT ADMIN PASSWORD:")
    processor.search_embeddings("operation manual instructions", k=3)