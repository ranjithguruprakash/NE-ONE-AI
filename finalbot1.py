import os
import sys
import json
from pathlib import Path

# ─── IMPORTANT PATCHES BEFORE ANY OTHER IMPORTS ─────────────────────────────────
# Disable Streamlit file watcher (avoids torch.classes instantiation errors)
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

# Patch out torch.classes if torch is installed to avoid custom class errors
try:
    import torch
    torch.classes.__path__ = []
except ImportError:
    pass
# ───────────────────────────────────────────────────────────────────────────────

import boto3
import chromadb
import fitz  # PyMuPDF
import pdfplumber
from unstructured.partition.pdf import partition_pdf
import pytesseract
from PIL import Image
import pandas as pd
import io
import streamlit as st

# ─── NOTE ─────────────────────────────────────────────────────────────────────
# Launch with:
#     streamlit run app.py
# ───────────────────────────────────────────────────────────────────────────────

# ─── Environment Variables ────────────────────────────────────────────────────
os.environ.setdefault('AWS_REGION',                  'us-east-1')
os.environ.setdefault('BEDROCK_TEXT_EMBEDDING_MODEL', 'amazon.titan-embed-text-v2:0')
os.environ.setdefault('BEDROCK_IMAGE_EMBEDDING_MODEL','amazon.titan-embed-text-v2:0')
os.environ.setdefault('BEDROCK_TABLE_EMBEDDING_MODEL','amazon.titan-embed-text-v2:0')
# Updated model ID for Claude 3.7 Sonnet:
os.environ.setdefault('BEDROCK_LLM_MODEL',           'us.anthropic.claude-3-7-sonnet-20250219-v1:0')
os.environ.setdefault('TESSERACT_PATH',               r'C:\Program Files\Tesseract-OCR\tesseract.exe')

pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_PATH')

# ─── Validate environment variables ─────────────────────────────────────────
REQUIRED_ENVS = [
    'AWS_REGION',
    'BEDROCK_TEXT_EMBEDDING_MODEL',
    'BEDROCK_IMAGE_EMBEDDING_MODEL',
    'BEDROCK_TABLE_EMBEDDING_MODEL',
    'BEDROCK_LLM_MODEL',
]
missing = [v for v in REQUIRED_ENVS if not os.getenv(v)]
if missing:
    st.error(f"Missing required environment variables: {', '.join(missing)}")
    sys.exit(1)

# ─── AWS Bedrock & ChromaDB clients ─────────────────────────────────────────
bedrock = boto3.client('bedrock-runtime', region_name=os.getenv('AWS_REGION'))
chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection    = chroma_client.get_or_create_collection("multimodal_docs")

# ─── Helper functions ────────────────────────────────────────────────────────
def make_serializable(obj):
    if isinstance(obj, frozenset):
        return sorted(obj)
    if hasattr(obj, '__dict__'):
        return {k: make_serializable(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(make_serializable(v) for v in obj)
    return obj


def extract_pdf_content(pdf_path: Path, ocr_lang='eng') -> dict:
    result = {"chunks": [], "tables": [], "images": []}

    # Text + metadata via unstructured
    chunks = partition_pdf(
        filename=str(pdf_path),
        chunking_strategy="by_title",
        extract_images_in_pdf=True,
        infer_table_structure=True
    )
    for c in chunks:
        text = getattr(c, 'text', '').strip()
        if text:
            result['chunks'].append({
                'text': text,
                'metadata': make_serializable(getattr(c, 'metadata', {}))
            })

    # Tables via pdfplumber (any failure is non-fatal)
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages):
                for tbl in page.extract_tables():
                    result['tables'].append({'page': i+1, 'table': tbl})
    except Exception:
        _ = fitz.open(str(pdf_path)).page_count  # fallback no-op

    # Images → OCR via PyMuPDF + Tesseract
    doc = fitz.open(str(pdf_path))
    for i in range(doc.page_count):
        for img in doc.get_page_images(i, full=True):
            base = doc.extract_image(img[0])
            image = Image.open(io.BytesIO(base['image']))
            ocr_text = pytesseract.image_to_string(image, lang=ocr_lang).strip()
            if ocr_text:
                result['images'].append({'page': i+1, 'ocr_text': ocr_text})

    return result


def embed_text(text: str) -> list:
    text = text.strip()
    if not text:
        return []
    resp = bedrock.invoke_model(
        modelId     = os.getenv('BEDROCK_TEXT_EMBEDDING_MODEL'),
        contentType = 'application/json',
        accept      = 'application/json',
        body        = json.dumps({'inputText': text})
    )
    data = json.loads(resp['body'].read())
    return data.get('embedding') or data.get('embeddings') or []


def embed_table(df: pd.DataFrame) -> list:
    return embed_text(df.to_csv(index=False))


def embed_image(image_bytes: bytes) -> list:
    txt = pytesseract.image_to_string(Image.open(io.BytesIO(image_bytes)))
    return embed_text(txt)


def store_document(doc_id: str, embedding: list, metadata: dict):
    if embedding:
        collection.add(
            ids        = [doc_id],
            embeddings = [embedding],
            documents  = [metadata.get('content', '')],
            metadatas  = [metadata],
        )


def query_llm(prompt: str) -> str:
    prompt = prompt.strip()
    model_id = os.getenv('BEDROCK_LLM_MODEL')
    if not prompt or not model_id:
        return ""
    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}]} 
            ]
        })
        resp = bedrock.invoke_model(
            modelId     = model_id,
            contentType = 'application/json',
            accept      = 'application/json',
            body        = body
        )
        data = json.loads(resp['body'].read())
        return data['content'][0]['text']
    except Exception as e:
        print(f"LLM query failed: {e}", file=sys.stderr)
        st.error(f"LLM query failed: {e}")
        return ""

# ─── Streamlit App ───────────────────────────────────────────────────────────
def main():
    st.title("NE-ONE GenAI")

    pdf_path = Path(r"C:\Users\ranjith.guruprakash\OneDrive - Calnex Solutions\Desktop\ne-one-ai\OperatorManual.pdf")
    if not pdf_path.exists():
        st.error(f"Couldn’t find PDF at: {pdf_path}")
        st.stop()

    if 'indexed' not in st.session_state:
        with st.spinner("Indexing PDF…"):
            content = extract_pdf_content(pdf_path, ocr_lang='eng')
            count = 0
            for c in content['chunks']:
                emb = embed_text(c['text'])
                store_document(f"chunk_{count}", emb, {'content': c['text']})
                count += 1
            for t in content['tables']:
                emb = embed_table(pd.DataFrame(t['table']))
                store_document(f"table_{count}", emb, {'content': str(t['table'])})
                count += 1
            for img in content['images']:
                emb = embed_text(img['ocr_text'])
                store_document(f"image_{count}", emb, {'content': img['ocr_text']})
                count += 1
            st.session_state.indexed = True
            st.success(f"Indexed {count} items.")

    if st.session_state.get('indexed', False):
        question = st.text_input("Ask a question about the document:")
        if question:
            with st.spinner("Thinking…"):
                q_emb = embed_text(question)
                if not q_emb:
                    st.error("Could not embed question.")
                    st.stop()
                res = collection.query(
                    query_embeddings=[q_emb],
                    n_results=3,
                    include=['documents']
                )
                docs = res['documents'][0] if res.get('documents') else []
                if not docs:
                    st.warning("No relevant content found.")
                    return
                ctx = "\n---\n".join(docs)
                answer = query_llm(f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:")
                st.subheader("Answer")
                st.write(answer)
                st.markdown("*Powered by Amazon Bedrock Claude 3.7 Sonnet*")

if __name__ == "__main__":
    main()