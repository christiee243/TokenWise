"""
TokenWise - End-to-End Integration Pipeline Test
Tests complete M1 -> M2 workflow:
  1. Prompt Optimizer (M1)
  2. PDF Processor & Markdown Chunking (M1)
  3. RAG Retrieval & Semantic Search (M2)
"""
import io
import sys
import os

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure backend directory is in path
sys.path.insert(0, os.path.dirname(__file__))

import fitz
from pdf_processor import process_pdf
from prompt_optimizer import optimize_prompt
from rag import RAGRetriever


def test_pipeline():
    print("=" * 60)
    print("TokenWise Full Pipeline Integration Test (M1 + M2)")
    print("=" * 60)

    # 1. Test prompt optimizer (M1)
    print("\n--- 1. Testing Prompt Optimizer (M1) ---")
    raw_prompt = "Hello AI, can you please kindly act as an expert in Python and tell me how in order to optimize tokens due to the fact that they cost money? Make sure that you give a clear table."
    res = optimize_prompt(raw_prompt)
    print(f"Original tokens:  {res['original_tokens']}")
    print(f"Optimized tokens: {res['optimized_tokens']}")
    print(f"Reduction:        {res['reduction_percent']}%")
    print(f"Optimized text:\n{res['optimized']}\n")
    assert res["optimized_tokens"] > 0
    assert res["reduction_percent"] >= 0
    print("✓ Prompt Optimizer: PASSED")

    # 2. Test PDF processor with an in-memory generated PDF (M1)
    print("\n--- 2. Testing PDF Processing & Chunking (M1) ---")
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((50, 50), "Document Header - Page 1", fontsize=9)
    page1.insert_text((50, 100), "Attention Is All You Need", fontsize=20)
    page1.insert_text((50, 140), "Abstract: We propose the Transformer model, based solely on attention mechanisms.", fontsize=11)
    page1.insert_text((50, 200), "1. Introduction", fontsize=16)
    page1.insert_text((50, 230), "Recurrent neural networks have been firmly established in sequence modeling.", fontsize=11)
    page1.insert_text((50, 750), "Page 1 of 2", fontsize=9)

    page2 = doc.new_page()
    page2.insert_text((50, 50), "Document Header - Page 1", fontsize=9)  # repeated header
    page2.insert_text((50, 100), "2. Model Architecture", fontsize=16)
    page2.insert_text((50, 130), "The Transformer follows an encoder-decoder architecture with self-attention.", fontsize=11)
    page2.insert_text((50, 750), "Page 2 of 2", fontsize=9)  # page number

    pdf_bytes = doc.write()
    doc.close()

    pdf_res = process_pdf(pdf_bytes, compression_level="Balanced")
    print(f"Pages:            {pdf_res['page_count']}")
    print(f"Sections:         {pdf_res['section_count']}")
    print(f"Original tokens:  {pdf_res['original_tokens']}")
    print(f"Optimized tokens: {pdf_res['optimized_tokens']}")
    print(f"Saved:            {pdf_res['tokens_saved']} ({pdf_res['reduction_percent']}%)")
    print(f"Chunks produced:  {pdf_res['chunk_count']}")
    assert pdf_res["chunk_count"] > 0
    assert len(pdf_res["chunks"]) == pdf_res["chunk_count"]
    print("✓ PDF Processor & Chunking: PASSED")

    # 3. Test RAG Retriever with M1 Chunks (M2)
    print("\n--- 3. Testing RAG Vector Indexing & Semantic Search (M2) ---")
    retriever = RAGRetriever()
    doc_id = "test_transformer_paper"

    index_res = retriever.index_document(pdf_res["chunks"], document_id=doc_id)
    print(f"Indexed document ID: {index_res['document_id']}")
    print(f"Chunks indexed:      {index_res['chunks_indexed']}")
    print(f"Embedding dimension: {index_res['embedding_dimension']}")
    assert index_res["status"] == "indexed"
    assert index_res["chunks_indexed"] == pdf_res["chunk_count"]

    # Natural language query 1: architecture
    query1 = "How does the Transformer architecture work?"
    q1_res = retriever.retrieve(query1, document_id=doc_id)
    print(f"\nQuery: '{query1}'")
    print(f"Intent: {q1_res['query_analysis']['intent']} | Complexity: {q1_res['query_analysis']['complexity']}")
    print(f"Keywords: {q1_res['query_analysis']['keywords']}")
    print(f"Retrieved {q1_res['num_chunks_retrieved']} chunk(s):")
    for idx, c in enumerate(q1_res["retrieved_chunks"], 1):
        print(f"  [{idx}] ({c['relevance_score']*100:.1f}% match - § {c['metadata']['section']}): {c['text'][:80]}...")
    assert q1_res["num_chunks_retrieved"] > 0
    assert any("transformer" in c["text"].lower() or "architecture" in c["text"].lower() for c in q1_res["retrieved_chunks"])

    # Natural language query 2: attention mechanism
    query2 = "What mechanism does the model rely on?"
    q2_res = retriever.retrieve(query2, document_id=doc_id)
    print(f"\nQuery: '{query2}'")
    print(f"Retrieved top match: § {q2_res['retrieved_chunks'][0]['metadata']['section']} (Score: {q2_res['retrieved_chunks'][0]['relevance_score']:.4f})")
    assert q2_res["num_chunks_retrieved"] > 0
    print("✓ RAG Retriever & Semantic Search: PASSED")

    print("\n" + "=" * 60)
    print("SUCCESS: Full M1 + M2 Pipeline Verified!")
    print("=" * 60)


if __name__ == "__main__":
    test_pipeline()
