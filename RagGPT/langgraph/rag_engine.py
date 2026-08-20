from dataclasses import dataclass
import json
import os
import time
from typing import Optional, List, Dict, Any
from sentence_transformers import CrossEncoder
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch

@dataclass
class Config:
    model_name: str = "gpt-4.1-mini"
    chunk_size: int = 800
    chunk_overlap: int = 100
    chunk_k:int = 30 # approximate top-k search param
    rerank_top_n: int = 7 # rerank top-n
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2";
    embedding_model: str = "BAAI/bge-small-en-v1.5" # local sentence-transformers model
    docs_path: str = "docs";
    faiss_path: str = "faiss_index";


def load_config(config_path) -> Config:
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return Config(**config_dict)
    except:
        return Config()


class RagEngine(torch.nn.Module):
    def __init__(self, config_path:str) -> None:
        super().__init__()
        self.config:Config = load_config(config_path)
        self.cross_encoder: Optional[CrossEncoder] = None
        try:
            self.cross_encoder = CrossEncoder(self.config.cross_encoder_model)
        except Exception as e:
            print(f"CrossEncoder model {self.config.cross_encoder_model} not found, reranking disabled. ")

        print("RAG Engine initialized");
        print(f"Chunk size: {self.config.chunk_size}, Overlap: {self.config.chunk_overlap}, Chunk Nos: {self.config.chunk_k}")

        # Config paths are relative to this module, not to the caller's cwd.
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.docs_path: str = os.path.join(base_dir, self.config.docs_path)
        self.faiss_path: str = os.path.join(base_dir, self.config.faiss_path)

        # Embeddings run locally: building and searching the index needs no API key.
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=self.config.embedding_model)
            print(f"Local embeddings enabled with model: {self.config.embedding_model}")
        except Exception as e:
            print(f"Failed to load embedding model {self.config.embedding_model}: {e}. RAG search disabled.")

        self.vector_store: Optional[FAISS] = None
        self.init_vector_store()

    def init_vector_store(self, vector_store: Optional[FAISS] = None) -> Optional[FAISS]:
        """Attach `vector_store` if given, else load the index at faiss_path, else build one from docs_path."""
        if vector_store is not None:
            self.vector_store = vector_store
            return self.vector_store

        if self.embeddings is None:
            self.vector_store = None
            return None

        if os.path.isdir(self.faiss_path):
            try:
                print(f"Loading existing FAISS index from {self.faiss_path}")
                # Index metadata is pickled; deserializing is safe only because we wrote it.
                self.vector_store = FAISS.load_local(
                    self.faiss_path, self.embeddings, allow_dangerous_deserialization=True
                )
                return self.vector_store
            except Exception as e:
                # Most likely a dimension mismatch after changing embedding_model.
                print(f"Could not load index at {self.faiss_path} ({e}); rebuilding from docs.")

        print("Creating new FAISS index")
        os.makedirs(self.docs_path, exist_ok=True)
        documents = []
        try:
            documents.extend(PyPDFDirectoryLoader(self.docs_path).load())
        except Exception as e:
            print(f"No PDF files found or error loading PDFs: {e}")
        try:
            documents.extend(
                DirectoryLoader(
                    self.docs_path,
                    glob="**/*.md",
                    loader_cls=TextLoader,
                    loader_kwargs={"encoding": "utf-8"},
                ).load()
            )
        except Exception as e:
            print(f"No Markdown files found or error loading Markdown: {e}")
        print(f"Total documents loaded: {len(documents)}")

        chunks = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        ).split_documents(documents)

        # Release notes describe old versions; keep them out of the index.
        chunks = [
            chunk for chunk in chunks
            if "release_notes" not in str(chunk.metadata.get("source", "")).lower()
        ]
        if not chunks:
            print("No documents to index")
            self.vector_store = None
            return None

        last_page_id = ""
        chunk_index = 0
        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            # Markdown files carry no page numbers.
            page_id = f"{source}:{0 if page is None else page}"
            chunk_index = chunk_index + 1 if page_id == last_page_id else 0
            last_page_id = page_id
            chunk_id = f"{page_id}:{chunk_index}"

            prefix = f"Source: {source}"
            if page is not None:
                prefix += f", Page: {page}"
            prefix += f", ChunkID: {chunk_id}"

            chunk.metadata["id"] = chunk_id
            # Embed the metadata-enriched text, but keep the original for reranking.
            chunk.metadata["original_content"] = chunk.page_content
            chunk.page_content = f"{prefix}\n\n{chunk.page_content}"

        print(f"Creating FAISS index with {len(chunks)} chunks (with metadata enrichment)")
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.vector_store.save_local(self.faiss_path)
        print(f"FAISS index saved to {self.faiss_path}")
        return self.vector_store

    def forward(self, query: str) -> str:
        """Retrieve chunks for `query`, rerank them, and format them as one summary string."""
        vector_store = self.vector_store
        if vector_store is None:
            print("FAISS index not initialized")
            return ""

        print(f"Searching RAG Engine with query: {query}")
        rag_start = time.perf_counter()

        search_start = time.perf_counter()
        results = vector_store.similarity_search_with_score(query, k=self.config.chunk_k)
        print(f"Search time: {time.perf_counter() - search_start:.4f}s")

        cross_encoder = self.cross_encoder
        if cross_encoder is not None and results:
            try:
                rerank_start = time.perf_counter()
                docs = [doc for doc, _ in results]
                # Score against the pre-enrichment text, not the metadata-prefixed version.
                scores = cross_encoder.predict(
                    [(query, doc.metadata.get("original_content", doc.page_content)) for doc in docs],
                    show_progress_bar=False,
                    batch_size=self.config.chunk_k,
                    convert_to_numpy=True,
                    convert_to_tensor=False,
                )
                results = sorted(zip(docs, scores), key=lambda pair: -pair[1])
                print(f"Reranking time: {time.perf_counter() - rerank_start:.4f}s")
                print(f"Reranked {len(results)} documents using CrossEncoder")
            except Exception as e:
                print(f"Error during reranking: {e}")
                print("Falling back to original FAISS results")
        results = results[: self.config.rerank_top_n]

        print(f"\n{'='*60}")
        print(f"Retrieved {len(results)} chunks for query: {query}")
        print(f"{'='*60}")

        summary = ""
        for idx, (doc, score) in enumerate(results, 1):
            page = doc.metadata.get("page")
            source = doc.metadata.get("source")
            print(f"\nChunk {idx}:")
            print(f"  Score: {score:.4f}")
            print(f"  Source: {source}")
            print(f"  ID: {doc.metadata.get('id')}")
            if page is not None:
                print(f"  Page: {page}")
            print(f"  Content preview: {doc.page_content[:100]}...")
            summary += f"{source} (Relevance Score: {score:.4f}) : {doc.metadata.get('original_content', doc.page_content)}\n"

        print(f"\n{'='*60}\n")
        print(f"RAG time: {time.perf_counter() - rag_start:.4f}s")
        return summary
