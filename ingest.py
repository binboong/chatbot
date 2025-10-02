"""
DEBUG VERSION: Enhanced logging to find the crash cause
"""

import os
import sys
import argparse
import logging
import pandas as pd
from typing import List, Tuple
import unicodedata
import re
import time
import gc
import traceback
import signal
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from onnx_embedder import get_embedder

# Configure MORE verbose logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ingest_debug.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("ingest")

# Track last processed file globally
LAST_FILE = None
CHECKPOINT_FILE = "ingest_checkpoint.txt"


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    logger.critical(f"!!! SIGNAL RECEIVED: {signum} !!!")
    logger.critical(f"Last file being processed: {LAST_FILE}")
    logger.critical("Stack trace:")
    traceback.print_stack(frame)
    sys.exit(1)


# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


class ONNXEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """ChromaDB-compatible embedding function using ONNX"""
    
    def __init__(self, model_path="./models"):
        logger.info("Initializing ONNX embedding function")
        self.embedder = get_embedder(model_path)
        logger.info("ONNX embedding function ready")

    def __call__(self, input):
        try:
            logger.debug(f"Embedding {len(input)} documents")
            result = self.embedder(input)
            logger.debug(f"Embedding complete, shape: {len(result)}")
            return result
        except Exception as e:
            logger.error(f"!!! EMBEDDING FAILED: {e}")
            logger.error(traceback.format_exc())
            raise


def save_checkpoint(filename: str):
    """Save checkpoint of last processed file"""
    try:
        with open(CHECKPOINT_FILE, 'w') as f:
            f.write(filename)
    except:
        pass


def load_checkpoint() -> str:
    """Load last checkpoint"""
    try:
        if os.path.exists(CHECKPOINT_FILE):
            with open(CHECKPOINT_FILE, 'r') as f:
                return f.read().strip()
    except:
        pass
    return None


def load_file(path: str) -> List[str]:
    """Load and extract text from various file formats with memory cleanup"""
    ext = os.path.splitext(path)[1].lower()
    texts = []

    try:
        logger.debug(f"Loading file: {path}")
        
        if ext == ".txt":
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
                if content.strip():
                    texts.append(content)
                    logger.debug(f"Loaded {len(content)} chars from TXT")

        elif ext == ".docx":
            try:
                from docx import Document
                doc = Document(path)
                content = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                if content.strip():
                    texts.append(content)
                    logger.debug(f"Loaded {len(content)} chars from DOCX")
                del doc
            except Exception as e:
                logger.error(f"DOCX error in {path}: {e}")

        elif ext == ".pdf":
            try:
                from PyPDF2 import PdfReader
                logger.debug("Opening PDF reader")
                reader = PdfReader(path)
                logger.debug(f"PDF has {len(reader.pages)} pages")
                
                pages_content = []
                for page_num, page in enumerate(reader.pages):
                    try:
                        text = page.extract_text() or ""
                        pages_content.append(text)
                        logger.debug(f"Extracted page {page_num+1}/{len(reader.pages)}")
                    except Exception as e:
                        logger.warning(f"Failed to extract page {page_num}: {e}")
                
                content = "\n".join(pages_content)
                if content.strip():
                    texts.append(content)
                    logger.debug(f"Loaded {len(content)} chars from PDF")
                
                del reader, pages_content
                gc.collect()
                
            except Exception as e:
                logger.error(f"PDF error in {path}: {e}")
                logger.error(traceback.format_exc())

        elif ext in [".xlsx", ".xls"]:
            try:
                df = pd.read_excel(path)
                content = df.to_string()
                if content.strip():
                    texts.append(content)
                    logger.debug(f"Loaded {len(content)} chars from Excel")
                del df
            except Exception as e:
                logger.error(f"Excel error in {path}: {e}")

        else:
            logger.debug(f"Skipping unsupported file type: {path}")

    except Exception as e:
        logger.error(f"!!! CRITICAL: Failed to read {path}: {e}")
        logger.error(traceback.format_exc())

    return texts


def slugify(value: str) -> str:
    """Convert string to safe filename (ASCII only) with protection"""
    try:
        # Limit input length first
        if len(value) > 500:
            logger.warning(f"Truncating very long filename: {len(value)} -> 500 chars")
            value = value[:500]
        
        value = unicodedata.normalize('NFKD', value)
        value = value.encode('ascii', 'ignore').decode('ascii')
        
        # Use simpler regex to avoid catastrophic backtracking
        value = re.sub(r'[^\w\s-]', '_', value, flags=re.ASCII)
        value = re.sub(r'[-\s]+', '_', value, flags=re.ASCII)
        
        result = value.strip('_')[:100]
        logger.debug(f"Slugified: '{value[:50]}...' -> '{result}'")
        return result
    except Exception as e:
        logger.error(f"!!! Slugify failed: {e}")
        # Return safe fallback
        import hashlib
        return hashlib.md5(value.encode('utf-8', errors='ignore')).hexdigest()[:20]


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks for better context"""
    try:
        if len(text) <= chunk_size:
            logger.debug(f"Text fits in one chunk: {len(text)} chars")
            return [text]
        
        chunks = []
        start = 0
        iterations = 0
        max_iterations = len(text) // (chunk_size - overlap) + 10  # Safety limit
        
        while start < len(text):
            iterations += 1
            if iterations > max_iterations:
                logger.error(f"!!! Chunking infinite loop detected! Breaking at {iterations} iterations")
                break
            
            end = start + chunk_size
            
            if end < len(text):
                for delimiter in ['.', '?', '!', '\n']:
                    try:
                        chunk_end = text.rfind(delimiter, start, end)
                        if chunk_end != -1 and chunk_end > start:
                            end = chunk_end + 1
                            break
                    except Exception as e:
                        logger.warning(f"Error finding delimiter: {e}")
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
                logger.debug(f"Chunk {len(chunks)}: start={start}, end={end}, len={len(chunk)}")
            
            start = end - overlap if end < len(text) else end
        
        logger.debug(f"Split text into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"!!! CRITICAL: chunk_text failed: {e}")
        logger.error(traceback.format_exc())
        # Return original text as single chunk on error
        return [text]


def get_all_files(folder: str) -> List[Tuple[str, str, str]]:
    """Get all files with supported extensions recursively"""
    supported_ext = {'.txt', '.pdf', '.docx', '.xlsx', '.xls'}
    files = []
    
    print(f"🔍 Scanning folder structure recursively: {folder}")
    
    for root, dirs, filenames in os.walk(folder):
        rel_path = os.path.relpath(root, folder)
        if rel_path != '.':
            print(f"   📁 Scanning subfolder: {rel_path}")
        
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in supported_ext:
                path = os.path.join(root, fname)
                files.append((path, fname, rel_path))
    
    logger.info(f"Found {len(files)} supported files")
    return files


def is_already_ingested(collection, doc_id: str) -> bool:
    """Check if document already exists in collection"""
    try:
        result = collection.get(ids=[doc_id])
        return len(result['ids']) > 0
    except Exception as e:
        logger.debug(f"Check ingested failed for {doc_id}: {e}")
        return False


def main(
    folder: str = "./data", 
    force: bool = False, 
    chunk_size: int = 1000,
    batch_size: int = 10,
    resume: bool = True,
    gc_interval: int = 50
):
    """Main ingestion function with extensive debug logging"""
    
    global LAST_FILE
    
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("DEBUG MODE: Starting document ingestion with enhanced logging")
    logger.info(f"Folder: {folder}")
    logger.info(f"Force: {force}, Resume: {resume}, Batch size: {batch_size}")
    logger.info(f"GC interval: {gc_interval} files")
    logger.info("=" * 60)
    
    if not os.path.exists(folder):
        logger.error(f"Data folder not found: {folder}")
        sys.exit(1)

    # Initialize ChromaDB client
    logger.info("Initializing ChromaDB client...")
    try:
        client = chromadb.PersistentClient(
            path="./vector_db",
            settings=Settings(
                anonymized_telemetry=False, 
                allow_reset=True
            )
        )
        logger.info("ChromaDB client created")
        
        embedding_fn = ONNXEmbeddingFunction("./models")
        logger.info("Embedding function initialized")
        
    except Exception as e:
        logger.error(f"!!! CRITICAL: Failed to initialize ChromaDB: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

    # Handle force mode
    if force:
        logger.warning("FORCE MODE: Deleting existing collection")
        try:
            client.delete_collection("knowledge_base")
            logger.info("Previous collection deleted")
        except Exception as e:
            logger.debug(f"Could not delete collection (may not exist): {e}")

    # Get or create collection
    logger.info("Getting or creating collection...")
    try:
        collection = client.get_or_create_collection(
            name="knowledge_base",
            embedding_function=embedding_fn
        )
        count_before = collection.count()
        logger.info(f"Collection initialized: {count_before} existing documents")
    except Exception as e:
        logger.error(f"!!! CRITICAL: Failed to create collection: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

    # Get all files
    logger.info("Scanning for files...")
    all_files = get_all_files(folder)
    total_files_found = len(all_files)
    logger.info(f"Found {total_files_found} files to process")
    
    if total_files_found == 0:
        logger.warning("No files found to ingest!")
        return

    # Load checkpoint if resuming
    last_checkpoint = load_checkpoint()
    if last_checkpoint and resume:
        logger.info(f"Resuming from checkpoint: {last_checkpoint}")

    # Statistics
    stats = {
        'processed': 0,
        'skipped': 0,
        'failed': 0,
        'chunks_added': 0
    }

    # Process files in batches
    batch_docs = []
    batch_metas = []
    batch_ids = []
    
    current_folder = None
    
    for file_idx, (path, fname, rel_folder) in enumerate(all_files, 1):
        LAST_FILE = fname
        
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing file {file_idx}/{total_files_found}: {fname}")
            logger.info(f"Path: {path}")
            
            # Show folder change
            if rel_folder != current_folder:
                current_folder = rel_folder
                folder_display = rel_folder if rel_folder != '.' else 'root'
                print(f"\n📂 Processing folder: {folder_display}")
            
            # Progress update every 10 files
            if file_idx % 10 == 0:
                elapsed = time.time() - start_time
                rate = file_idx / elapsed if elapsed > 0 else 0
                eta = (total_files_found - file_idx) / rate if rate > 0 else 0
                print(f"📊 Progress: {file_idx}/{total_files_found} ({file_idx/total_files_found*100:.1f}%) | Rate: {rate:.1f} files/s | ETA: {eta/60:.1f} min")
            
            print(f"📄 [{file_idx}/{total_files_found}] {fname}")
            
            # Load file
            logger.info(f"Step 1: Loading file content...")
            texts = load_file(path)
            logger.info(f"Loaded {len(texts)} text sections")
            
            if not texts:
                logger.debug(f"No content: {fname}")
                stats['skipped'] += 1
                continue

            # Process each text content
            file_chunks_added = 0
            for i, text in enumerate(texts):
                try:
                    logger.info(f"Step 2: Processing text section {i+1}/{len(texts)}")
                    
                    if not text or not text.strip():
                        continue
                    
                    # Chunk large documents
                    logger.info(f"Step 3: Chunking text (length: {len(text)} chars)")
                    chunks = chunk_text(text, chunk_size=chunk_size)
                    logger.info(f"Created {len(chunks)} chunks")
                    
                    for j, chunk in enumerate(chunks):
                        try:
                            if not chunk.strip():
                                continue
                            
                            # Generate unique document ID
                            logger.debug(f"Step 3.1: Creating doc_id for chunk {j}")
                            doc_id = f"{slugify(fname)}_{i}_{j}"
                            logger.debug(f"Doc ID created: {doc_id}")
                            
                            # Skip if already exists (resume mode)
                            if resume and not force and is_already_ingested(collection, doc_id):
                                logger.debug(f"Chunk {j} already ingested, skipping")
                                continue
                            
                            logger.debug(f"Adding chunk {j}/{len(chunks)} to batch (ID: {doc_id})")
                            
                            # Add to batch
                            batch_docs.append(chunk)
                            batch_metas.append({
                                "source": fname,
                                "path": path,
                                "folder": rel_folder,
                                "chunk": j,
                                "total_chunks": len(chunks)
                            })
                            batch_ids.append(doc_id)
                            file_chunks_added += 1
                            
                            # When batch is full, add to collection
                            if len(batch_docs) >= batch_size:
                                logger.info(f"Step 4: Batch full ({len(batch_docs)} docs), adding to collection...")
                                
                                max_retries = 3
                                for retry in range(max_retries):
                                    try:
                                        logger.debug(f"Attempt {retry+1}/{max_retries} to add batch")
                                        collection.add(
                                            documents=batch_docs,
                                            metadatas=batch_metas,
                                            ids=batch_ids
                                        )
                                        stats['chunks_added'] += len(batch_docs)
                                        logger.info(f"✅ Batch added successfully: {len(batch_docs)} chunks")
                                        break
                                    except Exception as e:
                                        if retry < max_retries - 1:
                                            logger.warning(f"⚠️  Batch add failed (retry {retry+1}/{max_retries}): {e}")
                                            time.sleep(2 ** retry)
                                        else:
                                            logger.error(f"!!! CRITICAL: Batch add failed after {max_retries} retries: {e}")
                                            logger.error(traceback.format_exc())
                                            stats['failed'] += len(batch_docs)
                                
                                # Clear batch
                                logger.debug("Clearing batch arrays")
                                batch_docs.clear()
                                batch_metas.clear()
                                batch_ids.clear()
                        
                        except Exception as chunk_error:
                            logger.error(f"!!! Error processing chunk {j}: {chunk_error}")
                            logger.error(traceback.format_exc())
                            continue
                    
                    # Clean up text variables
                    logger.debug("Cleaning up text chunks")
                    del chunks
                    
                except Exception as text_error:
                    logger.error(f"!!! Error processing text section {i}: {text_error}")
                    logger.error(traceback.format_exc())
                    continue
            
            if file_chunks_added > 0:
                print(f"   ✅ Added {file_chunks_added} chunks from {fname}")
            else:
                print(f"   ⏭️  Skipped {fname} (no new content)")
            
            stats['processed'] += 1
            save_checkpoint(fname)
            
            # Clean up file data
            logger.debug("Cleaning up file data")
            del texts
            
            # Force garbage collection periodically
            if file_idx % gc_interval == 0:
                logger.info(f"Running garbage collection at file {file_idx}")
                gc.collect()
                logger.info("Garbage collection complete")
                    
        except Exception as e:
            print(f"   ❌ Failed: {fname} - {str(e)}")
            logger.error(f"!!! CRITICAL: Failed to process {fname}: {e}")
            logger.error(traceback.format_exc())
            stats['failed'] += 1
            gc.collect()
    
    # Add remaining batch
    if batch_docs:
        logger.info(f"Adding final batch: {len(batch_docs)} chunks")
        try:
            collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )
            stats['chunks_added'] += len(batch_docs)
            logger.info(f"Final batch added: {len(batch_docs)} chunks")
        except Exception as e:
            logger.error(f"!!! Final batch add failed: {e}")
            logger.error(traceback.format_exc())
    
    # Final cleanup
    batch_docs.clear()
    batch_metas.clear()
    batch_ids.clear()
    gc.collect()

    # Final stats
    count_after = collection.count()
    elapsed = time.time() - start_time
    
    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info(f"Files found: {total_files_found}")
    logger.info(f"Files processed: {stats['processed']}")
    logger.info(f"Files skipped: {stats['skipped']}")
    logger.info(f"Files failed: {stats['failed']}")
    logger.info(f"Chunks added: {stats['chunks_added']}")
    logger.info(f"Collection before: {count_before}, after: {count_after}")
    logger.info(f"Time elapsed: {elapsed/60:.2f} minutes")
    logger.info(f"Processing rate: {stats['processed']/elapsed:.2f} files/second")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB (DEBUG MODE)")
    parser.add_argument("--folder", type=str, default="./data", help="Data folder path")
    parser.add_argument("--force", action="store_true", help="Delete existing collection before ingestion")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Maximum chunk size in characters")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of documents to add per batch")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume mode")
    parser.add_argument("--gc-interval", type=int, default=50, help="Run garbage collection every N files")
    
    args = parser.parse_args()

    try:
        main(
            folder=args.folder, 
            force=args.force, 
            chunk_size=args.chunk_size,
            batch_size=args.batch_size,
            resume=not args.no_resume,
            gc_interval=args.gc_interval
        )
    except Exception as e:
        logger.critical(f"!!! FATAL ERROR: {e}")
        logger.critical(traceback.format_exc())
        logger.critical(f"Last file being processed: {LAST_FILE}")
        sys.exit(1)