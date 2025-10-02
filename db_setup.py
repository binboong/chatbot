"""
Database setup and validation script
Ensures ChromaDB is properly initialized with ONNX embeddings
"""

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import sys
import os
import logging

# Configure UTF-8 console output
if hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, "reconfigure") and sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("db_setup")

# Import embedder
from onnx_embedder import get_embedder


class ONNXEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """ChromaDB-compatible embedding function using ONNX"""
    
    def __init__(self, model_path="./models"):
        self.embedder = get_embedder(model_path)

    def __call__(self, input):
        return self.embedder(input)


def setup_database(force_recreate: bool = False):
    """
    Setup and validate ChromaDB
    
    Args:
        force_recreate: If True, delete and recreate collection
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("Initializing Vector Database...")

        # Get paths from environment or use defaults
        db_path = os.getenv("VECTORDIR", "./vector_db")
        data_dir = os.getenv("DATA_DIR", "./data")

        # Ensure directories exist
        os.makedirs(db_path, exist_ok=True)
        logger.info(f"Database path: {os.path.abspath(db_path)}")
        logger.info(f"Data directory: {os.path.abspath(data_dir)}")

        # Initialize ChromaDB client
        client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        logger.info("ChromaDB client initialized")

        # Initialize embedding function
        embedding_fn = ONNXEmbeddingFunction("./models")
        logger.info("ONNX embedding function loaded")

        # Handle force recreate
        if force_recreate:
            logger.warning("Force recreate enabled - deleting existing collection")
            try:
                client.delete_collection("knowledge_base")
                logger.info("Previous collection deleted")
            except Exception as e:
                logger.debug(f"Could not delete collection: {e}")

        # Get or create collection
        collection = client.get_or_create_collection(
            "knowledge_base",
            embedding_function=embedding_fn
        )

        # Get collection info
        count = collection.count()
        
        logger.info("=" * 60)
        logger.info("DATABASE STATUS")
        logger.info(f"Collection: {collection.name}")
        logger.info(f"Documents: {count}")
        logger.info(f"Location: {os.path.abspath(db_path)}")
        logger.info("=" * 60)

        if count == 0:
            logger.warning("Database is empty")
            logger.info("Run 'python ingest.py --folder ./data' to add documents")
        else:
            logger.info("Database is ready for queries")

        return True

    except Exception as e:
        logger.error(f"Database initialization failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup ChromaDB vector database")
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force recreate collection (deletes existing data)"
    )
    
    args = parser.parse_args()
    
    success = setup_database(force_recreate=args.force)
    
    if not success:
        logger.error("Setup failed")
        sys.exit(1)
    
    logger.info("Setup completed successfully")