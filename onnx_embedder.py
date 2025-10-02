"""
Custom ONNX Embedding Function for BGE-M3
Supports multilingual text including Vietnamese
FIXED: Memory leak issues
"""

import os
import numpy as np
from typing import List, Union
import logging
import gc

try:
    import onnxruntime as ort
    from transformers import AutoTokenizer
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    raise ImportError("Please install: pip install onnxruntime transformers sentencepiece")

logger = logging.getLogger(__name__)


class ONNXEmbedder:
    """Lightweight ONNX-based embedder supporting multilingual text"""
    
    def __init__(self, model_path: str = "./models"):
        if not HAS_ONNX:
            raise ImportError("onnxruntime and transformers are required")
        
        self.model_path = model_path
        self.onnx_path = os.path.join(model_path, "model.onnx")
        self.tokenizer = None
        self.session = None
        self.initialized = False
        
        self._load_model()
    
    def _load_model(self):
        """Load ONNX model and tokenizer"""
        try:
            # Load tokenizer with proper settings for BGE-M3
            logger.info(f"Loading tokenizer from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
                trust_remote_code=True
            )
            
            # Verify ONNX model exists
            if not os.path.exists(self.onnx_path):
                raise FileNotFoundError(f"ONNX model not found at {self.onnx_path}")
            
            # Load ONNX session with optimized settings
            logger.info(f"Loading ONNX model from {self.onnx_path}")
            
            # Session options for better memory management
            sess_options = ort.SessionOptions()
            sess_options.enable_cpu_mem_arena = False  # Disable memory arena for better cleanup
            sess_options.enable_mem_pattern = False    # Disable memory pattern optimization
            sess_options.enable_mem_reuse = True       # Enable memory reuse
            
            providers = ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(
                self.onnx_path, 
                sess_options=sess_options,
                providers=providers
            )
            
            self.initialized = True
            logger.info("ONNX embedder initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise
    
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling over token embeddings with memory cleanup"""
        token_embeddings = model_output[0]
        input_mask_expanded = np.broadcast_to(
            np.expand_dims(attention_mask, -1),
            token_embeddings.shape
        ).astype(np.float32)
        
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        
        result = sum_embeddings / sum_mask
        
        # Clean up intermediate arrays
        del token_embeddings, input_mask_expanded, sum_embeddings, sum_mask
        
        return result
    
    def _normalize(self, embeddings):
        """Normalize embeddings to unit length"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        result = embeddings / np.clip(norms, a_min=1e-9, a_max=None)
        del norms
        return result
    
    def encode(
        self, 
        sentences: Union[str, List[str]], 
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode sentences to embeddings with memory optimization
        
        Args:
            sentences: Single sentence or list of sentences
            batch_size: Batch size for encoding
            normalize: Whether to normalize embeddings
        
        Returns:
            numpy array of embeddings
        """
        if not self.initialized:
            raise RuntimeError("Model not initialized")
        
        # Convert single string to list
        single_input = isinstance(sentences, str)
        if single_input:
            sentences = [sentences]
        
        # Filter out empty strings
        valid_sentences = [s for s in sentences if s and s.strip()]
        if not valid_sentences:
            logger.warning("No valid sentences to encode")
            return np.array([])
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(valid_sentences), batch_size):
            batch = valid_sentences[i:i + batch_size]
            
            # Tokenize with proper encoding
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors="np",
                return_attention_mask=True
            )
            
            # Prepare ONNX inputs
            onnx_inputs = {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
            }
            
            # Add token_type_ids if present
            if "token_type_ids" in encoded:
                onnx_inputs["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)
            
            # Run inference
            outputs = self.session.run(None, onnx_inputs)
            
            # Apply mean pooling
            embeddings = self._mean_pooling(outputs, encoded["attention_mask"])
            
            # Normalize if requested
            if normalize:
                embeddings = self._normalize(embeddings)
            
            all_embeddings.append(embeddings)
            
            # CRITICAL: Clean up batch data
            del encoded, onnx_inputs, outputs, embeddings, batch
        
        # Concatenate all batches
        result = np.vstack(all_embeddings)
        
        # Clean up batch list
        del all_embeddings
        
        # Return single embedding if input was single string
        if single_input:
            single_result = result[0]
            del result
            return single_result
        
        return result
    
    def __call__(self, input: Union[str, List[str]]) -> List[List[float]]:
        """ChromaDB-compatible interface with memory cleanup"""
        embeddings = self.encode(input)
        
        # Handle empty result
        if embeddings.size == 0:
            return []
        
        # Convert to list format
        if len(embeddings.shape) == 1:
            result = [embeddings.tolist()]
            del embeddings
            return result
        
        result = embeddings.tolist()
        del embeddings
        return result
    
    def clear_cache(self):
        """Explicitly clear tokenizer cache"""
        if self.tokenizer is not None:
            # Clear tokenizer cache if it exists
            if hasattr(self.tokenizer, '_tokenizer') and hasattr(self.tokenizer._tokenizer, 'clear_cache'):
                self.tokenizer._tokenizer.clear_cache()
        
        # Force garbage collection
        gc.collect()
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            if self.session:
                del self.session
            if self.tokenizer:
                del self.tokenizer
            gc.collect()
        except:
            pass


# Singleton instance
_global_embedder = None


def get_embedder(model_path: str = "./models") -> ONNXEmbedder:
    """Get or create global embedder instance"""
    global _global_embedder
    if _global_embedder is None:
        _global_embedder = ONNXEmbedder(model_path)
    return _global_embedder


def clear_embedder_cache():
    """Clear global embedder cache"""
    global _global_embedder
    if _global_embedder is not None:
        _global_embedder.clear_cache()