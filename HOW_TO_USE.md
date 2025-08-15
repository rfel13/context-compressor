# Context Compressor - Complete How-To Guide

**Master the most powerful AI text compression library in Python**

*By Mohammed Huzaifa*

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Installation Guide](#installation-guide)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [Compression Strategies](#compression-strategies)
6. [Quality Evaluation](#quality-evaluation)
7. [Framework Integrations](#framework-integrations)
8. [REST API Usage](#rest-api-usage)
9. [Batch Processing](#batch-processing)
10. [Performance Optimization](#performance-optimization)
11. [Custom Strategies](#custom-strategies)
12. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### 1. Install the Package
```bash
pip install context-compressor
```

### 2. Basic Compression
```python
from context_compressor import ContextCompressor

# Initialize compressor
compressor = ContextCompressor()

# Compress text
text = """
Artificial Intelligence (AI) is revolutionizing industries through machine learning,
natural language processing, and computer vision. These technologies enable automated
decision-making, predictive analytics, and intelligent automation across healthcare,
finance, transportation, and entertainment sectors. Companies are investing heavily
in AI research and development to gain competitive advantages and improve efficiency.
"""

result = compressor.compress(text, target_ratio=0.5)

print(f"Original ({len(text)} chars): {text}")
print(f"Compressed ({len(result.compressed_text)} chars): {result.compressed_text}")
print(f"Compression ratio: {result.actual_ratio:.1%}")
print(f"Tokens saved: {result.tokens_saved}")
```

---

## 📦 Installation Guide

### Full Installation (Recommended)
```bash
# All features included by default
pip install context-compressor
```

### Verify Installation
```python
import context_compressor
print(f"Version: {context_compressor.__version__}")

# Test basic functionality
from context_compressor import ContextCompressor
compressor = ContextCompressor()
result = compressor.compress("Test text for compression.", target_ratio=0.8)
print("✓ Installation successful!")
```

### Development Installation
```bash
git clone https://github.com/MohammedHuzaifa785/context-compressor.git
cd context-compressor
pip install -e ".[dev]"
```

---

## 🎯 Basic Usage

### Simple Text Compression

```python
from context_compressor import ContextCompressor

compressor = ContextCompressor()

# Basic compression
text = "Your long text here..."
result = compressor.compress(text, target_ratio=0.6)

print(f"Compressed text: {result.compressed_text}")
print(f"Quality score: {result.quality_metrics.overall_score:.2f}")
```

### Query-Aware Compression

```python
# Compress with focus on specific topics
text = """
The field of artificial intelligence encompasses machine learning, deep learning,
natural language processing, computer vision, and robotics. Machine learning
algorithms can be supervised, unsupervised, or reinforcement-based. Deep learning
uses neural networks with multiple layers to process complex data patterns.
"""

query = "machine learning algorithms"

result = compressor.compress(
    text=text,
    target_ratio=0.4,
    query=query  # Prioritizes content related to this query
)

print(f"Query-focused result: {result.compressed_text}")
```

### Controlling Compression Strategy

```python
# Use specific strategy
result = compressor.compress(
    text=text,
    target_ratio=0.5,
    strategy="extractive"  # Options: "extractive", "auto"
)

print(f"Strategy used: {result.strategy_used}")
print(f"Processing time: {result.processing_time:.3f}s")
```

---

## 🔧 Advanced Features

### Comprehensive Configuration

```python
from context_compressor import ContextCompressor
from context_compressor.utils.cache import CacheManager

# Custom cache configuration
cache_manager = CacheManager(
    ttl=7200,  # 2 hours
    max_size=1000,
    cleanup_interval=300  # 5 minutes
)

# Advanced compressor setup
compressor = ContextCompressor(
    enable_caching=True,
    cache_manager=cache_manager,
    enable_quality_evaluation=True,
    default_strategy="auto",
    max_workers=4
)

# Compression with detailed options
result = compressor.compress(
    text=long_text,
    target_ratio=0.4,
    query="specific topic",
    preserve_entities=True,  # Keep named entities
    min_sentence_length=20,  # Minimum sentence length to keep
    position_bias=0.3  # Prefer sentences from beginning/end
)

# Access detailed metrics
print(f"ROUGE-1 Score: {result.quality_metrics.rouge_1:.3f}")
print(f"ROUGE-2 Score: {result.quality_metrics.rouge_2:.3f}")
print(f"Semantic Similarity: {result.quality_metrics.semantic_similarity:.3f}")
print(f"Entity Preservation: {result.quality_metrics.entity_preservation_rate:.3f}")
```

### Caching and Performance

```python
# Enable/disable caching
compressor_with_cache = ContextCompressor(enable_caching=True, cache_ttl=3600)
compressor_no_cache = ContextCompressor(enable_caching=False)

# Check cache status
if compressor.cache_manager:
    print(f"Cache hits: {compressor.cache_manager.hits}")
    print(f"Cache misses: {compressor.cache_manager.misses}")
    print(f"Cache size: {compressor.cache_manager.size}")
```

---

## 📊 Compression Strategies

### 1. Extractive Strategy (Default)

```python
from context_compressor.strategies import ExtractiveStrategy

# Configure extractive strategy
extractive = ExtractiveStrategy(
    scoring_method="combined",  # "tfidf", "frequency", "position", "combined"
    min_sentence_length=15,
    position_bias=0.2,  # Prefer sentences from beginning/end
    query_weight=0.3  # Weight for query relevance
)

compressor = ContextCompressor(strategies=[extractive])

result = compressor.compress(text, target_ratio=0.5, strategy="extractive")
```

### 2. Automatic Strategy Selection

```python
# Let the system choose the best strategy
compressor = ContextCompressor(default_strategy="auto")

result = compressor.compress(text, target_ratio=0.4)
print(f"Automatically selected strategy: {result.strategy_used}")
```

### 3. Strategy Performance Comparison

```python
strategies = ["extractive", "auto"]
text = "Your long text here..."

for strategy in strategies:
    result = compressor.compress(text, target_ratio=0.5, strategy=strategy)
    print(f"{strategy.capitalize()} Strategy:")
    print(f"  Ratio: {result.actual_ratio:.1%}")
    print(f"  Quality: {result.quality_metrics.overall_score:.2f}")
    print(f"  Time: {result.processing_time:.3f}s")
    print()
```

---

## 📈 Quality Evaluation

### Understanding Quality Metrics

```python
result = compressor.compress(text, target_ratio=0.5)

metrics = result.quality_metrics

print("=== Quality Metrics ===")
print(f"Overall Score: {metrics.overall_score:.3f}")
print(f"ROUGE-1 (Unigram): {metrics.rouge_1:.3f}")
print(f"ROUGE-2 (Bigram): {metrics.rouge_2:.3f}")
print(f"ROUGE-L (Longest): {metrics.rouge_l:.3f}")
print(f"Semantic Similarity: {metrics.semantic_similarity:.3f}")
print(f"Entity Preservation: {metrics.entity_preservation_rate:.3f}")
print(f"Readability Score: {metrics.readability_score:.3f}")
```

### Quality Threshold Filtering

```python
# Only accept high-quality compressions
min_quality = 0.7

result = compressor.compress(text, target_ratio=0.3)

if result.quality_metrics.overall_score >= min_quality:
    print("✓ High quality compression achieved")
    print(result.compressed_text)
else:
    print("⚠ Quality below threshold, trying different ratio...")
    result = compressor.compress(text, target_ratio=0.5)
```

---

## 🔌 Framework Integrations

### LangChain Integration

```python
from langchain.schema import Document
from context_compressor.integrations.langchain import ContextCompressorTransformer

# Create documents
docs = [
    Document(page_content="Document 1 content...", metadata={"source": "doc1"}),
    Document(page_content="Document 2 content...", metadata={"source": "doc2"}),
]

# Create transformer
transformer = ContextCompressorTransformer(
    compressor=compressor,
    target_ratio=0.6,
    preserve_metadata=True
)

# Transform documents
compressed_docs = transformer.transform_documents(docs)

for doc in compressed_docs:
    print(f"Source: {doc.metadata['source']}")
    print(f"Compressed: {doc.page_content}")
    print()
```

### OpenAI Integration

```python
from context_compressor.integrations.openai import compress_for_openai
import openai

# Compress text before sending to OpenAI
long_context = "Your very long context here..."

compressed_prompt = compress_for_openai(
    text=long_context,
    target_ratio=0.4,
    model="gpt-4",  # Automatically uses correct tokenizer
    preserve_structure=True
)

# Use with OpenAI API
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": compressed_prompt}
    ]
)

print(f"Original tokens: ~{len(long_context.split())}")
print(f"Compressed tokens: ~{len(compressed_prompt.split())}")
print(f"Token savings: {(1 - len(compressed_prompt)/len(long_context)):.1%}")
```

### Custom Integration Example

```python
class MyFrameworkIntegration:
    def __init__(self, compressor):
        self.compressor = compressor
    
    def process_documents(self, documents, compression_ratio=0.5):
        results = []
        
        for doc in documents:
            # Compress each document
            result = self.compressor.compress(
                text=doc['content'], 
                target_ratio=compression_ratio,
                query=doc.get('query', None)
            )
            
            # Add to results with metadata
            results.append({
                'id': doc['id'],
                'original_length': len(doc['content']),
                'compressed_content': result.compressed_text,
                'compression_ratio': result.actual_ratio,
                'quality_score': result.quality_metrics.overall_score
            })
        
        return results

# Usage
integration = MyFrameworkIntegration(compressor)
documents = [
    {'id': 1, 'content': 'Document content...', 'query': 'search term'},
    {'id': 2, 'content': 'Another document...'}
]
results = integration.process_documents(documents, compression_ratio=0.6)
```

---

## 🌐 REST API Usage

### Starting the API Server

```bash
# Start the FastAPI server
uvicorn context_compressor.api.main:app --reload --host 0.0.0.0 --port 8000
```

### API Endpoints

#### 1. Single Text Compression

```bash
curl -X POST "http://localhost:8000/compress" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your long text here for compression...",
    "target_ratio": 0.5,
    "strategy": "extractive",
    "query": "optional search query",
    "enable_quality_evaluation": true
  }'
```

#### 2. Batch Compression

```bash
curl -X POST "http://localhost:8000/compress/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "First document content...",
      "Second document content...",
      "Third document content..."
    ],
    "target_ratio": 0.4,
    "strategy": "auto",
    "parallel": true,
    "max_workers": 4
  }'
```

#### 3. Health Check

```bash
curl "http://localhost:8000/health"
```

#### 4. Available Strategies

```bash
curl "http://localhost:8000/strategies"
```

#### 5. Compression Statistics

```bash
curl "http://localhost:8000/stats"
```

### Python Client for API

```python
import requests
import json

class ContextCompressorClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def compress(self, text, target_ratio=0.5, **kwargs):
        response = requests.post(
            f"{self.base_url}/compress",
            json={
                "text": text,
                "target_ratio": target_ratio,
                **kwargs
            }
        )
        return response.json()
    
    def compress_batch(self, texts, target_ratio=0.5, **kwargs):
        response = requests.post(
            f"{self.base_url}/compress/batch",
            json={
                "texts": texts,
                "target_ratio": target_ratio,
                **kwargs
            }
        )
        return response.json()

# Usage
client = ContextCompressorClient()

result = client.compress(
    text="Your text here...",
    target_ratio=0.6,
    strategy="extractive"
)

print(f"Compressed: {result['compressed_text']}")
print(f"Quality: {result['quality_metrics']['overall_score']}")
```

---

## ⚡ Batch Processing

### Basic Batch Processing

```python
texts = [
    "First document about artificial intelligence and machine learning applications...",
    "Second document discussing natural language processing and text analysis...",
    "Third document covering computer vision and image recognition techniques...",
    "Fourth document about deep learning architectures and neural networks..."
]

# Process batch
batch_result = compressor.compress_batch(
    texts=texts,
    target_ratio=0.5,
    parallel=True,  # Enable parallel processing
    max_workers=4   # Number of parallel workers
)

# Access results
print(f"Processed {len(batch_result.results)} documents")
print(f"Average compression ratio: {batch_result.average_compression_ratio:.1%}")
print(f"Total tokens saved: {batch_result.total_tokens_saved}")
print(f"Total processing time: {batch_result.total_processing_time:.2f}s")

# Individual results
for i, result in enumerate(batch_result.results):
    if result.success:
        print(f"Document {i+1}: {result.actual_ratio:.1%} compression")
    else:
        print(f"Document {i+1}: Failed - {result.error}")
```

### Advanced Batch Configuration

```python
# Custom batch processing with different settings per document
batch_configs = [
    {"target_ratio": 0.3, "query": "machine learning"},
    {"target_ratio": 0.5, "query": "natural language"},
    {"target_ratio": 0.4, "query": "computer vision"},
    {"target_ratio": 0.6, "strategy": "extractive"}
]

batch_result = compressor.compress_batch_with_configs(
    texts=texts,
    configs=batch_configs,
    parallel=True
)
```

### Memory-Efficient Batch Processing

```python
# For very large batches, process in chunks
def process_large_batch(compressor, texts, chunk_size=100, **kwargs):
    all_results = []
    
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        print(f"Processing chunk {i//chunk_size + 1}/{(len(texts)-1)//chunk_size + 1}")
        
        batch_result = compressor.compress_batch(
            texts=chunk,
            parallel=True,
            **kwargs
        )
        
        all_results.extend(batch_result.results)
    
    return all_results

# Process 1000 documents in chunks of 100
large_texts = ["Document content..." for _ in range(1000)]
results = process_large_batch(compressor, large_texts, target_ratio=0.5)
```

---

## 🚀 Performance Optimization

### Memory Optimization

```python
# For memory-constrained environments
compressor = ContextCompressor(
    enable_caching=False,  # Disable caching to save memory
    enable_quality_evaluation=False,  # Skip quality evaluation
    max_workers=2  # Reduce parallel workers
)

# Process with minimal memory usage
result = compressor.compress(
    text=large_text,
    target_ratio=0.5,
    strategy="extractive"  # Most memory-efficient strategy
)
```

### Speed Optimization

```python
# For maximum speed
compressor = ContextCompressor(
    enable_caching=True,  # Enable caching for repeated texts
    cache_ttl=7200,  # Long cache time
    enable_quality_evaluation=False,  # Skip time-consuming evaluation
    max_workers=8  # Use more parallel workers
)

# Pre-warm cache with common patterns
common_texts = ["Common text pattern 1...", "Common text pattern 2..."]
for text in common_texts:
    compressor.compress(text, target_ratio=0.5)
```

### Monitoring Performance

```python
import time

def benchmark_compression(compressor, texts, target_ratio=0.5):
    start_time = time.time()
    
    results = []
    for text in texts:
        result = compressor.compress(text, target_ratio=target_ratio)
        results.append(result)
    
    end_time = time.time()
    
    print(f"=== Performance Benchmark ===")
    print(f"Documents processed: {len(texts)}")
    print(f"Total time: {end_time - start_time:.2f}s")
    print(f"Average time per document: {(end_time - start_time)/len(texts):.3f}s")
    print(f"Documents per second: {len(texts)/(end_time - start_time):.1f}")
    
    # Quality statistics
    quality_scores = [r.quality_metrics.overall_score for r in results if r.quality_metrics]
    if quality_scores:
        print(f"Average quality score: {sum(quality_scores)/len(quality_scores):.3f}")
    
    return results

# Run benchmark
test_texts = ["Test document content..." for _ in range(100)]
benchmark_results = benchmark_compression(compressor, test_texts)
```

---

## 🎨 Custom Strategies

### Creating a Custom Strategy

```python
from context_compressor.strategies.base import CompressionStrategy
from context_compressor.core.models import StrategyMetadata

class CustomCompressionStrategy(CompressionStrategy):
    def __init__(self, custom_param=0.5):
        self.custom_param = custom_param
        super().__init__()
    
    def _create_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="custom",
            description="Custom compression strategy",
            version="1.0.0",
            author="Your Name",
            parameters={
                "custom_param": self.custom_param
            }
        )
    
    def _compress_text(self, text: str, target_ratio: float, **kwargs) -> str:
        """
        Implement your custom compression logic here.
        
        Args:
            text: Input text to compress
            target_ratio: Desired compression ratio (0.0 to 1.0)
            **kwargs: Additional parameters
        
        Returns:
            Compressed text
        """
        sentences = text.split('. ')
        
        # Simple example: keep first N sentences based on target ratio
        target_sentences = max(1, int(len(sentences) * target_ratio))
        compressed_sentences = sentences[:target_sentences]
        
        return '. '.join(compressed_sentences) + '.'
    
    def _validate_input(self, text: str, target_ratio: float, **kwargs) -> bool:
        """Validate input parameters."""
        if not text or len(text.strip()) == 0:
            return False
        if not 0.0 < target_ratio <= 1.0:
            return False
        return True

# Register and use custom strategy
custom_strategy = CustomCompressionStrategy(custom_param=0.7)
compressor = ContextCompressor(strategies=[custom_strategy])

result = compressor.compress(
    text="Your text here...",
    target_ratio=0.5,
    strategy="custom"
)

print(f"Custom compression result: {result.compressed_text}")
```

### Advanced Custom Strategy with ML

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticCompressionStrategy(CompressionStrategy):
    def __init__(self, similarity_threshold=0.8):
        self.similarity_threshold = similarity_threshold
        self.vectorizer = TfidfVectorizer(stop_words='english')
        super().__init__()
    
    def _create_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="semantic",
            description="Semantic similarity-based compression",
            version="1.0.0",
            author="Your Name"
        )
    
    def _compress_text(self, text: str, target_ratio: float, **kwargs) -> str:
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if len(sentences) <= 2:
            return text
        
        # Create TF-IDF vectors
        tfidf_matrix = self.vectorizer.fit_transform(sentences)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Select diverse sentences
        selected_indices = self._select_diverse_sentences(
            similarity_matrix, 
            target_count=max(1, int(len(sentences) * target_ratio))
        )
        
        # Return selected sentences in original order
        selected_indices.sort()
        compressed_sentences = [sentences[i] for i in selected_indices]
        
        return '. '.join(compressed_sentences) + '.'
    
    def _select_diverse_sentences(self, similarity_matrix, target_count):
        """Select diverse sentences using greedy algorithm."""
        n_sentences = similarity_matrix.shape[0]
        selected = [0]  # Start with first sentence
        
        while len(selected) < target_count and len(selected) < n_sentences:
            min_similarity = float('inf')
            best_candidate = -1
            
            for candidate in range(n_sentences):
                if candidate in selected:
                    continue
                
                # Find minimum similarity to already selected sentences
                max_sim_to_selected = max(similarity_matrix[candidate][i] for i in selected)
                
                if max_sim_to_selected < min_similarity:
                    min_similarity = max_sim_to_selected
                    best_candidate = candidate
            
            if best_candidate != -1:
                selected.append(best_candidate)
        
        return selected

# Use the advanced custom strategy
semantic_strategy = SemanticCompressionStrategy(similarity_threshold=0.7)
compressor = ContextCompressor(strategies=[semantic_strategy])

result = compressor.compress(text, target_ratio=0.4, strategy="semantic")
```

---

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors
```python
# Problem: ModuleNotFoundError
# Solution: Check installation
try:
    from context_compressor import ContextCompressor
    print("✓ Package imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Solution: pip install context-compressor")
```

#### 2. Strategy Not Found
```python
# Problem: Strategy 'xyz' not found
# Solution: Check available strategies
compressor = ContextCompressor()
available_strategies = compressor.get_available_strategies()
print(f"Available strategies: {available_strategies}")

# Use a valid strategy
result = compressor.compress(text, target_ratio=0.5, strategy="extractive")
```

#### 3. Text Too Short
```python
# Problem: Text too short for compression
def safe_compress(compressor, text, target_ratio=0.5, min_length=100):
    if len(text) < min_length:
        print(f"⚠ Text too short ({len(text)} chars), returning original")
        return text
    
    try:
        result = compressor.compress(text, target_ratio=target_ratio)
        return result.compressed_text
    except Exception as e:
        print(f"✗ Compression failed: {e}")
        return text

# Usage
compressed = safe_compress(compressor, short_text)
```

#### 4. Memory Issues
```python
# Problem: Out of memory during batch processing
# Solution: Process in smaller chunks
def memory_safe_batch_compress(compressor, texts, chunk_size=50, **kwargs):
    results = []
    
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        try:
            chunk_results = compressor.compress_batch(
                texts=chunk,
                parallel=False,  # Reduce memory usage
                **kwargs
            )
            results.extend(chunk_results.results)
        except MemoryError:
            print(f"Memory error on chunk {i//chunk_size + 1}, processing individually")
            for text in chunk:
                try:
                    result = compressor.compress(text, **kwargs)
                    results.append(result)
                except Exception as e:
                    print(f"Failed to process text: {e}")
    
    return results
```

#### 5. Performance Issues
```python
# Problem: Slow compression
# Solution: Profile and optimize
import cProfile
import pstats

def profile_compression():
    compressor = ContextCompressor(enable_quality_evaluation=False)
    
    def compress_test():
        for _ in range(10):
            compressor.compress("Test text " * 100, target_ratio=0.5)
    
    # Profile the code
    profiler = cProfile.Profile()
    profiler.enable()
    compress_test()
    profiler.disable()
    
    # Print stats
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)

# Run profiling
profile_compression()
```

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

compressor = ContextCompressor()
result = compressor.compress("Debug text...", target_ratio=0.5)

# Check detailed information
print(f"Strategy used: {result.strategy_used}")
print(f"Processing time: {result.processing_time:.3f}s")
if result.quality_metrics:
    print(f"Quality metrics: {result.quality_metrics}")
```

### Health Check Function

```python
def health_check():
    """Comprehensive health check for Context Compressor."""
    print("=== Context Compressor Health Check ===")
    
    try:
        # 1. Import test
        from context_compressor import ContextCompressor
        print("✓ Package import successful")
        
        # 2. Initialization test
        compressor = ContextCompressor()
        print("✓ Compressor initialization successful")
        
        # 3. Basic compression test
        test_text = "This is a test sentence. This is another test sentence. This is a third test sentence."
        result = compressor.compress(test_text, target_ratio=0.6)
        print("✓ Basic compression successful")
        
        # 4. Batch processing test
        test_texts = [test_text, test_text + " Extended.", test_text + " More content."]
        batch_result = compressor.compress_batch(test_texts, target_ratio=0.5, parallel=False)
        print("✓ Batch processing successful")
        
        # 5. Strategy test
        strategies = compressor.get_available_strategies()
        print(f"✓ Available strategies: {strategies}")
        
        # 6. Cache test
        if compressor.cache_manager:
            print("✓ Caching enabled")
        else:
            print("⚠ Caching disabled")
        
        print("\n🎉 All health checks passed!")
        return True
        
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Run health check
health_check()
```

---

## 📚 Additional Resources

### Performance Guidelines
- Use `target_ratio` between 0.3-0.7 for optimal results
- Enable caching for repeated compression tasks
- Disable quality evaluation for maximum speed
- Use parallel processing for large batches

### Best Practices
- Always validate input text length before compression
- Monitor quality metrics for critical applications
- Use query-aware compression for focused content
- Implement error handling for production use

### Getting Help
- **GitHub Issues**: [Report bugs and request features](https://github.com/MohammedHuzaifa785/context-compressor/issues)
- **Documentation**: [Complete API reference](https://github.com/MohammedHuzaifa785/context-compressor#readme)
- **Examples**: Check the `examples/` directory for more code samples

---

**Made with ❤️ by Mohammed Huzaifa**

*This guide covers all major features of Context Compressor v1.0.0. For the latest updates and features, check the GitHub repository.*