# AI Context Compressor - Examples

This directory contains comprehensive examples demonstrating all features of the AI Context Compressor package.

## 📁 Example Files

### Core Usage Examples

#### [`batch_processing.py`](./batch_processing.py) 
**Demonstrates batch processing capabilities**
- Serial vs parallel processing comparison
- Different compression ratios
- Query-aware batch compression 
- Quality evaluation in batches
- Error handling strategies
- Performance optimization tips
- Large batch processing simulation
- Cache performance analysis

**Run with:**
```bash
python examples/batch_processing.py
```

#### [`quality_evaluation.py`](./quality_evaluation.py)
**Shows how to evaluate compression quality**
- Basic quality metrics (ROUGE, semantic similarity, entity preservation)
- Quality vs compression ratio analysis
- Quality evaluation by text type
- Query-aware quality evaluation
- Detailed quality analysis with interpretation
- Custom quality evaluator configuration
- Quality trends analysis across ratios
- Batch quality evaluation
- Quality score interpretation guide

**Run with:**
```bash
python examples/quality_evaluation.py
```

### Advanced Features

#### [`custom_strategy.py`](./custom_strategy.py)
**Guide to creating custom compression strategies**
- Three complete custom strategy implementations:
  - `RandomSelectionStrategy`: Random sentence selection
  - `KeywordBasedStrategy`: Keyword-based sentence scoring
  - `SummaryStrategy`: Extractive summarization
- Strategy registration and metadata
- Query-aware custom strategies
- Performance comparison between strategies
- Strategy validation and error handling
- Best practices for custom development

**Run with:**
```bash
python examples/custom_strategy.py
```

#### [`advanced_usage.py`](./advanced_usage.py)
**Production-ready patterns and advanced techniques**
- Custom tokenizers with domain-specific rules
- Adaptive compression strategy that analyzes content
- Compression pipelines with validation stages
- Advanced caching with multiple eviction strategies
- Production compressor with fallbacks and metrics
- Concurrent processing examples
- Performance monitoring and metrics collection

**Run with:**
```bash
python examples/advanced_usage.py
```

### Framework Integration

#### [`integration_examples.py`](./integration_examples.py)
**Integration with popular AI frameworks**
- LangChain document processing (simulated)
- OpenAI API integration with cost estimation
- Chat completion message compression
- Batch processing for API usage
- Custom AI service integration patterns
- RAG system integration example
- Performance monitoring integration

**Run with:**
```bash
python examples/integration_examples.py
```

#### [`api_client.py`](./api_client.py)
**REST API client examples and usage**
- Synchronous API client implementation
- Asynchronous API client with aiohttp
- Complete API endpoint coverage
- Concurrent request processing
- Error handling examples
- cURL command equivalents
- Production considerations

**Run with:**
```bash
# Note: Requires API server to be running
python examples/api_client.py
```

## 🚀 Getting Started

### 1. Basic Installation
```bash
pip install context-compressor
```

### 2. With Optional Dependencies
```bash
# For OpenAI integration
pip install "context-compressor[openai]"

# For LangChain integration  
pip install "context-compressor[langchain]"

# For REST API server
pip install "context-compressor[api]"

# All optional dependencies
pip install "context-compressor[all]"
```

### 3. Run Examples
```bash
# Basic usage
python examples/batch_processing.py

# Quality analysis
python examples/quality_evaluation.py

# Custom strategies
python examples/custom_strategy.py

# Framework integrations
python examples/integration_examples.py

# Advanced patterns
python examples/advanced_usage.py
```

### 4. Start API Server (for API client examples)
```bash
# Method 1: Direct module execution
python -m src.context_compressor.api.main

# Method 2: Using uvicorn
uvicorn src.context_compressor.api.main:app --reload

# Then run API client examples
python examples/api_client.py
```

## 📊 Example Outputs

### Compression Results
```
Original tokens: 150
Compressed tokens: 75  
Actual ratio: 50.0%
Tokens saved: 75
Strategy used: extractive
Processing time: 0.023s
```

### Quality Metrics
```
Overall Score: 0.742 ⭐⭐⭐
ROUGE-1: 0.834
ROUGE-L: 0.798  
Semantic Similarity: 0.652
Entity Preservation: 0.800
Readability: 67.3
```

### Batch Processing
```
Texts processed: 5
Success rate: 100.0%
Total tokens saved: 425
Average compression: 45.2%
Processing time: 0.156s
Throughput: 32.1 texts/second
```

## 🔧 Configuration Examples

### Custom Quality Evaluator
```python
from context_compressor.core.quality_evaluator import QualityEvaluator

evaluator = QualityEvaluator(
    semantic_weight=0.4,      # 40% semantic similarity
    rouge_weight=0.3,         # 30% ROUGE scores  
    entity_weight=0.2,        # 20% entity preservation
    readability_weight=0.1    # 10% readability
)
```

### Production Compressor
```python
from examples.advanced_usage import ProductionCompressor

compressor = ProductionCompressor(
    enable_caching=True,
    cache_config={
        'max_size': 1000,
        'ttl_seconds': 3600,
        'eviction_strategy': 'lru'
    },
    enable_metrics=True,
    enable_fallbacks=True,
    max_workers=4
)
```

### Custom Strategy
```python
from context_compressor.strategies.base import CompressionStrategy
from context_compressor.core.models import StrategyMetadata

class MyCustomStrategy(CompressionStrategy):
    def _create_metadata(self):
        return StrategyMetadata(
            name="my_strategy",
            description="My custom compression approach",
            version="1.0.0",
            # ... other metadata
        )
    
    def _compress_text(self, text, target_ratio, query=None, **kwargs):
        # Your compression logic here
        return compressed_text
```

## 🎯 Use Cases by Example

| Use Case | Example File | Key Features |
|----------|--------------|-------------|
| **API Integration** | `integration_examples.py` | OpenAI cost savings, LangChain workflows |
| **Batch Processing** | `batch_processing.py` | High-throughput processing, parallel execution |
| **Quality Control** | `quality_evaluation.py` | Comprehensive quality metrics, validation |
| **Custom Algorithms** | `custom_strategy.py` | Domain-specific strategies, algorithm development |
| **Production Systems** | `advanced_usage.py` | Caching, monitoring, error handling |
| **Web Services** | `api_client.py` | REST API integration, microservices |

## 🔍 Feature Coverage

### ✅ Implemented Features
- [x] Multiple compression strategies (extractive, adaptive, custom)
- [x] Comprehensive quality evaluation (ROUGE, semantic, entity preservation)
- [x] Batch processing with parallel execution
- [x] Query-aware compression for context relevance
- [x] Advanced caching with TTL and eviction strategies
- [x] REST API with FastAPI integration
- [x] Framework integrations (LangChain, OpenAI)
- [x] Production-ready patterns (monitoring, fallbacks, error handling)
- [x] Custom strategy development framework
- [x] Performance optimization techniques

### 🚧 Future Enhancements
- [ ] GPU-accelerated compression strategies
- [ ] Multi-language support beyond English
- [ ] Real-time streaming compression
- [ ] Advanced neural compression models
- [ ] Integration with more AI frameworks

## 📚 Documentation

- **Package Documentation**: See main README.md
- **API Reference**: Available when running the API server at `/docs`
- **Strategy Development**: See `custom_strategy.py` for detailed guide
- **Production Deployment**: See `advanced_usage.py` for patterns

## 🤝 Contributing Examples

To contribute new examples:

1. **Follow the Pattern**: Use existing examples as templates
2. **Add Documentation**: Include docstrings and inline comments
3. **Test Thoroughly**: Ensure examples run without errors
4. **Update README**: Add your example to this documentation
5. **Consider Use Cases**: Address real-world scenarios

### Example Template
```python
#!/usr/bin/env python3
"""
[Example Name] for AI Context Compressor.

Brief description of what this example demonstrates.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from context_compressor import ContextCompressor

def demonstrate_feature():
    """Demonstrate specific feature."""
    print("🚀 Feature Demonstration")
    print("-" * 25)
    
    compressor = ContextCompressor()
    # Your example code here
    
def main():
    """Run all demonstrations."""
    print("📦 AI Context Compressor - [Example Name]")
    print("=" * 50)
    
    demonstrate_feature()
    
    print("\n✅ Example completed!")
    print("\n💡 Key Takeaways:")
    print("   • Takeaway 1")
    print("   • Takeaway 2")

if __name__ == "__main__":
    main()
```

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Huzaifa785/context-compressor/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Huzaifa785/context-compressor/discussions)
- **PyPI**: [context-compressor](https://pypi.org/project/context-compressor/)

## 📄 License

All examples are provided under the same license as the main package. See LICENSE file for details.

---

**Happy compressing! 🎉**