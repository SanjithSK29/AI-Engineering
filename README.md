# AI Engineering: 

A comprehensive implementation of Retrieval Augmented Generation systems exploring PDF processing, text embedding, semantic search, and language model integration using modern AI tools.

## Project Overview

This project demonstrates the complete RAG pipeline, from document ingestion through semantic retrieval and generation. It provides hands-on implementations of:

- PDF document processing and text extraction
- Intelligent text chunking with overlap for context preservation
- Dense vector embedding using state-of-the-art models
- Semantic similarity search and retrieval
- Language model integration using LiteLLM for provider flexibility
- Prompt engineering techniques for improved output quality

## Project Structure

```
Ai_Engineering/
├── Basic_RAG.ipynb              # Foundational RAG implementation with core components
├── RAG.ipynb                    # Advanced RAG implementation with full pipeline
├── prompt_engineering.ipynb     # Techniques and strategies for prompt optimization
├── data/                        # Directory containing source documents
│   └── Epstein.pdf              # Sample PDF document for processing
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Core Components

### 1. PDF Extraction

The project includes robust PDF text extraction using PyMuPDF (fitz), which:
- Iterates through all pages in a PDF document
- Extracts text while preserving readability
- Consolidates content into a single unified text string
- Provides a foundation for downstream processing

### 2. Text Chunking

Intelligent text segmentation that:
- Divides large documents into manageable chunks
- Implements configurable overlap between chunks to maintain context continuity
- Respects token limits of embedding and language models
- Reduces the risk of splitting critical information across boundaries

### 3. Embedding Generation

Dense vector representation using embedding models that:
- Support multiple providers through LiteLLM (OpenAI, HuggingFace, Anthropic, etc.)
- Capture semantic meaning in high-dimensional vector space
- Enable similarity-based retrieval beyond keyword matching
- Use the same embedding space for both documents and queries

### 4. Semantic Retrieval

Context-aware information retrieval that:
- Computes similarity between query and document embeddings
- Ranks and retrieves the most relevant chunks
- Preserves semantic relationships between documents and queries
- Forms the foundation for accurate generation

### 5. Language Model Integration

Multi-provider LLM support using LiteLLM that:
- Enables easy switching between providers (OpenAI, Anthropic, Google, etc.)
- Maintains consistent API across different models
- Supports both completion and embedding endpoints
- Allows customization of model parameters and behavior

### 6. Prompt Engineering

Optimization techniques including:
- Few-shot prompting for improved task understanding
- Context window management for efficient token usage
- Temperature and parameter tuning
- Chain-of-thought prompting for complex reasoning

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager
- API keys for your chosen LLM and embedding providers
- PDF documents for processing

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Ai_Engineering.git
cd Ai_Engineering
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure API keys:
   - Create a `.env` file in the project root
   - Add your API keys:
   ```
   OPENAI_API_KEY=your_openai_key
   HUGGINGFACE_API_KEY=your_huggingface_key
   ANTHROPIC_API_KEY=your_anthropic_key
   ```

### Required Packages

Core dependencies for the RAG pipeline:

- `litellm`: Unified interface for multiple LLM providers
- `pymupdf` (fitz): PDF document processing
- `numpy`: Numerical computing
- `jupyter`: Interactive notebook environment

## Notebooks Overview

### Basic_RAG.ipynb

Foundational implementation covering:
- PDF text extraction from documents
- Text chunking with configurable parameters
- Basic embedding generation
- Vector storage and retrieval
- Simple query processing

This notebook is ideal for understanding the core RAG concepts and getting started with the system.

### RAG.ipynb

Production-ready implementation featuring:
- Advanced text preprocessing and cleaning
- Intelligent chunking strategies
- Batch embedding generation for efficiency
- Vector database integration
- Full pipeline from query to generation
- Evaluation metrics and performance analysis

This notebook demonstrates a complete end-to-end RAG system suitable for real-world applications.

### prompt_engineering.ipynb

Comprehensive exploration of prompt optimization:
- Template-based prompt construction
- Few-shot learning examples
- Chain-of-thought prompting strategies
- Parameter tuning and analysis
- Output formatting and parsing
- Error handling and fallback strategies

## Usage Examples

### Running the Basic RAG Pipeline

1. Open `Basic_RAG.ipynb` in Jupyter
2. Configure your PDF path in the notebook
3. Execute cells sequentially to:
   - Extract text from PDF
   - Chunk the text appropriately
   - Generate embeddings
   - Perform similarity search
   - Generate responses

### Adding Custom Documents

To process your own documents:

1. Place PDF files in the `data/` directory
2. Update the `pdf_path` variable in the notebook
3. Run the extraction and embedding cells
4. Perform queries against the new document

### Switching LLM Providers

LiteLLM makes provider switching seamless. Update the model parameter:

```python
# Using OpenAI
response = completion(model="gpt-4", messages=...)

# Using Claude
response = completion(model="claude-3-opus", messages=...)

# Using Gemini
response = completion(model="gemini-pro", messages=...)
```

## Key Concepts

### Vector Embeddings

Embeddings transform text into numerical vectors that capture semantic meaning. Two pieces of text with similar meaning will have vectors that are close together in vector space, enabling semantic similarity search.

### Retrieval Augmented Generation

RAG augments language models with external knowledge by:
1. Embedding user query in the same space as documents
2. Retrieving most similar document chunks
3. Passing retrieved context to the language model
4. Generating responses based on both query and context

This approach reduces hallucinations and improves accuracy compared to standalone language models.

### Text Chunking Strategy

Effective chunking balances multiple factors:
- Chunk size affects embedding quality and context window limits
- Overlap preserves context continuity between chunks
- Too large: loses retrieval precision
- Too small: loses contextual information

### Semantic Search

Finding relevant information through embedding similarity rather than keyword matching enables:
- Understanding user intent beyond exact wording
- Discovering related concepts
- Handling synonyms and paraphrasing naturally

## Configuration Options

### Embedding Model Selection

The default model is `huggingface/sentence-transformers/all-MiniLM-L6-v2`, a lightweight and efficient model. Alternatives include:

- `text-embedding-ada-002`: OpenAI's high-quality embeddings
- `text-embedding-3-large`: Latest OpenAI model with better performance
- `nomic-embed-text`: High-performance open-source option
- Custom fine-tuned models for domain-specific tasks

### Chunking Parameters

Adjust these in the notebooks based on your use case:

```python
chunk_size = 1000  # Characters per chunk
overlap = 200      # Character overlap between chunks
```

- Larger chunks: Better context but reduced retrieval precision
- Smaller chunks: Better precision but potential context loss
- Higher overlap: Better context continuity but increased redundancy

### LLM Parameters

Customize model behavior:

```python
response = completion(
    model="model-name",
    messages=messages,
    temperature=0.7,      # Creativity vs consistency
    max_tokens=500,       # Response length limit
    top_p=0.95           # Nucleus sampling parameter
)
```

## Troubleshooting

### API Key Issues
- Verify `.env` file is in project root
- Check API keys have appropriate permissions
- Ensure keys are not accidentally committed to Git

### PDF Processing Errors
- Verify PDF file is valid and readable
- Check file path is correct and accessible
- Some PDFs may have encoding issues requiring preprocessing

### Embedding Generation Failures
- Confirm API credentials are valid
- Check rate limits haven't been exceeded
- Verify text encoding is UTF-8 compatible

### Memory Issues
- Reduce chunk size for large documents
- Process documents in batches
- Use smaller embedding models for resource-constrained environments

## Performance Considerations

### Optimization Tips

- Use batch processing for multiple documents
- Cache embeddings to avoid recomputation
- Implement approximate nearest neighbor search for large datasets
- Choose embedding models based on speed vs quality tradeoff

### Scaling to Production

For production deployment:
- Use vector database (Pinecone, Weaviate, Milvus)
- Implement caching layers
- Monitor API costs and rate limits
- Add comprehensive logging and error handling
- Implement authentication and authorization

## Contributing

Contributions are welcome. Please:
1. Fork the repository
2. Create a feature branch
3. Commit changes with clear messages
4. Push to the branch
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## References

### RAG Systems
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2021)
- "In-Context Learning Makes Multitask Learning Trivial" (Min et al., 2022)

### Embeddings
- "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" (Reimers & Gurevych, 2019)
- "Dense Passage Retrieval for Open-Domain Question Answering" (Karpukhin et al., 2020)

### Language Models
- OpenAI GPT-4 Documentation
- Anthropic Claude Documentation
- LiteLLM Documentation

## Contact

For questions or feedback, please open an issue in the repository.

## Acknowledgments

This project leverages:
- PyMuPDF for PDF processing
- LiteLLM for unified LLM API
- Hugging Face Transformers for embedding models
- OpenAI, Anthropic, and other model providers
