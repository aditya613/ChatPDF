# Insurance Policy RAG System

## Overview

This project provides a robust Retrieval Augmented Generation (RAG) system focused on insurance policy documents. It supports:

- conversational Q&A over one or more insurance PDFs,
- focused extraction of policy information like inclusions/exclusions,
- detection of important features such as room rent limits, ICU limits, zero dep cover, waiting periods, co-pay, deductibles, and sub-limits,
- citation-friendly answers grounded in retrieved chunks.


## Retrieval Augmented Generation (RAG)

### Introduction

RAG combines a retriever (vector search over your policy text) and a generator (LLM) to produce answers grounded in your uploaded documents.

### Workflow

1. **PDF Input**: Upload one or more policy PDFs.
2. **Chunking + Metadata**: Text is split with overlap and tagged with source/page metadata.
3. **Vector Index**: Chunks are embedded with all-MiniLM-L6-v2 and indexed with FAISS.
4. **Conversational Retrieval**: Questions are answered through retrieval + LLM generation.
5. **Policy Extraction Mode**: A dedicated extraction flow summarizes critical insurance terms and constraints.

### Benefits

- **Adaptability**: RAG adapts to situations where facts may evolve over time, making it suitable for dynamic knowledge domains.
- **Efficiency**: By combining retrieval and generation, RAG provides access to the latest information without the need for extensive model retraining.
- **Reliability**: The methodology ensures reliable outputs by leveraging both retrieval-based and generative approaches.

## Project Features

1. **Insurance-specific extraction** for inclusions, exclusions, room rent, zero dep, waiting periods, co-pay, deductibles, and sub-limits.
2. **RAG summary with citations** to source pages.
3. **Pattern-based validation layer** for quick keyword hit detection alongside LLM output.
4. **Environment-driven OpenAI configuration** via `.env`.
5. **Streamlit UI** with separate tabs for Q&A and feature extraction.

## Getting Started

To use the PDF Intelligence System:

1. Clone the repository.
   ```bash
   git clone https://github.com/ArmaanSeth/ChatPDF.git
   ```

2. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```

3. Add your OpenAI key in `.env`.

   Example:

   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-4o-mini
   OPENAI_TEMPERATURE=0.1
   ```

4. Run the application.
   ```bash
   streamlit run app.py
   ```

5. Open your browser and navigate to `http://localhost:8501`.

## License

This project is licensed under the [Apache License](LICENSE).

## Notes

- Keep `.env` private and never commit real API keys.
- If extraction quality needs improvement for your policy format, increase retriever depth (`k` and `fetch_k`) in [app.py](app.py).
