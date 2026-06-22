### <img src="https://npkum.github.io/agenticai/images/docIQ_learning.gif" width="40" height="40"/> DocIQ: Knowledge Engine
#### Secure, cloud/On-Premise Generative AI utilizing Open Source RAG Architecture
**Architecture:** CPU-Optimized RAG

---

#### Overview
**DocIQ** represents a paradigm shift in enterprise document intelligence. By leveraging a high-efficiency **Retrieval-Augmented Generation (RAG)** architecture, DocIQ delivers accurate, context-aware answers from internal knowledge bases without data ever leaving the application boundary.

Unlike cloud-dependent solutions, DocIQ operates entirely on open-source technology, optimized for constrained environments (2 vCPU), demonstrating that enterprise-grade AI does not require massive GPU clusters—only smart architecture.

---

#### Technical Architecture

The system utilizes a decoupled "retriever-generator" architecture designed for maximum throughput on minimal hardware.

| Component | Technology / Model | Enterprise Function |
| :--- | :--- | :--- |
| **Generative LLM** | `MBZUAI/LaMini-Flan-T5-248M` | **Inference Engine:** A distilled 248M parameter model fine-tuned for instruction following, capable of running efficiently on CPUs while minimizing hallucinations. |
| **Embedding Engine** | `sentence-transformers/all-MiniLM-L6-v2` | **Semantic Indexing:** Converts text to vector embeddings locally with high semantic density and millisecond latency. |
| **Vector Database** | **ChromaDB** (Persistent) | **Knowledge Store:** Serverless, persistent vector storage allowing instant semantic search across thousands of document chunks. |
| **Orchestration** | Python 3.10 + Streamlit | **Frontend/Backend:** Unified interface for ingestion, querying, and administrative management. |
| **Caching Layer** | SQLite + Fuzzy Logic | **Performance:** Local caching database to store QA history and serve repeated questions instantly. |

---

#### Key Features

#### 1. Role-Based Access Control (RBAC)
Security is foundational. DocIQ implements session-based authentication to segregate duties, complying with standard data governance policies.

*   **Admin Role:** Full privileges for document ingestion, topic creation, sub-topic management, and database cleaning.
*   **Viewer Role:** Restricted "Read-Only" access. Viewers can query the knowledge base and provide feedback but cannot alter the vector index or delete audit logs.

#### 2. Intelligent Adaptive Chunking
To maximize the context window of efficient LLMs, DocIQ employs sophisticated segmentation strategies during ingestion:

*   **Adaptive NLTK Chunking:** Utilizes Natural Language Processing to respect sentence boundaries, ensuring semantic context is never severed mid-sentence.
*   **Recursive & Sliding Window:** Fallback mechanisms ensure robust handling of unstructured text and OCR data.
*   **Metadata Tagging:** Every chunk is tagged with `topic`, `sub_topic`, and `chunking_strategy` for precise retrieval filtering.

#### 3. Integrated Hallucination Metrics & QA
Trust is the barrier to AI adoption. DocIQ builds trust through transparent observability displayed with every answer:

*   **Cosine Similarity Score:** Mathematically verifies how closely the generated answer aligns with the source documents.
*   **ROUGE Metrics (2 & L):** Measures the n-gram overlap to ensure the model isn't inventing new facts.
*   **Source Citation:** Every answer includes a JSON log of the exact `source_docs` used to generate the response.

#### 4. Smart Caching & Feedback Loops
The system accelerates over time using a local Metadata Layer.

*   **Fuzzy Match Caching:** If a user asks a question similar to a previous one (e.g., >78% similarity), the system serves a pre-validated answer instantly, bypassing the inference engine.
*   **RLHF-Lite (Feedback):** Users can vote "Helpful" 👍 or "Not Helpful" 👎. This data is stored in SQLite to build a dataset for future model fine-tuning.

---

#### Operational Capabilities

#### **Ingestion Module**
> *"Turn static PDFs into dynamic knowledge."*
*   Auto-detection of topics and sub-topics from filenames.
*   Extraction of tabular data utilizing `Camelot`.
*   **OCR Fallback:** Automatically utilizes `Tesseract` and `pdf2image` if no text layer is detected in scanned PDFs.

#### **Admin Console & Audit**
> *"Full Observability."*
*   **Strategy Overview:** Visual analytics showing which chunking strategies are applied across the document estate.
*   **Topic Management:** Granular deletion capabilities (delete entire topics or specific sub-topics).
*   **Audit Logging:** Automatic generation of deletion reports confirming when and what data was purged.

---

#### Infrastructure & Cost Efficiency

This application is engineered to run on :

*   **Compute:** 2 vCPU / 16GB RAM.
*   **GPU Requirement:** None.
*   **Data Sovereignty:** All processing is local. No data is transmitted to OpenAI, Anthropic, or Azure.
*   **API tokens Cost:** $0.00.

---

#### Conclusion

DocIQ validates that secure, private, and intelligent Knowledge Management can be achieved using open-source tools. by intelligently combining `ChromaDB` for storage, `Sentence-Transformers` for understanding, and `LaMini` for generation, it provides an Enterprise-Grade RAG solution compliant with strict data privacy requirements.