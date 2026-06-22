#### <img src="https://npkum.github.io/DocIQ-pub/images/kmai_animated.gif" width="35" height="35"/> DocIQ - User Guide

---

#### 1. System Overview
**DocIQ** is an enterprise-grade Retrieval-Augmented Generation (RAG) platform designed to ingest PDF documentation, create searchable vector embeddings, and allow users to query their knowledge base using natural language.

The system features a **Smart Caching Layer** that learns from previous questions to deliver faster answers over time, and a **Role-Based Access Control (RBAC)** system to ensure secure management of data.

---

#### 2. Access & Authentication
DocIQ separates functionality based on user roles. Use the sidebar navigation to switch between modules.

#### Roles and Permissions

| Feature | **General User** (Public) | **Viewer** (Read-Only) | **Admin** (Full Access) |
| :--- | :---: | :---: | :---: |
| **Query Knowledge Base** | ✓ | ✓ | ✓ |
| **View Upload Configuration** | ✗ | ✓ | ✓ |
| **Upload Documents** | ✗ | ✗ (Disabled) | ✓ |
| **View Strategies Overview** | ✗ | ✓ | ✓ |
| **View Audit Logs/Cache** | ✗ | ✓ | ✓ |
| **Delete Topics/Clear DB** | ✗ | ✗ (Disabled) | ✓ |

#### Login Credentials
*   **Admin:** Full control over uploads and database management.
*   **Viewer:** Audit and verification access only.

*(Read only login details - user-id: viewer, password: view123)*

---

#### 3. Module: Upload PDFs
**Access:** Gated (Requires Login)  
**Purpose:** Ingest new knowledge (PDFs) into the system.

#### Step-by-Step Upload Process

1.  **Authentication:** Navigate to "Upload PDFs" in the sidebar and log in.
2.  **Topic Assignment:**
    *   *Select existing topic:* Choose from the dropdown to add documents to an existing collection (e.g., "HR_Policies").
    *   *Create new topic:* Enter a unique name (no spaces recommended, e.g., "Engineering_Specs_2025") to start a new collection.
3.  **File Selection:** Drag and drop a PDF file. The system supports text-based PDFs and will utilize **OCR (Optical Character Recognition)** for scanned images if the checkbox is selected.
4.  **Configuration: Chunking Strategies**
    Select the strategy that best fits your document structure:

    *   **Adaptive Chunking (NLTK):** *Best for dense text.* Uses Natural Language Toolkit logic to respect sentence boundaries intelligently.
    *   **Recursive Sentence Chunking:** *Good balance.* Splits text by paragraphs and sentences using Regex logic.
    *   **Sliding Window:** *Legacy/Simple.* Strictly cuts text at fixed character limits.

    > **Tip:** For documents with complex legal clauses, use **Adaptive Chunking** with a larger overlap (e.g., 200 chars) to maintain context.

5.  **Execution:** Click **"Upload and Process"**.
    *   *Note for Viewers:* This button is disabled.
    *   The system will extract text, tables, and generate vector embeddings. A sample of processed chunks will be displayed upon success.

---

#### 4. Module: Ask Questions
**Access:** Public (No Login Required)  
**Purpose:** Query the vectorized data and retrieve answers.

#### How to Query
1.  **Select Context:** Choose the **Topic** you wish to search.
2.  **Filter (Optional):** Select a specific **Sub-topic** (derived automatically from filenames) to narrow the search scope.
3.  **Search Parameters:**
    *   *Relevance Threshold:* Adjusts strictness. Lower (0.4) is more inclusive; Higher (0.8) requires exact wording matches.
    *   *Top-k:* Number of document chunks to retrieve for the AI to analyze.
4.  **Fuzzy Match:** Controls how strictly the system looks for a *cached* answer before generating a new one.

#### Understanding the Results
*   **Generative Answer:** An AI-synthesized response based strictly on the provided context.
*   **Source Citations:** The specific PDF filenames used to generate the answer are listed.
*   **Metrics Dashboard:**
    *   **Cosine:** Similarity between the context and the answer. (Green > 0.85 is excellent).
    *   **ROUGE scores:** Text overlap statistics used to measure hallucinations.
*   **Feedback Loop:** Users can click **👍 (Helpful)** or **👎 (Not Helpful)**. This tags the entry in the cache database to improve future retrieval.

---

#### 5. Module: DocIQ-Admin
**Access:** Gated (Requires Login)  
**Purpose:** Database maintenance, auditing, and strategy review.

#### A. Chunking Strategies Overview
Provides a high-level table view of all documents in the system.
*   **Use Case:** verify consistency in data ingestion.
*   **Columns:** Document Name, Sub-topic, Strategy Used, Total Vector Records.

#### B. Manage Topics (Vector DB)
Manage the ChromaDB vector collections.
*   **Delete Entire Topic:** Permanently removes a collection and all associated embeddings.
*   **Delete Sub-topics:** Removes specific documents (grouped by sub-topic) while keeping the rest of the topic intact.
*   **Audit Report:** Upon deletion, the system generates a temporary audit table confirming the number of records removed.
*   *Permissions:* **Viewer** role can see these options but the "Delete" buttons are disabled.

#### C. Manage Cached DB
Manage the Q&A history and feedback loop.
*   **Manage Q&A:** View a full list of questions asked.
    *   View metrics (Cosine/ROUGE).
    *   Manually override feedback (Helpful/Not Helpful).
    *   Delete specific bad Q&A pairs.
*   **Filter & Sort:** Use sliders to find "low quality" answers (e.g., Cosine < 0.5) for review.
*   **Clear DB:** Options to clear the in-memory summary cache or wipe the entire SQLite history.
*   *Permissions:* **Viewer** role can filter and view data, but cannot delete rows or clear the database.

---

#### 6. Module: Knowledge Catalog
**Access:** Public (No Login Required)  
**Purpose:** Provide a structured overview of all knowledge assets available within the system.

#### Overview
The **Knowledge Catalog** module presents a comprehensive catalogue of all available **Topics** and **Sub-topics**, along with their associated reference documents and source information. This module serves as the authoritative index for understanding the scope and coverage of the knowledge base.

Each entry includes direct access to the underlying documents, which can be opened by selecting the hyperlink in the **Document** column. Source details are also provided to ensure transparency, traceability, and content authenticity.

In addition, the module includes **example questions** for each topic and sub-topic. These questions are intended to guide users in understanding the type of queries supported by the knowledge base and to assist with effective information discovery.

#### Key Capabilities
*   Centralised inventory of Topics and Sub-topics
*   Direct access to source documents via embedded links
*   Clear visibility of document origins and authoritative sources
*   Example question sets to support learning, validation, and AI-assisted querying

---

#### 7. Best Practices & Troubleshooting

| Issue | Recommendation |
| :--- | :--- |
| **"Answer not stated in context"** | The AI restricts itself to your documents to prevent lying. Try lowering the **Relevance Threshold** slider to 0.45 or 0.50 to pull in more broad context. |
| **PDF Tables not appearing** | Ensure the PDF is native (selectable text). While OCR is enabled, complex tables in scanned images are difficult to parse. |
| **Upload Fails** | Check if the file is password protected. Remove PDF passwords before uploading. |
| **Slow Performance** | If running on CPU, large "Top-k" values (e.g., > 10) will slow down generation. Keep Top-k between 3 and 5 for optimal speed. |

***
*DocIQ Enterprise Support*