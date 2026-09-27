# RAG Document QA System

A **Retrieval-Augmented Generation (RAG) Document Question Answering system** designed to provide students with accurate, context-aware answers from uploaded educational documents.

The system retrieves relevant content from documents, processes the retrieved context, generates answers using **Google Gemini**, and provides **references to the original document pages**. Students can click a reference to directly access the corresponding document page stored in **AWS S3**.

The system also supports **streaming answers**, allowing students to receive the generated response progressively instead of waiting for the complete answer.

---

## Key Features

### 1. Retrieval-Augmented Generation

The system follows a RAG pipeline:

```text
Student Question
       │
       ▼
Query Processing
       │
       ▼
Query Decomposition
       │
       ▼
Document Retrieval
       │
       ▼
Semantic Search
       │
       ▼
Reranking
       │
       ▼
Relevant Context
       │
       ▼
Google Gemini
       │
       ▼
Generated Answer
       │
       ├──────────────► References
       │                  │
       │                  ▼
       │              AWS S3 Page
       │
       ▼
Student
```

Instead of generating an answer only from the LLM's internal knowledge, the system retrieves relevant information from the available educational documents and uses that information as context for answer generation.

---

## 2. Source References with Page-Level Navigation

A key feature of the system is **source/reference-based answers**.

When an answer is generated from retrieved document content, the response can include references to the source documents.

For example:

```text
Answer:

Newton's Second Law states that force is equal to mass
multiplied by acceleration.

References:
[Physics Chapter 2 - Page 14]
[Physics Chapter 2 - Page 15]
```

When the student clicks a reference, the system can direct the student to the corresponding document/page stored in **AWS S3**.

### Reference Flow

```text
Retrieved Chunk
      │
      ├── Document Name
      ├── Page Number
      ├── Retrieved Text
      └── S3 Location
              │
              ▼
        Reference URL
              │
              ▼
       Student clicks
              │
              ▼
       AWS S3 document/page
```

This provides **traceability between the generated answer and the original learning material**.

Students can therefore verify the answer by opening the referenced source.

---

## 3. Streaming Answers

The system supports **streaming answer generation**.

Instead of waiting for the complete LLM response:

```text
Request
   │
   ▼
LLM Processing
   │
   ▼
Complete Answer
   │
   ▼
Student
```

the system can stream the generated response progressively:

```text
Request
   │
   ▼
LLM Processing
   │
   ├── Token/Chunk 1 ──► Student
   ├── Token/Chunk 2 ──► Student
   ├── Token/Chunk 3 ──► Student
   ├── Token/Chunk 4 ──► Student
   └── ...
```

This improves the perceived response time and allows students to start reading the answer while the model is still generating the remaining content.

---

## 4. Multilingual Question Processing

The system includes multilingual processing capabilities.

The pipeline can process queries in different languages and use the retrieved context to generate an appropriate response.

The architecture includes:

* Language detection
* Multilingual query processing
* Multilingual retrieval
* Language-aware answer generation

This is useful for educational applications where students may ask questions in English or regional languages.

---

## 5. Query Decomposition

Complex questions can be divided into smaller subqueries before retrieval.

For example:

```text
Original Question:

"Explain Newton's second law, give its mathematical formula,
and provide a practical example."
```

can be decomposed into:

```text
Subquery 1:
What is Newton's second law?

Subquery 2:
What is the mathematical formula?

Subquery 3:
What is a practical example?
```

The system can then retrieve relevant context for the individual subqueries.

---

## 6. Semantic Retrieval

The system uses semantic retrieval to identify document content relevant to the student's question.

Instead of relying only on exact keyword matching, semantic retrieval helps identify content with similar meaning.

Example:

```text
Student Query:
"What happens when the force applied to an object increases?"

Relevant document:
"According to Newton's second law, acceleration
is directly proportional to the applied force..."
```

The query and document do not need to use exactly the same words to be considered relevant.

---

## 7. Reranking

Retrieved results can be reranked before being passed to the LLM.

The reranking stage helps prioritize the most relevant document chunks.

```text
Initial Retrieval
       │
       ▼
Multiple Candidate Chunks
       │
       ▼
Reranking
       │
       ▼
Most Relevant Context
       │
       ▼
LLM
```

This helps improve the quality of the context provided to the language model.

---

## 8. Semantic Chunking

Documents are divided into meaningful chunks before indexing.

Instead of blindly splitting text at fixed character boundaries, semantic chunking attempts to preserve meaningful portions of the document.

```text
PDF
 │
 ▼
Text Extraction
 │
 ▼
Semantic Chunking
 │
 ├── Chunk 1
 ├── Chunk 2
 ├── Chunk 3
 └── ...
       │
       ▼
Retrieval / Indexing
```

---

## 9. Redis-Based Retrieval

Redis is used as part of the retrieval architecture.

The system contains dedicated Redis search and storage components for:

* Document retrieval
* Search operations
* Cached data
* Retrieval optimization

---

## 10. Semantic Cache

The system includes semantic caching.

When a new question is semantically similar to a previously processed question, cached information can potentially be reused.

```text
New Query
   │
   ▼
Semantic Cache
   │
   ├── Similar result found
   │        │
   │        ▼
   │      Return cached result
   │
   └── No similar result
            │
            ▼
       RAG Pipeline
```

This can reduce unnecessary processing and improve response performance.

---

# Technology Stack

| Component            | Technology                     |
| -------------------- | ------------------------------ |
| Programming Language | Python                         |
| LLM                  | Google Gemini                  |
| RAG                  | Retrieval-Augmented Generation |
| Retrieval            | Redis / Semantic Search        |
| Database             | PostgreSQL                     |
| Object Storage       | AWS S3                         |
| Embeddings           | Gemini Embedding               |
| Evaluation           | RAGAS                          |
| Caching              | Semantic Cache                 |
| Document Processing  | PDF Processing                 |
| Response Generation  | Google Gemini                  |
| Streaming            | Streaming response pipeline    |

---

# System Architecture

```text
                         ┌──────────────────────┐
                         │       Student        │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Student Query     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Query Processing     │
                         │ Language Detection   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Query Decomposition  │
                         └──────────┬───────────┘
                                    │
                                    ▼
              ┌─────────────────────────────────────────┐
              │              Retrieval                  │
              │                                         │
              │  Redis Search + Semantic Retrieval     │
              └────────────────────┬────────────────────┘
                                   │
                                   ▼
                         ┌──────────────────────┐
                         │      Reranking       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Relevant Context    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Google Gemini     │
                         └──────────┬───────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     │                             │
                     ▼                             ▼
              Streaming Answer              Source References
                     │                             │
                     │                             ▼
                     │                       AWS S3 Document
                     │                       / Page
                     │
                     ▼
                  Student
```

---

# Project Structure

```text
rag-document-qa/
│
├── advanced_services/
│   ├── multilingual_pipeline.py
│   ├── query_decomposition.py
│   ├── reranker.py
│   └── semantic_chunking.py
│
├── .env.example
├── .env.ragas.example
├── .gitignore
│
├── config.py
├── database.py
├── init.py
├── language_utils.py
├── llm_helpers.py
├── main.py
├── models.py
├── pdf_processing.py
├── prompts.py
├── question_extraction.py
├── question_paper_generator.py
├── ragas_evaluator.py
├── ragas_local_eval.py
├── redis_search.py
├── redis_store.py
├── requirements.txt
├── s3.py
├── semantic_cache.py
└── utils.py
```

---

# Core Components

## `main.py`

Main application entry point.

It coordinates the application flow and connects the different RAG services.

---

## `redis_search.py`

Provides Redis-based search and retrieval functionality.

Responsibilities include:

* Query processing
* Document retrieval
* Similarity filtering
* Token overlap
* Context selection
* Gemini integration

---

## `redis_store.py`

Handles Redis storage and related operations.

---

## `advanced_services/multilingual_pipeline.py`

Provides multilingual processing capabilities.

---

## `advanced_services/query_decomposition.py`

Decomposes complex user questions into smaller searchable queries.

---

## `advanced_services/reranker.py`

Reranks retrieved document chunks based on relevance.

---

## `advanced_services/semantic_chunking.py`

Creates semantically meaningful document chunks.

---

## `semantic_cache.py`

Provides semantic caching for similar queries.

---

## `pdf_processing.py`

Handles PDF document processing and text extraction.

---

## `s3.py`

Handles AWS S3 integration and document storage/access.

S3 references can be associated with retrieved document pages so that students can navigate back to the original source.

---

## `ragas_evaluator.py`

Provides RAG evaluation functionality using RAGAS.

---

## `ragas_local_eval.py`

Supports local RAG evaluation workflows.

---

## `question_extraction.py`

Provides question extraction functionality.

---

## `question_paper_generator.py`

Provides question-paper generation functionality using the LLM.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/sadanandsawargave01-sys/rag-document-qa.git
cd rag-document-qa
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment on Linux/macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Environment Configuration

Create a local environment file from the example:

```bash
cp .env.example .env
```

Then update the values in `.env`.

Example:

```env
# Google / Gemini
GOOGLE_API_KEY=your-google-api-key

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rag_database
DB_USER=postgres
DB_PASSWORD=your-database-password

# Redis
REDIS_HOST=127.0.0.1
REDIS_PORT=6379

# AWS / S3
AWS_ACCESS_KEY=your-aws-access-key
AWS_SECRET_KEY=your-aws-secret-key
S3_BUCKET=your-s3-bucket
S3_REGION=ap-south-1

# RAGAS
RAGAS_EVALUATOR_MODEL=gemini-2.5-flash
RAGAS_EMBEDDING_MODEL=models/gemini-embedding-001
```

**Never commit the real `.env` file to GitHub.**

Only the example environment files should be committed.

---

# Redis Setup

Make sure Redis is running:

```bash
redis-server
```

Default configuration:

```env
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
```

---

# PostgreSQL Setup

Configure PostgreSQL using the environment variables:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rag_database
DB_USER=postgres
DB_PASSWORD=your-database-password
```

---

# AWS S3 Configuration

Configure AWS S3 through environment variables:

```env
AWS_ACCESS_KEY=your-aws-access-key
AWS_SECRET_KEY=your-aws-secret-key
S3_BUCKET=your-s3-bucket
S3_REGION=ap-south-1
```

AWS credentials should never be hard-coded in the source code.

---

# Running the Application

After configuring the required services:

```bash
python main.py
```

If the application is run using Uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Use the appropriate command for the configured deployment environment.

---

# RAGAS Evaluation

The project contains RAGAS evaluation components for evaluating RAG performance.

Create the local RAGAS environment:

```bash
cp .env.ragas.example .env.ragas
```

Configure:

```env
GOOGLE_API_KEY=your-google-api-key
RAGAS_EVALUATOR_MODEL=gemini-2.5-flash
RAGAS_EMBEDDING_MODEL=models/gemini-embedding-001
RAGAS_AUTO_EVALUATE=false
```

RAGAS can be used to evaluate aspects of retrieval and generated answers.

---

# Answer and Reference Flow

The complete student interaction can be summarized as:

```text
Student asks a question
          │
          ▼
Query processing
          │
          ▼
Relevant documents retrieved
          │
          ▼
Retrieved chunks reranked
          │
          ▼
Context sent to Gemini
          │
          ▼
Answer generated
          │
          ├──────────────► References generated
          │                       │
          │                       ▼
          │                  S3 document/page
          │
          ▼
Streaming answer shown to student
```

The student can read the answer while it is being generated and use the provided references to verify the information against the original document.

---

# Security

Sensitive configuration is intentionally excluded from the repository.

The following files should not be committed:

```text
.env
.env.ragas
*.pem
*.key
*.crt
```

The repository uses `.gitignore` to prevent accidental commits of these files.

If a credential is accidentally exposed:

1. Revoke or rotate the credential.
2. Generate a new credential.
3. Update the local environment.
4. Remove the exposed credential from Git history when necessary.

---

# Development Workflow

Check repository status:

```bash
git status
```

Stage changes:

```bash
git add .
```

Review staged changes:

```bash
git diff --cached
```

Commit:

```bash
git commit -m "Describe your change"
```

Push:

```bash
git push
```

---

# Future Improvements

Possible future enhancements include:

* Improved hybrid retrieval
* Advanced metadata filtering
* Better citation generation
* Page-level citation validation
* Streaming reference generation
* Retrieval observability
* Distributed Redis deployment
* Background document processing
* Automated RAG evaluation
* Improved multilingual evaluation
* Production containerization
* Improved monitoring and tracing

---

# License

Add the appropriate license for your project.

---

# Author

**Sadanand Sawargave**

GitHub:

https://github.com/sadanandsawargave01-sys

