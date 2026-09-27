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
```
