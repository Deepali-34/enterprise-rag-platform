
# Retrieval Evaluation Report

## 1. Overview

This document describes the retrieval evaluation process for the Enterprise RAG Platform.

The objective is to measure how effectively the retrieval pipeline identifies relevant source documents for user queries.

The evaluation uses the production retrieval pipeline, including:

- Multi-query retrieval
- Hybrid search
- BM25 keyword retrieval
- Dense vector retrieval
- Context compression
- Cross-encoder reranking

The evaluation is performed independently of LLM answer generation.

## 2. Evaluation Configuration

| Parameter | Configuration |
|---|---|
| Evaluation type | Multi-document retrieval |
| Evaluation queries | 6 |
| Retrieval depth (K) | 3 |
| Vector database | ChromaDB |
| Embedding model | all-MiniLM-L6-v2 |
| Reranking model | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Retrieval implementation | Production search_documents() pipeline |
| Evaluation output | JSON report |

## 3. Evaluation Dataset

The benchmark contains six queries targeting different indexed PDF documents.

| Test | Query Topic | Expected Source |
|---|---|---|
| 1 | Java fundamentals, JDK, JRE and JVM | JAVA SYLLABUS.pdf |
| 2 | Java functions, loops and programming exercises | Java Exercise 1.pdf |
| 3 | One-dimensional arrays | Java Lecture 10.pdf |
| 4 | Two-dimensional arrays and matrices | Java Lecture 11.pdf |
| 5 | RAG architecture and retrieval strategies | RAG_LLM_Architecture_Guide.pdf |
| 6 | Internetworking practical examination | big data.pdf |

### Duplicate Document Handling

The indexed files `RAG_LLM_Architecture_Guide.pdf` and `API_Test_RAG_Guide.pdf` have identical SHA256 hashes.

For evaluation purposes, both filenames are normalized to one canonical source:

`RAG_LLM_Architecture_Guide.pdf`

This prevents identical files from being counted as separate relevant sources.

The original PDFs and ChromaDB records are preserved.

## 4. Evaluation Metrics

### 4.1 Precision@3

Measures the proportion of the top three retrieved chunks attributed to the expected source document.

Precision@3 = Relevant source-attributed chunks / 3

### 4.2 Source Recall@3

Measures whether the expected source document appears among the top three retrieved chunks.

Source Recall@3 = Expected sources retrieved / Total expected sources

### 4.3 Mean Reciprocal Rank (MRR)

Measures the ranking position of the first retrieved chunk attributed to an expected source.

MRR = Average of (1 / rank of first relevant result)

### 4.4 Average Retrieval Latency

Measures the average time taken by the retrieval pipeline to process an evaluation query.

Average Latency = Total retrieval time / Number of successful queries

## 5. Evaluation Results

The following results were obtained on 23 November 2026.

| Test | Precision@3 | Source Recall@3 | Reciprocal Rank | Latency (s) |
|---|---:|---:|---:|---:|
| Java Fundamentals | 1.00 | 1.00 | 1.00 | 1.824 |
| Java Exercises | 0.33 | 1.00 | 1.00 | 0.748 |
| One-dimensional Arrays | 0.67 | 1.00 | 1.00 | 0.832 |
| Two-dimensional Arrays | 1.00 | 1.00 | 1.00 | 0.655 |
| RAG Architecture | 1.00 | 1.00 | 1.00 | 0.472 |
| Internetworking | 0.67 | 1.00 | 1.00 | 0.497 |

### Aggregate Results

| Metric | Result |
|---|---:|
| Total test cases | 6 |
| Successful test cases | 6 |
| Average Precision@3 | 0.7778 |
| Average Source Recall@3 | 1.0000 |
| Mean Reciprocal Rank | 1.0000 |
| Average Retrieval Latency | 0.838 seconds |

## 6. Result Analysis

### Strengths

- All six evaluation queries successfully retrieved their expected source document within the top three results.
- The first relevant source-attributed chunk appeared at rank one for every query.
- The RAG architecture and Java fundamentals queries achieved 100% Precision@3.
- The average retrieval latency was approximately 0.84 seconds.

### Areas for Improvement

**Java Exercises**

The query achieved Precision@3 of 0.33. The retrieval pipeline returned chunks from the Java syllabus and arrays lecture alongside the expected exercise document.

**One-dimensional Arrays**

The pipeline retrieved a chunk from the two-dimensional arrays lecture before the expected one-dimensional arrays document.

**Internetworking**

The expected document appeared twice in the top three, but an unrelated RAG document also appeared.

These results suggest opportunities to improve topic discrimination and reduce irrelevant retrieval results.

## 7. Limitations

The current benchmark uses document-level relevance labels.

A retrieved chunk is considered relevant when its source filename matches the expected document. This does not establish that the individual chunk is semantically relevant to the query.

Consequently:

- Precision@3 is a source-label proxy, not human-judged chunk relevance.
- Source Recall@3 measures expected document retrieval, not complete information recall.
- MRR measures the position of the first chunk from the expected source.
- The benchmark currently contains six queries and should be expanded before making broad performance claims.

A future evaluation should include manually verified relevant chunks and a larger, more diverse query dataset.

## 8. Reproducing the Evaluation

Run the following command from the backend directory:

```powershell
.\.venv\Scripts\python.exe .\evaluation\evaluate_retrieval.py
```

The script generates:

`backend/evaluation/retrieval_evaluation_report.json`

The preserved baseline is:

`backend/evaluation/retrieval_evaluation_baseline.json`

## 9. Conclusion

The initial multi-document benchmark demonstrates that the Enterprise RAG Platform can retrieve the expected source document within the top three results across six different queries.

The system achieved:

- 77.78% average Precision@3
- 100% Source Recall@3
- 1.00 Mean Reciprocal Rank
- 0.838-second average retrieval latency

These results establish a preliminary retrieval baseline for future optimization and evaluation.

The next improvement is to introduce manually judged chunk-level relevance labels and expand the benchmark dataset.
