# LLM4Rec

# RAG LLM Product Recommender

---

## 🚀 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system for product recommendations. It leverages a combination of **Sentence Transformers** for efficient product similarity search and a **Google Gemini 1.5 Flash Large Language Model (LLM)** for generating nuanced, contextualized, and explainable product recommendations. The entire application is presented through an interactive **Streamlit** web interface, allowing users to explore recommendations based on various criteria.

The system supports two recommendation modes:
1.  **ML Content-based Only:** Purely relies on product description similarity to find top-k relevant items.
2.  **ML + LLM Enhanced (Gemini):** First retrieves top-k similar products using content similarity, then augments this information into a prompt for the Gemini LLM to generate more sophisticated and human-like recommendations.

---

## ✨ Features

* **Hybrid Recommendation:** Combines traditional content-based filtering with advanced LLM capabilities for richer recommendations.
* **Semantic Search:** Uses Sentence Transformers to understand the meaning of product descriptions and user queries for accurate retrieval.
* **Contextualized LLM Recommendations:** Gemini LLM generates detailed recommendations, explaining why certain products are suitable based on retrieved context and user preferences.
* **Interactive UI:** A user-friendly Streamlit interface for selecting products, inputting queries, and choosing recommendation modes.
* **Single-File Deployment:** All core logic (UI and RAG backend) is contained within the Streamlit application for simplified local execution.

---

## 🛠️ Technologies Used

* **Python 3.10+** (Recommended: Python 3.10 or 3.11 for PyTorch compatibility)
* **Streamlit:** For building the interactive web UI.
* **Sentence Transformers:** For generating product embeddings and performing semantic similarity search.
* **PyTorch:** Underlying library for Sentence Transformers.
* **Google Generative AI SDK (`google-generativeai`):** For interacting with the Gemini LLM.
* **Pandas:** For data handling and manipulation of the product database.

---

## ⚙️ Setup and Installation

Follow these steps to get the RAG LLM Product Recommender up and running on your local machine.

### 1. Clone the Repository (or prepare your files)

If your code is in a Git repository, start by cloning it:
```bash
git clone <your-repository-url>
cd <your-repository-folder>
