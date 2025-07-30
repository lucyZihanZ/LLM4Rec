# LLM4Rec

# RAG LLM Product Recommender

---

## 🚀 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system for product recommendations. It leverages a combination of **Sentence Transformers** for efficient product similarity search and a **LLM** for generating nuanced, contextualized, and explainable product recommendations. The entire application is presented through an interactive **Streamlit** web interface, allowing users to explore recommendations based on various criteria.

The system supports two recommendation modes:
1.  **ML Content-based Only:** Purely relies on product description similarity to find top-k relevant items.
2.  **ML + LLM Enhanced:** First retrieves top-k similar products using content similarity, then augments this information into a prompt for the Gemini LLM to generate more sophisticated and human-like recommendations.

## 🚀 Work Flow:
1. We use a traditional content-based similarity search to retrieve relevant information.

2. Based on the top-k most relevant results, we further enhance recommendations by incorporating potential user preferences.

3. Leveraging both the top-k relevant information and the enhanced recommendations, we utilize RAG (Retrieval-Augmented Generation) and LLMs to infer additional potentially relevant items.

All results are presented in a user-friendly interface. If users prefer more precise and standard answers, they can select the "Precise Recommendation" mode. For more diverse and comprehensive recommendations, they can choose the "Diverse suggestions" mode.

## User-friendly interface
![Interface](images/interface.png)


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
```
### 2. Create and Activate a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies. Due to known compatibility issues with PyTorch on Python 3.12, Python 3.10 or 3.11 is strongly recommended.
```bash
# Create a new conda environment with Python 3.10
conda create -n rag_recommender_env python=3.10

# Activate the environment
conda activate rag_recommender_env
```
###3. Install Dependencies
Install all necessary Python packages. Ensure you install PyTorch first, specifically choosing the correct version for your CUDA setup (if you have an NVIDIA GPU) or for CPU-only.

## 🚀 How to Run
After completing the setup steps:
Activate your virtual environment:

Bash
```
conda activate rag_recommender_env
Run the Streamlit application:
```
Bash
```
streamlit run frontend.py
```


