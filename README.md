# NeuroSeek 🧠

**An AI-powered question-answering system built with Java, vector search, and local language models.**

NeuroSeek combines semantic search with locally running AI models to retrieve relevant information and generate context-aware answers. It uses vector embeddings, multiple search algorithms, and Ollama-powered language models to build a retrieval-based question-answering system.

## ✨ Features

* **AI-Powered Question Answering** — Generate responses using a local language model.
* **Vector Search** — Search for relevant information using Brute Force, KD-Tree, and HNSW approaches.
* **Semantic Embeddings** — Convert text into vector representations for similarity-based retrieval.
* **Retrieval-Augmented Generation (RAG)** — Use retrieved information as context for answer generation.
* **Local AI Inference** — Run embedding and language models through Ollama.
* **Web Interface** — Interact with the application through its browser-based interface.

## 🛠️ Tech Stack

* **Backend:** Java
* **Frontend:** HTML
* **AI Runtime:** Ollama
* **Embedding Model:** `nomic-embed-text`
* **Language Model:** `llama3.2`
* **Core Concepts:** Vector databases, similarity search, embeddings, and RAG

## ⚙️ Prerequisites

Before running NeuroSeek, install:

* A compatible Java Development Kit (JDK)
* [Ollama](https://ollama.com/download)

Download the required models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

Ensure Ollama is running before launching the application.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/NeuroSeek.git
cd NeuroSeek
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Start the application

On Windows, run:

```bat
run.bat
```

The startup script compiles the Java source files and launches the application.

### 3. Open NeuroSeek

Visit the following address in your browser:

```text
http://localhost:8080
```

Keep the Java application and Ollama running while using NeuroSeek.

## 🧠 How It Works

1. Input text is converted into vector embeddings.
2. The search system retrieves relevant information using vector similarity.
3. Retrieved information is passed to the language model as context.
4. The model generates a response based on the available context and its general knowledge.

## 🌐 Deployment

NeuroSeek requires a running Java backend and access to Ollama with the required models. Hosting the frontend alone will not provide the complete AI functionality.

For a fully functional online deployment, configure a server that can run the backend and securely access the Ollama service.

## 📌 Project Goals

* Explore practical applications of AI and semantic search.
* Understand vector-search algorithms and information retrieval.
* Build a locally runnable question-answering system using Java.
* Integrate Java applications with locally hosted language models.

## 👨‍💻 Author

**YOUR NAME**

* GitHub: https://github.com/muskanpandey77

## 📄 License

Add the applicable license information before distributing the project.
