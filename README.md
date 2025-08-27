# Multi-Modal Fashion Product Search Engine 👕

This project is a complete, end-to-end multi-modal search engine for a large catalog of fashion products. It allows users to find items using natural language, an uploaded image, or a combination of both. The system is built with a modern Python stack, featuring a high-performance vector search backend and a user-friendly web interface.

---

## How It Works

The project follows a robust, three-stage architecture to deliver fast and accurate search results.

### 1. Embedding Generation
First, we convert both the product images and their text descriptions into a common mathematical format called **vector embeddings**. This allows us to compare different types of data (like an image and a text query) directly.
* **Image Embeddings**: The powerful **OpenCLIP** model processes each of the 44,000+ product images, capturing their visual features (like color, shape, and style) into a high-dimensional vector.
* **Text Embeddings**: A highly efficient **SentenceTransformer** model converts all product titles and user queries into corresponding text embeddings.

### 2. Vector Indexing with FAISS
With tens of thousands of vectors, a simple search would be too slow. We use **FAISS (Facebook AI Similarity Search)** to build a highly optimized index of all the embeddings. This index partitions the vectors into clusters, allowing the system to perform a nearest-neighbor search in milliseconds instead of searching through the entire dataset every time.

### 3. API and User Interface
The system is decoupled into a backend and a frontend for scalability and maintainability.
* **FastAPI Backend**: A robust, asynchronous API built with **FastAPI** serves the core search logic. It exposes simple endpoints for text, image, and hybrid searches, handling all the heavy computation.
* **Gradio UI**: An interactive and intuitive web interface built with **Gradio** acts as the frontend. It communicates with the FastAPI backend to fetch results and display them to the user in a clean, visual format.

---

## Features

* **🔎 Text Search**: Find products using natural language queries (e.g., "red summer dress").
* **🖼️ Image Search**: Upload an image to find visually similar products from the catalog.
* **✨ Hybrid Search**: Combine an image upload with a text query for highly specific, refined results (e.g., upload a picture of a shirt and add the text "in blue").
* **❤️ Recommendations**: Get content-based product recommendations for simulated users.

---

## How to Run

### Prerequisites
* Python 3.8+
* Git

### 1. Clone the Repository and Install Dependencies
```bash
git clone [https://github.com/sohammandal1/MultiModal-Product-Search.git](https://github.com/sohammandal1/MultiModal-Product-Search.git)
cd MultiModal-Product-Search
pip install -r requirements.txt
```

### 2. Run the One-Time Setup Script
This script will download the dataset, generate all 44,000+ embeddings, and build the FAISS index. This may take some time depending on your hardware (GPU is recommended).
```bash
python main.py
```

### 3. Launch the Application
The application runs in two parts. You will need to open two separate terminals.

Terminal 1: Start the API Server
```bash
uvicorn api:app --reload
```

Leave this terminal running. This is your backend.

Terminal 2: Start the Gradio UI
```bash
python ui.py
```

This will provide a local URL (e.g., http://127.0.0.1:7860). Open it in your browser to use the application.

Tech Stack
Backend: Python, FastAPI, PyTorch

Frontend: Gradio

ML Models: OpenCLIP, SentenceTransformer

Vector Database: FAISS

Core Libraries: Pandas, NumPy, Pillow
