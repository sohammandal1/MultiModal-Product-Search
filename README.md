# Multi-Modal Fashion Search Engine

This project is a multi-modal search engine for fashion products, allowing users to search via text, images, or a hybrid approach. It also includes a recommendation system.

## Features

* **Text Search**: Find products using natural language queries.
* **Image Search**: Find visually similar products by uploading an image.
* **Hybrid Search**: Combine text and images for refined results.
* **Recommendations**: Get personalized product recommendations.

## How to Run

1.  **Clone the repository and install dependencies:**
    ```bash
    git clone https://github.com/sohammandal1/MultiModal-Product-Search.git
    cd fashion-search-engine
    pip install -r requirements.txt
    ```

2.  **Run the setup script to download data and build embeddings:**
    ```bash
    python main.py
    ```

3.  **Run the API and UI in two separate terminals:**
    ```bash
    # Terminal 1: Start the API
    uvicorn api:app --reload

    # Terminal 2: Start the UI
    python ui.py
    ````