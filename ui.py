import gradio as gr
from PIL import Image
import io
import requests # Used to make API calls
import os

# --- CONFIGURATION ---
# The address of your running FastAPI application
API_BASE_URL = "http://127.0.0.1:8000"

# --- HELPER FUNCTIONS TO INTERFACE WITH THE API ---

def format_results_for_gallery(hits):
    """
    Processes the list of result dictionaries to be compatible with gr.Gallery.
    Returns a list of image paths and their corresponding captions.
    NOTE: This assumes the Gradio UI is running on the same machine as the API,
    as it uses local file paths returned by the API.
    """
    if not hits:
        return [], []
    
    # Check if the image paths are valid on the machine running the UI
    image_paths = []
    for hit in hits:
        path = hit.get('image_path')
        if path and os.path.exists(path):
            image_paths.append(path)
        else:
            # Add a placeholder if the image is not found
            print(f"Warning: Image path not found: {path}")
            image_paths.append(None) # Gradio handles None as a blank image

    captions = []
    for hit in hits:
        caption = f"{hit.get('productDisplayName', 'N/A')}"
        if '_score' in hit:
            caption += f"\nScore: {hit['_score']:.3f}"
        captions.append(caption)
        
    return image_paths, captions

def handle_text_search(query, k):
    """Sends a request to the /search/text API endpoint."""
    if not query:
        return [], [], "Please enter a search query."
    
    try:
        payload = {"q": query, "k": int(k)}
        response = requests.post(f"{API_BASE_URL}/search/text", json=payload)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        
        results = response.json().get('results', [])
        images, captions = format_results_for_gallery(results)
        return images, captions, f"Found {len(images)} results for '{query}'."

    except requests.exceptions.RequestException as e:
        raise gr.Error(f"API request failed: {e}")

def handle_image_search(image, k):
    """Sends a request to the /search/image API endpoint."""
    if image is None:
        return [], [], "Please upload an image to search."

    try:
        # Convert PIL.Image to bytes for the request
        with io.BytesIO() as output:
            image.save(output, format="PNG")
            image_bytes = output.getvalue()
        
        files = {'file': ('image.png', image_bytes, 'image/png')}
        data = {'k': int(k)}
        response = requests.post(f"{API_BASE_URL}/search/image", files=files, data=data)
        response.raise_for_status()

        results = response.json().get('results', [])
        images, captions = format_results_for_gallery(results)
        return images, captions, f"Found {len(images)} similar items."

    except requests.exceptions.RequestException as e:
        raise gr.Error(f"API request failed: {e}")

def handle_hybrid_search(image, text, k, alpha):
    """Sends a request to the /search/hybrid API endpoint."""
    if image is None and not text:
        return [], [], "Please provide an image, text, or both."

    try:
        data = {'q': text, 'k': int(k), 'alpha': alpha}
        files = {}
        if image:
            with io.BytesIO() as output:
                image.save(output, format="PNG")
                image_bytes = output.getvalue()
            files = {'file': ('image.png', image_bytes, 'image/png')}

        response = requests.post(f"{API_BASE_URL}/search/hybrid", data=data, files=files)
        response.raise_for_status()

        results = response.json().get('results', [])
        images, captions = format_results_for_gallery(results)
        return images, captions, f"Found {len(images)} hybrid results."

    except requests.exceptions.RequestException as e:
        raise gr.Error(f"API request failed: {e}")

def handle_recommendations(user_id, k):
    """Sends a request to the /recommend/{user_id} API endpoint."""
    if not user_id:
        return [], [], "Please enter a User ID."
        
    try:
        params = {'k': int(k)}
        response = requests.get(f"{API_BASE_URL}/recommend/{user_id}", params=params)
        response.raise_for_status()

        results = response.json().get('results', [])
        images, captions = format_results_for_gallery(results)
        return images, captions, f"Showing {len(images)} recommendations for {user_id}."
        
    except requests.exceptions.RequestException as e:
        raise gr.Error(f"API request failed: {e}")

# --- BUILD THE GRADIO UI (Layout is unchanged) ---

with gr.Blocks(theme=gr.themes.Soft(), title="Fashion Search Engine") as demo:
    gr.Markdown("# 👕 Multi-Modal Fashion Search Engine")
    gr.Markdown("Search for products using text, images, or a combination of both. You can also get personalized recommendations.")

    with gr.Tabs():
        # -- Text Search Tab --
        with gr.TabItem("🔎 Text Search"):
            with gr.Row():
                text_query = gr.Textbox(label="Search Query", placeholder="e.g., blue summer dress", scale=4)
                text_k = gr.Slider(minimum=1, maximum=20, value=10, step=1, label="Results")
                text_search_btn = gr.Button("Search", variant="primary", scale=1)
            text_status = gr.Textbox(label="Status", interactive=False)
            text_gallery = gr.Gallery(label="Search Results", show_label=False, columns=5, object_fit="contain", height="auto")
            text_captions = gr.Textbox(visible=False)

        # -- Image Search Tab --
        with gr.TabItem("🖼️ Image Search"):
            with gr.Row():
                img_input = gr.Image(type="pil", label="Upload an Image")
                with gr.Column():
                    img_k = gr.Slider(minimum=1, maximum=20, value=10, step=1, label="Results")
                    img_search_btn = gr.Button("Find Similar", variant="primary")
            img_status = gr.Textbox(label="Status", interactive=False)
            img_gallery = gr.Gallery(label="Search Results", show_label=False, columns=5, object_fit="contain", height="auto")
            img_captions = gr.Textbox(visible=False)

        # -- Hybrid Search Tab --
        with gr.TabItem("✨ Hybrid Search"):
            with gr.Row():
                hybrid_img_input = gr.Image(type="pil", label="Upload Image (Optional)")
                with gr.Column():
                    hybrid_text_query = gr.Textbox(label="Refine with Text (Optional)", placeholder="e.g., in red, for men")
                    hybrid_alpha = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.05, label="Image vs. Text Weight")
                    hybrid_k = gr.Slider(minimum=1, maximum=20, value=10, step=1, label="Results")
                    hybrid_search_btn = gr.Button("Search", variant="primary")
            hybrid_status = gr.Textbox(label="Status", interactive=False)
            hybrid_gallery = gr.Gallery(label="Search Results", show_label=False, columns=5, object_fit="contain", height="auto")
            hybrid_captions = gr.Textbox(visible=False)

        # -- Recommendations Tab --
        with gr.TabItem("❤️ Recommendations"):
            with gr.Row():
                rec_user_id = gr.Textbox(label="User ID", placeholder="e.g., user_1", scale=4)
                rec_k = gr.Slider(minimum=1, maximum=20, value=10, step=1, label="No. of Recs")
                rec_btn = gr.Button("Get Recommendations", variant="primary", scale=1)
            rec_status = gr.Textbox(label="Status", interactive=False)
            rec_gallery = gr.Gallery(label="Recommended Products", show_label=False, columns=5, object_fit="contain", height="auto")
            rec_captions = gr.Textbox(visible=False)

    # --- CONNECT UI COMPONENTS TO FUNCTIONS ---
    text_search_btn.click(fn=handle_text_search, inputs=[text_query, text_k], outputs=[text_gallery, text_captions, text_status])
    img_search_btn.click(fn=handle_image_search, inputs=[img_input, img_k], outputs=[img_gallery, img_captions, img_status])
    hybrid_search_btn.click(fn=handle_hybrid_search, inputs=[hybrid_img_input, hybrid_text_query, hybrid_k, hybrid_alpha], outputs=[hybrid_gallery, hybrid_captions, hybrid_status])
    rec_btn.click(fn=handle_recommendations, inputs=[rec_user_id, rec_k], outputs=[rec_gallery, rec_captions, rec_status])

if __name__ == "__main__":
    # Check if the API is running before launching
    try:
        requests.get(f"{API_BASE_URL}/")
        print("API server is reachable. Launching Gradio UI...")
        demo.launch()
    except requests.exceptions.ConnectionError:
        print("="*50)
        print("ERROR: Could not connect to the API server.")
        print(f"Please ensure the FastAPI server is running at {API_BASE_URL}")
        print("You can start it by running this command in a separate terminal:")
        print("uvicorn api:app --reload")
        print("="*50)

