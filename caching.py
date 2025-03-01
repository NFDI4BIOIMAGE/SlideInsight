from huggingface_hub import create_repo, login, HfApi
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import pdfplumber
import torch
import os
import shelve
import io
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import (
                SystemMessage,
                UserMessage,
                TextContentItem,
                ImageContentItem,
                ImageUrl,
                ImageDetailLevel,
            )
from azure.core.credentials import AzureKeyCredential 
from PIL import Image
from pdf2image import convert_from_path
import base64
from datasets import Dataset, DatasetDict, concatenate_datasets, Features, Image, load_dataset, Value, Sequence
import tempfile
import requests
import shutil
import pdfplumber
from io import BytesIO
import yaml
import re
from pdf2image import convert_from_bytes


# Initialize models
text_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# Function to check and create a repository if it doesn't exist
def ensure_repo_exists(repo_name):
    api = HfApi()
    user = api.whoami()["name"]
    
    # Check if repo_name already includes the namespace 
    if "/" not in repo_name:
        full_repo_name = f"{user}/{repo_name}"
    else:
        full_repo_name = repo_name  # Assume it's already correctly formatted
    
    try:
        # Check if the repository already exists
        api.repo_info(full_repo_name, repo_type="dataset")
        print(f"Repository '{full_repo_name}' already exists.")
    except Exception:
        # Create the repository if it doesn't exist
        create_repo(repo_name, repo_type="dataset", private=False)
        print(f"Repository '{full_repo_name}' created.")
    return full_repo_name


# Load or initialize Hugging Face cache
def load_cache_dataset(repo_name):
    try:
        # Try to load the dataset
        return load_dataset(repo_name, split="train")
    except Exception:
        # If it doesn't exist, create a new dataset
        return Dataset.from_dict({"key": [], "text": [], "visual": [], "mixed": [], "image": []})

# Embedding functions
def embed_text(pdf_filename, slide_number, text_model):
    with pdfplumber.open(pdf_filename) as pdf:
        text = pdf.pages[slide_number].extract_text() or ""
        return text_model.encode(text)


def embed_visual(pdf_filename, slide_number, clip_processor, clip_model):
    with pdfplumber.open(pdf_filename) as pdf:
        image = pdf.pages[slide_number].to_image().original
        inputs = clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            return clip_model.get_image_features(**inputs).squeeze().tolist()


def embed_mixed(image, text_model, token, use_openai):
    """
    Generates an embedding of a structured description of an image using GPT-4o.

    Parameters
    ----------
    image : PIL.Image
    text_model: str
    token: str
    use_openai: bool

    Returns
    -------
    mixed_embedding:
        Text Embedding of the models anwser. 
    """
 
    endpoint = "https://models.inference.ai.azure.com"
    token = token
    
    # Convert PIL image to byte stream
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
    image_data_uri = f"data:image/png;base64,{img_base64}"
    
    if use_openai:
        from openai import OpenAI
        
        client = OpenAI(api_key = token) 
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional Data Scientist. Provide a structured description of the image in 1-2 sentences. Focus on what you can see in the image."},
                {"role": "user", "content": [{"type": "image_url", "image_url": {
                    "url": image_data_uri}}]}
            ]
        )
        
    else:
        client = ChatCompletionsClient(endpoint=endpoint,credential=AzureKeyCredential(token))
    
        response = client.complete(
            messages=[
                SystemMessage(
                    content="You are a professional Data Scientist. Provide a structured description of the image in 1-2 sentences. Focus on what you can see in the image."
                ),
                UserMessage(
                    content=[
                        ImageContentItem(
                            image_url=ImageUrl(url=image_data_uri, detail=ImageDetailLevel.LOW)
                        ),
                    ],
                ),
            ],
            model="gpt-4o",
        )
        
    # Parse structured description from response
    structured_response = response.choices[0].message.content
    
    # Convert the textual response into an embedding
    mixed_embedding = text_model.encode(structured_response)

    return mixed_embedding
            

def caching_local(pdf_path):
    """
    Caches embeddings for each slide of a PDF, including the images.
    
    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.

    Returns
    -------
    None
    """
    with shelve.open("local_cache.db", writeback=True) as cache:
        slides = convert_from_path(pdf_path)

        for slide_number, image in enumerate(slides, start=1):
            key = f"{pdf_path}_slide{slide_number}"
            
            # Initialize the cache entry if not exists
            if key not in cache:
                cache[key] = {}

            cached_data = cache[key]
            updated = False  # Track if we update the cache

            # Check and compute missing values
            if "text" not in cached_data:
                print(f"Generating text embedding for Slide {slide_number}")
                cached_data["text"] = embed_text(pdf_path, slide_number, text_model)
                updated = True

            if "visual" not in cached_data:
                print(f"Generating visual embedding for Slide {slide_number}")
                cached_data["visual"] = embed_visual(pdf_path, slide_number, clip_processor, clip_model)
                updated = True

            if "mixed" not in cached_data:
                print(f"Generating mixed embedding for Slide {slide_number}")
                cached_data["mixed"] = embed_mixed(image, text_model)
                updated = True

            if "image" not in cached_data:
                print(f"Storing image for Slide {slide_number}")
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format="PNG")
                cached_data["image"] = img_byte_arr.getvalue()
                updated = True

            # If we updated anything, commit changes
            if updated:
                cache[key] = cached_data  # Assign back to shelve
                print(f"Slide {slide_number} cached successfully!")

            else:
                print(f"All data already cached for Slide {slide_number}")



def load_local_cache(pdf_path, slide_number):
    """
    Loads embeddings and the image from the cache.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.
    slide_number : int
        The slide number to retrieve.

    Returns
    -------
    dict or None
        A dictionary containing:
        - "text": Text embedding
        - "visual": Visual embedding
        - "mixed": Mixed embedding
        - "image": PIL.Image.Image
        Returns None if no cached data is found.
    """
    with shelve.open("local_cache.db") as cache:
        key = f"{pdf_path}_slide{slide_number}"
        
        if key in cache:
            cached_data = cache[key]

            # Convert stored image bytes back to PIL Image
            if "image" in cached_data:
                img_byte_arr = io.BytesIO(cached_data["image"])
                cached_data["image"] = Image.open(img_byte_arr)

            return cached_data  # Returns the entire dictionary
        else:
            print(f"No cached data found for {key}")
            return None


def load_hf_cache(record_id, slide_number, pdf_number = 1, repo_name="your-hf-repo"):
    """
    Loads embeddings and metadata from the Hugging Face cache.

    Parameters
    ----------
    record_id: str
    slide_number : int
        The slide number to retrieve.
    pdf_number: int, optional
    repo_name : str, optional
        Name of the Hugging Face Hub dataset repository.

    Returns
    -------
    dict or None
        A dictionary containing:
        - "text": Text embedding
        - "visual": Visual embedding
        - "mixed": Mixed embedding
        - "zenodo_record_id": Zenodo record ID
        - "zenodo_filename": Original Zenodo file name
        - "page_number": Slide number
        Returns None if no cached data is found.
    """

    # Load dataset from Hugging Face Hub
    try:
        cache_dataset = load_dataset(repo_name, split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

    key = f'record{record_id}_pdf{pdf_number}_slide{slide_number}'
    # Search for the key in the dataset
    for idx, cached_key in enumerate(cache_dataset["key"]):
        if cached_key == key:
            cached_data = {
                "key": cache_dataset["key"][idx],
                "text": cache_dataset["text"][idx],
                "visual": cache_dataset["visual"][idx],
                "mixed": cache_dataset["mixed"][idx],
                "zenodo_record_id": cache_dataset["zenodo_record_id"][idx], 
                "zenodo_filename": cache_dataset["zenodo_filename"][idx], 
                "page_number": cache_dataset["page_number"][idx]
            }

            return cached_data

    print(f"No cached data found for {key}")
    return None




def append_rows_to_dataset(existing_dataset, new_data):

    # Convert new data to Dataset
    new_dataset = Dataset.from_list(new_data)

    # If the existing dataset is empty, return the new dataset
    if existing_dataset is None or len(existing_dataset) == 0:
        return new_dataset

    # Ensure both datasets have the same schema
    if set(existing_dataset.column_names) != set(new_dataset.column_names):
        raise ValueError("Column names in the new data do not match the existing dataset.")

    # Concatenate datasets (append new rows)
    updated_dataset = concatenate_datasets([existing_dataset, new_dataset])

    return updated_dataset




# Function to extract Zenodo record IDs from URLs
def extract_zenodo_ids(url_list):
    zenodo_ids = []
    if isinstance(url_list, str):  # Single URL case
        url_list = [url_list]  # Convert to list for consistency
        
    for url in url_list:
        if "zenodo.org" in url:  # Ensure it's a Zenodo link
            match = re.search(r"zenodo.org/records/(\d+)", url)           
            if match:
                zenodo_ids.append(match.group(1))
                continue
            match = re.search(r"zenodo\.org/doi/\d+\.\d+/zenodo\.(\d+)", url)
            if match:
                zenodo_ids.append(match.group(1))
                
    return sorted(zenodo_ids)
    
# Load YAML file and extract Zenodo record IDs correctly
def get_zenodo_ids_from_yaml(yaml_file):
    with open(yaml_file, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    zenodo_ids = []

    for entry in data.get("resources",[]):
        if "url" in entry and "type" in entry:
            # Ensure 'type' is always treated as a list
            entry_type = entry["type"]
            if isinstance(entry_type, str):
                entry_type = [entry_type]  # Convert single string to a list
        
            # Check if 'Slides' is in the type list
            if "Slides" in entry_type:
                zenodo_ids.extend(extract_zenodo_ids(entry["url"]))  # Extract IDs safely
    sorted_ids = sorted(zenodo_ids)
   
    return sorted_ids
    
# Function to fetch Zenodo record files (PDFs)
def get_zenodo_pdfs(record_id):
    api_url = f"https://zenodo.org/api/records/{record_id}"
    response = requests.get(api_url)

    if response.status_code != 200:
        print(f"Failed to fetch Zenodo record {record_id}, skipping...")
        return []

    record_data = response.json()
    pdf_files = sorted(
        [file for file in record_data.get("files", []) if file["key"].lower().endswith(".pdf")],
        key=lambda x: x["key"]  # Sort by filename
    )


    return pdf_files

# Function to download a PDF file from Zenodo
def download_pdf(pdf_url):
    response = requests.get(pdf_url)
    if response.status_code == 200:
        return BytesIO(response.content)
    else:
        print(f"Failed to download PDF: {pdf_url}")
        return None

# Main function to process each slide and store embeddings with metadata
def cache_hf(zenodo_record_id, token, use_openai, repo_name="your-hf-repo"):
    """
    Processes all PDF slides from a Zenodo record and stores embeddings with metadata.

    Parameters
    ----------
    zenodo_record_id : str
        Zenodo record ID to fetch PDFs from.
    token: str
        Token for OpenAI or GH Models 
    use_openai: bool
        if True: uses OpenAI API, otherwise GH Models
    repo_name : str, optional
        Hugging Face dataset repository name.

    Returns
    -------
    None
    """
    
    # Ensure the repository exists
    full_repo_name = ensure_repo_exists(repo_name)
    
    # Load existing dataset
    cache_dataset = load_cache_dataset(full_repo_name)

    # Get existing keys from the dataset
    existing_keys = set(cache_dataset["zenodo_record_id"]) if "zenodo_record_id" in cache_dataset.column_names else set()

    # **Check if the record already exists**
    if zenodo_record_id in existing_keys:
        print(f"Skipping Zenodo Record {zenodo_record_id}: Already in dataset.")
        return  # Skip processing this record
        
    # Fetch all PDFs from the Zenodo record
    pdf_files = get_zenodo_pdfs(zenodo_record_id)

    for pdf_number,pdf_file in enumerate(pdf_files):
        pdf_url = pdf_file["links"]["self"]
        pdf_filename = pdf_file["key"]

        print(f"Processing {pdf_filename} from Zenodo Record {zenodo_record_id}")

        # Download PDF
        pdf_bytes = download_pdf(pdf_url)
        if not pdf_bytes:
            continue

        # Open PDF and extract slides
        slides = []
        with pdfplumber.open(pdf_bytes) as pdf:
            slides = pdf.pages  # List of all slides

        # Initialize embedding models
        text_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Process each slide
        new_data = []

        for i, slide in enumerate(slides):
            slide_number = i + 1
            slide_key = f"record{zenodo_record_id}_pdf{pdf_number + 1}_slide{slide_number}"

            text_embedding = embed_text(pdf_bytes, i, text_model)
            visual_embedding = embed_visual(pdf_bytes, i, clip_processor, clip_model)

            page_image = slide.to_image().original  # Convert to PIL Image
            mixed_embedding = embed_mixed(page_image, text_model, token, use_openai)

            # Store metadata
            new_data.append({
                "key": slide_key,
                "zenodo_record_id": zenodo_record_id,
                "zenodo_filename": pdf_filename,
                "page_number": slide_number,
                "text": text_embedding,
                "visual": visual_embedding,
                "mixed": mixed_embedding
            })
            
        cache_dataset = append_rows_to_dataset(cache_dataset, new_data)
    
    # Push dataset to Hugging Face Hub
    cache_dataset.push_to_hub(repo_name)

    print(f"Finished processing Zenodo Record {zenodo_record_id}.")






