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
from datasets import Dataset, DatasetDict, concatenate_datasets, Features, Image, load_dataset, Value, Sequence, load_dataset
import tempfile
import requests
import shutil
import pdfplumber
from io import BytesIO
import yaml
import re
from pdf2image import convert_from_bytes
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# Initialize models
#text_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", trust_remote_code=True)
text_model = SentenceTransformer("microsoft/harrier-oss-v1-0.6b", model_kwargs={"dtype": "auto"}, trust_remote_code=True)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", trust_remote_code=True, use_safetensors=True)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


mixed_embed_prompt = """
    You are an expert slide and presentation analyst. 
    Your task is to examine a single presentation slide (provided as image) and describe its characteristics in a structured JSON-like dictionary.
    
    Return only a valid JSON object with these fields:
    Content, Style, Language, Knowledge Level, Learning Perspective.
    
    Content: One simple sentence about the slides content
    Style: What kind of slide is this? Options: Title, Closing, Divider, Content, Image, Infographic, Code, Table, Link, Placeholder, Quiz, None
    Language: What language is used in the slide? Options: English, German, Mixed, None
    Knowledge Level: What's the target group for this slide? Options: Beginner, Intermediate, Expert, None
    Learning Perspective: What's the main learning objective for this slide? Options: Agenda/Learning Objectives, Tips/Recommendation, Considerations, Note, Guidelines, Criteria, 
    Definition/Explanation, Components, Comparison, Examples, Options, Overview, Structure, Goals, Challenges, Purpose/Intent, Motivation/Rationale, Pros and Cons/Evaluation, 
    Further Reading/Literature, Credits/Contacts, Questions, Summary/Conclusion, How-To/Demonstration, Informative/Descriptive, Introduction, None
    
    Different Options are seperated by commata, words seperated by backslashes belong to one option.
    
    For each characteristic, choose EXACT one of the options. Don't come up with new ideas, only choose the predifined options. If no option is suitable, choose None.
    DON'T output anything else than the VALID JSON FORMAT dictionary that looks like this:
    
    {
      "Content": "",
      "Style": "",
      "Language": "",
      "Knowledge Level": "",
      "Learning Perspective": ""
    }
    """


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
        return Dataset.from_list([])
        

# Embedding functions
def embed_and_extract_text(pdf_filename, slide_number, text_model):
    with pdfplumber.open(pdf_filename) as pdf:
        text = pdf.pages[slide_number].extract_text() or ""
        return text_model.encode(text), text


def embed_visual(pdf_filename, slide_number, clip_processor, clip_model):
    with pdfplumber.open(pdf_filename) as pdf:
        image = pdf.pages[slide_number].to_image().original
        inputs = clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            return clip_model.get_image_features(**inputs).squeeze().tolist()


def embed_mixed(image, text_model, token, use_api):
    """
    Generates an embedding of a structured description of an image.

    Parameters
    ----------
    image : PIL.Image
    text_model: str
    token: str
    use_api: str

    Returns
    -------
    mixed_embedding:
        Text Embedding of the models anwser. 
    """
    
    # Convert PIL image to byte stream
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
    image_data_uri = f"data:image/png;base64,{img_base64}"
    
    if use_api == "use_openai":
        from openai import OpenAI
        
        client = OpenAI(api_key = token) 
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": mixed_embed_prompt},
                {"role": "user", "content": [{"type": "image_url", "image_url": {
                    "url": image_data_uri}}]}
            ]
        )
        
    elif use_api == "use_gh_models":
        endpoint = "https://models.inference.ai.azure.com"
        client = ChatCompletionsClient(endpoint=endpoint,credential=AzureKeyCredential(token))
    
        response = client.complete(
            messages=[
                SystemMessage(
                    content=mixed_embed_prompt
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

    elif use_api == "use_scads_api":
        from openai import OpenAI
        client = OpenAI(base_url="https://llm.scads.ai/v1", api_key=token)
        response = client.chat.completions.create(
            model="Qwen/Qwen3-VL-8B-Instruct",
            messages=[
                {"role": "system", "content": mixed_embed_prompt},
                {"role": "user", "content": [{"type": "image_url", "image_url": {
                    "url": image_data_uri}}]}
            ]
        )


    else:
        print("No valid API KEY defined")
        
    # Parse structured description from response
    structured_response = response.choices[0].message.content
    
    # Convert the textual response into an embedding
    mixed_embedding = text_model.encode(structured_response)

    return mixed_embedding, structured_response


def load_single_hf_cache(record_id, slide_number, parquet_path, pdf_number=1):
    """
    Loads a single row from the Hugging Face cache as a Pandas DataFrame.

    Parameters
    ----------
    record_id : str
        The Zenodo record ID.
    slide_number : int
        The slide number to retrieve.
    parquet_path : str
        Path to the .parquet file on the Hugging Face Hub.
    pdf_number : int, optional
        The PDF number, defaults to 1.

    Returns
    -------
    pd.DataFrame or None
        A single-row DataFrame with the matched cache entry.
        Returns None if no match is found.
    """
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

    key = f"record{record_id}_pdf{pdf_number}_slide{slide_number}"
    match = df[df["key"] == key]

    if match.empty:
        print(f"No cached data found for {key}")
        return None

    return match.reset_index(drop=True)




def load_full_hf_cache(repo_name):
    """
    Loads the entire dataset from the Hugging Face cache.
    Function has to be adapted, but it works right now.
    
    Parameters
    ----------
    repo_name : str, optional
        Hugging Face dataset repository name.

    Returns
    -------
    Pandas DataFrame of the Huggingface Dataset.
    """
    import pandas as pd
    from datasets import load_dataset

    ds = load_dataset(repo_name)
    df = ds["train"].to_pandas()

    return df



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
def get_zenodo_ids_from_yaml(yaml_file, valid_licenses, unclear_licenses):
    with open(yaml_file, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    zenodo_ids = []

    for entry in data.get("resources",[]):
        if "url" in entry and "type" in entry:
            # Ensure 'type' is always treated as a list
            entry_type = entry["type"]
            if isinstance(entry_type, str):
                entry_type = [entry_type]  # Convert single string to a list
        
            # Check if 'Slides' is in the type list and if entry has a license
            if "Slides" in entry_type:
                license = entry.get("license")
                if license:
                    license_lower = license.lower()  # normalize case
                    if (license_lower in [l.lower() for l in valid_licenses]) or \
                       (license_lower in [l.lower() for l in unclear_licenses]):
                        zenodo_ids.extend(extract_zenodo_ids(entry["url"]))
                        
    sorted_ids = sorted(zenodo_ids)
   
    return sorted_ids


# Function to fetch Zenodo record files (PDFs)
def get_zenodo_pdfs(record_id):
    api_url = f"https://zenodo.org/api/records/{record_id}"
    headers = {"User-Agent": "Mozilla/5.0 (Zenodo PDF Fetcher)"}
    response = requests.get(api_url, headers=headers)

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
def cache_hf(zenodo_record_id, token, use_api, repo_name):
    """
    Processes all PDF slides from a Zenodo record and stores embeddings with metadata.

    Parameters
    ----------
    zenodo_record_id : str
        Zenodo record ID to fetch PDFs from.
    token: str
        Token for OpenAI or GH Models 
    use_api: str
        Name of API to use SCADS, OpenAI API or GH Models
    repo_name : str
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
    existing_keys = set(cache_dataset["key"]) if "key" in cache_dataset.column_names else set()
        
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

        # Process each slide
        new_data = []

        for i, slide in enumerate(slides):
            slide_number = i + 1
            slide_key = f"record{zenodo_record_id}_pdf{pdf_number + 1}_slide{slide_number}"

            # Check if the slide already exists
            if slide_key in existing_keys:
                print(f"Skipping {slide_key}: already cached.")
                continue

            text_embedding, extracted_text = embed_and_extract_text(pdf_bytes, i, text_model)
            visual_embedding = embed_visual(pdf_bytes, i, clip_processor, clip_model)

            page_image = slide.to_image().original  # Convert to PIL Image
            mixed_embedding, structured_response = embed_mixed(page_image, text_model, token, use_api)

            # Store metadata
            new_data.append({
                "key": slide_key,
                "zenodo_record_id": zenodo_record_id,
                "zenodo_filename": pdf_filename,
                "page_number": slide_number,
                "text_embedding": text_embedding,
                "visual_embedding": visual_embedding,
                "mixed_embedding": mixed_embedding,
                "structured_description": structured_response,
                "extracted_text": extracted_text,  
            })
            
        if new_data:
            cache_dataset = append_rows_to_dataset(cache_dataset, new_data)
            cache_dataset.push_to_hub(repo_name)
            existing_keys.update(row["key"] for row in new_data)
        
        #cache_dataset = append_rows_to_dataset(cache_dataset, new_data)
    
    # Push dataset to Hugging Face Hub
    #cache_dataset.push_to_hub(repo_name)


    print(f"Finished processing Zenodo Record {zenodo_record_id}.")
    
