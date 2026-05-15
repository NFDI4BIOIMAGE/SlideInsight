from datasets import Dataset, Features, Value, Sequence
from datasets import load_dataset
from huggingface_hub import create_repo, login, HfApi
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import pdfplumber
import torch
import os
import shelve
import requests
from tqdm import tqdm
from caching import load_full_hf_cache, get_zenodo_pdfs


def load_pdf(pdf_name):
    """
    Load pdf and convert it to a list of its pages as images.

    Parameters
    ----------
    pdf_name: str
        Name of the pdf (must be in the same folder as this notebook)

    Returns
    -------
    slides: list
        List of Images 
    """
    from pdf2image import convert_from_path

    slides = convert_from_path(pdf_name)
    return slides



def save_images(filepath, pdf, new_width=None):
    """
    Save a list of images (from pdf file) to a specified directory , optionally with a resized width.

    Parameters
    ----------
    filepath : str
        The directory path where the images will be saved.
    pdf : str
        Name of the pdf (must be in the same folder as this notebook)
    new_width : int, optional
        The new width for resizing each image while maintaining its aspect ratio.
        If not provided, the original width of each image will be used.

    Returns
    -------
    None
        Saves each image in the specified directory as PNG files named slide1.png, slide2.png, etc.
    """
    from PIL import Image
    import os
    from pdf2image import convert_from_path
    import re
    
    # Convert pdf to images
    slides = convert_from_path(pdf)
    
    for i, img in enumerate(slides, start=1):
        # Use original width if new_width is not specified
        if new_width is None:
            new_width = img.width
        
        # Calculate the new height to maintain the aspect ratio
        aspect_ratio = img.height / img.width
        new_height = int(new_width * aspect_ratio)
        
        # Resize the image
        img_resized = img.resize((new_width, new_height))
        
        # Ensure filepath directory exists
        os.makedirs(filepath, exist_ok=True)

        # Saving the image in the specified filepath
        image_filename = os.path.join(filepath, f"{pdf}_slide{i}.png")
        img_resized.save(image_filename, "PNG")


def text_extraction(pdf_name, images, slide_dict=None):
    """
    Extract text from a PDF and pair it with corresponding slide images, updating the given dictionary
    or creating a new one if none is provided. The result is saved as a YAML file.

    Parameters
    ----------
    pdf_name : str
        The name or path of the PDF file from which to extract text.
    images : list
        A list of images corresponding to the slides in the PDF.
    slide_dict : dict, optional
        A dictionary mapping slide image file names to the extracted text. 
        If None, a new dictionary is created.

    Returns
    -------
    dict
        The updated slide_dict containing the new text-image pairs.
    """
    import yaml
    import pdfplumber
    import re

    # Initialize slide_dict if not provided
    if slide_dict is None:
        slide_dict = {}

    slide_texts = []

    # Extract the original text from each slide
    with pdfplumber.open(pdf_name) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            slide_texts.append(text)

    # Update the dictionary with new text-image pairs
    for i, (img, text) in enumerate(zip(images, slide_texts), start=1):
        # Define the path for each image file
        image_path = f"{pdf_name}_slide{i}.png"

        # Add the image path and corresponding text to the dictionary only if it doesn't already exist
        if image_path not in slide_dict:
            slide_dict[image_path] = text

    # Save the updated dictionary as a YAML file
    with open("dict_slides_text.yml", "w") as yaml_file:
        yaml.dump(slide_dict, yaml_file, default_flow_style=False)
        
    return slide_dict



def text_extract_from_pdfs(downloads_folder="downloads", yaml_file_path="dict_slides_text.yml"):
    """
    Extract text from PDFs based on their corresponding slide images and save the text-image pairs in a YAML file.
    Processes PDFs and images in sorted order. Skips PDFs whose slides are already processed and stored in the YAML file.

    Parameters
    ----------
    downloads_folder : str, optional
        The path to the folder containing the PDF files and the 'images' subfolder (default is "downloads").
    yaml_file_path : str, optional
        The path to the YAML file where text-image pairs will be stored (default is "dict_slides_text.yml").

    Returns
    -------
    None
    """
    import os
    import yaml
    import re
    from natsort import natsorted

    # Define paths
    images_folder = os.path.join(downloads_folder, "images")

    # Ensure the YAML dictionary exists or initialize an empty one
    if os.path.exists(yaml_file_path):
        with open(yaml_file_path, "r") as yaml_file:
            slide_dict = yaml.safe_load(yaml_file) or {}
    else:
        slide_dict = {}

    # Group images by their corresponding PDFs
    pdf_image_map = {}
    for image_file in os.listdir(images_folder):
        if image_file.lower().endswith('.png'):
            # Extract the base PDF name from the image filename (e.g., `example_slide1.png` -> `example`)
            match = re.match(r"(.+)_slide\d+\.png", image_file)
            if match:
                pdf_name = match.group(1)
                pdf_image_map.setdefault(pdf_name, []).append(os.path.join(images_folder, image_file))

    # Sort the PDFs alphabetically and their corresponding images
    sorted_pdf_image_map = {pdf_name: natsorted(images) for pdf_name, images in natsorted(pdf_image_map.items())}

    # Process each group of images
    for pdf_name, image_list in sorted_pdf_image_map.items():
        # Check if these slides are already in the dictionary
        processed_slides = [key for key in slide_dict if pdf_name in key]
        if processed_slides:
            print(f"Slides for {pdf_name} already processed. Skipping.")
            continue

        # Construct the PDF file path
        pdf_path = os.path.join(downloads_folder, f"{pdf_name}.pdf")
        if not os.path.exists(pdf_path):
            print(f"PDF file {pdf_path} corresponding to the images not found. Skipping.")
            continue

        # Process and extract text
        try:
            print(f"Processing slides for {pdf_name}...")
            slide_dict = text_extraction(pdf_path, image_list, slide_dict)
        except Exception as e:
            print(f"Error processing slides for {pdf_name}: {e}")

    # Save the updated dictionary to the YAML file
    with open(yaml_file_path, "w") as yaml_file:
        yaml.dump(slide_dict, yaml_file, default_flow_style=False, allow_unicode=True, sort_keys=False)


def OLD_process_slides(pdf_path, slides, client, clip_processor, clip_model, text_model, repo_name="lea-33/SlightInsight_Cache"):
    """
    Processes PDF slides to compute visual and mixed-modal embeddings, caching results on Hugging Face.

    Parameters
    ----------
    pdf_path: str
        Name of the PDF file.
    slides : list
        List of images representing the slides.
    clip_processor: CLIPProcessor
        The processor for the CLIP model.
    clip_model: CLIPModel
        The CLIP model for generating visual embeddings.
    text_model: SentenceTransformer
        The text embedding model.
    repo_name: str, optional
        Name of the Hugging Face Hub repository for caching.

    Returns
    -------
    list
        A list of dictionaries containing embeddings and slide numbers for all slides.
    """
    import torch
    import os
    from huggingface_hub import HfApi
    from datasets import Dataset
    from PIL import Image
    from transformers import CLIPProcessor, CLIPModel
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

    endpoint = "https://models.inference.ai.azure.com"
    token = os.environ["GITHUB_TOKEN"]
    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(token),
    )

    # Ensure Hugging Face repository exists
    full_repo_name = ensure_repo_exists(repo_name)

    # Load or initialize cache dataset
    cache_dataset = load_cache_dataset(full_repo_name)
    cached_keys = set(cache_dataset["key"]) if "key" in cache_dataset.column_names else set()

    slide_embeddings = []  # Store all embeddings as a list of dictionaries

    for slide_number, slide_image in enumerate(slides):
        # Save slide image temporarily
        image_path = f"images/slide_{slide_number}.png"
        slide_image.save(image_path)

        key = f"{pdf_path}_page{slide_number}"
        
        vision_embedding, text_embedding, mixed_embedding = None, None, None

        # Check if the key exists in the cache
        if key in cached_keys:
            print(f"Fetching cached embeddings for slide {slide_number}.")
            cached_value = cache_dataset.filter(lambda x: x["key"] == key)["value"][0]

            if (
                "vision_embedding" in cached_value and cached_value["vision_embedding"] and
                "text_embedding" in cached_value and cached_value["text_embedding"]
            ):
                vision_embedding = cached_value["vision_embedding"]
                text_embedding = cached_value["text_embedding"]
            else:
                print(f"Key found but entry is incomplete. Removing old entry for slide {slide_number}.")
                cache_dataset = cache_dataset.filter(lambda x: x["key"] != key)

        # Compute embeddings if the entry does not exist or was removed
        if vision_embedding is None or text_embedding is None:
            print(f"Computing embeddings for slide {slide_number}.")
            inputs = clip_processor(images=slide_image, return_tensors="pt")
            with torch.no_grad():
                vision_embedding = clip_model.get_image_features(**inputs).squeeze().tolist()

            text_embedding = calculate_text_embeddings(pdf_path, text_model).get(slide_number, [])

            # Cache the new entry
            new_entry = {
                "key": key,
                "value": {
                    "vision_embedding": vision_embedding,
                    "text_embedding": text_embedding
                }
            }
            cache_dataset = cache_dataset.add_item(new_entry)
            print(f"Cached embeddings for slide {slide_number}.")

        # Generate 'mixed-modal' embedding using GPT-4o
        try:
            mixed_embedding = get_mixed_embedding(client, image_path, text_model)
        except Exception as e:
            print(f"Error generating GPT-4o embedding for slide {slide_number}: {e}")
            mixed_embedding = None

        # Add all embeddings for this slide to the main list
        slide_embeddings.append({
            "slide_number": slide_number,
            "vision_embedding": vision_embedding,
            "text_embedding": text_embedding,
            "mixed_modal_embedding": mixed_embedding
        })

    # Push the updated cache dataset to Hugging Face Hub
    cache_dataset.push_to_hub(full_repo_name)
    print(f"Finished caching embeddings for all slides.")

    return slide_embeddings




def download_all_pdfs(repo_name, save_dir="zenodo_pdfs"):
    """
    Downloads all unique PDFs listed in a dataframe using Zenodo record metadata.

    Parameters:
        repo_name (str): Name of the HF Repository that stores information about the PDFs.
        save_dir (str): Directory where PDFs will be saved.

    Returns:
        None (creates a folder "zenodo_pdfs" that stores the PDFs from the Cache File)
    """
    
    df = load_full_hf_cache(repo_name=repo_name)

    os.makedirs(save_dir, exist_ok=True)
    pdf_names = set()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading PDFs"):
        record_id = row["zenodo_record_id"]
        pdf_filename = row["zenodo_filename"]
        pdf_path = os.path.join(save_dir, pdf_filename)

        if os.path.exists(pdf_path) or pdf_filename in pdf_names:
            continue

        # Get list of available PDFs for this record
        pdf_files = get_zenodo_pdfs(record_id)
        pdf_file = next((file for file in pdf_files if file["key"] == pdf_filename), None)

        if not pdf_file:
            print(f"PDF {pdf_filename} not found in record {record_id}")
            continue

        try:
            response = requests.get(pdf_file["links"]["self"], stream=True)
            response.raise_for_status()

            with open(pdf_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"Downloaded {pdf_filename}")
            pdf_names.add(pdf_filename)

        except Exception as e:
            print(f"Error downloading {pdf_filename}: {e}")

    print("All PDFs downloaded!")


def download_zenodo_pdf(zenodo_record_id, pdf_number=1, save_dir=None):
    """
    Downloads a specific PDF from a Zenodo record and saves it locally.

    Parameters:
    ----------
    zenodo_record_id : str
        Zenodo record ID to fetch PDFs from.
    pdf_number : int, optional
        The number of the PDF to download (1-based index).
    save_dir : str, optional
        Directory where the PDF will be saved. If not provided, the PDF is saved in the current directory.

    Returns:
    --------
    str : Path to the downloaded PDF file.
    """
    import os
    import requests

    # If save_dir is provided, create the directory if it doesn't exist
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"zenodo_{zenodo_record_id}_pdf{pdf_number}.pdf")
    else:
        # If save_dir is not provided, save in the current working directory
        save_path = f"zenodo_{zenodo_record_id}_pdf{pdf_number}.pdf"

    # Fetch list of PDFs from Zenodo
    pdf_files = get_zenodo_pdfs(zenodo_record_id)
    if not pdf_files:
        raise ValueError(f"No PDFs found for Zenodo record {zenodo_record_id}")

    # Ensure the requested PDF exists
    if pdf_number > len(pdf_files):
        raise ValueError(f"Requested PDF {pdf_number} not available, only {len(pdf_files)} found.")

    # Get the selected PDF file
    pdf_url = pdf_files[pdf_number - 1]["links"]["self"]

    # Download and save PDF
    response = requests.get(pdf_url)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded PDF saved at: {save_path}")
        return save_path
    else:
        raise ValueError(f"Failed to download PDF (Status Code: {response.status_code})")













