from pdf2image import convert_from_path
from PIL import Image
import os
import json
import pdfplumber


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
    
    slides = convert_from_path(pdf_name)
    return slides



def save_images(filepath, slides, new_width=None):
    """
    Save a list of images to a specified directory , optionally with a resized width.

    Parameters
    ----------
    filepath : str
        The directory path where the images will be saved.
    slides : list
        A list of images to be saved.
    new_width : int, optional
        The new width for resizing each image while maintaining its aspect ratio.
        If not provided, the original width of each image will be used.

    Returns
    -------
    None
        Saves each image in the specified directory as PNG files named slide1.png, slide2.png, etc.
    """
    
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
        image_filename = os.path.join(filepath, f"slide{i}.png")
        img_resized.save(image_filename, "PNG")

def text_extraction(pdf_name, images):
    """
    Extract text from a PDF and pair it with corresponding slide images, saving the result as a JSON file.
    
    Parameters
    ----------
    pdf_name : str
        The name or path of the PDF file from which to extract text.
    images : list
        A list of images corresponding to the slides in the PDF.
    
    Returns
    -------
    None
        Saves a dictionary mapping slide image file names (e.g., slide1.png) to the extracted text
        in a JSON file named 'dict_slides_text.json'.
    """

    slide_texts = []

    # Extract the original text from each Slide
    with pdfplumber.open(pdf_name) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            slide_texts.append(text)
            
    slide_dict = {}
    for i, (img, text) in enumerate(zip(images, slide_texts), start=1):
        # Define the path for each image file
        image_path = f"slide{i}.png"
        
        # Add the image path and corresponding text to the dictionary
        slide_dict[image_path] = text
    
    # Save the dictionary to a JSON file
    with open(f"dict_{pdf_name}.json", "w") as json_file:
        json.dump(slide_dict, json_file)




