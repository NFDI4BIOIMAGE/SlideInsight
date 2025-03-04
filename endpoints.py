"""
helper functions to prompt different language models.
"""

def prompt_chatgpt(image, prompt):
    """
    Generates a response from Azure's ChatGPT model using a text prompt and an image.
    
    Parameters
    ----------
    image : str
        Path to the image file to be analyzed.
    prompt : str
        Text prompt to guide the models response.
    
    Returns
    -------
    str
        The response content generated by the model.
    """
    
    import os
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
    
    response = client.complete(
        messages=[
            SystemMessage(
                content="You are a professional Data Scientist specializing in Image Data Analysis and Research Data Management."
            ),
            UserMessage(
                content=[
                    TextContentItem(text = prompt),
                    ImageContentItem(
                        image_url=ImageUrl.load(
                            image_file = image,
                            image_format="png",
                            detail=ImageDetailLevel.LOW)
                    ),
                ],
            ),
        ],
        model="gpt-4o",
    )

    return response.choices[0].message.content


def prompt_llama_11b(image, prompt):
    """
    Generates a response from Azure's Llama-3.2-11B-Vision-Instruct model using a text prompt and an image.
    
    Parameters
    ----------
    image : str
        Path to the image file to be analyzed.
    prompt : str
        Text prompt to guide the models response.
    
    Returns
    -------
    str
        The response content generated by the model.
    """
    
    import os
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
    
    response = client.complete(
        messages=[
            SystemMessage(
                content="You are a professional Data Scientist specializing in Image Data Analysis and Research Data Management."
            ),
            UserMessage(
                content=[
                    TextContentItem(text = prompt),
                    ImageContentItem(
                        image_url=ImageUrl.load(
                            image_file = image,
                            image_format="png",
                            detail=ImageDetailLevel.LOW)
                    ),
                ],
            ),
        ],
        model="Llama-3.2-11B-Vision-Instruct",
    )

    return response.choices[0].message.content



def prompt_phi(image, prompt):
    """
    Generates a response from Azure's Phi-3.5-vision-instruct model using a text prompt and an image.
    
    Parameters
    ----------
    image : str
        Path to the image file to be analyzed.
    prompt : str
        Text prompt to guide the models response.
    
    Returns
    -------
    str
        The response content generated by the model.
    """
    
    import os
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
    
    response = client.complete(
        messages=[
            SystemMessage(
                content="You are a professional Data Scientist specializing in Image Data Analysis and Research Data Management."
            ),
            UserMessage(
                content=[
                    TextContentItem(text = prompt),
                    ImageContentItem(
                        image_url=ImageUrl.load(
                            image_file = image,
                            image_format="png",
                            detail=ImageDetailLevel.LOW)
                    ),
                ],
            ),
        ],
        model="Phi-3.5-vision-instruct",
    )

    return response.choices[0].message.content



def prompt_llama_90b(image, prompt):
    """
    Generates a response from Azure's Llama-3.2-90B-Vision-Instruct model using a text prompt and an image.
    
    Parameters
    ----------
    image : str
        Path to the image file to be analyzed.
    prompt : str
        Text prompt to guide the models response.
    
    Returns
    -------
    str
        The response content generated by the model.
    """
    
    import os
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
    
    response = client.complete(
        messages=[
            SystemMessage(
                content="You are a professional Data Scientist specializing in Image Data Analysis and Research Data Management."
            ),
            UserMessage(
                content=[
                    TextContentItem(text = prompt),
                    ImageContentItem(
                        image_url=ImageUrl.load(
                            image_file = image,
                            image_format="png",
                            detail=ImageDetailLevel.LOW)
                    ),
                ],
            ),
        ],
        model="Llama-3.2-90B-Vision-Instruct",
    )

    return response.choices[0].message.content



def prompt_gpt_mini(image, prompt):
    """
    Generates a response from Azure's gpt-4o-mini model using a text prompt and an image.
    
    Parameters
    ----------
    image : str
        Path to the image file to be analyzed.
    prompt : str
        Text prompt to guide the models response.
    
    Returns
    -------
    str
        The response content generated by the model.
    """
    
    import os
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
    
    response = client.complete(
        messages=[
            SystemMessage(
                content="You are a professional Data Scientist specializing in Image Data Analysis and Research Data Management."
            ),
            UserMessage(
                content=[
                    TextContentItem(text = prompt),
                    ImageContentItem(
                        image_url=ImageUrl.load(
                            image_file = image,
                            image_format="png",
                            detail=ImageDetailLevel.LOW)
                    ),
                ],
            ),
        ],
        model="gpt-4o-mini",
    )

    return response.choices[0].message.content