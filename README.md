# Slide Insight

This repository contains a collection of notebooks for gaining insights into presentation slides collected in the [NFDI4BIOIMAGE Training Material](https://nfdi4bioimage.github.io/training/readme.html) through multimodal AI models. Some goals are:

## 1. Explore different multimodal models and use slide embeddings to gain insights about contents
- [Comparison of different models](Test_Models.ipynb) and their performance on summarizing the content of presentation slides. This is not implemented through text-to-text models but rather through image-to-text (multimodal) models. As a first test pdf I3D:bio's Training Material ['WhatIsOMERO.pdf'](https://doi.org/10.5281/zenodo.8323588) (Schmidt, C., Bortolomeazzi, M. et al., 2023) is used.

- Improve our understanding about [how different types of embeddings represent the same content](Inspect_Embeddings/Compare_distorted_Embeddings.ipynb). For this task, some presentation slides are adapted to see whether text, visual or mixed-modal embeddings perform comparably well in representating a slides features, when the slide is changed in a specific manner. For this, slides are adapted from the Bio-image Data Science Lectures.

- Establishing a [cache](https://huggingface.co/datasets/ScaDS-AI/SlideInsight_Cache_v2) that stores a text embedding, visual embedding, mixed embedding and the extracted text for each slide from the collection. Besides that, there is also a dataset available that stores each [slide as an image](https://huggingface.co/datasets/ScaDS-AI/Slide_Insight_Images_v2) and shares a corresponding key with the embeddings from the cache dataset.

- The cached embeddings can then be used to [visualize the embeddings](Inspect_Embeddings/Visualize_HF_Embeddings.ipynb) and gain insights into the contents.

## 2. Building a multimodal RAG to efficiently search for specific contents
- Two different approaches were tested for this task:
    - Using the [Byaldi](https://github.com/AnswerDotAI/byaldi) package, see [Notebook](RAG/RAG_with_byaldi.ipynb).
    - Using an [Open AI CLIP Model](https://huggingface.co/openai/clip-vit-base-patch32) to compare image and query embeddings, see [Notebook](RAG/RAG_with_CLIP.ipynb).

- The two different approaches were compared against each other, as seen in the [corresponding Notebook](RAG/Compare_RAG_approaches.ipynb). To further evaluate their performance on the desired task, a [simple Benchmark](RAG/Benchmark_Byaldi_CLIP.ipynb) was created.


## Github Models
To access the AI models used in this repository, this [free Service from Github](https://github.com/marketplace/models) is used.

***Be aware that there are certain [rate limits](https://docs.github.com/en/github-models/prototyping-with-ai-models#rate-limits) for each model!***



### Before getting started:
Make sure to generate a developer key / personal access token on Github and set it as an environment variable. You can generate the token via the [Github website](github.com) under user settings and afterwards set it like this for your current session:


##### bash:
```export GITHUB_TOKEN= "your-github-token-goes-here"```

##### powershell:
```$Env:GITHUB_TOKEN= "your-github-token-goes-here"```

##### Windows command prompt:
```set GITHUB_TOKEN= your-github-token-goes-here```
