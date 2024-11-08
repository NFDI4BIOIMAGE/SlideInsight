## Slide Insight

This repository contains a collection of notebooks for gaining insights into presentation slides through multimodal AI models. The goal is to compare different models and how they perform on summarizing the content of presentation slides. This is not implemented through text-to-text models but rather through image-to-text (multimodal) models. 

To access the models, a [free Service from Github](https://github.com/marketplace/models) is used.

***Be aware that there are certain [rate limits](https://docs.github.com/en/github-models/prototyping-with-ai-models#rate-limits) for each model!***

 - ### First:
Make sure to generate a developer key / personal access token on Github and set it as an environment variable. You can generate the token via the [Github website](github.com) under user settings and afterwards set it like this for your current session:


##### bash:
```export GITHUB_TOKEN= "your-github-token-goes-here"```

##### powershell:
```$Env:GITHUB_TOKEN= "your-github-token-goes-here"```

##### Windows command prompt:
```set GITHUB_TOKEN= your-github-token-goes-here```

 - ### Second:

Install Azure AI Inference SDK:

```pip install azure-ai-inference```

 - ### Third:
Set up the Model. How this is done is shown in the [first Notebook](Test_Models.ipynb)
