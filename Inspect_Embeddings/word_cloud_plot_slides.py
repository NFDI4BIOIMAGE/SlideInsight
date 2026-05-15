import sys
import os

# Add the scripts directory to Python path to import functions
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import plotly.express as px
import dash
from dash import Dash, dcc, html, Input, Output
import numpy as np
from umap import UMAP
from wordcloud import WordCloud
from io import BytesIO
import base64
import pandas as pd
from tqdm.auto import tqdm
import plotly.graph_objects as go
from caching import load_full_hf_cache
import random

# Set up
repo_name_cache = "ScaDS-AI/SlideInsight_Cache_v2"
tqdm.pandas()
umap_model = UMAP(n_components=2, random_state=42)
print("Finished Setup")

print("Loading DataFrame from Hugging Face cache...")
df = load_full_hf_cache(repo_name=repo_name_cache)
print(f"Loaded DataFrame with {len(df)} rows and {len(df.columns)} columns.")

# Use Plotly's "Safe" palette (colorblind-safe)
plotly_colors = px.colors.qualitative.Safe

def plotly_color_func(*args, **kwargs):
    return random.choice(plotly_colors)


# Ensure embeddings are numpy arrays
df["mixed_embedding"] = df["mixed_embedding"].apply(np.array)
X = np.vstack(df["mixed_embedding"].values)
embedding_2d = umap_model.fit_transform(X)

df["umap_x"] = embedding_2d[:, 0]
df["umap_y"] = embedding_2d[:, 1]

print("Finished computing UMAP.")


# --- Build initial scatter plot ---
fig = px.scatter(
    df,
    x="umap_x",
    y="umap_y",
    color_discrete_sequence=["rgb(194, 184, 203)"],   #vorher: rgb(151, 203, 184)
    #color_discrete_sequence=["rgb(174, 39, 106)"], 
    hover_data=["zenodo_filename", "page_number"],
    title="Mixed Embedding UMAP Projection",
    width=900,
    height=700,
)


# --- Dash app setup ---
app = dash.Dash(__name__)
app.title = "UMAP + WordCloud Explorer"

app.layout = html.Div([
    html.H1("Interactive UMAP + Word Cloud Explorer", style={
        "textAlign": "center",
        "marginBottom": "20px"
    }),

    html.Div([
        html.Label("Neighborhood Radius:", style={"fontSize": "18px", "marginRight": "10px"}),
        dcc.Slider(
            id="radius-slider",
            min=0.2, max=1.5, step=0.1, value=0.5,
            marks={str(round(i, 2)): str(round(i, 2)) for i in np.arange(0.05, 1.05, 0.15)},
            tooltip={"placement": "bottom", "always_visible": True},
        )
    ], style={
        "width": "60%", "margin": "auto", "marginBottom": "40px"
    }),

    html.Div([
        # Left: UMAP plot
        html.Div([
            dcc.Graph(id="umap-plot", figure=fig)
        ], style={"width": "48%", "display": "inline-block"}),

        # Right: Word cloud
        html.Div([
            html.Img(id="wordcloud", style={
                "width": "100%", "height": "700px",
                "objectFit": "contain", "borderRadius": "10px"
            })
        ], style={"width": "48%", "display": "inline-block", "paddingLeft": "2%"})
    ], style={
        "display": "flex",
        "justifyContent": "center",
        "alignItems": "center",
        "textAlign": "center"
    })
])



# --- Helper to create a wordcloud image as base64 ---
def generate_wordcloud(texts):
    text = " ".join(str(t) for t in texts if isinstance(t, str))
    if not text.strip():
        return None
    #wc = WordCloud(width=800, height=600, background_color="white", color_func=plotly_color_func).generate(text)
    wc = WordCloud(width=800, height=600, background_color="white").generate(text)
    buffer = BytesIO()
    wc.to_image().save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


# --- Callback: update wordcloud when hovering ---
@app.callback(
    [Output("wordcloud", "src"),
     Output("umap-plot", "figure")],
    [Input("umap-plot", "hoverData"),
     Input("radius-slider", "value")]
)
def update_wordcloud_and_highlight(hoverData, radius):
    # base scatter
    updated_fig = px.scatter(
        df,
        x="umap_x",
        y="umap_y",
        color_discrete_sequence=["rgb(151, 203, 184)"],
        #color_discrete_sequence=["rgb(174, 39, 106)"],
        width=900,   
        height=700, 
    )

    if hoverData is None:
        return dash.no_update, updated_fig

    # Hovered point
    x_hover = hoverData["points"][0]["x"]
    y_hover = hoverData["points"][0]["y"]

    # Compute distance and find nearby points
    mask = np.sqrt((df["umap_x"] - x_hover)**2 + (df["umap_y"] - y_hover)**2) < radius
    nearby_texts = df.loc[mask, "extracted_text"]

    # Highlight neighbors
    updated_fig.add_trace(go.Scatter(
        x=df.loc[mask, "umap_x"],
        y=df.loc[mask, "umap_y"],
        mode="markers",
        marker=dict(size=14, color="black", symbol="circle-open", line=dict(width=2)),
        name="Selected neighborhood",
        hovertemplate=None,
        showlegend=False,
    ))

    updated_fig.update_layout(
        title=f"UMAP of Mixed Embeddings (r = {radius:.2f})",
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )

    # Generate new word cloud
    img = generate_wordcloud(nearby_texts)
    return img, updated_fig


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
