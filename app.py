import gradio as gr
from Models import VisionModel
import huggingface_hub
from PIL import Image
import torch.amp.autocast_mode
from pathlib import Path
import torch
import torchvision.transforms.functional as TVF
import os

MODEL_REPO = "fancyfeast/joytag"
THRESHOLD = 0.4
DESCRIPTION = """
This application uses the JoyTag model to automatically tag images with relevant labels. 
You can upload multiple images at once for batch processing.

How to use:
1. Click the "Upload Images" button or drag and drop images onto the designated area.
2. Click the "Process Images" button to start tagging.
3. View the results in the table below, including tags and scores.

Model: [JoyTag](https://huggingface.co/fancyfeast/joytag)
"""

def prepare_image(image: Image.Image, target_size: int) -> torch.Tensor:
    # Pad image to square
    image_shape = image.size
    max_dim = max(image_shape)
    pad_left = (max_dim - image_shape[0]) // 2
    pad_top = (max_dim - image_shape[1]) // 2

    padded_image = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
    padded_image.paste(image, (pad_left, pad_top))

    # Resize image
    if max_dim != target_size:
        padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)
    
    # Convert to tensor
    image_tensor = TVF.pil_to_tensor(padded_image) / 255.0

    # Normalize
    image_tensor = TVF.normalize(image_tensor, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

    return image_tensor

@torch.no_grad()
def predict_single(image: Image.Image):
    image_tensor = prepare_image(image, model.image_size)
    batch = {
        'image': image_tensor.unsqueeze(0),
    }

    with torch.amp.autocast_mode.autocast('cpu', enabled=True):
        preds = model(batch)
        tag_preds = preds['tags'].sigmoid().cpu()
    
    scores = {top_tags[i]: tag_preds[0][i].item() for i in range(len(top_tags))}
    predicted_tags = [tag for tag, score in scores.items() if score > THRESHOLD]
    tag_string = ', '.join(predicted_tags)

    return tag_string, scores

def process_images(image_files):
    results = []
    for img_file in image_files:
        img = Image.open(img_file.name).convert('RGB')
        tag_string, scores = predict_single(img)
        
        # Format scores for display
        formatted_scores = ", ".join([f"{tag}: {score:.2f}" for tag, score in sorted(scores.items(), key=lambda x: x[1], reverse=True) if score > THRESHOLD])
        
        results.append([os.path.basename(img_file.name), tag_string, formatted_scores])
    return results

print("Downloading model...")
path = huggingface_hub.snapshot_download(MODEL_REPO)
print("Loading model...")
model = VisionModel.load_model(path)
model.eval()

with open(Path(path) / 'top_tags.txt', 'r') as f:
    top_tags = [line.strip() for line in f.readlines() if line.strip()]

print("Starting server...")

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏷️ JoyTag Batch Image Tagger")
    gr.Markdown(DESCRIPTION)
    
    with gr.Row():
        input_images = gr.File(label="Upload Images", file_count="multiple")
        submit_btn = gr.Button("Process Images", variant="primary")

    output = gr.Dataframe(
        headers=["Filename", "Tags", "Scores (score > 0.4)"],
        label="Results",
    )
    
    submit_btn.click(
        fn=process_images,
        inputs=input_images,
        outputs=output,
    )

    gr.Markdown("## How it works")
    gr.Markdown("""
    1. The JoyTag model analyzes each uploaded image.
    2. It predicts relevant tags based on the image content.
    3. Tags with a confidence score above 0.4 are displayed.
    4. Results are presented in a table for easy viewing.
    
    For more information about the JoyTag model, visit the [model page](https://huggingface.co/fancyfeast/joytag).
    """)

if __name__ == "__main__":
    demo.launch(share=True)
