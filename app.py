import gradio as gr
from Models import VisionModel
import huggingface_hub
from PIL import Image
import torch.amp.autocast_mode
from pathlib import Path
import torch
import torchvision.transforms.functional as TVF


MODEL_REPO = "fancyfeast/joytag"
THRESHOLD = 0.4
DESCRIPTION = """
Demo for the JoyTag model: https://huggingface.co/fancyfeast/joytag
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
def predict(image: Image.Image):
	image_tensor = prepare_image(image, model.image_size)
	batch = {
		'image': image_tensor.unsqueeze(0),
	}

	with torch.amp.autocast_mode.autocast('cpu', enabled=True):
		preds = model(batch)
		tag_preds = preds['tags'].sigmoid().cpu()
	
	scores = {top_tags[i]: tag_preds[0][i] for i in range(len(top_tags))}
	predicted_tags = [tag for tag, score in scores.items() if score > THRESHOLD]
	tag_string = ', '.join(predicted_tags)

	return tag_string, scores


print("Downloading model...")
path = huggingface_hub.snapshot_download(MODEL_REPO)
print("Loading model...")
model = VisionModel.load_model(path)
model.eval()

with open(Path(path) / 'top_tags.txt', 'r') as f:
	top_tags = [line.strip() for line in f.readlines() if line.strip()]

print("Starting server...")

gradio_app = gr.Interface(
	predict,
	inputs=gr.Image(label="Source", sources=['upload', 'webcam'], type='pil'),
	outputs=[
		gr.Textbox(label="Tag String"),
		gr.Label(label="Tag Predictions", num_top_classes=100),
	],
	title="JoyTag",
	description=DESCRIPTION,
	allow_flagging="never",
)


if __name__ == '__main__':
	gradio_app.launch(share=True)
