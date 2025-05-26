import torch
from cog import BasePredictor, Input, Path
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image

class Predictor(BasePredictor):
    def setup(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        ).to("cuda")

        # Load the LoRA weights
        self.pipe.load_lora_weights(".", weight_name="luna_style_training-10.safetensors")
        self.pipe.fuse_lora()

    def predict(
        self,
        prompt: str = Input(description="Prompt for the cartoon baby image"),
    ) -> Path:
        image = self.pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
        output_path = "/tmp/output.png"
        image.save(output_path)
        return Path(output_path)
