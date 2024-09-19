import torch
from PIL.Image import Image
from diffusers import StableDiffusionXLPipeline
from pipelines.models import TextToImageRequest
from torch import Generator
from diffusers.models.attention_processor import AttnProcessor2_0


def load_pipeline() -> StableDiffusionXLPipeline:
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "./models/newdream-sdxl-21",
        torch_dtype=torch.float16,
        local_files_only=True,
    ).to("cuda")
    pipeline.unet = torch.compile(pipeline.unet, mode='reduce-overhead', fullgraph=True)
    pipeline.unet.set_attn_processor(AttnProcessor2_0())

    pipeline(prompt="")

    return pipeline


def infer(request: TextToImageRequest, pipeline: StableDiffusionXLPipeline) -> Image:
    generator = Generator(pipeline.device).manual_seed(request.seed) if request.seed else None

    return pipeline(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        generator=generator,
    ).images[0]
