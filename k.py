from diffusers import DiffusionPipeline
import torch
import os
import sys
import shlex

# https://stackoverflow.com/questions/75641074/i-run-stable-diffusion-its-wrong-runtimeerror-layernormkernelimpl-not-implem
# commandline_args = os.environ.get('COMMANDLINE_ARGS', "--skip-torch-cuda-test --no-half")
commandline_args = os.environ.get('COMMANDLINE_ARGS', "--precision full --no-half")
sys.argv+=shlex.split(commandline_args)


print("Torch version:",torch.__version__)
print("Is CUDA enabled?",torch.cuda.is_available())

# 'LayerNorm' is one of the layers in the Model. Looks like you're trying to load the diffusion model in float16(Half) format on CPU which is not supported. For float16 format, GPU needs to be used. For CPU run the model in float32 format.
# Reference: https://github.com/pytorch/pytorch/issues/52291

# pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
# pipeline.to("cuda")

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32)
pipeline.safety_checker = lambda images, clip_input: (images, False)
pipeline("An image of a squirrel in Picasso style").images[0]