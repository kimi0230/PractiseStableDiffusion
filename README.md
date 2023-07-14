# PractiseStableDiffusion

## Python env

* https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/

```sh
# Installing virtualenv
python3 -m pip install --user virtualenv

# Creating a virtual environment
python3 -m venv env

# Activating a virtual environment
source env/bin/activate
```

## PyTorch & Flax
* https://huggingface.co/docs/diffusers/v0.18.2/en/installation

PyTorch
```sh
pip install 'diffusers[torch]'
```

Flax
```sh
pip install 'diffusers[flax]'
```

```sh
pip install transformers
```

## 

### 'LayerNorm' is one of the layers in the Model.
```
'LayerNorm' is one of the layers in the Model. Looks like you're trying to load the diffusion model in float16(Half) format on CPU which is not supported. For float16 format, GPU needs to be used. For CPU run the model in float32 format.
Reference: https://github.com/pytorch/pytorch/issues/52291
```

### Potential NSFW content was detected in one or more images. A black image will be returned instead. Try again with a different prompt and/or seed.

* https://github.com/CompVis/stable-diffusion/issues/239
```py
pipe.safety_checker = lambda images, clip_input: (images, False)
```

## Reference
* [huggingface/diffusers](https://github.com/huggingface/diffusers)
* [AUTOMATIC1111 / stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)