import logging
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from diffusers import StableDiffusionPipeline
from fastcore.all import concat
from huggingface_hub import notebook_login
from PIL import Image
from ipdb import set_trace as st
from tqdm.auto import tqdm
from diffusers import StableDiffusionImg2ImgPipeline
from fastdownload import FastDownload
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
from torchvision import transforms
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import LMSDiscreteScheduler

def main():
    path = "fastdownloads"
    dler = FastDownload(base=path)
    dler.download('https://iadsb.tmgrup.com.tr/7ddb86/0/0/0/0/1926/1086?u=https://idsb.tmgrup.com.tr/2018/05/22/horses-the-wings-of-mankind-1527015927739.jpg')
    img = Image.open(f'{path}/archive/horses-the-wings-of-mankind-1527015927739.jpg').crop((0, 0, 1200, 1086)).resize((512, 512))
    # plt.imshow(img)
    # plt.savefig(f'{path}/horse_with_wings.jpg')
    # plt.close()
    img.save(f'{path}/horse_with_wings.jpg')
    im_tensor = transforms.ToTensor()(img)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16)
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to("cuda")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema", torch_dtype=torch.float16).to("cuda")
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", torch_dtype=torch.float16).to("cuda")
    beta_start, beta_end = 0.00085, 0.012
    scheduler = LMSDiscreteScheduler(beta_start = beta_start, beta_end = beta_end, beta_schedule = "scaled_linear", num_train_timesteps = 1000)

if __name__ == "__main__":
    main()
