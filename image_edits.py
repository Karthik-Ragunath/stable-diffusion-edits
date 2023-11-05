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

def add_gaussian_noise(latents, std_dev = 0.5):
    noise = torch.normal(mean = 0, std = std_dev, size = latents.size(), device = latents.device, dtype = latents.dtype) # torch.Size([1, 4, 64, 64])
    return latents * 0.18215 + noise

def decode(latents, vae):
    with torch.no_grad():
        images = vae.decode(1 / 0.18215 * latents).sample
    return images

def get_image(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    image = (image * 255).round().astype("uint8")
    return Image.fromarray(image)

def encode_text(tokenizer, text_encoder, inputs = ['']):
    # uncond_inputs = [''] * len(inputs) # ['']
    # all_inputs = uncond_inputs + inputs # ['', 'Horse'] # tokenizer.model_max_length = 77
    all_inputs = inputs
    all_inputs_tokenized = tokenizer(all_inputs, padding = 'max_length', max_length = tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to('cuda') # all_inputs_tokenized.shape - torch.Size([2, 77])
    embeddings = text_encoder(all_inputs_tokenized)[0].half() # torch.Size([2, 77, 768]) # text_encoder(all_inputs_tokenized)[1].shape - torch.Size([1, 768]) # len(text_encoder(all_inputs_tokenized)) = 2
    return embeddings

def add_gaussian_noise_to_image(image, std_dev=0.5):
    """
    Add Gaussian noise with a given standard deviation to an image.

    Args:
    - image (torch.Tensor): The image tensor of shape [3, H, W].
    - std_dev (float): Standard deviation of the Gaussian noise.

    Returns:
    - torch.Tensor: Noisy image.
    """
    noise = torch.normal(mean=0, std=std_dev, size=image.size(), device=image.device, dtype = image.dtype)
    noisy_image = image + noise
    # Clip values to ensure they are within valid range (assuming [0, 1] range for image)
    noisy_image = torch.clamp(noisy_image, 0, 1)
    return noisy_image

def convert_to_latents(image_tensor, vae):
  # Make sure image_tensor is a batch object
  with torch.no_grad():
    latents = vae.encode(image_tensor).latent_dist.sample()
  return latents

def main():
    path = "fastdownloads"
    image_save_path = "save_images"
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
    scheduler = LMSDiscreteScheduler(beta_start=beta_start, beta_end=beta_end, beta_schedule="scaled_linear", num_train_timesteps=1000)
    device = torch.device('cuda')
    im_tensor = im_tensor.to(device).to(torch.float16)
    # with torch.no_grad():
        # latents = vae.encode(im_tensor[None, :]).latent_dist.sample() # torch.Size([1, 4, 64, 64]) # im_tensor[None, :].shape - torch.Size([1, 3, 512, 512])
    latents = convert_to_latents(im_tensor[None, :], vae) # torch.Size([1, 4, 64, 64])
    
    # visualizing the latents. # vae.encode(im_tensor[None, :]).latent_dist - 'DiagonalGaussianDistribution' object
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    for i in range(4):
        axs[i].imshow(latents[0][i].to('cpu').detach().numpy(), cmap='gray')
        axs[i].axis('off')  # Hide axes for better visualization
        axs[i].set_title(f"Channel {i + 1}")
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{image_save_path}/latent_channels.jpg")
    plt.close()

    latents_with_noise = add_gaussian_noise(latents) # torch.Size([1, 4, 64, 64])
    deocded_latents = decode(latents_with_noise, vae)[0] # torch.Size([3, 512, 512])
    pil_image = get_image(deocded_latents)
    pil_image.save(f"{image_save_path}/latent_decoded_image.jpg")

    noisy_img = add_gaussian_noise_to_image(im_tensor)
    pil_noisy_image = ToPILImage()(noisy_img)
    pil_noisy_image.save(f"{image_save_path}/latent_decoded_noisy_image.jpg")

    noisy_latents = convert_to_latents(noisy_img[None, :], vae)
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    for i in range(4):
        axs[i].imshow(noisy_latents[0][i].to('cpu').detach().numpy(), cmap='gray')
        axs[i].axis('off')
        axs[i].set_title(f"Channel {i + 1}")
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{image_save_path}/random_noisy_latent_image_channels.jpg")
    plt.close()

    ref_embs = encode_text(tokenizer, text_encoder, ['Horse'])
    query_embs = encode_text(tokenizer, text_encoder, ['Zebra'])

if __name__ == "__main__":
    main()
