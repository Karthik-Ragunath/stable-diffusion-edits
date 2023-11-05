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
    uncond_inputs = [''] * len(inputs) # ['']
    all_inputs = uncond_inputs + inputs # ['', 'Horse'] # tokenizer.model_max_length = 77
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

def get_noise_inhalf(im_latents, prompt, scheduler, unet, vae, tokenizer, text_encoder, strength = 0.5, seed = 100, guidance_scale = 7.5, num_timesteps = 50):

  torch.manual_seed(seed)
  p_embs = encode_text(tokenizer, text_encoder, [prompt])

  #scale the embeddings
  scaled_latents = im_latents.clone() * vae.config.scaling_factor

  # Get the timestep
  scheduler.set_timesteps(num_timesteps)
  start_timestep_index = int(num_timesteps * strength)

  # Generate Noise and add the noise to the scaled latents
  noise = torch.randn_like(scaled_latents, dtype = scaled_latents.dtype, device = scaled_latents.device)
  noisy_latents = scheduler.add_noise(scaled_latents, noise, torch.tensor([scheduler.timesteps[start_timestep_index]], device = scaled_latents.device))

  # Run the diffusion process once on that timestep and get the results

  ts = scheduler.timesteps[start_timestep_index]
  input = torch.cat([noisy_latents]*2, dim = 0)
  input = scheduler.scale_model_input(input, ts)

  with torch.no_grad():
      noise_pred = unet(input, ts, encoder_hidden_states = p_embs).sample

  u, p = noise_pred.chunk(2)
  pred = u + guidance_scale * (p - u)
  noisy_latents = scheduler.step(pred, ts, noisy_latents).prev_sample

  return pred

def get_mask(latents, ref, query, scheduler, unet, vae, tokenizer, text_encoder, n = 10):

    seeds = torch.randint(low = 0, high = 7*10**6, size = (10,))
    noise_diffs = []
    for i, sd in enumerate(tqdm(seeds)):
        ref_noise = get_noise_inhalf(latents, 'Horse', scheduler, unet, vae, tokenizer, text_encoder, strength = 0.5, seed = sd, guidance_scale = 7.5, num_timesteps = 50)
        query_noise = get_noise_inhalf(latents, 'Zebra', scheduler, unet, vae, tokenizer, text_encoder, strength = 0.5, seed = sd, guidance_scale = 7.5, num_timesteps = 50)

        ## Calculate Eucledian distance between two noises and then average it on axis 0

        noise_diff = (query_noise - ref_noise)[0].pow(2).sum(dim = 0).pow(0.5)
        noise_diffs.append(noise_diff[None, :])

    noise_diff_t = torch.cat(noise_diffs, dim = 0)
    noise_diff_mean = torch.mean(noise_diff_t, dim = 0)
    # Normalization
    normalized_difference = (noise_diff_mean - noise_diff_mean.min()) / (noise_diff_mean.max() - noise_diff_mean.min())
    # Masking
    mask = (normalized_difference > 0.1).float()
    return mask

def edit_image(image_latents, mask, query_text, scheduler, unet, vae, tokenizer, text_encoder, strength = 0.5, num_timesteps = 50):
    mask = mask.to(torch.float16)
    embs = encode_text(tokenizer, text_encoder, [query_text])

    scaled_latents = image_latents.clone() * vae.config.scaling_factor

    scheduler.set_timesteps(num_timesteps)
    start_timestep_index = int(num_timesteps * strength)

    # Generate Noise and add the noise to the scaled latents
    noise = torch.randn_like(scaled_latents, dtype = scaled_latents.dtype, device = scaled_latents.device)

    yt = None

    for i, ts in enumerate(tqdm(scheduler.timesteps[start_timestep_index:])):
        xt = scheduler.add_noise(scaled_latents, noise, torch.tensor([ts], device = scaled_latents.device))

        if yt == None:
            yt = xt.clone()

        input = torch.cat([yt]*2, dim = 0)
        input = scheduler.scale_model_input(input, ts)

        with torch.no_grad():
            pred = unet(input, ts, encoder_hidden_states=embs).sample

        n_u, n_t = pred.chunk(2)
        noise = n_u + 7.5 * (n_t - n_u)

        yt_hat = scheduler.step(noise, ts, yt).prev_sample
        yt = mask * yt_hat + (1 - mask) * xt

    return yt



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

    # 2. Schedueler
    num_inference_steps = 30
    im_tensor = transforms.ToTensor()(img)
    im_tensor = im_tensor.to(device).to(torch.float16)
    # latents = convert_to_latents(im_tensor[None, :], vae) # torch.Size([1, 4, 64, 64])
    # latents_for_computation = add_gaussian_noise(latents) # torch.Size([1, 4, 64, 64])

    noisy_img = add_gaussian_noise_to_image(im_tensor)
    latents_for_computation = convert_to_latents(noisy_img[None, :], vae)
    
    # Denoising Part
    
    guidance_scale = 7.5
    scheduler.set_timesteps(num_inference_steps)
    latents_for_computation = latents_for_computation * scheduler.init_noise_sigma # torch.Size([1, 4, 64, 64])

    for i, t in enumerate(tqdm(scheduler.timesteps)):
        input = torch.cat([latents_for_computation]*2) # torch.Size([2, 4, 64, 64])
        input = scheduler.scale_model_input(input, t) # torch.Size([2, 4, 64, 64])

        # Predict the Noise
        with torch.no_grad():
            noise_pred = unet(input, t, encoder_hidden_states = ref_embs).sample # ref_embs.shape - torch.Size([2, 77, 768])

        n_u, n_p = noise_pred.chunk(2)
        pred = n_u + guidance_scale * (n_p - n_u)

        latents_for_computation = scheduler.step(pred, t, latents_for_computation).prev_sample
    
    deocded_latents = decode(latents_for_computation, vae)[0]
    pil_image = get_image(deocded_latents)
    pil_image.save(f"{image_save_path}/latent_decoded_image_ref_embeds.jpg")

    # scheduler.timesteps
    # tensor([999.0000, 964.5517, 930.1035, 895.6552, 861.2069, 826.7586, 792.3104,
    #     757.8621, 723.4138, 688.9655, 654.5172, 620.0690, 585.6207, 551.1724,
    #     516.7241, 482.2758, 447.8276, 413.3793, 378.9310, 344.4828, 310.0345,
    #     275.5862, 241.1379, 206.6897, 172.2414, 137.7931, 103.3448,  68.8966,
    #      34.4483,   0.0000])
    # scheduler.timesteps.shape
    # torch.Size([30])

    im_tensor = transforms.ToTensor()(img)
    im_tensor = im_tensor.to(device).to(torch.float16)
    latents = convert_to_latents(im_tensor[None, :], vae) # torch.Size([1, 4, 64, 64])
    ref_embs = text_encoder(tokenizer(['Horse'], padding = 'max_length', max_length = tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to('cuda'))[0].half()
    query_embs = text_encoder(tokenizer(['Zebra'], padding = 'max_length', max_length = tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to('cuda'))[0].half()
    noise_std_dev = torch.linspace(0.4, 0.6, 10)
    difference = torch.zeros(latents.size(), dtype = latents.dtype, device = latents.device)
    for noise_std in tqdm(noise_std_dev):
        noisy_latents = latents + torch.normal(mean=0, std = noise_std, size=latents.size(), device=latents.device, dtype = latents.dtype)
        with torch.no_grad():
            noise_r = unet(noisy_latents, scheduler.timesteps[0], ref_embs).sample
            noise_q = unet(noisy_latents, scheduler.timesteps[0], query_embs).sample

    difference += noise_q - noise_r
    difference = difference/10
    clipped_difference = torch.clamp(difference, min = -0.5, max = 0.5)
    # visualizing the latents.
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    for i in range(4):
        axs[i].imshow(clipped_difference[0][i].to('cpu').detach().numpy(), cmap='gray')
        axs[i].axis('off')  # Hide axes for better visualization
        axs[i].set_title(f"Channel {i + 1}")

    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{image_save_path}/difference_image.jpg')
    plt.close()
    normalized_difference = (clipped_difference - clipped_difference.min()) / (clipped_difference.max() - clipped_difference.min())
    mask = (normalized_difference > 0.5).float()

    im_tensor = transforms.ToTensor()(img)
    im_tensor = im_tensor.to(device).to(torch.float16)
    latents = convert_to_latents(im_tensor[None, :], vae) # torch.Size([1, 4, 64, 64])
    ref_embs = text_encoder(tokenizer(['Horse'], padding = 'max_length', max_length = tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to('cuda'))[0].half()
    query_embs = text_encoder(tokenizer(['Zebra'], padding = 'max_length', max_length = tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to('cuda'))[0].half()
    # mask = get_mask(latents, 'Horse', 'Zebra')
    mask = get_mask(latents=latents, ref=ref_embs, query=query_embs, scheduler=scheduler, unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder)
    plt.imshow(mask.cpu().detach().numpy(), cmap = 'gray')
    plt.savefig(f'{image_save_path}/mask_image.jpg')
    plt.close()

    # yt = edit_image(latents, mask, 'A zebra on a grass field with a cloudy sky', strength = 0.3, num_timesteps = 70)
    yt = edit_image(latents, mask, 'A zebra on a grass field with a cloudy sky', scheduler=scheduler, unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, strength = 0.3, num_timesteps = 70)
    with torch.no_grad():
        image = vae.decode(1 / 0.18215 * yt).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image[0].detach().cpu().permute(1, 2, 0).numpy()
    image = (image * 255).round().astype("uint8")
    zebra_pil_image = Image.fromarray(image)
    zebra_pil_image.save(f"{image_save_path}/zebra.jpg")

if __name__ == "__main__":
    main()
