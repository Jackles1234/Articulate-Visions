from PIL import Image
import torch as th
import os
import glob

# Assuming the required modules for model creation and checkpoint loading are properly defined elsewhere
from download import load_checkpoint
from create_model import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)

def generate_image(prompt, batch_size, guidance_scale, head_channels, xf_heads):
    try:
        # Check if the folder 'imgs' exists, if not create one
        if not os.path.exists('../imgs'):
            os.makedirs('../imgs')

        has_cuda = th.cuda.is_available()
        device = th.device('cpu' if not has_cuda else 'cuda')

        options = model_and_diffusion_defaults()
        options['num_head_channels'] = head_channels
        options['xf_heads'] = xf_heads

        options['use_fp16'] = has_cuda
        options['timestep_respacing'] = '100'  # Fast sampling with 100 diffusion steps
        model, diffusion = create_model_and_diffusion(**options)
        model.eval()
        if has_cuda:
            model.convert_to_fp16()
        model.to(device)
        model.load_state_dict(load_checkpoint('base', device))

        options_up = model_and_diffusion_defaults_upsampler()
        options_up['use_fp16'] = has_cuda
        options_up['timestep_respacing'] = 'fast27'  # Very fast sampling with 27 diffusion steps
        model_up, diffusion_up = create_model_and_diffusion(**options_up)
        model_up.eval()
        if has_cuda:
            model_up.convert_to_fp16()
        model_up.to(device)
        model_up.load_state_dict(load_checkpoint('upsample', device))

        tokens = model.tokenizer.encode(prompt)
        tokens, mask = model.tokenizer.padded_tokens_and_mask(
            tokens, options['text_ctx']
        )
        
        full_batch_size = batch_size * 2
        uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
            [], options['text_ctx']
        )

        model_kwargs = {
            'tokens': th.tensor(
                [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
            ),
            'mask': th.tensor(
                [mask] * batch_size + [uncond_mask] * batch_size,
                dtype=th.bool,
                device=device,
            ),
        }

        def model_fn(x_t, ts, **kwargs):
            half = x_t[: len(x_t) // 2]
            combined = th.cat([half, half], dim=0)
            model_out = model(combined, ts, **kwargs)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = th.cat([half_eps, half_eps], dim=0)
            return th.cat([eps, rest], dim=1)

        model.del_cache()
        samples = diffusion.p_sample_loop(
            model_fn,
            (full_batch_size, 3, options["image_size"], options["image_size"]),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None
        )[:batch_size]
        model.del_cache()
        files = glob.glob('/imgs')
        for f in files:
            os.remove(f)
        # Save generated images
        for i, sample in enumerate(samples):
            image = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(1, 2, 0).cpu().numpy()
            img = Image.fromarray(image)
            img.save(f"imgs/image_batch{i}.png")
            x = f"imgs/image_batch{i}.png"
        return x
    except ValueError:
        # Reset to default values if there's an error
        generate_image(prompt="an oil painting of a corgi", batch_size=1, guidance_scale=3.0)

#generate_image(prompt="an oil painting of a corgi", batch_size=2, guidance_scale=3.0)