import sys
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image

model = "stable-diffusion-v1-5/stable-diffusion-v1-5"
ref_path = sys.argv[1] if len(sys.argv) > 1 else "data/check_FX490LPq4f4/f0055.jpg"
p = AutoPipelineForText2Image.from_pretrained(model, torch_dtype=torch.float16,
                                              safety_checker=None).to("cuda")
p.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
p.set_ip_adapter_scale(0.6)
# Introspect: how many attention processors became IP-Adapter-aware?
procs = p.unet.attn_processors
ipn = sum(1 for v in procs.values() if "IPAdapter" in type(v).__name__)
print(f"attn processors: {len(procs)} total, {ipn} IP-Adapter-aware", flush=True)
ref = Image.open(ref_path).convert("RGB")
o = p(prompt="a photo of a woman, photorealistic", ip_adapter_image=ref,
      num_inference_steps=20).images[0]
o.save("/tmp/ipt2i.png")
print("OK wrote /tmp/ipt2i.png", flush=True)
