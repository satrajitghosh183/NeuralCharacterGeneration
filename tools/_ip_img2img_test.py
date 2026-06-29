import sys
import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image

p = AutoPipelineForImage2Image.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16,
    safety_checker=None).to("cuda")
p.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
p.set_ip_adapter_scale(0.6)
procs = p.unet.attn_processors
ipn = sum(1 for v in procs.values() if "IPAdapter" in type(v).__name__)
print(f"procs {len(procs)} ip {ipn}", flush=True)
base = Image.open(sys.argv[1]).convert("RGB").resize((512, 512))
ref = Image.open(sys.argv[2]).convert("RGB")
o = p(prompt="a photo of a woman, photorealistic skin", image=base, ip_adapter_image=ref,
      strength=0.5, num_inference_steps=20).images[0]
o.save("/tmp/ipi2i.png")
print("OK wrote /tmp/ipi2i.png", flush=True)
