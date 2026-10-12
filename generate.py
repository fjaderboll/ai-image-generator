#!/usr/bin/env python3
"""Generate an image from a text prompt using a locally stored diffusers model."""

import argparse
import os
import random
import sys
import time
from pathlib import Path

MODEL_PATH = os.environ.get('MODEL_PATH', '/model')
DEFAULT_STEPS = int(os.environ.get('DEFAULT_STEPS', '4'))
DEFAULT_GUIDANCE = float(os.environ.get('DEFAULT_GUIDANCE', '0.0'))


def log(message):
	print(message, file=sys.stderr, flush=True)


def parse_args():
	parser = argparse.ArgumentParser(description='Generate an image from a text prompt.')
	parser.add_argument('-p', '--prompt', required=True, help='Text describing the image')
	parser.add_argument('-n', '--negative-prompt', help='Text describing what to avoid (ignored when guidance is 0)')
	parser.add_argument('-o', '--output', default='output.jpg', help='Output file, format given by extension (default: %(default)s)')
	parser.add_argument('--width', type=int, default=512, help='Image width in pixels (default: %(default)s)')
	parser.add_argument('--height', type=int, default=512, help='Image height in pixels (default: %(default)s)')
	parser.add_argument('--steps', type=int, default=DEFAULT_STEPS, help='Number of inference steps (default: %(default)s)')
	parser.add_argument('--guidance', type=float, default=DEFAULT_GUIDANCE, help='Guidance scale (default: %(default)s)')
	parser.add_argument('--seed', type=int, help='Seed for reproducible results (default: random)')
	parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto', help='Device to run on (default: %(default)s)')
	return parser.parse_args()


def select_device(requested):
	import torch

	if requested == 'auto':
		return 'cuda' if torch.cuda.is_available() else 'cpu'
	if requested == 'cuda' and not torch.cuda.is_available():
		sys.exit('CUDA requested but not available (use the GPU image and run with --gpus=all)')
	return requested


def load_pipeline(device):
	import torch
	from diffusers import AutoPipelineForText2Image

	# fp16 on GPU; fp32 on CPU since half precision is slow or unsupported there
	dtype = torch.float16 if device == 'cuda' else torch.float32
	has_fp16_weights = any(Path(MODEL_PATH).glob('*/*.fp16.safetensors'))

	pipe = AutoPipelineForText2Image.from_pretrained(
		MODEL_PATH,
		dtype=dtype,
		variant='fp16' if has_fp16_weights else None,
		use_safetensors=True,
		local_files_only=True,
	)
	return pipe.to(device)


def round_to_multiple(value, multiple=8):
	return max(multiple, round(value / multiple) * multiple)


def main():
	args = parse_args()

	import torch

	device = select_device(args.device)
	seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
	# models need dimensions divisible by 8, resize afterwards if adjusted
	width = round_to_multiple(args.width)
	height = round_to_multiple(args.height)

	log(f'Device: {device}')
	log(f'Prompt: {args.prompt}')
	log(f'Size: {args.width}x{args.height}, steps: {args.steps}, guidance: {args.guidance}, seed: {seed}')

	start = time.time()
	pipe = load_pipeline(device)
	log(f'Model loaded in {time.time() - start:.1f}s')

	start = time.time()
	image = pipe(
		prompt=args.prompt,
		negative_prompt=args.negative_prompt,
		width=width,
		height=height,
		num_inference_steps=args.steps,
		guidance_scale=args.guidance,
		generator=torch.Generator('cpu').manual_seed(seed),
	).images[0]
	log(f'Image generated in {time.time() - start:.1f}s')

	if image.size != (args.width, args.height):
		from PIL import Image
		image = image.resize((args.width, args.height), Image.LANCZOS)

	output = Path(args.output)
	output.parent.mkdir(parents=True, exist_ok=True)
	if output.suffix.lower() in ('.jpg', '.jpeg'):
		image.convert('RGB').save(output, quality=95)
	else:
		image.save(output)
	log(f'Saved: {output}')


if __name__ == '__main__':
	main()
