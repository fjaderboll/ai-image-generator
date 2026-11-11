# AI image generator
A Docker containerized simple tool to generate an image using AI.
Runs entirely locally no network access is needed at runtime.
Based on the model [SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo).

| Image tag        | Size   | Runs on                                    |
| ---------------- | ------ | ------------------------------------------ |
| `gpu` / `latest` | ~13 GB | NVIDIA GPU (~8 GB VRAM), falls back to CPU |
| `cpu`            | ~8 GB  | CPU only (~16 GB RAM)                      |

An 800x600 image takes about 9 seconds on an RTX 4070 Super and
on a 8-core CPU about 40 seconds. Most of that is loading the model, so to create
several images use `--prompts-file`, which loads it only once.

## Running
```shell
mkdir -p output

# single image
#  GPU accelerated
docker run --rm --gpus=all --user $(id -u):$(id -g) -v ./output:/data ghcr.io/fjaderboll/ai-image-generator --prompt "A happy unicorn"
#  CPU only (smaller image size)
docker run --rm --user $(id -u):$(id -g) -v ./output:/data ghcr.io/fjaderboll/ai-image-generator:cpu --prompt "A happy unicorn" --width 800 --height 600

# write to a specific file
docker run --rm --gpus=all --user $(id -u):$(id -g) -v ./output:/data ghcr.io/fjaderboll/ai-image-generator -p "A red fox" -o fox.png

# multiple images (there is like 90%+ overhead loading the model)
#  one image per line in output/prompts.txt, saved as animal-01.png, animal-02.png, ...
printf 'A red fox\nA happy unicorn\nA lonely wolf\n' > output/prompts.txt
docker run --rm --gpus=all --user $(id -u):$(id -g) -v ./output:/data ghcr.io/fjaderboll/ai-image-generator -f prompts.txt -o animal.png
```

`--user` makes the generated files owned by you instead of `root`. Create the output directory before running, otherwise Docker creates it owned by `root`.

GPU usage requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### Options
| Option | Default | Description |
|---|---|---|
| `-p`, `--prompt` | | Text describing the image, this or `--prompts-file` is required |
| `-f`, `--prompts-file` | | File relative to `/data` with one prompt per line, empty lines are skipped |
| `-n`, `--negative-prompt` | | Text describing what to avoid (ignored when guidance is 0), used for all prompts |
| `-o`, `--output` | `output.jpg` | Output file relative to `/data`, format given by extension. With `--prompts-file` numbered by line: `output-01.jpg`, `output-02.jpg`, ... (at least two digits) |
| `--width`, `--height` | `512` | Image size in pixels. SDXL-Turbo works best around 512 |
| `--steps` | `4` | Number of inference steps (1-4 is enough for SDXL-Turbo) |
| `--guidance` | `0.0` | Guidance scale (SDXL-Turbo is trained without guidance) |
| `--seed` | random | Seed for reproducible results, increased by one for each prompt in `--prompts-file` |
| `--device` | `auto` | `auto`, `cpu` or `cuda` |

## Building
```shell
# with GPU support
docker build -t ai-image-generator:gpu .

# CPU support only
docker build -t ai-image-generator:cpu --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu . 
```

Other diffusers models from [Hugging Face](https://huggingface.co/models?pipeline_tag=text-to-image&library=diffusers) can be bundled instead, with matching defaults:
```shell
docker build -t ai-image-generator:sdxl \
  --build-arg MODEL_ID=stabilityai/stable-diffusion-xl-base-1.0 \
  --build-arg DEFAULT_STEPS=30 --build-arg DEFAULT_GUIDANCE=7.0 .

# gated models need a token
HF_TOKEN=hf_... docker build --secret id=hf_token,env=HF_TOKEN --build-arg MODEL_ID=... .
```

## License
The bundled SDXL-Turbo model is licensed under the [Stability AI Non-Commercial Research Community License](https://huggingface.co/stabilityai/sdxl-turbo/blob/main/LICENSE.md).
