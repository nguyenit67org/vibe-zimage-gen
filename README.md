# Vibe Z-Image Generator

A lightweight Tkinter GUI for generating images from text prompts using the quantized [tongsuo/Z-image-Turbo](https://huggingface.co/tongsuo/Z-image-Turbo) model. The app reproduces the setup shown in the reference snippet while providing a simple interface for experimenting with prompts.

> **Note:** The pipeline is optimized for CUDA GPUs and requires a machine with an NVIDIA GPU that supports bfloat16. CPU execution is not recommended.

## Prerequisites

- Python 3.10+
- An NVIDIA GPU with CUDA drivers and at least ~16 GB of VRAM
- [uv](https://github.com/astral-sh/uv) package manager installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Access to the `tongsuo/Z-image-Turbo` model on Hugging Face (ensure you have accepted the model license if required)

## Installation

1. Clone this repository and enter the directory.
2. Install dependencies with uv:

```bash
uv sync
```

This will create and populate a `.venv` managed by uv.

## Running the app

Launch the GUI with uv:

```bash
uv run vibe-zimage-gen
```

The interface allows you to:

- Enter a text prompt
- Adjust the guidance scale and number of inference steps
- Generate and preview the resulting image

Generated images are saved to the `outputs/` directory with timestamped filenames.

## Notes

- Model weights are downloaded on the first run; subsequent generations reuse the cached pipeline.
- If you need deterministic outputs, set a custom seed in the GUI before generating.
- The GUI uses the exact quantization configuration shown in the reference code to reduce VRAM usage.
