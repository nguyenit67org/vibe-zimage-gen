"""Tkinter GUI for generating images with the quantized Z-image Turbo pipeline."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import torch
from diffusers import PipelineQuantizationConfig, ZImagePipeline
from diffusers.quantizers import quantize
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox

MODEL_ID = "tongsuo/Z-image-Turbo"
OUTPUT_DIR = Path("outputs")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


def _build_pipeline() -> ZImagePipeline:
    if DEVICE != "cuda":
        raise RuntimeError(
            "This app requires a CUDA-enabled GPU. Please run on a machine with an NVIDIA GPU."
        )

    pipeline_quant_config = PipelineQuantizationConfig(
        backend="bitsandbytes_4bit",
        weight_bits=4,
        weight_quants=quantize.Int4QuantizerConfig(
            calibration_tasks=["text_to_image"]
        ),
        activations_bits=8,
        activations_quants=quantize.Uint8QuantizerConfig(
            calibration_tasks=["text_to_image"]
        ),
        attn_implementation="flash_attention_2",
        attn_dtype="bfloat16",
        components_to_quantize="all",
        attn_quantization="per_tensor",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        # Z-image components are already in B DT for flash-attn
        components_to_keep_in_fp32=["vae.decoder", "vae.encoder"],
        components_to_keep_in_bf16=[
            "transformers",
            "text_encoder",
            "text_encoder_2",
        ],
    )

    pipeline = quantize(
        ZImagePipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
        ),
        pipeline_quant_config,
    ).to(DEVICE)

    pipeline.unet.set_default_attn_processor("flash_attention_2")
    return pipeline


class ZImageApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Vibe Z-Image Generator")
        self.root.geometry("640x720")

        self.pipeline: Optional[ZImagePipeline] = None
        self.generating = False

        self.prompt_var = tk.StringVar(value="A super-serious cat CEO in a business suit.")
        self.guidance_var = tk.DoubleVar(value=2.0)
        self.steps_var = tk.IntVar(value=4)
        self.seed_var = tk.IntVar(value=4325)

        self.status_var = tk.StringVar(value="Ready. GPU required: CUDA + bfloat16")

        self._image_label: Optional[ttk.Label] = None
        self._photo_image: Optional[ImageTk.PhotoImage] = None

        self._build_layout()

    def _build_layout(self) -> None:
        main_frame = ttk.Frame(self.root, padding=12)
        main_frame.pack(fill=tk.BOTH, expand=True)

        prompt_label = ttk.Label(main_frame, text="Prompt")
        prompt_label.pack(anchor=tk.W)

        prompt_entry = ttk.Entry(main_frame, textvariable=self.prompt_var)
        prompt_entry.pack(fill=tk.X, pady=(0, 8))

        controls = ttk.Frame(main_frame)
        controls.pack(fill=tk.X, pady=4)

        ttk.Label(controls, text="Guidance scale").grid(row=0, column=0, sticky=tk.W)
        ttk.Scale(
            controls,
            variable=self.guidance_var,
            orient=tk.HORIZONTAL,
            from_=0.0,
            to=10.0,
        ).grid(row=0, column=1, padx=8, sticky=tk.EW)

        ttk.Label(controls, text="Steps").grid(row=1, column=0, sticky=tk.W)
        ttk.Spinbox(
            controls,
            from_=1,
            to=20,
            textvariable=self.steps_var,
            width=10,
        ).grid(row=1, column=1, padx=8, sticky=tk.W)

        ttk.Label(controls, text="Seed").grid(row=2, column=0, sticky=tk.W)
        ttk.Spinbox(
            controls,
            from_=0,
            to=2**31 - 1,
            textvariable=self.seed_var,
            width=10,
        ).grid(row=2, column=1, padx=8, sticky=tk.W)

        controls.columnconfigure(1, weight=1)

        generate_btn = ttk.Button(
            main_frame, text="Generate", command=self._on_generate_clicked
        )
        generate_btn.pack(fill=tk.X, pady=8)

        status_label = ttk.Label(
            main_frame, textvariable=self.status_var, foreground="gray"
        )
        status_label.pack(anchor=tk.W, pady=(0, 8))

        separator = ttk.Separator(main_frame, orient=tk.HORIZONTAL)
        separator.pack(fill=tk.X, pady=4)

        self._image_label = ttk.Label(main_frame, text="Image preview will appear here")
        self._image_label.pack(fill=tk.BOTH, expand=True)

    def _load_pipeline(self) -> None:
        if self.pipeline is not None:
            return
        self.status_var.set("Loading pipeline (first run may take a while)...")
        self.root.update_idletasks()
        self.pipeline = _build_pipeline()
        self.status_var.set("Pipeline ready. Enter a prompt and generate.")

    def _on_generate_clicked(self) -> None:
        if self.generating:
            return

        prompt = self.prompt_var.get().strip()
        if not prompt:
            messagebox.showwarning("Missing prompt", "Please enter a prompt first.")
            return

        self.generating = True
        self.status_var.set("Preparing to generate...")

        thread = threading.Thread(target=self._generate_image, args=(prompt,))
        thread.daemon = True
        thread.start()

    def _generate_image(self, prompt: str) -> None:
        try:
            self._load_pipeline()
            assert self.pipeline is not None
        except Exception as exc:  # noqa: BLE001 - surface pipeline load failures
            self._update_status(f"Error loading pipeline: {exc}")
            self.generating = False
            return

        seed = int(self.seed_var.get())
        guidance = float(self.guidance_var.get())
        steps = int(self.steps_var.get())

        generator = torch.Generator(device=DEVICE).manual_seed(seed)

        self._update_status("Generating image...")
        start = time.time()
        try:
            result = self.pipeline(
                prompt=prompt,
                guidance_scale=guidance,
                num_inference_steps=steps,
                height=1024,
                width=1024,
                generator=generator,
                output_type="pil",
            )
            image: Image.Image = result.images[0]
        except Exception as exc:  # noqa: BLE001 - surface generation failures
            self._update_status(f"Generation failed: {exc}")
            self.generating = False
            return

        duration = time.time() - start
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        filename = OUTPUT_DIR / f"zimage_{int(time.time())}.png"
        image.save(filename)

        self.root.after(0, lambda: self._show_image(image, filename, duration))

    def _show_image(self, image: Image.Image, filename: Path, duration: float) -> None:
        display = image.copy().resize((512, 512))
        self._photo_image = ImageTk.PhotoImage(display)
        if self._image_label:
            self._image_label.configure(image=self._photo_image, text="")

        self._update_status(
            f"Saved {filename} (guidance={self.guidance_var.get():.2f}, steps={self.steps_var.get()}, {duration:.1f}s)"
        )
        self.generating = False

    def _update_status(self, message: str) -> None:
        self.status_var.set(message)
        self.root.after(0, self.root.update_idletasks)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = ZImageApp()
    app.run()


if __name__ == "__main__":
    main()
