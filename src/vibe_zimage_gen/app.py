"""Tkinter GUI for generating images with the quantized Z-image Turbo pipeline."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import torch
from diffusers import PipelineQuantizationConfig, ZImagePipeline
from packaging import version as pkg_version
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox

MODEL_ID = "Tongyi-MAI/Z-image-Turbo"
OUTPUT_DIR = Path("outputs")

# Prefer running on CUDA for correct performance; let the runtime verify availability.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

TORCH_VERSION = pkg_version.Version(torch.__version__.split("+")[0])
if TORCH_VERSION < pkg_version.Version("2.5.0"):
    # Torch < 2.5.0 can't handle the `enable_gqa` kwarg, so strip it for compatibility.
    import torch.nn.functional as F  # type: ignore  # noqa: WPS433

    _orig_sdpa = F.scaled_dot_product_attention

    def _sdpa_no_gqa(*args, **kwargs):  # type: ignore[override]
        kwargs.pop("enable_gqa", None)
        return _orig_sdpa(*args, **kwargs)

    F.scaled_dot_product_attention = _sdpa_no_gqa


def _build_pipeline() -> ZImagePipeline:
    if DEVICE != "cuda":
        raise RuntimeError("This app requires a CUDA-enabled GPU. Please run on a machine with an NVIDIA GPU.")

    pipeline_quant_config = PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={
            "load_in_4bit": True,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
        },
    )

    pipeline = ZImagePipeline.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        quantization_config=pipeline_quant_config,
    ).to(DEVICE)

    return pipeline


class ZImageApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Vibe Z-Image Generator")
        self.root.geometry("640x720")

        self.pipeline: Optional[ZImagePipeline] = None
        self.generating = False

        self.prompt_var = tk.StringVar(value="A super-serious cat CEO in a business suit.")
        self.guidance_var = tk.DoubleVar(value=0.0)
        self.guidance_display_var = tk.StringVar(value=f"{self.guidance_var.get():.2f}")
        self.guidance_var.trace_add("write", lambda *_: self._sync_guidance_display())
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
        guidance_display = ttk.Entry(
            controls,
            textvariable=self.guidance_display_var,
            width=6,
            state="readonly",
            justify="center",
        )
        guidance_display.grid(row=0, column=1, padx=(8, 4), sticky=tk.W)
        ttk.Scale(
            controls,
            variable=self.guidance_var,
            orient=tk.HORIZONTAL,
            from_=0.0,
            to=10.0,
        ).grid(row=0, column=2, padx=8, sticky=tk.EW)

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

        controls.columnconfigure(2, weight=1)

        generate_btn = ttk.Button(main_frame, text="Generate", command=self._on_generate_clicked)
        generate_btn.pack(fill=tk.X, pady=8)

        status_label = ttk.Label(main_frame, textvariable=self.status_var, foreground="gray")
        status_label.pack(anchor=tk.W, pady=(0, 8))

        separator = ttk.Separator(main_frame, orient=tk.HORIZONTAL)
        separator.pack(fill=tk.X, pady=4)

        self._image_label = ttk.Label(main_frame, text="Image preview will appear here")
        self._image_label.pack(fill=tk.BOTH, expand=True)

    def _sync_guidance_display(self) -> None:
        self.guidance_display_var.set(f"{self.guidance_var.get():.2f}")

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
