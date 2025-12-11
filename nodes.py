# LongCat-Image nodes
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from comfy_api.latest._io import comfytype, ComfyTypeIO
import torch
import folder_paths
import os

# HuggingFace model repository mapping
LONGCAT_MODEL_REPOS = {
    "LongCat-Image": "meituan/LongCat-Image",
    "LongCat-Image-Edit": "meituan/LongCat-Image-Edit",
    "LongCat-Image-Dev": "meituan/LongCat-Image-Dev",
}

# Available models for download
LONGCAT_DOWNLOADABLE_MODELS = list(LONGCAT_MODEL_REPOS.keys())


def check_and_download_model(model_name: str) -> str:
    """
    Check if the model exists locally, if not download from HuggingFace.
    Returns the local path to the model.
    """
    # Default download directory
    download_dir = os.path.join(folder_paths.models_dir, "diffusion_models")
    os.makedirs(download_dir, exist_ok=True)
    
    model_path = os.path.join(download_dir, model_name)
    
    # Check if model already exists locally
    if os.path.exists(model_path) and os.path.isdir(model_path):
        # Verify it has the required files
        transformer_path = os.path.join(model_path, "transformer")
        if os.path.exists(transformer_path):
            print(f"[LongCat] Model '{model_name}' found locally at: {model_path}")
            return model_path
    
    # Model not found, need to download
    if model_name not in LONGCAT_MODEL_REPOS:
        raise ValueError(f"Unknown model '{model_name}'. Available models: {list(LONGCAT_MODEL_REPOS.keys())}")
    
    repo_id = LONGCAT_MODEL_REPOS[model_name]
    print(f"[LongCat] Model '{model_name}' not found locally. Downloading from HuggingFace: {repo_id}")
    
    try:
        from huggingface_hub import snapshot_download
        
        # Download the model
        snapshot_download(
            repo_id=repo_id,
            local_dir=model_path,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"[LongCat] Model '{model_name}' downloaded successfully to: {model_path}")
        return model_path
        
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for model download. "
            "Install it with: pip install huggingface_hub"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to download model '{model_name}' from {repo_id}: {e}")


@comfytype(io_type="LONGCAT_PIPE")
class LongCatPipe(ComfyTypeIO):
    Type = dict

def get_available_longcat_models():
    # Start with downloadable models (will be auto-downloaded if selected)
    models = list(LONGCAT_DOWNLOADABLE_MODELS)
    
    # Add locally existing models
    base_paths = [
        os.path.join(folder_paths.models_dir, "diffusion_models"),
        os.path.join(folder_paths.models_dir, "checkpoints"),
        folder_paths.models_dir,
    ]
    for base_path in base_paths:
        if not os.path.exists(base_path):
            continue
        try:
            for item in os.listdir(base_path):
                item_path = os.path.join(base_path, item)
                if os.path.isdir(item_path):
                    if "longcat" in item.lower() or os.path.exists(os.path.join(item_path, "transformer")):
                        if item not in models:
                            models.append(item)
        except OSError:
            continue
    
    # Add manual path option at the end
    if "(manual path)" not in models:
        models.append("(manual path)")
    
    return models


class MeituanLongCatLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        available_models = get_available_longcat_models()
        return io.Schema(
            node_id="MeituanLongCatLoader",
            display_name="LongCat Model Loader",
            category="Meituan/LongCat",
            inputs=[
                io.Combo.Input("model_name", options=available_models, default=available_models[0]),
                io.String.Input("custom_model_path", default="", multiline=False),
                io.Combo.Input("dtype", options=["bfloat16", "float16", "float32"], default="bfloat16"),
                io.Combo.Input("enable_cpu_offload", options=["true", "false"], default="true"),
                io.Combo.Input("attention_backend", options=["default", "sage"], default="default"),
            ],
            outputs=[LongCatPipe.Output(display_name="LongCat Pipeline")],
        )

    @classmethod
    def execute(cls, model_name, custom_model_path, dtype, enable_cpu_offload, attention_backend) -> io.NodeOutput:
        try:
            from transformers import AutoProcessor
            from longcat_image.models import LongCatImageTransformer2DModel
            from longcat_image.pipelines import LongCatImagePipeline, LongCatImageEditPipeline
        except ImportError as e:
            raise ImportError(f"LongCat-Image not installed: {e}")
        
        # Determine model path
        if model_name == "(manual path)" or not model_name:
            model_path = custom_model_path
            if not model_path:
                raise ValueError("Model path required when using manual path option")
        else:
            model_path = model_name
        
        # Handle absolute paths
        if os.path.isabs(model_path):
            checkpoint_dir = model_path
            if not os.path.exists(checkpoint_dir):
                raise ValueError(f"Model not found at path: {checkpoint_dir}")
        # Check if it's a downloadable model
        elif model_path in LONGCAT_MODEL_REPOS:
            # Auto-download if not exists
            checkpoint_dir = check_and_download_model(model_path)
        else:
            # Search in local directories
            checkpoint_dir = None
            for base in [os.path.join(folder_paths.models_dir, "diffusion_models"), folder_paths.models_dir]:
                p = os.path.join(base, model_path)
                if os.path.exists(p):
                    checkpoint_dir = p
                    break
            if not checkpoint_dir:
                raise ValueError(f"Model not found: {model_path}. If this is a HuggingFace model, please use one of the supported models: {list(LONGCAT_MODEL_REPOS.keys())}")
        
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        torch_dtype = dtype_map[dtype]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if attention_backend == "sage" and device.type == "cuda":
            try:
                import torch.nn.functional as F
                from sageattention import sageattn
                if not getattr(F.scaled_dot_product_attention, "_sage_wrapped", False):
                    orig = F.scaled_dot_product_attention
                    def wrapper(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
                        if attn_mask is not None or dropout_p not in (0, 0.0) or query.shape[-1] > 256:
                            return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)
                        try:
                            return sageattn(query, key, value, tensor_layout="HND", is_causal=is_causal, sm_scale=scale)
                        except:
                            return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)
                    wrapper._sage_wrapped = True
                    F.scaled_dot_product_attention = wrapper
            except ImportError:
                pass
        
        text_processor = AutoProcessor.from_pretrained(checkpoint_dir, subfolder="tokenizer")
        is_edit = "edit" in model_path.lower()
        cpu_off = enable_cpu_offload == "true"
        
        transformer = LongCatImageTransformer2DModel.from_pretrained(checkpoint_dir, subfolder="transformer", torch_dtype=torch_dtype, use_safetensors=True)
        if not cpu_off:
            transformer = transformer.to(device)
        
        PipeClass = LongCatImageEditPipeline if is_edit else LongCatImagePipeline
        pipe = PipeClass.from_pretrained(checkpoint_dir, transformer=transformer, text_processor=text_processor, torch_dtype=torch_dtype)
        
        if cpu_off:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device, torch_dtype)
        
        return io.NodeOutput({"pipe": pipe, "device": device, "dtype": torch_dtype, "is_edit": is_edit})


class MeituanLongCatT2I(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MeituanLongCatT2I",
            display_name="LongCat Text to Image",
            category="Meituan/LongCat",
            inputs=[
                LongCatPipe.Input("longcat_pipeline", display_name="LongCat Pipeline"),
                io.String.Input("prompt", default="", multiline=True),
                io.String.Input("negative_prompt", default="", multiline=True),
                io.Int.Input("width", default=1344, min=64, max=4096, step=64),
                io.Int.Input("height", default=768, min=64, max=4096, step=64),
                io.Int.Input("steps", default=50, min=1, max=200, step=1, display_mode=io.NumberDisplay.number),
                io.Float.Input("guidance_scale", default=4.5, min=0.0, max=20.0, step=0.1),
                io.Int.Input("seed", default=43, min=0, max=0xffffffffffffffff),
                io.Combo.Input("enable_cfg_renorm", options=["true", "false"], default="true"),
                io.Combo.Input("enable_prompt_rewrite", options=["true", "false"], default="true"),
                io.Int.Input("batch_size", default=1, min=1, max=8, step=1),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, longcat_pipeline, prompt, negative_prompt, width, height, steps, guidance_scale, seed, enable_cfg_renorm, enable_prompt_rewrite, batch_size) -> io.NodeOutput:
        if not longcat_pipeline:
            raise ValueError("Pipeline required")
        if longcat_pipeline.get("is_edit"):
            raise ValueError("Use LongCat Image Edit node for edit models")
        
        import numpy as np
        result = longcat_pipeline["pipe"](
            prompt, negative_prompt=negative_prompt, height=height, width=width,
            guidance_scale=guidance_scale, num_inference_steps=steps, num_images_per_prompt=batch_size,
            generator=torch.Generator("cpu").manual_seed(seed),
            enable_cfg_renorm=enable_cfg_renorm == "true", enable_prompt_rewrite=enable_prompt_rewrite == "true"
        )
        return io.NodeOutput(torch.stack([torch.from_numpy(np.array(img).astype(np.float32) / 255.0) for img in result.images], dim=0))


class MeituanLongCatEdit(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MeituanLongCatEdit",
            display_name="LongCat Image Edit",
            category="Meituan/LongCat",
            inputs=[
                LongCatPipe.Input("longcat_pipeline", display_name="LongCat Pipeline"),
                io.Image.Input("image"),
                io.String.Input("prompt", default="", multiline=True),
                io.String.Input("negative_prompt", default="", multiline=True),
                io.Int.Input("steps", default=50, min=1, max=200, step=1, display_mode=io.NumberDisplay.number),
                io.Float.Input("guidance_scale", default=4.5, min=0.0, max=20.0, step=0.1),
                io.Int.Input("seed", default=43, min=0, max=0xffffffffffffffff),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, longcat_pipeline, image, prompt, negative_prompt, steps, guidance_scale, seed) -> io.NodeOutput:
        if not longcat_pipeline:
            raise ValueError("Pipeline required")
        if not longcat_pipeline.get("is_edit"):
            raise ValueError("Use LongCat Text to Image node for T2I models")
        
        from PIL import Image
        import numpy as np
        pil_image = Image.fromarray((image[0].cpu().numpy() * 255).astype(np.uint8)).convert("RGB")
        result = longcat_pipeline["pipe"](
            pil_image, prompt, negative_prompt=negative_prompt, guidance_scale=guidance_scale,
            num_inference_steps=steps, num_images_per_prompt=1, generator=torch.Generator("cpu").manual_seed(seed)
        )
        return io.NodeOutput(torch.from_numpy(np.array(result.images[0]).astype(np.float32) / 255.0)[None,])


class MeituanLongCatExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MeituanLongCatLoader, MeituanLongCatT2I, MeituanLongCatEdit]

async def comfy_entrypoint() -> MeituanLongCatExtension:
    return MeituanLongCatExtension()

NODE_CLASS_MAPPINGS = {
    "MeituanLongCatLoader": MeituanLongCatLoader,
    "MeituanLongCatT2I": MeituanLongCatT2I,
    "MeituanLongCatEdit": MeituanLongCatEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MeituanLongCatLoader": "LongCat Model Loader",
    "MeituanLongCatT2I": "LongCat Text to Image",
    "MeituanLongCatEdit": "LongCat Image Edit",
}
