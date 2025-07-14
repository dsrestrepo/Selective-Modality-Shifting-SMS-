import torch
try:
    from transformers import BitsAndBytesConfig
except:
    print("Could not import BitsAndBytesConfig from transformers. Quantization will not be available.")
    
from transformers import AutoProcessor
from transformers import AutoTokenizer
import os
import sys


def load_qwen2_vl(quantization=None, use_flash_attention=True, model_id="Qwen/Qwen2-VL-2B-Instruct", return_attention=False, return_logits=False):
    """
    Load the Qwen2VLForConditionalGeneration model with specified quantization and attention type.

    Args:
        quantization (str, optional): Quantization type. Options are "4b", "8b", "16b", or None (no quantization).
        use_flash_attention (bool, optional): Whether to use flash attention. Defaults to True.

    Returns:
        tuple: A tuple containing the model and processor.
    """
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
    # Configure quantization
    bnb_config = None
    if quantization == "4b":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8b":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type='nf4',
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "16b":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # Load the model with specified options
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config if bnb_config else None,
        torch_dtype=torch.bfloat16 if quantization in ["4b", "8b"] else torch_dtype,
        attn_implementation="flash_attention_2" if use_flash_attention else "eager",
        output_attentions=return_attention,
        output_scores=return_logits,
        return_dict_in_generate=True
    )
    
    if quantization not in ["4b", "8b"]:
        model = model.to(torch.device("cuda"))
    
    model.eval()
    torch.cuda.empty_cache()

    # Load the processor
    processor = Qwen2VLProcessor.from_pretrained(model_id)
    
    return model, processor


def load_llava(model_id="llava-hf/llava-1.5-7b-hf", quantization=None, use_flash_attention=True, return_attention=False, return_logits=False):
    """
    Load the LlavaForConditionalGeneration model with specified version, backbone, quantization, and attention type.

    Args:
        model_id (str, optional): Model version identifier. Defaults to f"llava-hf/llava-1.5-{backbone}-hf".
        quantization (str, optional): Quantization type. Options are "4b", "8b", "16b", or None (no quantization).
        use_flash_attention (bool, optional): Whether to use flash attention (only available for specific models). Defaults to True.

    Returns:
        tuple: A tuple containing the model and processor.
    """
    from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaNextProcessor, LlavaNextForConditionalGeneration
    
    # Configure quantization
    bnb_config = None
    if quantization == "4b":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8b":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type='nf4',
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "16b":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    if "1.5" in model_id:
        # Load the model with specified options
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=bnb_config if bnb_config else None,
            torch_dtype=torch.bfloat16 if quantization in ["4b", "8b"] else torch_dtype,
            attn_implementation="flash_attention_2" if use_flash_attention else "eager",
            output_attentions=return_attention,
            output_scores=return_logits,
            return_dict_in_generate=True
        )
    else:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=bnb_config if bnb_config else None,
            torch_dtype=torch.bfloat16 if quantization in ["4b", "8b"] else torch_dtype,
            attn_implementation="flash_attention_2" if use_flash_attention else "eager",
            output_attentions=return_attention,
            output_scores=return_logits,
            return_dict_in_generate=True
        )

    if quantization not in ["4b", "8b"]:
        model = model.to(torch.device("cuda"))
        
    model.eval()
    torch.cuda.empty_cache()

    # Load the processor
    if "1.5" in model_id:        
        processor = AutoProcessor.from_pretrained(model_id)
    else:
        processor = LlavaNextProcessor.from_pretrained(model_id)

    return model, processor



def load_pali_gemma(model_id="google/paligemma2-10b-pt-224", quantization=None, return_attention=False, return_logits=False):
    """
    Load the PaliGemma model with specified version and quantization options.

    Args:
        model_id (str, optional): Model version identifier. Defaults to "google/paligemma2-10b-pt-224".
        quantization (str, optional): Quantization type. Options are "4b", "8b", "16b", or None (no quantization).

    Returns:
        tuple: A tuple containing the loaded model and processor.
    """
    from dotenv import load_dotenv
    import huggingface_hub

    # Load the .env file
    load_dotenv()

    # Access environment variables
    hf_key = os.getenv('hf_key')
    huggingface_hub.login(token=hf_key)


    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
    
    # Configure quantization
    bnb_config = None
    if quantization == "4b":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8b":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf4",
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "16b":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    # Load the model with specified options
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config if bnb_config else None,
        torch_dtype=torch.bfloat16 if quantization in ["4b", "8b"] else torch_dtype,
        output_attentions=return_attention,
        output_scores=return_logits,
        return_dict_in_generate=True
    )
    
    if quantization not in ["4b", "8b"]:
        model = model.to(torch.device("cuda"))
        
        
    # Load the processor
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    #processor = AutoProcessor.from_pretrained(model_id)

    return model, processor

import sys
import os

# For interactive environments like Jupyter Notebook
current_dir = os.getcwd()  # Use the current working directory
janus_folder = os.path.join(current_dir, "Janus")
if janus_folder not in sys.path:
    sys.path.insert(0, janus_folder)

def load_janus_pro(
    model_id: str = "deepseek-ai/Janus-Pro-7B",
    return_attention: bool = False,
    return_logits: bool = False,
    quantization: str = None,
):
    """
    Load the Janus Pro multi-modality model and its associated VLChatProcessor.

    Args:
        model_id (str): Identifier or path of the Janus Pro model.
            For example: "deepseek-ai/Janus-Pro-1B" or "deepseek-ai/Janus-Pro-7B".
        return_attention (bool): If True, the processor/model will be configured to return attention maps.
        return_logits (bool): If True, the model will be configured to return logits/scores.

    Returns:
        tuple: A tuple containing:
            - model: The loaded MultiModalityCausalLM model.
            - processor: The loaded VLChatProcessor.
            - tokenizer: The tokenizer used by the VLChatProcessor.
    """
    import torch
    from transformers import AutoModelForCausalLM
    # Import the Janus objects from your package
    from janus.models import MultiModalityCausalLM, VLChatProcessor

    # Load the processor (this includes loading the tokenizer)
    processor = VLChatProcessor.from_pretrained(
        model_id
    )
    
    tokenizer = processor.tokenizer


    # Configure quantization
    bnb_config = None
    if quantization == "4b":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8b":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf4",
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "16b":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32


    # Load the model using Hugging Face's AutoModelForCausalLM.
    # Here, trust_remote_code=True is used because Janus may contain custom model code.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
 
        device_map="cuda",
        quantization_config=bnb_config if bnb_config else None,
        torch_dtype=torch.bfloat16 if quantization in ["4b", "8b"] else torch_dtype,
        output_attentions=return_attention,
        output_scores=return_logits,
        return_dict_in_generate=True
    )

    # Move model to GPU using bfloat16 and set it to evaluation mode.
    model = model.to(torch.bfloat16).cuda().eval()

    return model, processor

# Usage examples:
### Qwen2VL
#model, processor = load_qwen2_vl(quantization="4b", use_flash_attention=True, model_id="Qwen/Qwen2-VL-2B-Instruct")

### Llava
#model, processor = load_llava(model_id="llava-hf/llava-1.5-7b-hf", quantization="16b", use_flash_attention=True)
#model, processor = load_llava(model_id="llava-hf/llava-v1.6-mistral-7b-hf", quantization="16b", use_flash_attention=True)

### PaliGemma
#model, processor = load_pali_gemma(model_id="google/paligemma2-3b-pt-224", quantization="16b")

### Janus
#model, processor = load_janus_pro(model_id="deepseek-ai/Janus-Pro-7B", quantization="16b")



def load_biomedgpt(model_id="PanaceaAI/BiomedGPT-Base-Pretrained",
                   quantization=None,
                   return_attention=False,
                   return_logits=False,
                   device="cuda"):
    """
    Loads the BiomedGPT model (based on OFA) with optional quantization.

    Returns:
        model, processor
    """
    from transformers import OFATokenizer, OFAModel  # For BiomedGPT
    from torchvision import transforms
    from PIL import Image

    # BitsAndBytes config
    torch_dtype = torch.float16



    # Load BiomedGPT Model
    model = OFAModel.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        )

    # Move to GPU if not in 4-bit or 8-bit
    if quantization not in ["4b", "8b"]:
        model = model.to(torch.device(device))
    else:
        model = model.to(torch.device(device))

    model.eval()
    torch.cuda.empty_cache()

    # Create a simple "processor" object that wraps
    # (1) the tokenizer and (2) any image transforms needed
    class BiomedGPTProcessor:
        def __init__(self, model_id, device="cuda"):
            self.tokenizer = OFATokenizer.from_pretrained(model_id,
                            torch_dtype=torch.float16,
                            output_attentions=return_attention,
                            output_scores=return_logits,
                            return_dict_in_generate=True
             )
            mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
            resolution = 480
            self.patch_transform = transforms.Compose([
                lambda image: image.convert("RGB"),
                transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            self.device = torch.device(device) # if torch.cuda.is_available() else torch.device("cpu")

        def __call__(self, text=None, image=None):
            """
            Return a dict with keys:
               input_ids: Torch tensor for text
               patch_images: Torch tensor for image
            """
            input_ids = None
            patch_images = None

            if text is not None:
                input_ids = self.tokenizer([text], return_tensors="pt").input_ids.to(self.device)
            if image is not None:
                patch_images = self.patch_transform(image).unsqueeze(0).to(self.device)
                # convert to torch_dtype if not in 4-bit or 8-bit
                if quantization not in ["4b", "8b"]:
                    patch_images = patch_images.to(torch_dtype)

            return {
                "input_ids": input_ids,
                "patch_images": patch_images
            }

    processor = BiomedGPTProcessor(model_id)

    return model, processor



def load_llava_med(
    model_id="microsoft/llava-med-v1.5-mistral-7b",
    quantization=None,
    return_attention=False,
    return_logits=False,
    conv_mode="llava_v0"
):
    """
    Loads the specialized LLaVA-Med model, using the official LLaVA builder code
    (rather than Transformers).
    Returns:
        (model, processor)
    """
    from llava.mm_utils import (
        KeywordsStoppingCriteria,
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    #from llava.conversation import SeparatorStyle, conv_templates
    from utils import SeparatorStyle, conv_templates
    from llava.model.builder import load_pretrained_model

    # If you do 4-bit or 8-bit quantization
    torch_dtype = torch.float32

    # Use the LLaVA builder
    model_name = get_model_name_from_path(model_id)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_id,
        model_base=None,
        model_name=model_name, 
        load_8bit=(quantization == "8b"),
        load_4bit=(quantization == "4b"),
        device_map="auto",
        device="cuda"
    )


    #
    # Create a minimal "processor" object that mirrors your typical pattern:
    #
    class LLaVAMedProcessor:
        def __init__(self, tokenizer, image_processor, conv_mode="llava_v2", device="cuda"):
            self.tokenizer = tokenizer
            self.image_processor = image_processor
            self.device = device
            self.conv_mode = conv_mode

        def apply_chat_template(self, conversation, assistant_message=None, system_message=None):
            # The standard LLaVA approach is to do "conv.get_prompt()",
            # but you might want to replicate exactly what your code does.
            # This is the function that your "prompts.generate_text()" calls.
            # roles = conversation[0].get("role", "user")
            # If you want the same approach as your "standard" llava, replicate
            # that logic here or see how you do it in your existing code.
            #
            # A simple approach is: build a single big string with <image> token
            # Then rely on the conversation's "conv_templates" from LLaVA.

            # For example, if you want to replicate the code in llava_med.py:
            #   conv = conv_templates["llava_v0"].copy()
            #   conv.append_message(conv.roles[0], conversation[0]["content"])
            #   conv.append_message(conv.roles[1], None)
            #   prompt = conv.get_prompt()
            #   return prompt
            # 
            # We'll do a minimal version below.
            
            # conversation[0]["content"] = [ {"type": "image"}, {"type": "text", "text": "..."} ]
            user_content_list = conversation[0]["content"]
            
            # Build a single string out of the list
            content_pieces = []
            for chunk in user_content_list:
                chunk_type = chunk.get("type", "text")
                if chunk_type == "image":
                    # Insert the LLaVA default image token
                    content_pieces.append(DEFAULT_IMAGE_TOKEN)
                elif chunk_type == "text":
                    text_str = chunk.get("text", "")
                    content_pieces.append(text_str)
                else:
                    # If you have any other chunk types, handle them here
                    content_pieces.append("")

            # Combine everything into one string, with a newline or space
            user_text = "\n".join(content_pieces)

            
            conv = conv_templates[self.conv_mode].copy()
            
            if system_message:
                conv.system = system_message
                
            #conv.system += " Always reply something."

            conv.append_message(conv.roles[0], user_text)
            conv.append_message(conv.roles[1], assistant_message)
            prompt = conv.get_prompt()

            return prompt, conv
        
        def process_image(self, image):
            
            args = {"image_aspect_ratio": "pad"}
            
            if image is list:
                image_tensor = process_images(image, self.image_processor, args)
            else:
                image_tensor = process_images([image], self.image_processor, args)
            
            return image_tensor.to(self.device, dtype=torch.float16)



        def prepare_inputs(self, text_prompt, image, conv):
            # This is how your code transforms the prompt + image into tokens
            # and returns the device-friendly dictionary of tensors to feed
            # into model.generate.
            # 
            # Example, from llava_med.py style:
            input_ids = (
                tokenizer_image_token(
                text_prompt,
                self.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt"
            ).unsqueeze(0).to(self.device))
    
            # image_processor is a transform function that converts PIL images to tensors
            if image is None:
                image_tensor = None
            else:
                image_tensor = self.process_image(image)
            
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            
            stopping_criteria = KeywordsStoppingCriteria(
                keywords=[stop_str], tokenizer=self.tokenizer, input_ids=input_ids
            )

            return {
                "input_ids": input_ids,
                "images": image_tensor,
                "stopping_criteria": stopping_criteria
            }

    processor = LLaVAMedProcessor(tokenizer, image_processor, device=model.device, conv_mode=conv_mode)

    return model, processor




def _get_bnb_config(quantization: str | None, gemma: bool = False):
    """Utility – return a BitsAndBytesConfig or None."""
    
    if BitsAndBytesConfig is None or quantization is None:
        return None, torch.float32

    if quantization == "4b":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ), torch.bfloat16
    if quantization == "8b":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf4",
            bnb_8bit_compute_dtype=torch.bfloat16,
        ), torch.bfloat16
    if quantization == "16b":
        if gemma:
            return None, torch.bfloat16
        else:
            return None, torch.float16

    return None, torch.float32



# MedGemma 4B‑IT 
def load_medgemma(
    model_id: str = "google/medgemma-4b-it",
    quantization: str | None = None,
    return_attention: bool = False,
    return_logits: bool = False,
):
    """Load **MedGemma** – Google’s medical‑tuned Gemma vision‑language model (4B‑IT)."""

    from transformers import AutoProcessor, AutoModelForImageTextToText

    bnb_config, torch_dtype = _get_bnb_config(quantization, gemma=True)

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch_dtype if bnb_config is None else torch.bfloat16,
        #attn_implementation="eager",
        output_attentions=return_attention,
        output_scores=return_logits,
        return_dict_in_generate=True,
    )
    
    if quantization not in ["4b", "8b"]:
        model = model.to(torch.device("cuda"))
        
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    print(f'Model device: {model.device}')

    model.eval()
    torch.cuda.empty_cache()
    return model, processor


# CheXagent‑8B (Stanford AIMI)
def load_chexagent(
    model_id: str = "StanfordAIMI/CheXagent-8b",
    quantization: str | None = None,
    return_attention: bool = False,
    return_logits: bool = False,
):
    """Load **CheXagent‑8B**, a chest‑X‑ray specialist VLM.

    Returns
    -------
    model, processor, generation_config
        ``generation_config`` is required for faithful reproduction of the
        authors’ decoding settings, so we include it in the return tuple.
    """

    from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
    import torch, psutil, humanize

    bnb_config = None
    if quantization == "4b":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        #print(f'''Using 4-bit quantization for {model_id}''')
    elif quantization == "8b":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf4",
            bnb_8bit_compute_dtype=torch.float16,
        )
        #print(f'''Using 8-bit quantization for {model_id}''')
        
    torch_dtype = torch.float16


    device = torch.device("cuda")

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    generation_config = GenerationConfig.from_pretrained(model_id)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch_dtype if bnb_config is None else None,
        output_attentions=return_attention,
        output_scores=return_logits,
        return_dict_in_generate=True,
    )
    
    # Only move non-quantised checkpoints manually
    if bnb_config is None:
        model = model.to(torch.device("cuda"), dtype=torch_dtype)

    model.eval()
    
    #print("CUDA allocated:",
    #    humanize.naturalsize(torch.cuda.memory_allocated(), binary=True))
    #print("CUDA reserved:",
    #    humanize.naturalsize(torch.cuda.memory_reserved(), binary=True))

    
    torch.cuda.empty_cache()
    return model, processor, generation_config


# MAIRA‑2
def load_maira2(
    model_id: str = "microsoft/maira-2",
    token: str | None = None,
    quantization: str | None = None,
    return_attention: bool = False,
    return_logits: bool = False,
):
    """Load **MAIRA‑2** – Multimodal AI Radiology Assistant v2.

    Parameters
    ----------
    model_id : str
        Hugging Face model repo. Default: ``"microsoft/maira-2"``.
    token : str | None
        HF access token *if* the repo requires gated access. If *None*, we fall back
        to the ``HF_TOKEN`` environment variable.
    quantization : str | None
        "4b", "8b", "16b", or *None* (fp32).
    return_attention / return_logits : bool
        Forward these diagnostics flags to ``model.generate``.

    Returns
    -------
    model, processor
        Ready‑to‑use ``(AutoModelForCausalLM, AutoProcessor)`` pair on the best CUDA
        device available.
    """

    from transformers import AutoModelForCausalLM, AutoProcessor

    # Bits and Bytes
    bnb_config, torch_dtype = _get_bnb_config(quantization)

    access_token = token or os.getenv("HF_TOKEN")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=access_token,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        output_attentions=return_attention,
        output_scores=return_logits,
        return_dict_in_generate=True,
    )

    # Processor requires 1 additional CLS image token depending on the HF version
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=access_token,
        num_additional_image_tokens=1, ##### # MAIRA-2 requires 1 additional image token
    )

    if quantization not in ("4b", "8b"):
        model = model.to(torch.device("cuda"))

    model.eval()
    torch.cuda.empty_cache()
    return model, processor



def load_llama3_2(
    model_id: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
    quantization: str | None = None,
    token: str | None = None,
    return_attention: bool = False,
    return_logits: bool = False,
):
    """
    Load LLaMA 3.2 Vision-Instruct with optional quantization and flash attention.

    Args:
        model_id (str): HF repo ID.
        quantization (str|None): "4b", "8b", "16b" or None.
        use_flash_attention (bool): use flash_attention_2 vs eager.
        return_attention (bool): include attention maps in output.
        return_logits (bool): include token scores in output.

    Returns:
        model, processor
    """
    from transformers import MllamaForConditionalGeneration, AutoProcessor
    
    # — configure BitsAndBytes if requested —
    bnb_config = None
    torch_dtype = torch.float32
    if BitsAndBytesConfig and quantization == "4b":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        torch_dtype = torch.bfloat16
    elif BitsAndBytesConfig and quantization == "8b":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf4",
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
        torch_dtype = torch.bfloat16
    elif quantization == "16b":
        torch_dtype = torch.float16
    
    # — load the model —
    access_token = token or os.getenv("HF_TOKEN")
    
    if access_token:
        os.environ["HF_TOKEN"] = access_token
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        token=access_token,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        output_attentions=return_attention,
        output_scores=return_logits,
        return_dict_in_generate=True,
    )
    if quantization not in ["4b", "8b"]:
        model = model.to(torch.device("cuda"))

    model.eval()
    torch.cuda.empty_cache()

    # — load the processor —
    processor = AutoProcessor.from_pretrained(model_id)

    return model, processor
