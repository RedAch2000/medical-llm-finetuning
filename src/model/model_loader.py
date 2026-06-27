from unsloth import FastModel
import torch
from unsloth.chat_templates import get_chat_template


class ModelLoader:
  
    def __init__(self, config):
        self.model = None
        self.tokenizer = None
        self.config = config


    def run_model_loader_step(self):
        # Loading model 
        print("*** Loading Model ...\n")
        self.loading_model()
        self.add_lora_adapters()
        self.chat_template_tokenizer()
        print("*** Model Loaded Successfully ***")
        return self.tokenizer, self.model


    def loading_model(self):
        """
        Loading the model and the tokenizer of Gemma 3 (4B parameters)
        """
        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name = self.config["model"]["name"],
            max_seq_length = 2048, # Choose any for long context!
            load_in_4bit = True,  # 4 bit quantization to reduce memory
            load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
            full_finetuning = False, # [NEW!] We have full finetuning now!
            # token = "YOUR_HF_TOKEN", # HF Token for gated models
        )

    
    def add_lora_adapters(self):
        """
        We now add LoRA adapters so we only need to update a small amount of parameters!
        """
        self.model = FastModel.get_peft_model(
            self.model,
            finetune_vision_layers     = False, # Turn off for just text!
            finetune_language_layers   = True,  # Should leave on!
            finetune_attention_modules = True,  # Attention good for GRPO
            finetune_mlp_modules       = True,  # Should leave on always!

            r = 8,           # Larger = higher accuracy, but might overfit
            lora_alpha = 8,  # Recommended alpha == r at least
            lora_dropout = 0,
            bias = "none",
            random_state = 3407,
        )


    def chat_template_tokenizer(self):
        """
        Chat Template tokenizer
        """
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template = "gemma-3",
        )


 
