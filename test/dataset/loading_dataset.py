from src.dataset.data_processor import MedicalDatasetPreprocessor
from src.model.model_loader import ModelLoader
from src.config import Config
# Loading configuration
yml_config = Config("configs/config.yaml")
import torch
print(torch.cuda.is_available())
print("**************************")
model_loader = ModelLoader(config=yml_config)
tokenizer, model = model_loader.run_model_loader_step()

# Testing The Load of dataset
data_processor = MedicalDatasetPreprocessor(config=yml_config, tokenizer=tokenizer)
data_processor.run_dataset_step()
