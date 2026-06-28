import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from datasets import load_dataset
from src.config import Config
from tqdm import tqdm
import numpy as np


class MedicalDatasetPreprocessor:
    def __init__(self, config, tokenizer):
        self.config = config
        self.dataset = None
        self.tokenizer = tokenizer
    

    def run_dataset_step(self):
        """
        Dataset Pipeline
        """
        # Load dataset
        self.load_dataset()

        # preprocess dataset
        self.preprocess_loaded_dataset()

        # max length
        # self.get_max_length_dataset()

        # formatting dataset
        self.dataset_formatter()


    def load_dataset(self):
        """
        Load the dataset for Hugging Face based on the provided configuration.
        """

        # Loading dataset from Hugging Face
        dataset = load_dataset(self.config['dataset']['name'], split="train")
        self.dataset = dataset
        print("*** Dataset Loaded Successfully")
        print(self.dataset)
        print("\n")


    def preprocess_loaded_dataset(self):
        """
        Preprocessing of the dataset
        """
        # remove unused columns
        self.remove_unused_columns()

        # remove empty or null values
        self.remove_null_empty_row()

        # remove duplicated row
        self.remove_duplicated_row()

        # get max length of dataset
        # self.get_max_length_dataset()

        # changing the format of dataset
        self.dataset_formatter()

        # filtering and keeping only tokenized text where length is under 1024
        self.remove_long_tokenized_row() 

        
        

    def remove_unused_columns(self):
        """
        Remove Unused Columns
        """
        # keeping only the two columns
        self.dataset = self.dataset.select_columns(["Patient", "Doctor"])

        print("*** Keeping only columns : Patient , Doctor ***")
        print(self.dataset)
        print("\n")


    def remove_null_empty_row(self):
        """
        Remove null or empty row from the dataset
        """

        # filtering null and empty values
        self.dataset = self.dataset.filter(
            lambda x: (
                x["Patient"] is not None
                and x["Patient"].strip() != ""
                and x["Doctor"] is not None
                and x["Doctor"].strip() != ""
            )
        )

        print("*** Filtering out null and empty values ***")
        print(self.dataset)
        print("\n")

    
    def remove_duplicated_row(self):
        """
        Removing Duplicated Rows from dataset
        """
        seen = set()

        def is_unique_row(row):
            key = (
                row["Patient"].strip(),
                row["Doctor"].strip()
            )

            if key in seen:
                return False
            
            seen.add(key)
            
            return True

        self.dataset = self.dataset.filter(is_unique_row)
        print("*** Deleting Duplicated Rows ***")
        print(self.dataset)
        print("\n")


    def get_max_length_dataset(self):
        
        lengths = []

        for example in tqdm(self.dataset):
            messages = [
                {"role": "user", "content": example["Patient"]},
                {"role": "assistant", "content": example["Doctor"]},
            ]

            text = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
            )

            tokens = self.tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=False,
            )

            current_length = tokens["input_ids"].shape[-1]
            lengths.append(current_length)

        lengths = np.array(lengths)

        print(f"Maximum length       : {lengths.max()}")
        print(f"Average length       : {lengths.mean():.2f}")
        print(f"Median length        : {np.median(lengths):.2f}")
        print(f"95th percentile      : {np.percentile(lengths, 95):.0f}")
        print(f"99th percentile      : {np.percentile(lengths, 99):.0f}")

        for seq_len in [512, 1024, 1536, 2048, 3072, 4096]:
            count = np.sum(lengths <= seq_len)
            percentage = count / len(lengths) * 100
            print(
                f"<= {seq_len:4d} tokens : "
                f"{count:6d}/{len(lengths)} ({percentage:.2f}%)"
            )


    def remove_long_tokenized_row(self):
        """
        keeping text rows that their lengths are under 1024
        """
        def is_under_max_length(examples):
            
            tokenized = self.tokenizer(
                examples["text"],
                add_special_tokens=False,
            )

            return [
                len(ids) <= self.config["dataset"]["max_length"]
                for ids in tokenized["input_ids"]
            ]     
           
        self.dataset = self.dataset.filter(
            is_under_max_length,
            batched=True,
            batch_size=5000,
            desc="Filtering long examples"
        )
        
        print(f"*** Keeping only rows whose tokenized length is under {str(self.config['dataset']['max_length'])} ***")
        print(self.dataset)

        

    def dataset_formatter(self):
        """
        Format the dataset into a structure suitable for model training.
        """
        def formatting_prompts_func(examples):
            texts = []

            for patient_msg, doctor_msg in zip(examples["Patient"], examples["Doctor"]):

                conversation = [
                    {"role": "user", "content": patient_msg},
                    {"role": "assistant", "content": doctor_msg},
                ]

                text = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False,
                )

                texts.append(text.removeprefix("<bos>"))

            return {"text": texts}
        
        self.dataset = self.dataset.map(
            formatting_prompts_func,
            batched=True,
            batch_size=5000,
            desc="Formatting dataset",
        )
       

    def splitter(self):
        """
        Split the dataset into training, validation, and test sets.
        """
        pass