import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from datasets import load_dataset
from src.config import Config


class MedicalDatasetPreprocessor:
    def __init__(self):
        self.config = Config("configs/dataset.yaml")
        self.dataset = None

    def run_dataset_pipeline(self):
        """
        Dataset Pipeline
        """
        # Load dataset
        self.load_dataset()

        # preprocess dataset
        self.preprocess_loaded_dataset()


    def load_dataset(self):
        """
        Load the dataset for Hugging Face based on the provided configuration.
        """

        # Loading dataset from Hugging Face
        dataset = load_dataset(self.config['dataset_name'], split="train")
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


    def formatter(self):
        """
        Format the dataset into a structure suitable for model training.
        """
        pass

    def splitter(self):
        """
        Split the dataset into training, validation, and test sets.
        """
        pass