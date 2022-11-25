import os
import sys
import json
import csv
import glob
import pprint
import numpy as np
import random
import argparse
import pandas as pd
from tqdm import tqdm
from .utils import DataProcessor
from .utils import SemEvalSingleSentenceExample
from transformers import (
    AutoTokenizer,
)


class SemEvalDataProcessor(DataProcessor):
    """Processor for Sem-Eval 2020 Task 4 Dataset.
    Args:
        data_dir: string. Root directory for the dataset.
        args: argparse class, may be optional.
    """

    def __init__(self, data_dir=None, args=None, **kwargs):
        """Initialization."""
        self.args = args
        self.data_dir = data_dir

    def get_labels(self):
        """See base class."""
        return 2  # Binary.

    def _read_data(self, data_dir=None, split="train"):
        """Reads in data files to create the dataset."""
        if data_dir is None:
            data_dir = self.data_dir

        ##################################################
        # NOTE: this TODO is optional
        # TODO: Use csv.DictReader or pd.read_csv to load
        # the csv file and process the data properly.
        # We recommend separately storing the correct and
        # the incorrect statements into two individual
        # `examples` using the provided class
        # `SemEvalSingleSentenceExample` in `utils.py`.
        # e.g. example_1 = ...
        #      example_2 = ...
        #      examples.append(example_1)
        #      examples.append(example_2)
        # For the guid, simply use the row number (0-
        # indexed) for each data instance.
        df = pd.read_csv(data_dir + "/" + split + ".csv")
        # print(len(df))
        # return
        examples = []
        for i in range(len(df)):
            data = df.iloc[i]
            # data = dataset[i]
            guid = i
            correct_statement = data['Correct Statement']
            # print("###",correct_statement)
            incorrect_statement = data['Incorrect Statement']
            right_reason1 = data['Right Reason1']
            right_reason2 = data['Right Reason2']
            right_reason3 = data['Right Reason3']
            confusing_reason1 = data['Confusing Reason1']
            confusing_reason2 = data['Confusing Reason2']

            example_1 = SemEvalSingleSentenceExample(
                guid = guid,
                text = correct_statement,
                label = 1,
                right_reason1 = right_reason1,
                right_reason2 = right_reason2,
                right_reason3 = right_reason3,
                confusing_reason1 = confusing_reason1,
                confusing_reason2 = confusing_reason2
            )
            examples.append(example_1)
            # print(example_1.to_json_string())

            example_2 = SemEvalSingleSentenceExample(
                guid = guid,
                text = incorrect_statement,
                label = 0,
                right_reason1 = right_reason1,
                right_reason2 = right_reason2,
                right_reason3 = right_reason3,
                confusing_reason1 = confusing_reason1,
                confusing_reason2 = confusing_reason2
            )
            examples.append(example_2)


        # End of TODO.
        ##################################################

        return examples

    def get_train_examples(self, data_dir=None):
        """See base class."""
        return self._read_data(data_dir=data_dir, split="train")

    def get_dev_examples(self, data_dir=None):
        """See base class."""
        return self._read_data(data_dir=data_dir, split="dev")

    def get_test_examples(self, data_dir=None):
        """See base class."""
        return self._read_data(data_dir=data_dir, split="test")


if __name__ == "__main__":

    # Test loading data.
    proc = SemEvalDataProcessor(data_dir="datasets/semeval_2020_task4")
    train_examples = proc.get_train_examples()
    val_examples = proc.get_dev_examples()
    test_examples = proc.get_test_examples()
    print()
    for i in range(8):
        print(train_examples[i])
    print()
