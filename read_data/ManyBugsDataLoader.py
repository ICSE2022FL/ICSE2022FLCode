import os
import re

import pandas as pd

from utils.file_util import *
from read_data.DataLoader import DataLoader


class ManyBugsDataLoader(DataLoader):

    def __init__(self, base_dir, program, bug_id):
        super().__init__(base_dir, program, bug_id)

    def load(self):
        self.file_dir = os.path.join(self.base_dir,
                                     "manybugs",
                                     self.program,
                                     )
        self._load_columns()
        self._load_features()
        self.data_df = pd.concat([self.feature_df, self.label_df], axis=1)
        self._load_fault_line()

    def _load_features(self):

        feature_path = os.path.join(self.file_dir, 'covMatrix.txt')
        feature_data = process_coding(feature_path)
        feature_data = self._process_feature_data(feature_data)
        self.feature_df = pd.DataFrame(feature_data, columns=self.concrete_columns[:])

        self._load_labels()

    def _load_labels(self):
        label_path = os.path.join(self.file_dir, 'error.txt')
        label_data = process_coding(label_path)
        label_data = self._process_label_data(label_data)
        self.label_df = pd.DataFrame(label_data, columns=['error'])

    def _load_columns(self):
        columns_path = os.path.join(self.file_dir, 'componentinfo.txt')
        self.concrete_columns = self._process_content(columns_path)

    def _load_fault_line(self):
        fault_line_data = process_coding(os.path.join(self.file_dir, "faultLine.txt"))
        self.fault_line = self._process_fault_line_data(fault_line_data)

    def _process_fault_line_data(self, fault_line_data):
        temp_data = re.findall("\"(.*?)\"", fault_line_data)[0]
        temp_data = temp_data.strip().split()
        return list(map(int, temp_data))

    def _process_label_data(self, label_data):
        token = choose_newlines(label_data)
        label_data = label_data.split(token)

        label_data = [list(map(int, arr)) for arr in label_data]
        return label_data

    def _process_content(self, columns_path):
        columns = process_coding(columns_path)
        token = choose_newlines(columns)
        if token in columns:
            temp_content = columns.split(token)
            total_line = int(temp_content[0])
            concrete_columns = temp_content[1].split()[:total_line]
        else:
            temp_content = columns.split()
            total_line = int(temp_content[0])
            concrete_columns = columns.split()[1:total_line + 1]
        return concrete_columns

    def _process_feature_data(self, feature_data):
        token = choose_newlines(feature_data)
        feature_data = feature_data.split(token)

        feature_data = [feature_str.strip().split() for feature_str in feature_data]
        feature_data = [list(map(int, arr)) for arr in feature_data]
        feature_data = [[0 if a == 0 else 1 for a in elem] for elem in feature_data]

        return feature_data

