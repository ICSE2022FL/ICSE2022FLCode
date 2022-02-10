import os
import re
import pandas as pd

from utils.file_util import *
from read_data.DataLoader import DataLoader


class Defects4JDataLoader(DataLoader):

    def __init__(self, base_dir, program, bug_id):
        super().__init__(base_dir, program, bug_id)

    def load(self):
        self.file_dir = os.path.join(self.base_dir,
                                     "d4j",
                                     "data",
                                     self.program, str(self.bug_id),
                                     "gzoltars",
                                     self.program, str(self.bug_id))
        self._load_columns()
        self._load_features()
        self._load_fault_line()

    def _load_features(self):
        self.matrix_path = os.path.join(self.file_dir, 'matrix')
        feature_data = process_coding(self.matrix_path)
        feature_data, label_data = self._process_feature_data(feature_data)
        self.feature_df = pd.DataFrame(feature_data, columns=self.concrete_column[:])
        self.label_df = pd.DataFrame(label_data, columns=['error'])
        self.data_df = pd.concat([self.feature_df, self.label_df], axis=1)

    def _load_columns(self):
        columns_path = os.path.join(self.file_dir, 'spectra')
        concrete_columns = self._process_content(columns_path)
        self.concrete_column, self.columnmap = self._getnewcolumns(concrete_columns)

    def _load_fault_line(self):
        fault_dir = os.path.join(self.base_dir,"d4j", "buggy-lines", self.program + "-" + str(self.bug_id) + ".buggy.lines")

        fault_line_data = process_coding(fault_dir)
        fault_line_data = self._process_fault_line_data(fault_line_data)
        self.fault_line = [self._cal_column(i, self.columnmap) for i in fault_line_data]

    def _getnewcolumns(self, classnames):
        names = []
        for i in classnames:
            name = re.sub('#.*', '', str(i))
            if name not in names:
                names.append(str(name))
        columns = []
        for i in classnames:
            columns.append(int(self._cal_column(str(i), names)))
        return columns, names

    def _cal_column(self, s, data):
        s = str(s)
        classname = re.sub('#.*', '', s)
        num = int(re.sub('.*#', '', s))
        classnum = data.index(classname)
        column = (classnum + 1) * 100000 + num
        return int(column)

    def _process_fault_line_data(self, fault_line_data):
        temp_data = re.findall(".*#\d+", fault_line_data)
        temp_data = [i.replace(r'.java', '') for i in temp_data]
        temp_data = [i.replace('/', '.') for i in temp_data]
        temp_data = [i.strip() for i in temp_data]
        return list(map(str, temp_data))

    def _process_content(self, columns_path):
        columns = process_coding(columns_path)
        token = choose_newlines(columns)

        concrete_columns = columns.split(token)

        return concrete_columns

    def _process_feature_data(self, feature_data):
        token = choose_newlines(feature_data)
        feature_data = feature_data.split(token)

        feature_data = [feature_str.strip().split() for feature_str in feature_data]

        label_data = [arr[-1] for arr in feature_data]

        label_data = [0 if a == '+' else 1 for a in label_data]

        feature_data = [arr[:-1] for arr in feature_data]
        feature_data = [list(map(int, arr)) for arr in feature_data]

        return feature_data, label_data
