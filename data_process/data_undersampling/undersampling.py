import random

import numpy as np
import pandas as pd
from data_process.ProcessedData import ProcessedData

class UndersamplingData(ProcessedData):

    def __init__(self, raw_data):
        super().__init__(raw_data)
        self.rest_columns = raw_data.rest_columns

    def process(self):
        equal_zero_index = (self.label_df != 1).values
        equal_one_index = ~equal_zero_index

        pass_feature = np.array(self.feature_df[equal_zero_index])
        fail_feature = np.array(self.feature_df[equal_one_index])

        select_num = len(fail_feature)
        if select_num >= len(pass_feature):
            return

        pass_i = []
        while len(pass_i) <= select_num:
            random_i = random.randint(0, len(pass_feature) - 1)
            if random_i not in pass_i:
                pass_i.append(random_i)

        temp_array = np.zeros([select_num, len(self.feature_df.values[0])])
        for i in range(select_num):
            temp_array[i] = pass_feature[pass_i[i]]

        compose_feature = np.vstack((fail_feature, temp_array))

        label_np = np.ones(select_num).reshape((-1, 1))
        gen_label = np.zeros(select_num).reshape((-1, 1))
        compose_label = np.vstack((label_np, gen_label))

        self.label_df = pd.DataFrame(compose_label, columns=['error'], dtype=float)
        self.feature_df = pd.DataFrame(compose_feature, columns=self.feature_df.columns, dtype=float)

        self.data_df = pd.concat([self.feature_df, self.label_df], axis=1)
