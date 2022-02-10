import numpy as np
import pandas as pd
from data_process.ProcessedData import ProcessedData

class ResamplingData(ProcessedData):

    def __init__(self, raw_data):
        super().__init__(raw_data)
        self.rest_columns = raw_data.rest_columns

    def process(self):
        equal_zero_index = (self.label_df != 1).values
        equal_one_index = ~equal_zero_index

        pass_feature = np.array(self.feature_df[equal_zero_index])
        fail_feature = np.array(self.feature_df[equal_one_index])

        diff_num = len(pass_feature) - len(fail_feature)

        if diff_num < 1 or len(fail_feature) <= 0:
            return

        temp_array = np.zeros([diff_num, len(self.feature_df.values[0])])
        for i in range(diff_num):
            temp_array[i] = fail_feature[i % len(fail_feature)]

        features_np = np.array(self.feature_df)
        compose_feature = np.vstack((features_np, temp_array))

        label_np = np.array(self.label_df)
        gen_label = np.ones(diff_num).reshape((-1, 1))
        compose_label = np.vstack((label_np, gen_label))

        self.label_df = pd.DataFrame(compose_label, columns=['error'], dtype=float)
        self.feature_df = pd.DataFrame(compose_feature, columns=self.feature_df.columns, dtype=float)

        self.data_df = pd.concat([self.feature_df, self.label_df], axis=1)
