import os
import numpy as np
from metrics.calc_corr import calc_corr
from utils.write_util import write_corr_to_txt, write_rank_to_txt
from utils.read_util import get_corr, find_closest_num


class CalculateSuspiciousness():
    def __init__(self, data_obj, method, save_rank_path, experiment):
        self.data_obj = data_obj
        self.method = method
        self.sava_rank_path = save_rank_path
        self.suspicious_list = None
        self.state = experiment

    def run(self):
        self._calculate_susp_for_method_list()
        self._calculate_rank()
        self._save_rank()

    def _calculate_susp_for_method_list(self):
        for method in self.method:
            self._calculate_susp_for_method(method)

    def _calculate_susp_for_method(self, method):
        self.suspicious_list = calc_corr(self.data_obj.data_df, method)
        for col in self.data_obj.rest_columns:
            self.suspicious_list[col] = 0
        write_corr_to_txt(method, self.suspicious_list, self.data_obj.file_dir, self.state)

    def _calculate_rank(self):
        all_df_dict = get_corr(self.data_obj.file_dir, self.method, self.state)
        self.rank_MFR_dict = self.__calculate_rank(all_df_dict, self.data_obj.fault_line, self.method)

    def _save_rank(self):
        save_rank_filename = os.path.join(self.sava_rank_path, f"{self.state}_MFR.txt")
        write_rank_to_txt(self.rank_MFR_dict, save_rank_filename, self.data_obj.program, self.data_obj.bug_id)

    def __calculate_rank(self, all_df_dict, fault_line_data, method_list):
        real_fault_line_data = list()

        real_line_data = all_df_dict[method_list[0]]['line_num'].tolist()
        for line in fault_line_data:
            if line in real_line_data:
                real_fault_line_data.append(line)
            else:
                real_fault_line_data.extend(find_closest_num(real_line_data, line))
        real_fault_line_data = list(set(real_fault_line_data))

        result_dict = dict()
        for method in method_list:
            result_dict[method] = float('-inf')
        for method in method_list:
            concrete_df = all_df_dict[method]
            temp_df = concrete_df[concrete_df["line_num"].isin(real_fault_line_data)]
            rank = temp_df.index.values[0]
            val = temp_df[method].values[0]
            result_dict[method] = rank + 1
        return result_dict

    def __calc_MAR_rank(self, all_df_dict, fault_line_data, method_list):
        real_fault_line_data = list()

        real_line_data = all_df_dict[method_list[0]]['line_num'].tolist()
        for line in fault_line_data:
            if line in real_line_data:
                real_fault_line_data.append(line)
            else:
                real_fault_line_data.extend(find_closest_num(real_line_data, line))
        real_fault_line_data = list(set(real_fault_line_data))

        result_dict = dict()
        for method in method_list:
            result_dict[method] = float('-inf')
        for method in method_list:
            concrete_df = all_df_dict[method]
            temp_df = concrete_df[concrete_df["line_num"].isin(real_fault_line_data)]

            result_dict[method] = np.mean(temp_df.index.values + 1)
        return result_dict
