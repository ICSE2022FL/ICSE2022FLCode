import os

import numpy as np
import pandas as pd

from utils.file_util import *

def process_corr_data(corr_data):
    token = choose_newlines(corr_data)
    corr_data = [x.strip().split() for x in corr_data.strip().split(token)]
    for elem in corr_data:
        elem[0] = int(elem[0])
        elem[1] = float(elem[1])
    return corr_data

def get_corr(path, method_list, state):
    all_df_dict = dict()

    for method in method_list:
        file_name = method + "-" + state + ".txt"
        corr = process_coding(os.path.join(path, state, file_name))
        corr = process_corr_data(corr)
        all_df_dict[method] = pd.DataFrame(corr, columns=["line_num", method])

    return all_df_dict


def find_closest_num(real_line_data, target):
    target = int(target)
    line_data_np = np.array(real_line_data, dtype=int)
    min_diff_val = min(abs(line_data_np - target))

    if int(target + min_diff_val) in real_line_data and int(target - min_diff_val) in real_line_data:
        return list([int(target + min_diff_val), int(target - min_diff_val)])
    if int(target + min_diff_val) in real_line_data:
        return list([int(target + min_diff_val)])
    if int(target - min_diff_val) in real_line_data:
        return list([int(target - min_diff_val)])
