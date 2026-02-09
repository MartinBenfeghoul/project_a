import numpy as np


class Logger():
    def __init__(self, layer_idx = None):
        self.layer_idx = layer_idx
        self.log_dict = {}

    def add_log(self, key, value):
        if key not in self.log_dict:
            self.log_dict[key] = []
        self.log_dict[key].append(value)

    def add_dict(self, input_dict):
        for key, value in input_dict.items():
            self.add_log(key, value)

    def get_log_list(self, key):
        return self.log_dict.get(key, [])

    def get_log_mean(self, key, std=False, return_dtype=np.float64):
        values = self.get_log_list(key)
        if values:
            if isinstance(values[0], (int, float)):
                res = np.mean(values, dtype=return_dtype)
                if std:
                    res = (res, np.std(values, dtype=return_dtype))
            elif isinstance(values[0], np.ndarray):
                res = np.mean(np.stack(values), axis=0, dtype=return_dtype)
                if std:
                    res = (res, np.std(np.stack(values), axis=0, dtype=return_dtype))
            else:
                raise ValueError(f"Unexpected value type: {type(values[0])}")
            return res
        return None

    def get_dict_mean(self, std=False):
        mean_dict = {}
        for key in self.log_dict.keys():
            mean_dict[key] = self.get_log_mean(key, std=std)
        return mean_dict

    def keys(self):
        return self.log_dict.keys()

    def values(self):
        return self.log_dict.values()

    @property
    def length(self):
        if self.log_dict:
            first_key = next(iter(self.log_dict))
            return len(self.log_dict[first_key])
        return 0

    def clear(self):
        self.log_dict = {}