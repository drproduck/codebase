import time
import os

def get_time():
    t = time.localtime()
    return f'{t.tm_year}{t.tm_mon}{t.tm_mday}'

def name_experiment(prefix, **kwargs):
    res_str = prefix + '_' + get_time()
    for k, w in **kwargs:
        res_str += f'{k}={w}'
        
    return res_str

def fix_duplicate(path, name):
    if name in os.listdir(path):
        # how many duplicates?
        n_dup = sum([pname == name] for pname in path)
        return name + f'r{n_dup+1}'
    
    return name
