import time
import os


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_time():
    t = time.localtime()
    return f'{t.tm_year}{t.tm_mon}{t.tm_mday}'


def fix_duplicate(path, name):
    n_dup = sum([name in pname for pname in os.listdir(path)])
    return name + f'_run={n_dup+1}'
    


def name_experiment(prefix, root, **kwargs):
    makedirs(root)
    res_str = prefix + '_' + get_time()
    for k, w in kwargs.items():
        res_str += f'_{k}={w}'
        
    name = fix_duplicate(root, res_str)
    
    path = os.path.join(root, name)
    makedirs(path)
    return path

