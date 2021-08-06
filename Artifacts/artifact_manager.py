import os
from os.path import join, exists

DIR, _ = os.path.split(os.path.abspath(__file__))


def get_folder_path(run_name):
    path = join(DIR, run_name)
    if not exists(path):
        os.mkdir(path)
    return path


