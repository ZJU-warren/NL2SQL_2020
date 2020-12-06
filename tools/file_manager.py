import shutil
import os


def generate_new_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    print('mkdir', path)
