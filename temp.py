import DataLinkSet as DLSet
from tools.file_manager import generate_new_folder


def clean_models():
    generate_new_folder(DLSet.model_folder_link)


def mkdir_result():
    generate_new_folder(DLSet.base_folder_link + '/Result')
    generate_new_folder(DLSet.base_folder_link + '/Result' + '/Select')
    generate_new_folder(DLSet.base_folder_link + '/Result' + '/From')
    generate_new_folder(DLSet.base_folder_link + '/Result' + '/Where')
    generate_new_folder(DLSet.base_folder_link + '/Result' + '/GroupBy')
    generate_new_folder(DLSet.base_folder_link + '/Result' + '/Having')
    generate_new_folder(DLSet.base_folder_link + '/Result' + '/Limit')
    generate_new_folder(DLSet.base_folder_link + '/Result' + '/GroupBy')
    generate_new_folder(DLSet.base_folder_link + '/Result' + '/Combination')


if __name__ == '__main__':
    clean_models()
    mkdir_result()
