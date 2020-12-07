import DataLinkSet as DLSet
from tools.file_manager import generate_new_folder
import json
import random


class DataGenerator:
    @staticmethod
    def __store(data: list, content: list, schema: list, store_target: str):
        # clear the folder and generate new one
        generate_new_folder(DLSet.base_folder_link + '/' + store_target)
        generate_new_folder(DLSet.raw_folder_link % store_target)

        # store
        with open(DLSet.raw_folder_link % store_target + '/data.json', 'w') as f:
            f.write(json.dumps(data, ensure_ascii=False, indent=4, separators=(',', ':')))
        with open(DLSet.raw_folder_link % store_target + '/db_content.json', 'w') as f:
            f.write(json.dumps(content, ensure_ascii=False, indent=4, separators=(',', ':')))
        with open(DLSet.raw_folder_link % store_target + '/db_schema.json', 'w') as f:
            f.write(json.dumps(schema, ensure_ascii=False, indent=4, separators=(',', ':')))

    @staticmethod
    def __sub_data_generate(data: list, choice_list: list):
        result_data = []
        for each in choice_list:
            result_data.append(data[each])

        return result_data

    @staticmethod
    def data_generate(data_source_list: list, cheat_mode: bool = True, test_data_ratio: float = 0.3):
        # 1. load the raw data
        data = []
        content = []
        schema = []

        for data_source in data_source_list:
            with open(DLSet.origin_source_folder_link % data_source + '/data.json', 'r') as f:
                data.extend(json.load(f))
            with open(DLSet.origin_source_folder_link % data_source + '/db_content.json', 'r') as f:
                content.extend(json.load(f))
            with open(DLSet.origin_source_folder_link % data_source + '/db_schema.json', 'r') as f:
                schema.extend(json.load(f))

        # whether cheat_mode
        if cheat_mode:
            # raw data size
            total = len(data)
            idx = [i for i in range(total)]

            # random sample
            test_idx = random.sample(idx, int(total * test_data_ratio))
            train_idx = list(set(idx) - set(test_idx))

            # store
            print('-' * 20 + 'Train Data generate' + '-' * 20)
            DataGenerator.__store(DataGenerator.__sub_data_generate(data, train_idx),
                                  content, schema, store_target='Train')

            print('-' * 20 + 'Validation Data generate' + '-' * 20)
            DataGenerator.__store(DataGenerator.__sub_data_generate(data, test_idx),
                                  content, schema, store_target='Validation')

        else:
            print('-' * 20 + 'Test Data generate' + '-' * 20)
            DataGenerator.__store(data, content, schema, store_target='Test')



