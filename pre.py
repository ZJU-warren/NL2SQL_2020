from preprocessing.data_preprocessor import DataPreprocessor
from preprocessing.data_generator import DataGenerator
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_ratio', type=float, default=0.3, help='Batch size')
    args = parser.parse_args()

    DataGenerator.data_generate([1, 2, 3], cheat_mode=True, test_data_ratio=args.test_data_ratio)
    DataGenerator.data_generate([4], cheat_mode=False)

    DataPreprocessor.preprocess(data_source='Train')
    DataPreprocessor.preprocess(data_source='Validation')
    DataPreprocessor.preprocess(data_source='Test', cheat_mode=False)
