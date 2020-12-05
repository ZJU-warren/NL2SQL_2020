from preprocessing.data_preprocessor import DataPreprocessor
#
if __name__ == '__main__':
    DataPreprocessor.preprocess(data_source='Train')
    DataPreprocessor.preprocess(data_source='Validation')
