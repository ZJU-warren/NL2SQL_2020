""" the link of the dataset and files used """
# --------------------------- folders --------------------------
base_folder_link = '../DataSet'
origin_folder_link = base_folder_link + '/Origin'
origin_source_folder_link = origin_folder_link + '/%d'

train_folder_link = base_folder_link + '/Train'
validation_folder_link = base_folder_link + '/Validation'
test_folder_link = base_folder_link + '/Test'
raw_folder_link = base_folder_link + '/%s' + '/Raw'
main_folder_link = base_folder_link + '/%s' + '/Main'
model_folder_link = base_folder_link + '/Models'
tmp_folder_link = base_folder_link + '/Tmp'
result_folder_link = base_folder_link + '/Result'

# --------------------------- raw files ------------------------
raw_data_link = raw_folder_link + '/data.json'
raw_content_link = raw_folder_link + '/db_content.json'
raw_schema_link = raw_folder_link + '/db_schema.json'

# --------------------------- main files -----------------------
X_link = main_folder_link + '/X'

# --------------------------- others ---------------------------
pre_train_model_path = "../Bert"
config_path = pre_train_model_path + '/bert_config.json'
vocab_path = pre_train_model_path + '/vocab.txt'

