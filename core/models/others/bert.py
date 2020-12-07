import transformers as trf
import DataLinkSet as DLSet
from GlobalParameters import cuda_id


class BertEncoder(trf.BertPreTrainedModel):
    def __init__(self, config_path=DLSet.config_path, pre_train_model_path=DLSet.pre_train_model_path, gpu=True):
        super(BertEncoder, self).__init__(trf.BertConfig.from_pretrained(config_path))
        self.bert_model = trf.BertModel.\
            from_pretrained(pre_train_model_path, config=trf.BertConfig.from_pretrained(config_path))
        self.gpu = gpu

    def forward(self, input_ids, token_type_ids, attention_mask):
        if self.gpu:
            input_ids = input_ids.cuda(cuda_id)
            attention_mask = attention_mask.cuda(cuda_id)
            token_type_ids = token_type_ids.cuda(cuda_id)

        outputs = self.bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return outputs

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
