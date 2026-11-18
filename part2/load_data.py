import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0
MODEL_NAME = "google-t5/t5-small"
PREFIX = "translate English to SQL: "
BOS_TOKEN = "<extra_id_0>"
MAX_LENGTH = 512
# The T5 SentencePiece vocabulary has no "<" piece: it is encoded as <unk> and lost on decoding,
# so "<" and "<=" comparisons could never be generated. We train on a placeholder word instead
# and map it back to "<" after decoding (see sql_to_target / target_to_sql).
LT_PLACEHOLDER = "LESSTHAN"


def sql_to_target(sql):
    return sql.replace("<", LT_PLACEHOLDER)


def target_to_sql(text):
    return text.replace(LT_PLACEHOLDER, "<")

_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
    return _tokenizer


def encode_text(tokenizer, text):
    return tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_LENGTH,
    ).input_ids


class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.split = split
        self.tokenizer = get_tokenizer()
        self.bos_id = self.tokenizer.convert_tokens_to_ids(BOS_TOKEN)
        self.encoder_ids = []
        self.decoder_inputs = []
        self.decoder_targets = []
        self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_lines = load_lines(os.path.join(data_folder, f"{split}.nl"))
        sql_lines = None if split == "test" else load_lines(os.path.join(data_folder, f"{split}.sql"))
        if sql_lines is not None and len(sql_lines) != len(nl_lines):
            raise ValueError(f"{split} has {len(nl_lines)} natural language lines and {len(sql_lines)} SQL lines")

        for i, nl in enumerate(nl_lines):
            enc_ids = encode_text(tokenizer, PREFIX + nl)
            self.encoder_ids.append(torch.tensor(enc_ids, dtype=torch.long))
            if split == "test":
                continue
            # T5 appends </s>. Decoder inputs are BOS plus every SQL token except that </s>;
            # targets are the SQL tokens, so each input position predicts the next one.
            sql_ids = encode_text(tokenizer, sql_to_target(sql_lines[i]))
            decoder_input = [self.bos_id] + sql_ids[:-1]
            self.decoder_inputs.append(torch.tensor(decoder_input, dtype=torch.long))
            self.decoder_targets.append(torch.tensor(sql_ids, dtype=torch.long))

    def __len__(self):
        return len(self.encoder_ids)

    def __getitem__(self, idx):
        if self.split == "test":
            return self.encoder_ids[idx], torch.tensor([self.bos_id], dtype=torch.long)
        return self.encoder_ids[idx], self.decoder_inputs[idx], self.decoder_targets[idx]


def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = pad_sequence([item[0] for item in batch], batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    decoder_inputs = pad_sequence([item[1] for item in batch], batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence([item[2] for item in batch], batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = decoder_inputs[:, :1].contiguous()
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs


def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = pad_sequence([item[0] for item in batch], batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    initial_decoder_inputs = torch.stack([item[1] for item in batch], dim=0)
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    dev_x = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))
    test_x = load_lines(os.path.join(data_folder, "test.nl"))
    return train_x, train_y, dev_x, dev_y, test_x