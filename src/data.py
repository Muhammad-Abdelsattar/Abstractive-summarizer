from typing import Optional
import hydra
from omegaconf import DictConfig
from dataclasses import dataclass
import os
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import (PreTrainedTokenizer,
                          AutoTokenizer)
from lightning import LightningDataModule

os.environ['CURL_CA_BUNDLE'] = ''

@dataclass
class DataPoint:
    input_ids: Tensor
    input_mask: Tensor
    target_ids: Tensor
    target_mask: Tensor


class SummarizationDataset(Dataset):
    def __init__(self,
                 dataset: pd.DataFrame,
                 tokenizer: PreTrainedTokenizer,
                 source_key = "source",
                 target_key = "target",
                 source_max_len = 2048,
                 target_max_len = 256,
                 data_slice = None
                 ):
        if(data_slice):
            self.dataset = dataset.iloc[data_slice[0]:data_slice[1]]
        else:
            self.dataset = dataset

        self.tokenizer = tokenizer
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len
        self.source_key = source_key
        self.target_key = target_key


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self,index):
        source = self.tokenizer("Doc to summarize: "+self.dataset.iloc[index][self.source_key],
                                return_tensors="pt",
                                truncation=True,
                                max_length=self.source_max_len)

        target = self.tokenizer(self.dataset.iloc[index][self.target_key],
                                return_tensors="pt",
                                truncation=True,
                                max_length=self.target_max_len)

        source_ids = source["input_ids"].squeeze(0)
        source_mask = source["attention_mask"].squeeze(0)
        target_ids = target["input_ids"].squeeze(0)
        target_ids[target_ids == 0] = -100 #this replaces the padding token id (0) by -100 so that it doesn't contribute to the cross entropy loss.
        target_mask = target["attention_mask"].squeeze(0)
        return DataPoint(source_ids,source_mask,target_ids,target_mask)


class SummarizationDataModule(LightningDataModule):
    def __init__(self,
                 config: DictConfig):

        super().__init__()
        self.config = config
        self.train_data_source = config.data.files.train_data_source
        self.valid_data_source = config.data.files.valid_data_source
        self.test_data_source = config.data.files.test_data_source
        self.model_id = config.model.model_id
        self.source_key = config.data.params.source_key
        self.target_key = config.data.params.target_key
        self.source_max_len = config.data.params.source_max_len
        self.target_max_len = config.data.params.target_max_len
        self.train_batch_size = config.data.params.train_batch_size
        self.valid_test_batch_size = config.data.params.valid_test_batch_size
        self.slices = config.data.params.slices

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)


    def prepare_data(self):
        pass


    def setup(self,stage: Optional[str] = None):
        self.train_df = pd.read_csv(self.train_data_source)
        self.valid_df = pd.read_csv(self.valid_data_source)
        self.test_df = pd.read_csv(self.test_data_source)

        self.train_dataset = SummarizationDataset(dataset = self.train_df,
                                                  source_key = self.source_key,
                                                  target_key = self.target_key,
                                                  source_max_len = self.source_max_len,
                                                  target_max_len = self.target_max_len,
                                                  tokenizer = self.tokenizer,
                                                  data_slice=self.slices.train)

        self.valid_dataset = SummarizationDataset(dataset = self.valid_df,
                                                  source_key = self.source_key,
                                                  target_key = self.target_key,
                                                  source_max_len = self.source_max_len,
                                                  target_max_len = self.target_max_len,
                                                  tokenizer = self.tokenizer,
                                                  data_slice=self.slices.valid)

        self.test_dataset = SummarizationDataset(dataset = self.test_df,
                                                  source_key = self.source_key,
                                                  target_key = self.target_key,
                                                  source_max_len = self.source_max_len,
                                                  target_max_len = self.target_max_len,
                                                  tokenizer = self.tokenizer,
                                                  data_slice=self.slices.test)


    def collate_fn(self,batch: list[DataPoint]):
        input_ids_list,input_mask_list,target_ids_list,target_mask_list = [],[],[],[]

        for datapoint in batch:
            input_ids_list.append(datapoint.input_ids)
            input_mask_list.append(datapoint.input_mask)
            target_ids_list.append(datapoint.target_ids)
            target_mask_list.append(datapoint.target_mask)

        input_ids_list = pad_sequence(input_ids_list, batch_first=True, padding_value=-100)
        input_mask_list = pad_sequence(input_mask_list, batch_first=True, padding_value=0)
        target_ids_list = pad_sequence(target_ids_list, batch_first=True, padding_value=-100)
        target_mask_list = pad_sequence(target_mask_list, batch_first=True, padding_value=0)

        return {"input_ids":input_ids_list,
                "input_attention_mask":input_mask_list,
                "target_ids":target_ids_list,
                "target_attention_mask":target_mask_list
               }


    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.train_batch_size,
                          num_workers=self.config.data.params.num_workers,
                          collate_fn=self.collate_fn)


    def val_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size=self.valid_test_batch_size,
                          num_workers=self.config.data.params.num_workers,
                          collate_fn=self.collate_fn)


    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.valid_test_batch_size,
                          num_workers=self.config.data.params.num_workers,
                          collate_fn=self.collate_fn)