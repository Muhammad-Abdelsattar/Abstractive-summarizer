from typing import Optional
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ConstantLR
from transformers import (AutoModelForSeq2SeqLM,
                          AutoTokenizer)

from peft import (PrefixTuningConfig,
                  LoraConfig,
                  get_peft_model)
from lightning import LightningModule


class AbstractiveSummarizationModule(LightningModule):
    def __init__(self,
                 config: DictConfig):
        super().__init__()
        self.config = config
        self.model = self._prepare_model()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.model_id)


    def _prepare_model(self):
        base_model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model.model_id)
        if(self.config.model.use_peft_config == "prefix"):
            prefix_tuning_conf = self.config.model.prefix_tuning_config
            peft_config = PrefixTuningConfig(task_type=prefix_tuning_conf.task_type,
                                            num_virtual_tokens=prefix_tuning_conf.num_virtual_tokens,
                                            inference_mode=prefix_tuning_conf.inference_mode,
                                            prefix_projection=prefix_tuning_conf.prefix_projection)

            print("Using prefix tuning.")


            return get_peft_model(base_model,peft_config)

        elif(self.config.model.use_peft_config=="lora"):
            lora_conf = self.config.model.lora_config
            peft_config = LoraConfig(task_type=lora_conf.task_type,
                                    inference_mode=lora_conf.inference_mode,
                                    lora_dropout=lora_conf.lora_dropout,
                                    lora_alpha=lora_conf.lora_alpha,
                                    r=lora_conf.r

                                    )

            print("Using LoRa for fine tuning.")


            return get_peft_model(base_model,peft_config)

        else:
            print("Using full fine tuning")

            return base_model


    def common_step(self,
                    input_dict: dict[str,Tensor],
                    ):

        output = self.model(
            input_ids=input_dict["input_ids"],
            attention_mask=input_dict["input_attention_mask"],
            labels=input_dict["target_ids"],
            decoder_attention_mask = input_dict["target_attention_mask"]
        )
        return output.loss, output.logits


    def configure_optimizers(self):
        optim_conf = self.config.training.optim
        optimizer = AdamW(self.model.parameters(),
                          **optim_conf.optimizers.AdamW)
        scheduler = ConstantLR(optimizer = optimizer,
                               **optim_conf.lr_schedulers.ConstantLR)
        return {"optimizer": optimizer,
                "lr_scheduler": scheduler}


    def training_step(self,batch, batch_idx):
        loss,logits = self.common_step(batch)
        self.log("Training Loss",loss,prog_bar = True,on_step=True,on_epoch=True)
        return loss


    def validation_step(self,batch,batch_idx):
        loss,logits = self.common_step(batch)
        self.log("Validation Loss",loss,prog_bar = True, on_epoch=True)
        return loss


    def test_step(self,batch,batch_idx):
        loss,logits = self.common_step(batch)
        self.log("Test Loss",loss,prog_bar = True, on_epoch=True)
        return loss

    
    def generate_from_dataloader(self,dataloader: DataLoader):

        generation_config = self.config.model.generation_config.copy()

        refs = []
        hyps = []
        for item in dataloader:
            out = self.model.generate(input_ids=item["input_ids"].to(device=self.device),
                                      **generation_config)
            
            ref = self.tokenizer.decode(item["target_ids"][0],skip_special_tokens=True)
            hyp = self.tokenizer.decode(out[0],skip_special_tokens=True)
            refs.append(ref)
            hyps.append(hyp)

        return hyps,refs