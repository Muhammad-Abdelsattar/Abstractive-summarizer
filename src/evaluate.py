from rouge import Rouge
from torch.utils.data import DataLoader
from .model import AbstractiveSummarizationModule



def calculate_rouge_score(hyps,refs,avg=True):
    scorer = Rouge()
    score = scorer.get_scores(hyps=hyps,refs=refs,avg=avg)

    return score


def evaluate(model:AbstractiveSummarizationModule,
             dataloader: DataLoader):
    hyps, refs = model.generate_from_dataloader(dataloader=dataloader)
    score = calculate_rouge_score(hyps=hyps,refs=refs,avg=True)

    return score