from torch.utils.data import IterableDataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext as tt

from typing import Optional, Callable
from csv import DictReader
from pathlib import Path
import functools as ft
import torch.nn as nn
import torch

TAB = "\t"


class TSVRelationExtractionDataset(IterableDataset):
    PA3_TSV_COLUMNS = ["relation", "e1_idx", "e2_idx", "sentence"]

    def __init__(
        self,
        file_path,
        column_names: Optional[list[str]] = None,
        sentence_column: str = "sentence",
        relation_column: str = "relation",
        e1_idx_column: str = "e1_idx",
        e2_idx_column: str = "e2_idx",
        tokenize_fn: Optional[Callable] = None,
        add_entity_tags: bool = False,
        truncate: bool = False,
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        train_dataset: Optional["TSVRelationExtractionDataset"] = None,
    ):
        if not tokenize_fn:
            # Default tokenizer should just split on spaces, None input ids
            def tokenize(sentence: str) -> list[str]:
                return sentence.split(), None
        else:
            # LLM tokenizer passed in returns base tokens and input ids
            def tokenize(sentence: str) -> list[str]:
                return sentence.split(), tokenize_fn(sentence)['input_ids']
        
        self.llm = True if tokenize_fn else False
        # Make the input arguments instance variables
        self.file_path = file_path
        self.column_names = column_names or self.PA3_TSV_COLUMNS
        self.tokenize = tokenize
        self.sentence_column = sentence_column
        self.relation_column = relation_column

        self.add_entity_tags = add_entity_tags
        self.truncate = truncate

        self.e1_idx_column = e1_idx_column
        self.e2_idx_column = e2_idx_column

        # Some important constants
        self.e1_tag = "<e1>"
        self.e1_tag_close = "</e1>"
        self.e2_tag = "<e2>"
        self.e2_tag_close = "</e2>"

        self.pad_token = pad_token
        self.unk_token = unk_token

        # If a train dataset was passed, we take note of it for vocab purposes
        self.train_dataset = train_dataset

        if not self.train_dataset:
            self.vocab = build_vocab_from_iterator(
                self.tokens, specials=[self.pad_token, self.unk_token]
            )
            self.vocab.set_default_index(self.vocab[self.unk_token])
            self.label_index = list(set(self.relations))        # added for test referencing
            self.label_vocab = {rel: idx for idx, rel in enumerate(self.label_index)}
        else:
            self.vocab = self.train_dataset.vocab
            self.label_index = self.train_dataset.label_index
            self.label_vocab = self.train_dataset.label_vocab

    def get_tsv_reader(self, file_in) -> DictReader:
        return DictReader(f=file_in, fieldnames=self.column_names, delimiter=TAB)

    def truncate_tokens(
        self, tokens: list[str], input_ids, e1_idx: int, e2_idx: int
    ) -> tuple[list[str], int, int]:
        # INCLUSIVE slice, keeps entity2 in the tokens
        tokens = tokens[e1_idx:e2_idx+1]

        if input_ids:
            input_ids = input_ids[e1_idx:e2_idx+1]

        e1_idx, e2_idx = 0, len(tokens)-1
        return tokens, input_ids, e1_idx, e2_idx

    def surround_entities_with_tags(
        self, tokens: list[str], e1_idx: int, e2_idx: int
    ) -> list[str]:
        tokens = (
            tokens[:e1_idx]
            + [self.e1_tag, tokens[e1_idx], self.e1_tag_close]
            + tokens[(e1_idx + 1) : e2_idx]
            + [self.e2_tag, tokens[e2_idx], self.e2_tag_close]
            + tokens[(e2_idx + 1) :]
        )

        return tokens

    def __iter__(self):
        with open(self.file_path) as file_in:
            for d in self.get_tsv_reader(file_in):
                # Handle the entity id conversion

                d[self.e1_idx_column] = int(d[self.e1_idx_column])
                d[self.e2_idx_column] = int(d[self.e2_idx_column])
                e1_idx, e2_idx = d[self.e1_idx_column], d[self.e2_idx_column]

                # Handle tokenization
                sentence = d[self.sentence_column]
                tokens, input_ids = self.tokenize(sentence)
                del d[self.sentence_column]

                # Indicate the two entities for clarity
                d["entity1"] = tokens[e1_idx]
                d["entity2"] = tokens[e2_idx]

                if self.truncate:
                    tokens, input_ids, e1_idx, e2_idx = self.truncate_tokens(
                        tokens, input_ids, e1_idx, e2_idx
                    )

                if self.add_entity_tags:
                    tokens = self.surround_entities_with_tags(tokens, e1_idx, e2_idx)

                d["tokens"] = tokens
                d["input_ids"] = input_ids

                yield d

    @property
    def relations(self):
        it = iter(self)

        for d in it:
            yield d[self.relation_column]

    @property
    def tokens(self):
        it = iter(self)

        for d in it:
            yield d["tokens"]


class CollateCallable:
    """
    An alternative to a collate_fn that keeps things a little bit more modular.
    """

    def __init__(
        self,
        vocab: tt.vocab.Vocab,
        label_vocab: dict,
        pad_value: int = 0,
        llm: bool = False,
    ):
        self.vocab = vocab
        self.label_vocab = label_vocab
        self.pad_value = pad_value
        self.llm = llm

    def pad(self, token_indices):
        return nn.utils.rnn.pad_sequence(
            [torch.tensor(indices) for indices in token_indices], batch_first=True
        )

    def __call__(self, examples):
        if self.llm:
            token_indices = self.pad(
                [example['input_ids'] for example in examples]  
            )   # input_ids from LLM instead of tokens
        else:
            token_indices = self.pad(
                [self.vocab(example["tokens"]) for example in examples]
            )

        labels = torch.tensor(
            [
                self.label_vocab.get(example["relation"], self.pad_value)

                for example in examples
            ]
        )

        return token_indices, labels
