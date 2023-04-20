import dataset as ds
import model as m
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import IterableDataset, DataLoader
from sklearn.metrics import classification_report

import click
import skorch as sk
import torch.nn as nn
import torch
import pandas as pd

from skorch.helper import predefined_split

from skorch.callbacks import EpochScoring
from torch.optim.lr_scheduler import LambdaLR


@click.command()
@click.option("--batch-size", default=1)
@click.option("--max-epochs", default=1)
@click.option("--truncate", is_flag=True)
@click.option("--add-entity-tags", is_flag=True)
@click.option("--debug", is_flag=True)
@click.option("--train-file", type=click.Path(readable=True), required=True)
@click.option("--dev-file", type=click.Path(readable=True), required=True)
@click.option("--test-file", type=click.Path(readable=True), default=None)
@click.option("--hidden-layer-sizes", default="200,100")
@click.option("--embedding-dim", default=300)
@click.option("--lr", default=5e-5)
@click.option("--device", type=click.Choice(["cpu", "cuda"]))
@click.option("--llm", is_flag=True)
@click.option("--llm-choice", type=click.Choice(["distilbert-base-uncased", "bert-base-uncased", "openai-gpt"]))
@click.option("--model", type=click.Choice(['dan',"cnn", "lstm"]), default='dan')

def main(
    batch_size,
    max_epochs,
    truncate,
    add_entity_tags,
    debug,
    train_file,
    dev_file,
    test_file,
    hidden_layer_sizes,
    embedding_dim,
    lr,
    device,
    llm,
    llm_choice,
    model,
):
    if model == 'cnn':
        classifier = m.LLMCNNTextClassifier if llm else m.CNNTextClassifier
    elif model == 'lstm':
        classifier = m.LLMLSTMTextClassifier if llm else m.LSTMTextClassifier
    else:
        classifier = m.DANClassifier
    
    if llm:
        from transformers import AutoTokenizer, DistilBertTokenizer, OpenAIGPTModel, BertModel, DistilBertModel
        match llm_choice:
            case 'openai-gpt':
                llmtoken = AutoTokenizer.from_pretrained(llm_choice)
                llmmodel = OpenAIGPTModel.from_pretrained(llm_choice)
            case 'bert-base-uncased':
                llmtoken = AutoTokenizer.from_pretrained(llm_choice)
                llmmodel = BertModel.from_pretrained(llm_choice)
            case 'distilbert-base-uncased':
                llmtoken = DistilBertTokenizer.from_pretrained(llm_choice)
                llmmodel = DistilBertModel.from_pretrained(llm_choice)
            case _:
                raise Exception("need to choose an llm")
        embedding_dim = 768
    
    hidden_layer_sizes = [int(hl) for hl in hidden_layer_sizes.split(",")]

    train = ds.TSVRelationExtractionDataset(
        file_path=train_file,
        truncate=truncate,
        add_entity_tags=add_entity_tags,
        tokenize_fn = llmtoken if llm else None,
    )
    dev = ds.TSVRelationExtractionDataset(
        file_path=dev_file,
        truncate=truncate,
        add_entity_tags=add_entity_tags,
        train_dataset=train,
        tokenize_fn = llmtoken if llm else None,
    )
    if test_file:
        test = ds.TSVRelationExtractionDataset(
            file_path=test_file,
            truncate=truncate,
            add_entity_tags=add_entity_tags,
            train_dataset=train,
            tokenize_fn = llmtoken if llm else None,
        )

    collate_callable = ds.CollateCallable(
        vocab=train.vocab, label_vocab=train.label_vocab, llm=llm,
    )

    # Define some metrics
    f1_micro = EpochScoring("f1_micro", lower_is_better=False,)
    f1_macro = EpochScoring("f1_macro", lower_is_better=False,)

    net = sk.NeuralNetClassifier(
        classifier,
        module__llm = llmmodel if llm else None,
        module__num_classes=len(train.label_vocab),
        module__vocab_size=len(train.vocab),
        module__hidden_layer_sizes=hidden_layer_sizes,
        module__embedding_dim=embedding_dim,
        optimizer=torch.optim.AdamW,
        lr=lr,
        max_epochs=max_epochs,
        criterion=nn.CrossEntropyLoss,
        batch_size=batch_size,
        iterator_train__shuffle=False,
        device=device,
        iterator_train=DataLoader,
        iterator_train__collate_fn=collate_callable,
        iterator_valid=DataLoader,
        iterator_valid__collate_fn=collate_callable,
        train_split=predefined_split(dev),
        callbacks=[f1_micro, 
                   f1_macro,
                   sk.callbacks.Checkpoint(monitor='valid_acc_best', fn_prefix=model.upper()+'_FINAL_'+llm_choice, dirname='best_performers', f_params="_model.pt", f_optimizer="_opt.pt", f_history='history.json')
        ],
    )

    # Training with skorch net
    print()
    print("-------TRAINING EXPERIMENT", model, llm_choice, "T:",truncate, "lr:", lr, "------------------------")
    net.fit(train, y=None)

    #load back best checkpoint
    f_path = 'best_performers/'+model.upper()+'_FINAL_'+llm_choice
    net.initialize()
    net.load_params(
        f_params=f_path+"_model.pt", f_optimizer=f_path+"_opt.pt", f_history=f_path+'history.json')

    #   Check the load's accuracy, get a report
    y_pred = net.predict(dev)
    y_true = [train.label_vocab[R] for R in dev.relations]

    print()
    print(list(y_pred[:15]))
    print(y_true[:15])
    print("-------- VALIDATION CLASSIFICATION REPORT --------------")
    print(classification_report(list(y_true), 
                                y_pred, 
                                labels=         list(train.label_vocab.values()),
                                target_names=   list(train.label_vocab.keys()),
                                zero_division=  True))

    if test_file:
        # Predict the Test data and store it in tsv
        print()
        print("-------------Predicting on the test set ---------------")

        test_pred = net.predict(test)
        test_sents = [(train.label_index[test_pred[i]], 
                       sample['e1_idx'],
                       sample['e2_idx'],
                       sample['original']) for i,sample in enumerate(test)]

        for sample in test_sents[20:25]: # looking at a random 5, novice comparison
            print(sample[0],sample[1],sample[2], sample[3], sep='\t')

        print()
        pred_df = pd.DataFrame(test_sents, columns=['predicted_relation','e1','e2','text'])
        print(pred_df)

        pred_df.to_csv(model+'test_predictions.tsv', sep='\t', encoding='utf-8', index=False)
    
if __name__ == "__main__":
    main()
