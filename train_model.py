import torch
import numpy as np
import evaluate
import numpy as np

from datasets import Dataset
from argparse import ArgumentParser
from typing import List
from prepare_data import prepare_data
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)


def preprocess_function(examples, tokenizer, enc_max_len, device):
    """
    Standard pre-process function from Hugginface. Thus, no typing. 
    Truncatin, padding and sending input tensors to current device.
    """
    inputs = [doc for doc in examples['document']]
    model_inputs = tokenizer(inputs, padding=True, max_length=enc_max_len, truncation=True, return_tensors='pt').to(device)

    labels = tokenizer(text_target=examples['summary'], padding=True, max_length=128, truncation=True, return_tensors='pt').to(device)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


def main() -> None:

    parser = ArgumentParser()
    parser.add_argument('model_save_folder', type=str, default='dialogue_summ_model')
    parser.add_argument('model_checkpoint', type=str)
    parser.add_argument('lr', type=float, default=3e-5)
    parser.add_argument('batch_size', type=int, default=32)
    parser.add_argument('weight_decay', type=float, default=0.01)
    parser.add_argument('num_epochs', type=int, default=4)

    args = parser.parse_args()

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)

    enc_max_len, dec_max_len = 400, 60 # Infered from the distribution of dialogue and summary lengths

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)  # Sending model to GPU

    # Preparing data
    train_data = prepare_data(['train.json', 'dialogsum.train.jsonl'])
    dev_data = prepare_data(['val.json', 'dialogsum.dev.jsonl'])
    test_data = prepare_data(['test.json', 'dialogsum.test.jsonl'], test=True)


    # This code was used to explore dialog- & summary-length histograms (to infer and max. encoder & decoder lengts)
    """
    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.hist(train_dialogues, bins=40, edgecolor='black')
    plt.title('Dialogue lengths')
    plt.ylabel('Counts')
    plt.xlabel('Length of dialogue (tokens)')

    plt.subplot(1,2,2)
    plt.hist(train_summaries, bins=40, edgecolor='black')
    plt.title('Sammary lengths')
    plt.ylabel('Counts')
    plt.xlabel('Length of dialogue (tokens)')

    plt.tight_layout()
    plt.show()
    """
    
    train_dateset = Dataset.from_dict(train_data)
    dev_dataset = Dataset.from_dict(dev_data)

    # Pre-processing the data
    tokenized_train = train_dateset.map(preprocess_function, tokenizer, enc_max_len, device, batched=True)
    tokenized_dev = dev_dataset.map(preprocess_function, tokenizer, enc_max_len, device, batched=True)

    # Data collator for padding
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=args.model_checkpoint)

    # Init. rouge eval. metric     
    rouge = evaluate.load('rouge')

    # Defining the function to compute the metric, following the instructions from HuggingFace
    def compute_metrics(eval_pred):
        """
        Function from Huggingface. To perform the computation,
        special tokeings (e.g. padding) should be removed. 

        Prediction lens computes the lenght of the prediction. 
        This is used to compute avg. generation length. 
        """

        preds, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}


    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.model_save_folder,
        evaluation_strategy='epoch',
        learning_rate=args.lr,
        per_device_eval_batch_size=args.batch_size,
        per_device_train_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        save_total_limit=3, 
        num_train_epochs=args.num_epochs,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False,  
    )

    # Init. trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print('training')
    trainer.train()

if __name__ == '__main__':
    main()