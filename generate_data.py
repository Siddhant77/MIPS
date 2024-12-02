import os
import sys
import csv
import argparse
import math

import time
import warnings
import datetime
import random
import pickle
import numpy as np
import pandas as pd

from datetime import timedelta

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

from transformers import pipeline, AutoTokenizer, TrainingArguments, Trainer
from accelerate import Accelerator

# Initialize parser
parser = argparse.ArgumentParser(description="Process some arguments.")

# Add arguments
parser.add_argument("--gpu", type=str, help="GPU to use")
parser.add_argument("--dataset", type=int, help="Dataset that should be converted")
parser.add_argument("--batch", type=int, help="Batch to continue from", default=0)
parser.add_argument("--port", type=str, help="port to use for gpu", default="29590")
parser.add_argument("--addr", type=str, help="address to use for gpu", default="127.0.0.1")

# Parse arguments
args = parser.parse_args()

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "2"
os.environ["MASTER_PORT"] = args.port
os.environ["MASTER_ADDR"] = args.addr

RANDOM_SEED = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4 #8 #16
MAX_FAILURES = 5
MAX_RESTARTS = 10

def get_vanilla_datasets():
    true_df = pd.read_csv("True.csv")
    fake_df = pd.read_csv("Fake.csv")
    train_df = pd.read_csv("WELFake_Dataset.csv")

    # drop columns that will not be used.
    true_df = true_df.drop(columns=["subject", "date", "title"])
    fake_df = fake_df.drop(columns=["subject", "date", "title"])
    train_df = train_df.drop(columns=["Unnamed: 0", "title"])    

    # add a 'label' columns
    true_df['label'] = 1
    fake_df['label'] = 0

    # concatinate 'true' and 'fake' news datasets
    test_df = pd.concat([true_df, fake_df])

    # clean dataframes 
    test_df = test_df.dropna()
    train_df = train_df.dropna()

    # flip labels for mix_df
    train_df['label'] = train_df["label"] ^ 1 # XOR operation flips 1 and 0

    # shuffle databases
    train_df = train_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    train1_df = train_df[len(train_df)//2 :]
    train2_df = train_df[: len(train_df)//2]

    train1_df = train1_df.iloc[::-1].reset_index(drop=True)
    train2_df = train2_df.iloc[::-1].reset_index(drop=True)
    test_df = test_df.iloc[::-1].reset_index(drop=True)

    train1_df.name = "vanilla_training1_dataset"
    train2_df.name = "vanilla_training2_dataset"
    test_df.name = "vanilla_testing_dataset"

    return [train1_df, train2_df, test_df]

def build_real_prompts(passages):
    messages = []
    for passage in passages:

        sys_message = "You are a writing AI assistant. Rewrite the provided text with official language. Respond only with the rewritten passage."
        user_message = "Respond only with the rewritten passage. Here is the passage: \n {}. ".format(passage)

        message = [
            { "role" : "system", "content" : sys_message},
            { "role" : "user", "content" : user_message}
            ]
        messages.append(message)
    return messages

def build_fake_prompts(passages):
    messages = []
    for passage in passages:

        sys_message = "You are a writing AI assistant. Rewrite the provided text like a scammer. Respond only with the rewritten passage."
        user_message = "Respond only with the rewritten passage. Here is the passage: \n {}. ".format(passage)

        message = [
            { "role" : "system", "content" : sys_message},
            { "role" : "user", "content" : user_message}
            ]
        messages.append(message)
    return messages

def chunk_dataframe(df, batch_size):
    for start in range(0, len(df), batch_size):
        yield df.iloc[start:start + batch_size]

def build_prompts(batch):
    messages = []

    real_prompt = "You are a writing AI assistant. Rewrite the provided text with official language. Respond only with the rewritten passage."
    fake_prompt = "You are a writing AI assistant. Rewrite the provided text like a scammer. Respond only with the rewritten passage."

    for _, (_,data) in enumerate(batch.iterrows()):
        sys_message = real_prompt if data['label'] else fake_prompt
        user_message = "Respond only with the rewritten passage. Here is the passage: \n {}. ".format(data['text'])

        message = [
            { "role" : "system", "content" : sys_message},
            { "role" : "user", "content" : user_message}
            ]
        
        messages.append(message)
    return messages

def get_llm_df(responses, batch):

    assert len(responses) == len(batch), "print len of batch and respones is not same"

    llm_responses = []
    for i in range(len(batch)):
        llm_response = responses[i][0]['generated_text']
        llm_responses.append(llm_response)

    llm_stylized_df = pd.DataFrame({
        'text' : batch['text'],
        'llm-text' : llm_responses,
        'label' : batch['label']
    })

    return llm_stylized_df

def make_llm_data(df, filename, batch_size, starting_batch=0):
    count = 0
    num_batches = math.ceil(len(df) / batch_size)
    failure_count = 0

    # define LLM
    model_id = "meta-llama/Llama-3.2-3B-Instruct" # better adherence to instructions
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    pipe = pipeline(
        task="text-generation", 
        model=model_id, 
        model_kwargs={"torch_dtype" : torch.bfloat16}, 
        device_map="auto",
        max_new_tokens=512,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
        batch_size=BATCH_SIZE
    )
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    pipe.tokenizer.padding_side = 'left'

    start_time = time.time()

    for batch in chunk_dataframe(df, batch_size):
        if count < starting_batch:
            count += 1
            continue
        try:
            msgs = build_prompts(batch)
            elapsed_time = time.time() - start_time
            elapsed_timedelta = timedelta(seconds=elapsed_time)
            print(f"Batch {count}/{num_batches} | Running time: {str(elapsed_timedelta)} ", end="\r")
            responses = pipe(msgs, batch_size=BATCH_SIZE, truncation=True, padding=True)            
            llm_df = get_llm_df(responses, batch)
            llm_df.to_csv(filename, mode='a', header=False, index=False)
            # reset failures
            failure_count = 0
            count += 1
        except Exception as e:
            failure_count+=1
            print(f"Error : {e} \n\n\n")
            print(f"Error processing batch {count+1}/{num_batches}")
            # clear un-used memory
            torch.cuda.empty_cache()

            #exit if too many consecutive failures
            if failure_count > MAX_FAILURES:
                print("Too many consecutive failures. Going to try and restart")
                return count
            continue

        # clear un-used memory
        torch.cuda.empty_cache()
    return count



def main():
    datasets = get_vanilla_datasets()
    target_dataset = datasets[args.dataset]

    # Access arguments
    print(f"GPU: {args.gpu}")
    print(f"PORT: {args.port}")
    print(f"ADDR: {args.addr}")
    print(f"Batch: {args.batch}")
    print(f"Dataset: {target_dataset.name}")
    print(f"length of dataset = {len(target_dataset)}")
    print(f"Using device: {DEVICE}")

    print("CUDA_LAUNCH_BLOCKING:", os.environ.get("CUDA_LAUNCH_BLOCKING"))
    print("TORCH_USE_CUDA_DSA:", os.environ.get("TORCH_USE_CUDA_DSA"))
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("RANK:", os.environ.get("RANK"))
    print("LOCAL_RANK:", os.environ.get("LOCAL_RANK"))
    print("WORLD_SIZE:", os.environ.get("WORLD_SIZE"))
    print("MASTER_PORT:", os.environ.get("MASTER_PORT"))
    print("MASTER_ADDR:", os.environ.get("MASTER_ADDR"))

    print("Is CUDA available?", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())
    print("Current GPU:", torch.cuda.current_device())
    print("GPU Name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    print(target_dataset.head())

    header = ["text", "llm-text", "label"]
    filename = "llm_" + target_dataset.name + ".csv"
    print(f"filename : {filename}")
    df = pd.DataFrame(columns=header)
    
    # create header
    df.to_csv(filename, mode='w', header=True, index=False)

    restart = 0
    c = args.batch
    while restart < MAX_RESTARTS:
        c = make_llm_data(target_dataset, filename=filename, batch_size=BATCH_SIZE, starting_batch=c)
        restart += 1


main()