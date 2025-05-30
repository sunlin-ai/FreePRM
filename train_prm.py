import re
import os
import torch
import random
import argparse
from datasets import load_from_disk
from transformers import DataCollatorWithPadding
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
# from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

parser = argparse.ArgumentParser()
parser.add_argument("--debug_mode", type=bool, default=False)

# parser.add_argument("--model_path", type=str, default="/home/sunl/app/Models/Qwen_Qwen2.5-Math-7B-Instruct")
parser.add_argument("--model_path", type=str, default="/home/sunl/reward_model/ckpts/Qwen_Qwen2.5-Math-7B-Instruct")
parser.add_argument("--per_device_train_batch_size", type=int, default=2)
parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
parser.add_argument("--total_batch_size", type=int, default=16)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--use_pretrained_lora", type=bool, default=False)

parser.add_argument("--reward_type", type=str, default='prm')
parser.add_argument("--data_name", type=str, default="Math-Shepherd",choices=["Math-Shepherd","openreasoner_MATH-APS","Math-Mutual","trl-lib_PRM800k"])
parser.add_argument("--percent_data", type=float, default=0.2)

parser.add_argument("--loss_type", type=str, default="gray",choices=['gray','classification'])
parser.add_argument("--use_pseudo_label", type=bool, default=True)

# useful only when gray
parser.add_argument("--use_random_beta", type=bool, default=False)

parser.add_argument("--add_last_weight", type=bool, default=False)
parser.add_argument("--last_weight_num", type=int, default=1)

args = parser.parse_args()


def load_data():
    data_path = os.path.join('datas', args.data_name)
    dataset = load_from_disk(data_path)

    if "PRM800k" in args.data_name:
        dataset = dataset.filter(lambda example: len(example["step_labels"]) > 0)

    if args.debug_mode:
        dataset['train'] = dataset['train'].select(range(150))
        dataset['test'] = dataset['test'].select(range(150))
    else:
        print(f'use {args.percent_data} of data!')
        dataset['train'] = dataset['train'].select(range(int(args.percent_data * len(dataset['train']))))

    print('start processing')
    tokenized_datasets = dataset.map(preprocess_function)

    if "PRM800k" in args.data_name:
        columns_to_delete = ['prompt', 'completions', 'step_labels', 'label']

    if "Math-Shepherd" in args.data_name:
        columns_to_delete = ['input', 'task', 'label']

    elif "Math-APS" in args.data_name:
        columns_to_delete = ['question', 'process', 'label']

    elif "Math-Mutual" in args.data_name:
        columns_to_delete = ['pid', 'question', 'solution', 'labels_math_shepherd', 'labels_math_mutual', 'label']

    tokenized_datasets['train'] = tokenized_datasets['train'].remove_columns(columns_to_delete)
    tokenized_datasets['test'] = tokenized_datasets['test'].remove_columns(columns_to_delete)
    return tokenized_datasets


def preprocess_function(example):
    special_token = "\n\n\n\n\n"

    if "PRM800k" in args.data_name:
        question = example['prompt']
        steps = example['completions']
        steps = [f'Step {i + 1}: ' + s.strip("\n").strip() for i, s in enumerate(steps)]
        if args.reward_type=='prm':
            process = f" {special_token} ".join(steps) + f" {special_token}"
        else:
            process = f"\n\n".join(steps) + f" {special_token}"
        input = f"{question} {process}"
        example['label'] = ['+' if label else '-' for label in example["step_labels"]]

    if "Math-Shepherd" in args.data_name:
        input_text = example['input']
        if args.reward_type=='prm':
            input = input_text.replace('ки\n', f'{special_token} ').replace('ки', special_token)
        else:
            input = input_text.replace('ки', '\n').strip()

        if not input.endswith(special_token):
            input += f' {special_token}'

        step_labels = re.split('Step \d+:', example['label'])
        step_labels = [s.strip() for s in step_labels[1:]]
        example['label'] = [l[-1] for l in step_labels]

    elif "Math-APS" in args.data_name:
        question = example['question']
        solution = example['process']
        process = f"{solution} {special_token}"
        input = f"{question} {process}"

    elif "Math-Mutual" in args.data_name:
        question = example['question']
        solution = example['solution']
        steps = re.split('Step \d+:', solution)
        steps = [s for s in steps if s.strip() != '']
        steps = [f'Step {i + 1}: ' + s.strip("\n").strip() for i, s in enumerate(steps)]
        if args.reward_type=='prm':
            process = f" {special_token} ".join(steps) + f" {special_token}"
        else:
            process = f"\n\n".join(steps) + f" {special_token}"
        input = f"{question} {process}"
        example['label'] = example["labels_math_mutual"]

    final_label=example['label'][-1]

    if args.use_pseudo_label:
        example['label'] =[final_label for _ in example['label']]

    if args.reward_type=='orm':
        example['label'] = [final_label]

    tokenized_inputs = tokenizer(
        input,
        truncation=True,
        padding='max_length',
        max_length=2048,
    )

    def find_all_indices(lst, element):
        return [i for i, x in enumerate(lst) if x == element]

    length = len(tokenized_inputs['input_ids'])
    indices = find_all_indices(tokenized_inputs['input_ids'], step_tag_id)

    if len(indices) != len(example['label']):
        if len(indices) < len(example['label']):
            example['label'] = example['label'][:len(indices)]
        else:
            label = ['+'] * len(indices)
            label[-len(example['label']):] = example['label']
            example['label'] = label

    assert len(indices) == len(example['label'])
    tokenized_inputs['labels'] = [-100] * length

    for i in range(len(indices)):
        if example['label'][i] == '+' or example['label'][i] == 1:
            tokenized_inputs['labels'][indices[i]] = candidate_tokens[0]
        elif example['label'][i] == '-' or example['label'][i] == 0:
            tokenized_inputs['labels'][indices[i]] = candidate_tokens[1]
        else:
            tokenized_inputs['labels'][indices[i]] = candidate_tokens[0]
            # raise ValueError('label is wrong')
        tokenized_inputs['attention_mask'][indices[i]] = 0

    return tokenized_inputs


def load_model():
    good_token = '+'
    neutral_token = 'o'
    bad_token = '-'
    step_tag = '\n\n\n\n\n'

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, add_eos_token=False)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    candidate_tokens = tokenizer.encode(f" {good_token} {bad_token} {neutral_token}")  # [488, 481]
    step_tag_id = tokenizer.encode(f" {step_tag}")[-1]  # 76325

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()
    return model, tokenizer, candidate_tokens, step_tag_id


def compute_metrics(eval_pred):
    pre, labels = eval_pred
    auc = roc_auc_score(pre[1], pre[0])
    ll = log_loss(pre[1], pre[0])
    acc = accuracy_score(pre[1], pre[0] > 0.5)
    result = {'auc': auc, 'll': ll, 'acc': acc}
    print(result)
    return result


def preprocess_logits_for_metrics(logits, labels):
    labels_index = torch.argwhere(torch.bitwise_or(labels == candidate_tokens[0], labels == candidate_tokens[1]))
    gold = torch.where(labels[labels_index[:, 0], labels_index[:, 1]] == candidate_tokens[1], 0, 1)

    if args.loss_type == 'gray':
        logits = logits[labels_index[:, 0], labels_index[:, 1]][:,[candidate_tokens[0], candidate_tokens[1], candidate_tokens[2]]]
    else:
        logits = logits[labels_index[:, 0], labels_index[:, 1]][:, [candidate_tokens[0], candidate_tokens[1]]]

    prob = torch.softmax(logits, dim=-1)

    if args.loss_type == 'gray':
        prob_good, gold_res = [],[]
        for id in range(len(gold)):
            if args.use_random_beta:
                beta_candidates = [1.0, 0.0]
                weights = [prob[id, 2], 1 - prob[id, 2]]
                beta = random.choices(beta_candidates, weights=weights, k=1)[0]
            else:
                beta=0.5

            ## the last label is clear
            is_last=labels_index[id][1]==labels.shape[1]-1
            if is_last:
                prob_good_item=prob[id, 0]
                if args.add_last_weight:
                    prob_good.extend([prob_good_item]*args.last_weight_num)
                    gold_res.extend([gold[id]]*args.last_weight_num)
                else:
                    prob_good.append(prob_good_item)
                    gold_res.append(gold[id])
            
            ## the process lable need gray
            else:
                if gold[id] == 0:
                    prob_good_item = 1 - (prob[id, 1] + beta * prob[id, 2])
                else:
                    prob_good_item = prob[id, 0] + beta * prob[id, 2]
                prob_good.append(prob_good_item)
                gold_res.append(gold[id])

        prob_good = torch.tensor(prob_good)
        gold_res = torch.tensor(gold_res)

    else:
        prob_good = prob[:, 0]
        gold_res=gold

    return prob_good, gold_res


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs["logits"].squeeze() if isinstance(outputs, dict) else outputs[0]
        preprocess_logits_for_metrics(logits, labels)
        print('sunlin')


def train():
    tokenized_datasets = load_data()

    if args.loss_type=='gray':
        output_path = f'runs/prm_results_qwen/{args.data_name}_{args.percent_data}_{args.loss_type}_{args.reward_type}_lastweight_{args.last_weight_num}'
    else:
        output_path = f'runs/prm_results_qwen/{args.data_name}_{args.percent_data}_{args.loss_type}_{args.reward_type}'

    output_path +="_pseudo_label" if args.use_pseudo_label else "_true_label"

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_path,
        evaluation_strategy="no",  # Evaluate at the end of each epoch
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.total_batch_size // args.per_device_train_batch_size,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        data_collator=DataCollatorWithPadding(tokenizer),
        tokenizer=tokenizer,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        # compute_metrics=compute_metrics,
    )

    trainer.train()
    # trainer.evaluate()

    save_file_path = os.path.join(output_path, 'fine_tuned_math_shepherd_lora_8bit')
    model.save_pretrained(save_file_path)
    tokenizer.save_pretrained(save_file_path)


if __name__ == '__main__':
    model, tokenizer, candidate_tokens, step_tag_id = load_model()
    train()
