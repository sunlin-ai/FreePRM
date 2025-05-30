import json
import torch
from peft import PeftModel
from flask import request, Flask
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)


@app.route('/get_reward', methods=['POST'])
def get_reward():
    return_type = "logit"

    rec_json = request.get_json(force=True)
    inputs = rec_json['inp']
    batch_size = rec_json['batch_size']
    output_scores = []
    for i in range(0, len(inputs), batch_size):
        inputs_batch = inputs[i: i + batch_size]
        inputs_batch = tokenizer(inputs_batch, padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inputs_batch).logits[:, :, candidate_tokens]

            if return_type == 'score':
                scores = logits.softmax(dim=-1)[:, :, 0]
            elif return_type == 'indice':
                values, indices = torch.max(logits, dim=-1)
                scores = indices
            elif return_type == 'logit':
                scores = logits

            step_scores_flat = scores[inputs_batch.input_ids == step_tag_id].tolist()
            step_scores = []
            counter = 0
            for i in range(len(inputs_batch.input_ids)):
                count = inputs_batch.input_ids[i].tolist().count(step_tag_id)
                step_score = step_scores_flat[counter: counter + count]
                step_scores.append(step_score)
                counter += count
        output_scores.extend(step_scores)
    return json.dumps({'rewards': output_scores})


def load_shepherd_model():
    good_token = '+'
    bad_token = '-'
    step_tag = 'ки'
    model_path = "/home/sunl/reward_model/ckpts/peiyi9979_math-shepherd-mistral-7b-prm"
    print(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    candidate_tokens = tokenizer.encode(f"{good_token} {bad_token}")[1:]  # [648, 387]
    step_tag_id = tokenizer.encode(f"{step_tag}")[-1]  # 12902
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, ).eval()
    return model, tokenizer, candidate_tokens, step_tag_id


def load_qwen_model():
    good_token = '+'
    neutral_token = 'o'
    bad_token = '-'
    step_tag = '\n\n\n\n\n'

    server_id=21
    
    if server_id==20:
        model_path = "/home/sunl/app/Models/Qwen_Qwen2.5-Math-7B-Instruct"
        lora_path="/home/sunl/app/sunl/reward_model/runs/prm_results_qwen/Math-Mutual_0.2_classification/fine_tuned_math_shepherd_lora_8bit"
    
    elif server_id==21:
        model_path ="/home/sunl/reward_model/ckpts/Qwen_Qwen2.5-Math-7B-Instruct"
        lora_path="/home/sunl/reward_model/runs/prm_results_qwen/Math-Mutual_1.0_gray_prm_lastweight_3/fine_tuned_math_shepherd_lora_8bit"

    tokenizer = AutoTokenizer.from_pretrained(model_path, add_eos_token=False)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    candidate_tokens = tokenizer.encode(f" {good_token} {bad_token} {neutral_token}")

    print(f'load lora: {lora_path}')
    step_tag_id = tokenizer.encode(f" {step_tag}")[-1]  # 76325
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        # attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(model, lora_path)
    return model, tokenizer, candidate_tokens, step_tag_id


if __name__ == '__main__':
    data_type = "Mutual"

    if data_type == "shepherd":
        model, tokenizer, candidate_tokens, step_tag_id = load_shepherd_model()
    else:
        model, tokenizer, candidate_tokens, step_tag_id = load_qwen_model()

    try:
        app.run('0.0.0.0', port=8000, threaded=True)
    except Exception as e:
        print('ERROR!!' + str(e))
