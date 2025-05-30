import os
import json
import numpy as np
from tqdm import tqdm


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()


def cal_step_reward(question_list, steps_list, format_type='aps', batch_size=4):
    if format_type == 'aps':
        special_token = '\n\n\n\n\n'
    else:
        special_token = 'ки'

    examples = []
    for question, steps in zip(question_list, steps_list):
        output = ""
        for idx, step in enumerate(steps):
            step = remove_step_ids(step)
            if idx == 0:
                step = f"Step {idx + 1}: {step}"
            else:
                if format_type == 'aps':
                    step = f" {special_token} Step {idx + 1}: {step}"
                else:
                    step = f" {special_token}\nStep {idx + 1}: {step}"
            output += step
        output += f" {special_token}"
        example = f"{question} {output}"
        examples.append(example)

    step_rewards = get_reward(examples, batch_size)
    return step_rewards


def save_cache(cache_file, cache_jsonl, cache):
    with open(cache_file, "a") as f:
        try:
            f.write(json.dumps(cache, indent=2, separators=(',', ': ')) + "\n")
        except Exception as e:
            print(e)
            print(cache)

    with open(cache_jsonl, "a") as f:
        try:
            json.dump(cache, f)
            f.write('\n')
        except Exception as e:
            print(e)
            print(cache)


def cal_reward():
    batch_size = 8
    format_type = 'aps'
    file_dir = r'D:\papers\13_MathR\data\processbench'

    save_dir = f'D:/papers/13_MathR/runs/process_bench/{reward_model}'
    os.makedirs(save_dir, exist_ok=True)

    configs = ['gsm8k', 'math', 'olympiadbench', 'omnimath']
    # configs = ['omnimath']
    for config in configs:
        file = os.path.join(file_dir, f'{config}.json')
        cache_file = os.path.join(save_dir, f"{config}.json")
        cache_jsonl = os.path.join(save_dir, f"{config}.jsonl")

        with open(file, 'r', encoding='utf-8') as f:
            datas = json.load(f)

        question_list, steps_list, pid_list, labels = [], [], [], []
        counter = 0

        tbar = tqdm(datas)
        for data in tbar:
            counter += 1
            question = data['problem']
            steps = data['steps']

            question_list.append(question)
            steps_list.append(steps)
            pid_list.append(counter - 1)

            if len(pid_list) == batch_size or counter == len(datas):
                step_rewards_list = cal_step_reward(question_list, steps_list, format_type, batch_size)
                for idx, step_rewards in zip(pid_list, step_rewards_list):
                    datas[idx]['step_rewards'] = step_rewards
                    save_cache(cache_file, cache_jsonl, datas[idx])
                question_list, steps_list, pid_list = [], [], []


def cal_acc(threshold, configs):
    return_score = True

    save_dir = f'D:/papers/13_MathR/runs/process_bench/{reward_model}'

    f1_list, acc_error_list, acc_correct_list, arithacc_list = [], [], [], []
    for config in configs:
        cache_jsonl = os.path.join(save_dir, f"{config}.jsonl")
        predictions = load_jsonl(cache_jsonl)

        res_data = []
        for idx, d in enumerate(predictions):
            step_rewards = d['step_rewards']

            pred = -1
            for id, reward in enumerate(step_rewards):
                if return_score:
                    if reward < threshold:
                        pred = id
                        break
                else:
                    # reward = torch.softmax(torch.tensor(reward), dim=0)[0]
                    # if reward < threshold:
                    #     pred = id
                    #     break

                    indices = np.argmax(reward)
                    if indices == 1:
                        pred = id
                        break

            new_d = {'label': d['label'], 'match': pred == d['label']}
            res_data.append(new_d)

        data1 = [e for e in res_data if e['label'] != -1]
        data2 = [e for e in res_data if e['label'] == -1]

        acc1 = np.mean([e['match'] for e in data1]) * 100
        acc2 = np.mean([e['match'] for e in data2]) * 100

        num_data1 = len(data1)
        num_data2 = len(data2)
        arithacc = num_data1 / (num_data1 + num_data2) * acc1 + num_data2 / (num_data1 + num_data2) * acc2
        arithacc_list.append(arithacc)

        f1 = 2 * acc1 * acc2 / (acc1 + acc2)
        f1_list.append(f1)
        acc_error_list.append(acc1)
        acc_correct_list.append(acc2)
        # print(f'{config} error acc: {acc1:.1f}, correct acc: {acc2:.1f}, f1: {f1:.1f}')
    return f1_list, acc_error_list, acc_correct_list, arithacc_list


def cal_acc_compare():
    configs = ['gsm8k', 'math', 'olympiadbench', 'omnimath']
    f1_avg_list, acc_error_avg_list, acc_correct_avg_list = [], [], []
    for threshold in np.arange(0.0, 1.05, 0.05):
        f1_list, acc_error_list, acc_correct_list, arithacc_list = cal_acc(threshold, configs)

        f1_list = [round(num, 1) for num in f1_list]
        acc_error_list = [round(num, 1) for num in acc_error_list]
        acc_correct_list = [round(num, 1) for num in acc_correct_list]
        arithacc_list = [round(num, 1) for num in arithacc_list]

        f1_avg = np.mean(f1_list)
        acc_error_avg = np.mean(acc_error_list)
        acc_correct_avg = np.mean(acc_correct_list)
        arithacc_avg = np.mean(arithacc_list)

        text_detail = f'acc_error: {acc_error_list} | acc_correct: {acc_correct_list} | f1: {f1_list} | arithacc: {arithacc_list}'
        text_simple = f'acc_error_avg: {acc_error_avg:.2f} | acc_correct_avg: {acc_correct_avg:.2f} | f1_avg: {f1_avg:.2f} | arithacc_avg: {arithacc_avg:.2f}'

        print(f"threshold: {round(threshold, 2)}")
        print(text_detail)
        print(text_simple)
        print()

        f1_avg_list.append(f1_avg)
        acc_error_avg_list.append(acc_error_avg)
        acc_correct_avg_list.append(acc_correct_avg)

    print(f1_avg_list)
    print(acc_error_avg_list)
    print(acc_correct_avg_list)


if __name__ == '__main__':
    # reward_model = "Math-Mutual_0.8-classification_prm"
    # reward_model = 'Math-Mutual_1.0_gray_prm_lastweight_5'
    # reward_model = "Math-Mutual_1.0_classification"

    reward_model = "Math-Mutual_0.2_gray_prm_lastweight_3"

    # reward_model = "Math-Mutual_0.2_deepseek-ai_DeepSeek-R1-Distill-Qwen-7B_gray_prm_lastweight_3"
    # reward_model="Math-Mutual_0.2_Qwen_Qwen2.5-3B-Instruct_gray_prm_lastweight_3"
    # reward_model = "trl-lib_PRM800k_1.0_Qwen_Qwen2.5-Math-7B_gray_prm_lastweight_3_true_label"
    # reward_model = "trl-lib_PRM800k_1.0_Qwen_Qwen2.5-Math-7B_classification_prm_lastweight_3_true_label"
    # reward_model = "Math-Mutual_1.0_gray_prm_lastweight_3"
    # reward_model = "Math-Mutual_0.2_gray_prm_lastweight_1_pseudo_label"
    # reward_model = "Math-Mutual_1.0_gray_prm_lastweight_3_logit"

    cal_reward()

    # cal_acc(threshold=0.95)

    cal_acc_compare()
