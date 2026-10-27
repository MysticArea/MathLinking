import json, os

def load_jsonl(path: str):
    if not os.path.exists(path):
        with open(path, "w", encoding='utf-8') as f:
            pass
    with open(path, "r", encoding='utf-8') as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def save_jsonl(path: str, data: list, t_stamp=True) -> None:
    if t_stamp:
        file_name = f"{path.replace('.jsonl','')}{timestamp()}.jsonl"
    else:
        file_name = path
    with open(file_name, 'w', encoding='utf-8') as f:
        for line in tqdm(data, desc='save'):
            f.write(json.dumps(line) + '\n')

def create_step_path(step):
    step_path = f"step_{step}_of_train_80k.jsonl"
    return os.path.join("data", step_path)

def append_and_create(path: str, data: dict):
    with open(path, "a", encoding='utf-8') as f:
        f.write(json.dumps(data) + '\n')  # 仅会写入字符串到文件中

def get_lce_step(data_point):
    lce_chains = data_point["messages"][2]["content"]
    temp_step = 0
    for lce_chain in lce_chains:
        if lce_chain["type"] == "code":
            temp_step += 1
    return temp_step

origin_data_path = r"data\MathCoder-Instruction\train_80k.jsonl"
datas = load_jsonl(origin_data_path)

accumulate = 0
for step in range(17):  # calc num of every step
    step_path = create_step_path(step)
    das = load_jsonl(step_path)
    temp_accumu = len(das)
    accumulate += temp_accumu
    print(f"lce链步长：{step}，共有：{temp_accumu}")

print("final end")