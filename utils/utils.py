import yaml

def load_sys_prompt(role):
    with open(f"prompts/{role}.txt", 'r', encoding='utf-8') as fin:
        sys_prompt = fin.read()
    return sys_prompt

def load_model_config(model):
    with open(f"configs/{model}.yaml", 'r', encoding='utf-8') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    return configs

def load_user_config(path="configs/user.yaml"):
    with open(path, 'r', encoding='utf-8') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    return configs