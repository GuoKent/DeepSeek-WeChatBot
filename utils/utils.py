import os
import re
import yaml
import jieba

STOPWORDS_PATH = "configs/stopwords.txt"

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

def load_stopwords(stopwords_path=STOPWORDS_PATH):
    '''
    加载停用词
    '''
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        return set([line.strip() for line in f])
    
def remove_stopwords(text, stopwords: set=None):
    '''
    去除停用词
    '''
    if stopwords is None:
        stopwords = load_stopwords()
    words = jieba.cut(text)
    return ' '.join([word for word in words if word not in stopwords])

def extract_name(text, name_set=None):
    """
    提取文本中包含在集合内的关键词
    """
    if name_set is None:
        name_list = os.listdir("pre_info")
        name_set = set([file.split('.')[0] for file in name_list])
        # print(f"name set: {name_set}")
    # 转义特殊字符并构建正则模式
    escaped = [re.escape(abbr) for abbr in name_set]
    pattern = r'\b(' + '|'.join(escaped) + r')\b'
    # 忽略大小写，确保匹配所有变体
    regex = re.compile(pattern, flags=re.IGNORECASE)
    # 提取所有匹配项
    matches = regex.findall(text)
    # 去重并统一为小写（若需保留原格式，移除 lower()）
    seen = set()
    result = []
    for match in matches:
        lower_match = match.lower()
        if lower_match not in seen:
            seen.add(lower_match)
            result.append(lower_match)
    return result

def text_to_vector(model, text):
    # model = SentenceTransformer('sbert-base-chinese-nli')
    return model.encode(text)

def load_search_file(file_name):
    try:
        with open(f"pre_info/{file_name}.txt", 'r', encoding='utf-8') as fin:
            pre_info = fin.read()
        return pre_info
    except:
        return ""

def load_preinfo(file_names):
    res = ""
    for name in file_names:
        res = res + load_search_file(name) + "\n"
    return res