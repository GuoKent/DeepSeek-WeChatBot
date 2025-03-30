import os
import sys
import faiss
import numpy as np

sys.path.append("..")
from utils.utils import text_to_vector, extract_name
from sentence_transformers import SentenceTransformer

EMBEDDING_SIZE = 768

class FaissIndexer:
    def __init__(self, bert_path="sbert-base-chinese-nli"):
        self.model = SentenceTransformer(bert_path)
        # self.index = faiss.IndexFlatL2(dimension)
        self.index = None
        self.filenames = []
    
    def create_index(self, file_list):
        """
        创建索引并保存关联数据
        """
        # 生成文件名向量
        file_vectors = self.model.encode(file_list)
        
        # 创建Faiss索引
        self.index = faiss.IndexFlatL2(file_vectors.shape[1])
        self.index.add(file_vectors.astype('float32'))
        self.filenames = file_list

    def add_vectors(self, vectors):
        self.index.add(np.array(vectors))
    
    def search(self, query, top_k=5):
        """
        执行搜索
        Return:
            [(filname, distance), ...]
        """
        query_vector = self.model.encode([query])
        distances, indices = self.index.search(query_vector.astype('float32'), top_k)
        
        return [(self.filenames[idx], float(dist)) for idx, dist in zip(indices[0], distances[0])]
    
    def search_filename(self, query, topk=5):
        """
        按照 List[filename] 形式返回
        Return:
            [filename1, filename2, ...]
        """
        res = self.search(query, top_k=topk)
        # filenames = [item[0] for item in res]
        return [item[0] for item in res]

    def save_index(self, save_dir="index"):
        """
        保存索引和文件名数据
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 保存Faiss索引
        faiss.write_index(self.index, os.path.join(save_dir, "index.faiss"))
        # 保存文件名列表
        np.save(os.path.join(save_dir, "filenames.npy"), self.filenames)

    def load_index(self, save_dir="index"):
        """
        加载已有索引
        """
        # 加载Faiss索引
        self.index = faiss.read_index(os.path.join(save_dir, "index.faiss"))
        # 加载文件名列表
        self.filenames = np.load(os.path.join(save_dir, "filenames.npy"), allow_pickle=True)
    
if __name__ == "__main__":
    # 读取目录下文件名
    file_names = os.listdir("../pre_info/")
    # 去除后缀
    file_names = [file.split('.')[0] for file in file_names]
    name_set = set(file_names)
    # model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    # model = SentenceTransformer('../sbert-base-chinese-nli')
    # vectors = text_to_vector(model, file_names)  # emb=768
    # print(len(vectors[0]))
    
    # 构建索引
    indexer = FaissIndexer(bert_path="../sbert-base-chinese-nli")
    # indexer.create_index(file_names)
    # indexer.save_index(save_dir="../index")
    
    # 加载使用
    indexer.load_index(save_dir="../index")
    query = "锐评一下 zhw 和 lzy "
    names = extract_name(query, name_set)
    print(names)  # ['zhw']
    query_names = ",".join(names)
    print(query_names)  # ['zhw']
    results = indexer.search(query_names)
    print(results)