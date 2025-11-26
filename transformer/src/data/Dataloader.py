from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets.translation import Multi30k
import torchtext
import spacy
import os
import ssl

'''
请确保已安装 spacy 及对应的语言模型，例如：
    pip install spacy
    python -m spacy download en_core_web_sm
    python -m spacy download de_core_news_sm（这个更好：conda install -c conda-forge spacy-model-de_core_news_sm）
    用conda下载更好，一定要注意版本；GPU用3090不然会版本冲突
    服务器上最好本地下载,再上传到服务器，再安装
    英语模型：https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0-py3-none-any.whl
    德语模型：https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.7.0/de_core_news_sm-3.7.0-py3-none-any.whl
输入输出说明：
--------------
输入：
    - text: 原始文本字符串
    - lang: 语言代码（如 "en", "zh", "fr"）
输出：
    - token_list: 分词结果（List[str]）
'''
class Tokenizer:
    def __init__(self):
        """
        初始化 tokenizer,加载所需的spacy模型
        """
        try:
            self.spacy_de = spacy.load('de_core_news_sm')
        except OSError:
            raise RuntimeError("请先运行：'python -m spacy download de_core_news_sm")

        try:
            self.spacy_en = spacy.load('en_core_web_sm')
        except OSError:
            raise RuntimeError("请先运行：'python -m spacy download en_core_web_sm")

    def tokenize_de(self, text: str) -> list:
        return [tok.text for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text: str) -> list:
        return [tok.text for tok in self.spacy_en.tokenizer(text)]


class Dataset:
    source: Field = None
    target: Field = None
    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        self.ext = ext  # 扩展名（如 ".de" 或 ".en"）
        self.tokenize_en = tokenize_en  #英语分词函数
        self.tokenize_de = tokenize_de  #德语分词函数
        self.init_token = init_token  #初始化 token（如 "<sos>"）
        self.eos_token = eos_token  #结束 token（如 "<eos>"）
        print(f"数据集开始初始化...")
        current_dir = os.path.dirname(os.path.abspath(__file__))  #Dataloader.py的父目录，即data包
        self.root_path = os.path.join(current_dir, '.data')  # 这样得到的是 data/.data

    def make_dataset(self):
        if self.ext == ('.de', '.en'):
            self.source = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
                        lower=True, batch_first=True)
            self.target = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
                        lower=True, batch_first=True)

        elif self.ext == ('.en', '.de'):
            self.source = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
                        lower=True, batch_first=True)
            self.target = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
                        lower=True, batch_first=True)
        #数据很难下载，有本地文件他就不会下载了直接使用本地文件
        # 禁用下载，直接使用本地文件

        try:
            train_data, valid_data, test_data = Multi30k.splits(
                exts=self.ext,
                fields=(self.source, self.target),
                root=self.root_path
            )
            print("本地数据加载成功！")
            return train_data, valid_data, test_data
        except Exception as e:
            raise RuntimeError(f"数据加载失败: {e}")

    def build_vocab(self, train_data, min_freq):
        """
        构建词汇表
        train_data: torchtext.datasets.TranslationDataset
        min_freq (int): 最小词频（默认为2）
        """
        self.source.build_vocab(train_data, min_freq=min_freq)
        self.target.build_vocab(train_data, min_freq=min_freq)

    def make_iter(self, train, validate, test, batch_size, device):
        """
         创建迭代器
         train: torchtext.datasets.TranslationDataset
         validate: torchtext.datasets.TranslationDataset
         test: torchtext.datasets.TranslationDataset
         """
        train_iter, valid_iter, test_iter = BucketIterator.splits(
            (train, validate, test),
            batch_size=batch_size,
            device=device)
        print(f"数据集初始化结束...")
        return train_iter, valid_iter, test_iter

if __name__ == "__main__":
    print("💬 Tokenizer demo:")
    tokenizer =Tokenizer()
    de_sentence = "Ich liebe natürliche Sprachverarbeitung."
    en_sentence = "I love natural language processing."
    print("Input:", de_sentence)
    print("Tokens:", tokenizer.tokenize_de(de_sentence))
    print("Input:", en_sentence)
    print("Tokens:", tokenizer.tokenize_en(en_sentence))
    print("Tokenizer demo finished.")