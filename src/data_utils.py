import unicodedata
from torch.utils.data import Dataset
import regex
import torch


def tokenize(text):
    """
    :param str text: Input text
    :return List[str]: List of words
    """
    normalized = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode()
    text = regex.sub('[^a-z ]', ' ', normalized.lower())
    text = text.strip()
    text = regex.split(' +', text)
    return text


class QuestionsDataset(Dataset):
    def __init__(self, dataset, vocab, max_len, size=None, pad_sos=False, pad_eos=False):
        super().__init__()

        self.pad_sos = pad_sos
        if self.pad_sos:
            self.sos_id = vocab['<sos>']
        self.pad_eos = pad_eos
        if self.pad_eos:
            self.eos_id = vocab['<eos>']

        self.vocab = vocab
        self.max_len = max_len
        self.dataset = dataset
        self.size = size

        self.texts = []
        self.tokens = []
        for record in self.dataset:
            if self.size is not None and len(self.texts) == self.size:
                break
            text = record[1].strip()
            tokens = self.vocab.lookup_indices(tokenize(text))
            self.texts.append(text)
            self.tokens.append(tokens)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokens[idx][:(self.max_len - self.pad_sos - self.pad_eos)]
        tokens = ([self.sos_id] if self.pad_sos else []) + tokens + ([self.eos_id] if self.pad_eos else [])
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens_len = torch.tensor(len(tokens), dtype=torch.long)

        return {'text': text, 'tokens': tokens, 'tokens_len': tokens_len}

    def __len__(self):
        return len(self.texts)
