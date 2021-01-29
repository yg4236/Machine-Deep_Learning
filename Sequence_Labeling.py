import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data
from torchtext import datasets
import time
import random

from torchtext.data import batch

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda')

# 3개의 필드 정의
TEXT = data.Field(lower = True)
UD_TAGS = data.Field(unk_token = None)
PTB_TAGS = data.Field(unk_token = None)

fields = (("text", TEXT), ("udtags", UD_TAGS), ("ptbtags", PTB_TAGS))

# 데이터 셋 만들기
train_data, valid_data, test_data = datasets.UDPOS.splits(fields)
# 단어 집합 만들기 ( GloVe 사용 )
# 최소 허용 빈도
MIN_FREQ = 5

# 사전 훈련된 워드 임베딩 GloVe 다운로드
TEXT.build_vocab(train_data, min_freq = MIN_FREQ, vectors = "glove.6B.100d")
UD_TAGS.build_vocab(train_data)
PTB_TAGS.build_vocab(train_data)

def tag_percentage(tag_counts): # 태그 레이블의 분포를 확인하는 함수
    total_count = sum([count for tag, count in tag_counts])
    tag_counts_percentages = [(tag, count, count/total_count) for tag, count in tag_counts]

    return tag_counts_percentages

# 데이터 로더 만들기
BATCH_SIZE = 64

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    device = device)

# 모델 구현하기
class RNNPOSTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers = n_layers, bidirectional = bidirectional)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [sent len, batch size]
        embedded = self.dropout(self.embedding(text))

        # embedded = [sent len, batch size, emb dim]
        outputs, (hidden, cell) = self.rnn(embedded)

        # output = [sent len, batch size, hid dim * n directions]
        # hidden/cell = [n layers * n directions, batch size, hid dim]
        predictions = self.fc(self.dropout(outputs))

        # predictions = [sent len, batch size, output dim]
        return predictions
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM = len(UD_TAGS.vocab)
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25

model = RNNPOSTagger(INPUT_DIM,
                     EMBEDDING_DIM,
                     HIDDEN_DIM,
                     OUTPUT_DIM,
                     N_LAYERS,
                     BIDIRECTIONAL,
                     DROPOUT)

# 사전 훈련된 워드 임베딩 사용
pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings) # 임베딩 벡터값 copy
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM) # 0번 임베딩 벡터에는 0값을 채운다.
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM) # 1번 임베딩 벡터에는 1값을 채운다.

# 옵티마이저와 비용함수 구현
TAG_PAD_IDX = UD_TAGS.vocab.stoi[UD_TAGS.pad_token]
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index = TAG_PAD_IDX) # 패딩토큰은 비용함수의 연산에 포함X

model = model.to(device)
criterion = criterion.to(device)
prediction = model(batch.text)
prediction = prediction.view(-1, prediction.shape[-1])

print(batch.udtags.view(-1).shape)