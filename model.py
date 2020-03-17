import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        decoder_output = self.decode(memory, src_mask, tgt, tgt_mask)
        return decoder_output

    def encode(self, src, src_mask):
        embedded_src = self.src_embed(src)
        return self.encoder(embedded_src, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        # memoryとはencoderの出力のこと
        embedded_tgt = self.tgt_embed(tgt)
        return self.decoder(embedded_tgt, memory, src_mask, tgt_mask)


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        # d_modelはDecoderの出力の次元数
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        # layerにはEncoderLayerを与える
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        # 論文の図とLayerNormを入れる位置が違うが論文と同じにするとlossが下がらない
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.size = size
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 論文の図とLayerNormを入れる位置が違うが論文と同じにするとlossが下がらない
        # sublayer1: self-attention
        input_x = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, mask)
        x = input_x + self.dropout(x)

        # sublayer2: feed forward
        input_x = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = input_x + self.dropout(x)

        return x


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.norm3 = LayerNorm(size)
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        # sublayer1: self-attention
        input_x = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, tgt_mask)
        x = input_x + self.dropout(x)

        # sublayer2: encoder-decoder attention
        # encoder decoder attentionはqueryはdecoderの出力で
        # keyとvalueはencoderの出力を指定
        input_x = x
        x = self.norm2(x)
        x = self.src_attn(x, memory, memory, src_mask)
        x = input_x + self.dropout(x)

        # sublayer3: feed forward
        input_x = x
        x = self.norm3(x)
        x = self.feed_forward(x)
        x = input_x + self.dropout(x)

        return x


def attention(query, key, value, mask=None, dropout=None):
    # scaled dot product attention
    # (query, key, value)の値を変えることでself_attnにもsrc_attnにもなる
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        # d_modelは全ヘッドを集めた次元になっている
        # d_kは各ヘッドの次元
        assert d_model % h == 0
        self.d_k = d_model // h

        # hはヘッドの数
        self.h = h

        # WQ, WK, WV, WOの4つ
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # self-attentionの場合は、query=key=value=xになっている
        # encoder-decoder attentionの場合は、query=x, key=value=memoryになっている
        query = self.wq(query)
        key = self.wk(key)
        value = self.wv(value)

        # ここで各ヘッドごとに独立にattentionを計算してから元のサイズにマージ
        query = query.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = key.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = value.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.wo(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].clone().detach()
        return self.dropout(x)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    # このAttention layerはforwardへのquery, key, valueの与え方に応じて
    # self-attentionにもencoder-decoder attentionにもなる
    attn = MultiHeadAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    # 各レイヤのパラメータは独立なのでdeepcopyが必要
    # encoder, decoder, src_embed, tgt_embed, generator
    encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
    decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)
    src_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
    tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), c(position))
    generator = Generator(d_model, tgt_vocab)

    model = EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator)

    # 重みの初期化
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
