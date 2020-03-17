import numpy as np
import torch


def subsequent_mask(size):
    # Decoderにおいて未来の出力を見ないようにするためのマスク
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    def __init__(self, src, tgt=None, pad=0):
        self.src = src  # [batch, seqlen]
        # padのところがFalseになるmaskを作成 => mask=Trueのところが有効
        self.src_mask = (src != pad).unsqueeze(-2)  # [batch, 1, seqlen]
        if tgt is not None:
            # tgtはteacher forcingのためのdecoderへの入力
            # 最後の要素は次の値がないので削除する
            self.tgt = tgt[:, :-1]
            # decoderは次の値を予測したいので1つずらす
            self.tgt_y = tgt[:, 1:]
            # 未来の値を使わないようにするmask [batch, seqlen-1, seqlen-1]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


def data_gen(V, batch_size, nbatches):
    for i in range(nbatches):
        # 1からV未満のランダムな数字を10個生成
        data = torch.from_numpy(np.random.randint(1, V, size=(batch_size, 10)))
        # 系列の最初の数字は1に固定
        data[:, 0] = 1
        # 入力と出力は全く同じ数字系列を出すように訓練する
        src = data.clone().detach()
        tgt = data.clone().detach()
        yield Batch(src, tgt, 0)
