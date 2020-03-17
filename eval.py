import torch
from data import subsequent_mask
from model import make_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


def main():
    V = 11
    model = make_model(V, V, N=2).to(device)
    model.load_state_dict(torch.load('checkpoint/model.pt'))
    model.to(device)
    model.eval()

    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).to(device)
    src_mask = torch.ones(1, 1, 10).to(device)

    result = greedy_decode(model, src, src_mask, max_len=10, start_symbol=1)
    print(result)


if __name__ == "__main__":
    main()
