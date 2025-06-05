import torch.nn as nn


class Seq2SeqLSTM(nn.Module):
    def __init__(self, vocab_sz, emb_sz, hid_sz):
        super().__init__()
        self.embed   = nn.Embedding(vocab_sz, emb_sz, padding_idx=0)
        self.encoder = nn.LSTM(emb_sz, hid_sz, batch_first=True)
        self.decoder = nn.LSTM(emb_sz, hid_sz, batch_first=True)
        self.out     = nn.Linear(hid_sz, vocab_sz)

    def forward(self, src, tgt_in):
        emb_src = self.embed(src)
        _, (h,c) = self.encoder(emb_src)
        emb_tgt = self.embed(tgt_in)
        dec_out, _ = self.decoder(emb_tgt, (h,c))
        return self.out(dec_out)

class Seq2SeqGRU(nn.Module):
    def __init__(self, vocab_sz, emb_sz, hid_sz):
        super().__init__()
        self.embed   = nn.Embedding(vocab_sz, emb_sz, padding_idx=0)
        self.encoder = nn.GRU(emb_sz, hid_sz, batch_first=True)
        self.decoder = nn.GRU(emb_sz, hid_sz, batch_first=True)
        self.out     = nn.Linear(hid_sz, vocab_sz)

    def forward(self, src, tgt_in):
        emb_src = self.embed(src)
        _, h = self.encoder(emb_src)
        emb_tgt = self.embed(tgt_in)
        dec_out, _ = self.decoder(emb_tgt, h)
        return self.out(dec_out)
