import torch
from torch import nn
import copy
from models.containers import ModuleList
from ..captioning_model import CaptioningModel
# from models.rstnet.grid_aug import PositionEmbeddingSine
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)



class Transformer(CaptioningModel):
    def __init__(self, bos_idx, encoder, decoder):
        super(Transformer, self).__init__()
        self.bos_idx = bos_idx
        self.encoder = encoder
        self.decoder = decoder
        self.grid_embedding = PositionalEncoding(self.decoder.d_model, 0.1)

        self.fc = nn.Linear(2048, self.decoder.d_model)
        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)
        self.init_weights()

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def att_embed(self, feats):
        # feats = self.bn1(feats)
        feats = self.fc(feats)
        feats = self.relu(feats)
        feats = self.dropout(feats)
        # feats = self.bn2(feats)
        return feats

    def get_pos_embedding(self, grids):
        grids = self.att_embed(grids)
        grid_embed = self.grid_embedding(grids)
        return grid_embed

    def forward(self, images, seq, *args):
        grid_embed = self.get_pos_embedding(images)
        enc_output, mask_enc = self.encoder(images, pos=grid_embed)
        dec_output = self.decoder(seq, enc_output, mask_enc, pos=grid_embed)
        return dec_output

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                self.grid_embed = self.get_pos_embedding(visual)
                self.enc_output, self.mask_enc = self.encoder(visual, pos=self.grid_embed)

                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()
            else:
                it = prev_output

        return self.decoder(it, self.enc_output, self.mask_enc, pos=self.grid_embed)


class TransformerEnsemble(CaptioningModel):
    def __init__(self, model: Transformer, weight_files):
        super(TransformerEnsemble, self).__init__()
        self.n = len(weight_files)
        self.models = ModuleList([copy.deepcopy(model) for _ in range(self.n)])
        for i in range(self.n):
            state_dict_i = torch.load(weight_files[i])['state_dict']
            self.models[i].load_state_dict(state_dict_i)

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        out_ensemble = []
        for i in range(self.n):
            out_i = self.models[i].step(t, prev_output, visual, seq, mode, **kwargs)
            out_ensemble.append(out_i.unsqueeze(0))

        return torch.mean(torch.cat(out_ensemble, 0), dim=0)
