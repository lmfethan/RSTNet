#!/usr/bin/env python
# coding: utf-8

# In[22]:


import torch
import torch.nn as nn
import numpy as np
from models.rstnet import Transformer, TransformerEncoder, TransformerDecoderLayer, ScaledDotProductAttention
from data.tokenizers import Tokenizer
from data.medicalDataloaders import R2DataLoader
from models.visual_extractor import VisualExtractor
import argparse
import evaluation


# In[2]:


sd = torch.load('/GPUFS/nsccgz_ywang_zfd/limengfei/RSTNet/saved_transformer_models/rstnet_250.pth')


# In[12]:


class args():
    def __init__(self):
        self.image_dir = '../ReportGen/data/iu_xray/images/'
        self.ann_path = '../ReportGen/data/iu_xray/annotation.json'

        self.dataset_name = 'iu_xray'
        self.max_seq_length = 60
        self.threshold = 3
        self.num_workers = 2
        self.batch_size = 8

        self.visual_extractor = 'resnet101'
        self.visual_extractor_pretrained = True

        self.d_model = 512
        self.d_ff = 512
        self.d_vf = 2048
        self.num_heads = 8
        self.num_layers = 3
        self.dropout = 0.1
        self.logit_layers = 1
        self.bos_idx = 0
        self.eos_idx = 0
        self.pad_idx = 0
        self.use_bn = 0
        self.drop_prob_lm = 0.5
      
        self.img_size = 224
        self.patch_size = 32
        self.num_channels = 3
        self.embedding_dim = 512
        self.num_heads_tnt = 8
        self.num_layers_tnt = 12
        self.hidden_dim = 512*4
        self.stride = 4
        self.num_class = 761

        self.sample_method = 'beam_search'
        self.beam_size = 3
        self.temperature = 1.0
        self.sample_n = 1
        self.group_size = 1
        self.output_logsoftmax = 1
        self.decoding_constraint = 0
        self.block_trigrams = 1

        self.n_gpu = 1
        self.epochs = 100
        self.save_dir = 'results/iu_xray/'
        self.record_dir = 'records/'
        self.save_period = 1
        self.monitor_mode = 'max'
        self.monitor_metric = 'BLEU_4'
        self.early_stop = 50

        self.optim = 'Adam'
        self.lr_ed = 1e-4
        self.weight_decay = 5e-5
        self.amsgrad = True

        self.lr_scheduler = 'StepLR'
        self.step_size = 50
        self.gamma = 0.1

        self.seed = 9223
        self.resume = True
arg = args()


# In[13]:


tokenizer = Tokenizer(arg)
dataloader_test = R2DataLoader(arg, tokenizer, split='test', shuffle=False)


# In[21]:


device = torch.device('cuda')
encoder = TransformerEncoder(3, 0, attention_module=ScaledDotProductAttention, attention_module_kwargs={'m': 40})
decoder = TransformerDecoderLayer(tokenizer.get_vocab_size(), 59, 3, tokenizer.token2idx['<pad>'])
ve = VisualExtractor(arg).to(device)
model = Transformer(tokenizer.token2idx['<bos>'], encoder, decoder).to(device)
model.load_state_dict(sd['tr_state_dict'])
ve.load_state_dict(sd['ve_state_dict'])


# In[24]:


model.eval()
gen = {}
gts = {}
for it, (images_id, images, captions, reports_masks) in enumerate(dataloader_test):
    images, captions, reports_masks = images.to(device), captions.to(device), reports_masks.to(device)
    with torch.no_grad():
        features = ve(images)
        out, _ = model.beam_search(features, arg.max_seq_length, tokenizer.token2idx['<eos>'], arg.beam_size, out_size=1)
    caps_gen = tokenizer.decode_batch(out[..., :-1])
    caps_gt = tokenizer.decode_batch(captions[..., 1:])
    for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
        # gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
        gen['%d_%d' % (it, i)] = [gen_i, ]
        # gen['%d_%d' % (it, i)] = [gen_i]
        # gts['%d_%d' % (it, i)] = gts_i
        gts['%d_%d' % (it, i)] = [gts_i, ]
gts = evaluation.PTBTokenizer.tokenize(gts)
gen = evaluation.PTBTokenizer.tokenize(gen)
scores, _ = evaluation.compute_scores(gts, gen)


# In[ ]:


print(scores)

