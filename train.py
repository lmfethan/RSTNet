import random
import evaluation
from evaluation import PTBTokenizer, Cider
from data.tokenizers import Tokenizer
from data.medicalDataloaders import R2DataLoader
from models.visual_extractor import VisualExtractor
from models.rstnet import Transformer, TransformerEncoder, TransformerDecoderLayer, ScaledDotProductAttention

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import pickle
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def evaluate_loss(model, visual_extractor, dataloader, loss_fn, tokenizer, device):
    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, (images_id, images, captions, reports_masks) in enumerate(dataloader):
                images, captions, reports_masks = images.to(device), captions.to(device), reports_masks.to(device)
                features = visual_extractor(images)
                out = model(features, captions)
                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                loss = loss_fn(out.view(-1, tokenizer.get_vocab_size()), captions.view(-1))
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss


def evaluate_metrics(model, visual_extractor, dataloader, tokenizer, args, device):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader)) as pbar:
         for it, (images_id, images, captions, reports_masks) in enumerate(dataloader):
            images, captions, reports_masks = images.to(device), captions.to(device), reports_masks.to(device)
            with torch.no_grad():
                features = visual_extractor(images)
                out, _ = model.beam_search(features, args.max_seq_length, tokenizer.token2idx['<eos>'], args.beam_size, out_size=1)
            caps_gen = tokenizer.decode_batch(out[..., :-1])
            caps_gt = tokenizer.decode_batch(captions[..., 1:])
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                # gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                # gen['%d_%d' % (it, i)] = [gen_i]
                # gts['%d_%d' % (it, i)] = gts_i
                gts['%d_%d' % (it, i)] = [gts_i, ]
            pbar.update()

    print(gen)
    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores


def train_xe(model, visual_extractor, dataloader, tokenizer, optim, device):
    # Training with cross-entropy
    model.train()
    scheduler.step()
    print('lr = ', optim.state_dict()['param_groups'][0]['lr'])
    
    running_loss = .0

    sum_eos = 0

    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (images_id, images, captions, reports_masks) in enumerate(dataloader_train):
            images, captions, reports_masks = images.to(device), captions.to(device), reports_masks.to(device)
            features = visual_extractor(images)
            out = model(features, captions)
            optim.zero_grad()

            # print('captions', (captions == tokenizer.token2idx['<eos>']).sum())

            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()
            
            sum_eos += int((captions == tokenizer.token2idx['<eos>']).sum())

            loss = loss_fn(out.view(-1, tokenizer.get_vocab_size()), captions_gt.view(-1))
            loss.backward()

            optim.step()
            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
            # scheduler.step()

    loss = running_loss / len(dataloader)

    # scheduler.step()

    return loss, sum_eos

if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument('--exp_name', type=str, default='rstnet')
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'],
                        help='the dataset to be used.')
    parser.add_argument('--logs_folder', type=str, default='language_tensorboard_logs')
    parser.add_argument('--ann_path', type=str, default='../ReportGen/data/iu_xray/annotation.json',
                        help='the path to the directory containing the data.')
    parser.add_argument('--image_dir', type=str, default='../ReportGen/data/iu_xray/images/',
                        help='the path to the directory containing the data.')
    # parser.add_argument('--ann_path', type=str, default='../datasets/mimic_cxr/annotation.json',
    #                     help='the path to the directory containing the data.')
    # parser.add_argument('--image_dir', type=str, default='../datasets/mimic_cxr/images/',
    #                     help='the path to the directory containing the data.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')

    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')

    parser.add_argument('--lr_model', type=float, default=0.5, help='the learning rate for the transformer.')
    parser.add_argument('--lr_ve', type=float, default=0.25, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_bert', type=float, default=1, help='the learning rate for BERT.')

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--patience', type=int, default=100)

    parser.add_argument('--xe_base_lr', type=float, default=0.0001)

    args = parser.parse_args()
    print(args)

    print('Transformer Training')
    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name))

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # Create the dataset
    tokenizer = Tokenizer(args)
    dataloader_train = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    dataloader_val = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    dataloader_test = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    # Model and dataloaders
    encoder = TransformerEncoder(3, tokenizer.token2idx['<pad>'], attention_module=ScaledDotProductAttention, attention_module_kwargs={'m': args.m})
    decoder = TransformerDecoderLayer(tokenizer.get_vocab_size(), args.max_seq_length - 1, 3, tokenizer.token2idx['<pad>'])
    ve = VisualExtractor(args).to(device)
    model = Transformer(tokenizer.token2idx['<bos>'], encoder, decoder).to(device)

    # ref_caps_train = list(train_dataset.text)
    # cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))

    def lambda_lr(s):
        print("s:", s)
        if s <= 3:
            lr = args.xe_base_lr * s / 4
        elif s <= 10:
            lr = args.xe_base_lr
        elif s <= 12:
            lr = args.xe_base_lr * 0.2
        else:
            lr = args.xe_base_lr * 0.2 * 0.2
        return lr

    # Initial conditions

    bert_params = list(map(id, model.decoder.language_model.parameters()))
    ed_params = filter(lambda x: id(x) not in bert_params, model.parameters())
    optim = Adam([
        {'params': model.decoder.language_model.parameters(), 'lr': args.lr_bert},
        {'params': ed_params, 'lr': args.lr_model, 'betas': (0.9, 0.98)},
        {'params': ve.parameters(), 'lr': args.lr_ve}
    ])
    scheduler = LambdaLR(optim, lambda_lr)

    loss_fn = NLLLoss(ignore_index=tokenizer.token2idx['<pad>'])
    # best_cider = .0
    # best_test_cider = 0.
    use_rl = False
    best_bleu4 = .0
    best_test_bleu4 = 0.
    patience = 0
    start_epoch = 0

    if args.resume_last or args.resume_best:
        if args.resume_last:
            fname = 'saved_transformer_models/%s_last.pth' % args.exp_name
        else:
            fname = 'saved_transformer_models/%s_best.pth' % args.exp_name

        if os.path.exists(fname):
            data = torch.load(fname)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['tr_state_dict'])
            ve.load_state_dict(data['ve_state_dict'])
            """
            optim.load_state_dict(data['optimizer'])
            scheduler.load_state_dict(data['scheduler'])
            """
            start_epoch = data['epoch'] + 1
            best_bleu4 = data['best_bleu4']
            best_test_bleu4 = data['best_test_bleu4']
            patience = data['patience']

            optim.load_state_dict(data['optimizer'])
            scheduler.load_state_dict(data['scheduler'])

            print('Resuming from epoch %d, validation loss %f, best bleu4 %f, and best_test_bleu4 %f' % (
                data['epoch'], data['val_loss'], data['best_bleu4'], data['best_test_bleu4']))
            print('patience:', data['patience'])

    print("Training starts")
    for e in range(start_epoch, start_epoch + args.epochs):

        train_loss, sum_eos = train_xe(model, ve, dataloader_train, tokenizer, optim, device)
        writer.add_scalar('data/train_loss', train_loss, e)

        # Validation loss
        val_loss = evaluate_loss(model, ve, dataloader_val, loss_fn, tokenizer, device)
        writer.add_scalar('data/val_loss', val_loss, e)

        # Validation scores
        scores = evaluate_metrics(model, ve, dataloader_val, tokenizer, args, device)
        print("Validation scores", scores)
        val_bleu4 = scores['BLEU'][3]
        writer.add_scalar('data/val_cider', scores['CIDEr'], e)
        writer.add_scalar('data/val_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/val_bleu2', scores['BLEU'][1], e)
        writer.add_scalar('data/val_bleu3', scores['BLEU'][2], e)
        writer.add_scalar('data/val_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/val_meteor', scores['METEOR'], e)
        writer.add_scalar('data/val_rouge', scores['ROUGE'], e)

        # Test scores
        scores = evaluate_metrics(model, ve, dataloader_test, tokenizer, args, device)
        print("Test scores", scores)
        test_bleu4 = scores['BLEU'][3]
        writer.add_scalar('data/test_cider', scores['CIDEr'], e)
        writer.add_scalar('data/test_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/test_bleu2', scores['BLEU'][1], e)
        writer.add_scalar('data/test_bleu3', scores['BLEU'][2], e)
        writer.add_scalar('data/test_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/test_meteor', scores['METEOR'], e)
        writer.add_scalar('data/test_rouge', scores['ROUGE'], e)

        # Prepare for next epoch
        best = False
        if val_bleu4 >= best_bleu4:
            best_bleu4 = val_bleu4
            patience = 0
            best = True
        else:
            patience += 1

        best_test = False
        if test_bleu4 >= best_test_bleu4:
            best_test_bleu4 = test_bleu4
            best_test = True

        exit_train = False

        if patience == args.patience:
            print('patience reached.')
            exit_train = True
        
        torch.save({
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': e,
            'val_loss': val_loss,
            'val_bleu4': val_bleu4,
            'tr_state_dict': model.state_dict(),
            've_state_dict': ve.state_dict(),
            'optimizer': optim.state_dict() if not use_rl else optim_rl.state_dict(),
            'scheduler': scheduler.state_dict() if not use_rl else scheduler_rl.state_dict(),
            'patience': patience,
            'best_bleu4': best_bleu4,
            'best_test_bleu4': best_test_bleu4,
            'use_rl': use_rl,
        }, 'saved_transformer_models/%s_last.pth' % args.exp_name)

        if best:
            copyfile('saved_transformer_models/%s_last.pth' % args.exp_name, 'saved_transformer_models/%s_best.pth' % args.exp_name)
        if best_test:
            copyfile('saved_transformer_models/%s_last.pth' % args.exp_name, 'saved_transformer_models/%s_best_test.pth' % args.exp_name)


        if exit_train:
            writer.close()
            break
