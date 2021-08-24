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
    print('lr = ', optim.state_dict()['param_groups'][0]['lr'])
    
    running_loss = .0

    sum_eos = 0

    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (images_id, images, captions, reports_masks) in enumerate(dataloader_train):
            images, captions, reports_masks = images.to(device), captions.to(device), reports_masks.to(device)
            print(captions)
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

    scheduler.step()

    return loss, sum_eos


# def train_scst(model, dataloader, optim, cider, text_field):
#     # Training with self-critical
#     tokenizer_pool = multiprocessing.Pool()
#     running_reward = .0
#     running_reward_baseline = .0

#     model.train()
#     scheduler_rl.step()
#     print('lr = ', optim_rl.state_dict()['param_groups'][0]['lr'])

#     running_loss = .0
#     seq_len = 20
#     beam_size = 5

#     with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
#         for it, (detections, caps_gt) in enumerate(dataloader):
#             detections = detections.to(device)
#             outs, log_probs = model.beam_search(detections, seq_len, text_field.vocab.stoi['<eos>'],
#                                                 beam_size, out_size=beam_size)
#             optim.zero_grad()

#             # Rewards
#             caps_gen = text_field.decode(outs.view(-1, seq_len))
#             caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt)))
#             caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
#             reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
#             reward = torch.from_numpy(reward).to(device).view(detections.shape[0], beam_size)
#             reward_baseline = torch.mean(reward, -1, keepdim=True)
#             loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)

#             loss = loss.mean()
#             loss.backward()
#             optim.step()

#             running_loss += loss.item()
#             running_reward += reward.mean().item()
#             running_reward_baseline += reward_baseline.mean().item()
#             pbar.set_postfix(loss=running_loss / (it + 1), reward=running_reward / (it + 1),
#                              reward_baseline=running_reward_baseline / (it + 1))
#             pbar.update()

#     loss = running_loss / len(dataloader)
#     reward = running_reward / len(dataloader)
#     reward_baseline = running_reward_baseline / len(dataloader)
#     return loss, reward, reward_baseline


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

    parser.add_argument('--lr_model', type=float, default=1e-4, help='the learning rate for the transformer.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')

    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--patience', type=int, default=1000)

    parser.add_argument('--xe_base_lr', type=float, default=0.0001)
    parser.add_argument('--rl_base_lr', type=float, default=5e-6)

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

    '''
    def lambda_lr(s):
        warm_up = args.warmup
        s += 1
        return (model.d_model ** -.5) * min(s ** -.5, s * warm_up ** -1.5)
    '''

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
    
    # def lambda_lr_rl(s):
    #     refine_epoch = args.refine_epoch_rl 
    #     print("rl_s:", s)
    #     if s <= refine_epoch:
    #         lr = args.rl_base_lr
    #     elif s <= refine_epoch + 3:
    #         lr = args.rl_base_lr * 0.2
    #     elif s <= refine_epoch + 6:
    #         lr = args.rl_base_lr * 0.2 * 0.2
    #     else:
    #         lr = args.rl_base_lr * 0.2 * 0.2 * 0.2
    #     return lr


    # Initial conditions
    # optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    optim = Adam([
        {'params': model.parameters(), 'lr': args.lr_model, 'betas': (0.9, 0.98)},
        {'params': ve.parameters(), 'lr': args.lr_ve}
    ])
    # scheduler = LambdaLR(optim, lambda_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, 10, 0.8)
    # optim_ve = Adam(ve.parameters(), lr=5e-5)
    # scheduler_ve = torch.optim.lr_scheduler.StepLR(optim_ve, args.step_size, args.gamma)

    # optim_rl = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    # scheduler_rl = LambdaLR(optim_rl, lambda_lr_rl)

    loss_fn = NLLLoss(ignore_index=tokenizer.token2idx['<pad>'])
    use_rl = False
    # best_cider = .0
    # best_test_cider = 0.
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
        
        # if not use_rl:
        #     train_loss = train_xe(model, dataloader_train, optim, text_field)
        #     writer.add_scalar('data/train_loss', train_loss, e)
        # else:
        #     train_loss, reward, reward_baseline = train_scst(model, dict_dataloader_train, optim_rl, cider_train, text_field)
        #     writer.add_scalar('data/train_loss', train_loss, e)
        #     writer.add_scalar('data/reward', reward, e)
        #     writer.add_scalar('data/reward_baseline', reward_baseline, e)
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

        # switch_to_rl = False
        exit_train = False

        # if patience == 5:
        #     if e < args.xe_least:   # xe stage train 15 epoches at least 
        #         print('special treatment, e = {}'.format(e))
        #         use_rl = False
        #         switch_to_rl = False
        #         patience = 0
        #     elif not use_rl:
        #         use_rl = True
        #         switch_to_rl = True
        #         patience = 0
                
        #         optim_rl = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
        #         scheduler_rl = LambdaLR(optim_rl, lambda_lr_rl)
                
        #         for k in range(e-1):
        #             scheduler_rl.step()

        #         print("Switching to RL")
        #     else:
        #         print('patience reached.')
        #         exit_train = True

        # if e == args.xe_most:     # xe stage no more than 20 epoches
        #     if not use_rl:
        #         use_rl = True
        #         switch_to_rl = True
        #         patience = 0
                
        #         optim_rl = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
        #         scheduler_rl = LambdaLR(optim_rl, lambda_lr_rl)

        #         for k in range(e-1):
        #             scheduler_rl.step()

        #         print("Switching to RL")

        if patience == args.patience:
            print('patience reached.')
            exit_train = True

        # if switch_to_rl and not best:
        #     data = torch.load('saved_transformer_models/%s_best.pth' % args.exp_name)
        #     torch.set_rng_state(data['torch_rng_state'])
        #     torch.cuda.set_rng_state(data['cuda_rng_state'])
        #     np.random.set_state(data['numpy_rng_state'])
        #     random.setstate(data['random_rng_state'])
        #     model.load_state_dict(data['state_dict'])
        #     print('Resuming from epoch %d, validation loss %f, best_cider %f, and best test_cider %f' % (
        #         data['epoch'], data['val_loss'], data['best_cider'], data['best_test_cider']))

        
        if e % 10 == 0 or best or best_test:
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
            if e > 50:
                copyfile('saved_transformer_models/%s_last.pth' % args.exp_name, 'saved_transformer_models/{}_{}.pth'.format(args.exp_name, e))

        if exit_train:
            writer.close()
            break
