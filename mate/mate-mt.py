#!/usr/bin/env python

import sys
import h5py
import argparse
from time import time
from pprint import pprint
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import xavier_uniform
from torch.nn.utils import clip_grad_norm
from numpy.random import permutation, seed

from mate import AttentionEncoder
from loss import TripletMarginCosineLoss, OrthogonalityLoss

class MultitaskAutoencoder(nn.Module):
    """The multitasking version of our Multitask Aspect Extractor.
    """
    def __init__(self, vocab_size, emb_size, num_aspects=10, neg_samples=10,
            w_emb=None, a_emb=None, recon_method='start', seed_w=None, num_seeds=None,
            attention=False, bias=True, M=None, b=None, fix_w_emb=True, fix_a_emb=False,
            num_domains=6, midlayer=50, dropout=0.0):
        """Initializes the multitasking autoencoder instance.

        Parameters:
            vocab_size (int): the vocabulary size
            emb_size (int): the embedding dimensionality
            num_aspects (int): the number of aspects
            neg_samples (int): the number of negative examples to use for the 
                               max-margin loss
            w_emb (matrix): a pre-trained embeddings matrix (None for random)
            a_emb (matrix): a pre-trained aspect matrix (None for random)
            recon_method (str): the segment reconstruction policy
                                - 'centr': uses centroid of seed words or single embeddings (ABAE)
                                - 'init': uses manually initialized seed weights
                                - 'fix': uses manually initialized seed weights, fixed during training
                                - 'cos': uses dynamic seed weights, obtained from cosine distance
            seed_w (matrix): seed weight matrix (for 'init' and 'fix')
            num_seeds (int): number of seed words
            attention (bool): use attention or not
            bias (bool): use bias vector for attention encoder
            M (matrix): matrix for attention encoder (optional)
            b (vector): bias vector for attention encoder (optional)
            fix_w_emb (bool): fix word embeddings throughout trainign
            fix_a_emb (bool): fix aspect embeddings throughout trainign
            num_domains (int): number of product domains
            midlayer (int): size of middle layer in domain classifier
            dropout (float): dropout probability for domain classifier
        """
        super(MultitaskAutoencoder, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.recon_method = recon_method
        self.num_seeds = num_seeds
        self.attention = attention
        self.bias = bias
        self.num_aspects = num_aspects
        self.neg_samples = neg_samples

        if not attention:
            self.seg_encoder = nn.EmbeddingBag(vocab_size, emb_size)
        else:
            self.seg_encoder = AttentionEncoder(vocab_size, emb_size, bias, M, b)
        self.encoder_hook = self.seg_encoder.register_forward_hook(self.set_targets)

        if w_emb is None:
            xavier_uniform(self.seg_encoder.weight.data)
        else:
            assert w_emb.size() == (vocab_size, emb_size), "Word embedding matrix has incorrect size"
            if not attention:
                self.seg_encoder.weight.data.copy_(w_emb)
                self.seg_encoder.weight.requires_grad = not fix_w_emb
            else:
                self.seg_encoder.set_word_embeddings(w_emb, fix_w_emb)

        if a_emb is None:
            self.a_emb = nn.Parameter(torch.Tensor(num_aspects, emb_size))
            xavier_uniform(self.a_emb.data)
        else:
            assert a_emb.size()[0] == num_aspects and a_emb.size()[-1] == emb_size, "Aspect embedding matrix has incorrect size"
            self.a_emb = nn.Parameter(torch.Tensor(a_emb.size()))
            self.a_emb.data.copy_(a_emb)
            self.a_emb.requires_grad = not fix_a_emb

        if recon_method == 'fix':
            self.seed_w = nn.Parameter(torch.Tensor(seed_w.size()))
            self.seed_w.data.copy_(seed_w)
            self.seed_w.requires_grad = False
        elif recon_method == 'init':
            self.seed_w = nn.Parameter(torch.Tensor(seed_w.size()))
            self.seed_w.data.copy_(seed_w)
        else:
            self.seed_w = None

        # autoencoder
        self.lin = nn.Linear(emb_size, num_aspects)
        self.softmax = nn.Softmax(dim=1)

        # domain classifier
        if dropout > 0:
            self.drop = nn.Dropout(p=dropout)
        else:
            self.drop = None

        self.dlin1 = nn.Linear(emb_size, midlayer)
        self.dlin2 = nn.Linear(midlayer, num_domains)

    def forward(self, inputs, domain_mask=None, batch_num=None):
        if self.training:
            # mask used for randomly selected negative examples
            self.cur_neg_mask = self._create_neg_mask(inputs.size(0))

            assert domain_mask is not None, 'When training, you must forward a domain mask'
            self.cur_domain_mask = domain_mask

        if not self.attention:
            offsets = Variable(torch.arange(0, inputs.numel(), inputs.size(1), out=inputs.data.new().long()))
            enc = self.seg_encoder(inputs.view(-1), offsets)
        else:
            enc = self.seg_encoder(inputs)

        # autoencoder
        x = self.lin(enc)
        a_probs = self.softmax(x)

        if self.recon_method == 'centr':
            r = a_probs.matmul(self.a_emb)
        elif self.recon_method == 'fix':
            a_emb_w = self.a_emb.mul(self.seed_w.view(self.num_aspects, self.num_seeds, 1))
            r = a_probs.view(-1, self.num_aspects, 1, 1).mul(a_emb_w).sum(dim=2).sum(dim=1)
        elif self.recon_method == 'init' or self.recon_method == 'rand':
            seed_w_norm = F.softmax(self.seed_w, dim=1)
            a_emb_w = self.a_emb.mul(seed_w_norm.view(self.num_aspects, self.num_seeds, 1))
            r = a_probs.view(-1, self.num_aspects, 1, 1).mul(a_emb_w).sum(dim=2).sum(dim=1)
        elif self.recon_method == 'cos':
            sim = F.cosine_similarity(enc.unsqueeze(1),
                    self.a_emb.view(1, self.num_aspects*self.num_seeds, self.emb_size),
                    dim=2).view(-1, self.num_aspects, self.num_seeds)
            self.seed_w = F.softmax(sim, dim=2)
            a_emb_w = self.a_emb.mul(self.seed_w.view(-1, self.num_aspects, self.num_seeds, 1))
            r = a_probs.view(-1, self.num_aspects, 1, 1).mul(a_emb_w).sum(dim=2).sum(dim=1)

        # mask out of domain instances out of autoencoder branch
        if self.training:
            r_stack = torch.stack([enc, r], 0)
            r = r_stack.gather(0, domain_mask.view(1, -1, 1).expand(1, -1, self.emb_size)).squeeze()

        # domain classifier
        x = self.dlin1(enc)
        if self.drop is not None:
            x = self.drop(x)
        d_out = self.dlin2(x)

        return r, d_out, a_probs

    def _create_neg_mask(self, batch_size):
        """Creates a mask for randomly selecting negative samples"""
        multi_weights = torch.ones(batch_size, batch_size) - torch.eye(batch_size)
        neg = min(batch_size - 1, self.neg_samples)

        neg_mask = torch.multinomial(multi_weights, neg)
        neg_mask = neg_mask.unsqueeze(2).expand(batch_size, neg, self.emb_size)
        neg_mask = Variable(neg_mask, requires_grad=False)
        return neg_mask

    def set_targets(self, module, input, output):
        """Sets positive and negative samples"""
        assert self.cur_neg_mask is not None, 'Tried to set targets without a mask'
        batch_size = output.size(0)

        if torch.cuda.is_available():
            neg_mask = self.cur_neg_mask.cuda()
        else:
            neg_mask = self.cur_neg_mask

        self.positive = Variable(output.data)
        neg_in = Variable(output.data).expand(batch_size, batch_size, self.emb_size).gather(1, neg_mask)
        neg_ood = Variable(-output.data.unsqueeze(1).expand_as(neg_in))
        neg_stack = torch.stack([neg_ood, neg_in], 0)
        self.negative = neg_stack.gather(0,
                self.cur_domain_mask.view(1, -1, 1, 1).expand(1, -1, neg_in.size(1), neg_in.size(2))).squeeze()
        self.cur_neg_mask = None

    def get_targets(self):
        assert self.positive is not None, 'Positive targets not set; needs a forward pass first'
        assert self.negative is not None, 'Negative targets not set; needs a forward pass first'
        return self.positive, self.negative

    def get_aspects(self):
        if self.a_emb.dim() == 2:
            return self.a_emb
        else:
            return self.a_emb.mean(dim=1)

    def train(self, mode=True):
        super(MultitaskAutoencoder,  self).train(mode)
        if self.encoder_hook is None:
            self.encoder_hook = self.seg_encoder.register_forward_hook(self.set_targets)
        return self

    def eval(self):
        super(MultitaskAutoencoder, self).eval()
        if self.encoder_hook is not None:
            self.encoder_hook.remove()
            self.encoder_hook = None
        return self


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      description =__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data', help="Multi-domain dataset name (without extension)", type=str)
    parser.add_argument('--test_data', help="hdf5 file of test segments", type=str, default='')
    parser.add_argument('--min_len', help="Minimum number of non-stop-words in segment (default: 2)", type=int, default=2)
    parser.add_argument('--aspect_seeds', help='file that contains aspect seed words (overrides number of aspects)',
            type=str, default='')
    parser.add_argument('--recon_method', help="Method of reconstruction (centr/init/fix/cos)",
            type=str, default='fix')
    parser.add_argument('--negative', help="Number of negative samples (default: 20)", type=int, default=20)
    parser.add_argument('--attention', help="Type of word attention to use (default: none)", action='store_true')
    parser.add_argument('--dropout', help="Dropout probability (default: 0)", type=float, default=0.0)
    parser.add_argument('--binarize', help="Binarize domain labels", action='store_true')
    parser.add_argument('--in_id', help="In-domain index (i.e., which is the domain of interest)", type=int, default=0)
    parser.add_argument('--midlayer', help="Size of middle layer", type=int, default=50)
    parser.add_argument('--fix_w_emb', help="Fix word embeddings (default: no)", action='store_true')
    parser.add_argument('--fix_a_emb', help="Fix aspect embeddings (default: no)", action='store_true')
    parser.add_argument('--epochs', help="Number of epochs (default: 15)", type=int, default=15)
    parser.add_argument('--lr', help="Learning rate (default: 0.001)", type=float, default=0.001)
    parser.add_argument('--k', help="Domain classification loss coefficient (default: 1)", type=float, default=1)
    parser.add_argument('--anneal_k', help="Anneal domain classification coefficient", action='store_true')
    parser.add_argument('--start_k', help="Coefficient at start, if anneal_k=True (default: 10)", type=float, default=10)
    parser.add_argument('--k_decay', help="Coefficient decay, if anneal_k=True (default: 0.5)", type=float, default=0.5)
    parser.add_argument('--l', help="Orthogonality loss coefficient (default: 1)", type=float, default=1)
    parser.add_argument('--savemodel', help="File to save model in (default: don't)", type=str, default='')
    parser.add_argument('--semeval', help="File to save semeval-style output (default: don't)", type=str, default='')
    parser.add_argument('--sumout', help="File to save semeval-style output (default: don't)", type=str, default='')
    parser.add_argument('-q', '--quiet', help="No information to stdout", action='store_true')
    parser.add_argument('--seed', help="Random seed (default: system seed, -1)", type=int, default=-1)
    args = parser.parse_args()

    if args.seed != -1:
        torch.manual_seed(args.seed)
        seed(args.seed)

    if not args.quiet:
        print 'Loading data...'

    id2word = {}
    word2id = {}
    fvoc = open(args.data + '_word_mapping.txt', 'r')
    for line in fvoc:
        word, id = line.split()
        id2word[int(id)] = word
        word2id[word] = int(id)
    fvoc.close()

    f = h5py.File(args.data + '.hdf5', 'r')
    batches = []
    domains = []
    original = []
    scodes = []
    num_domains = 0
    for b in f['data']:
        batches.append(Variable(torch.from_numpy(f['data/' +  b][()]).long()))
        original.append(list(f['original/' + b][()]))
        scodes.append(list(f['scodes/' + b][()]))
        domains.append(Variable(torch.from_numpy(f['domains/' +  b][()]).long()))
        num_domains = max(num_domains, domains[-1].data.max() + 1)

    w_emb_array = f['w2v'][()]
    w_emb = torch.from_numpy(w_emb_array)
    vocab_size, emb_size = w_emb.size()
    f.close()

    test_batches = []
    test_labels = []
    test_original = []
    test_scodes = []
    if args.test_data != '':
        f = h5py.File(args.test_data, 'r')
        for b in f['data']:
            test_batches.append(Variable(torch.from_numpy(f['data/' +  b][()]).long()))
            test_labels.append(Variable(torch.from_numpy(f['labels/' +  b][()]).long()))
            test_original.append(list(f['original/' + b][()]))
            test_scodes.append(list(f['scodes/' + b][()]))
    f.close()

    if args.aspect_seeds != '':
        fseed = open(args.aspect_seeds, 'r')
        aspects_ids = []
        if args.recon_method == 'fix' \
                or args.recon_method == 'init':
            seed_weights = []
        else:
            seed_weights = None

        for line in fseed:
            if args.recon_method == 'fix' \
                    or args.recon_method == 'init':
                seeds = []
                weights = []
                for tok in line.split():
                    word, weight = tok.split(':')
                    if word in word2id:
                        seeds.append(word2id[word])
                        weights.append(float(weight))
                    else:
                        seeds.append(0)
                        weights.append(0.0)
                aspects_ids.append(seeds)
                seed_weights.append(weights)
            else:
                seeds = [word2id[word] if word in word2id else 0 for word in line.split()]
                aspects_ids.append(seeds)
        fseed.close()

        if seed_weights is not None:
            seed_w = torch.Tensor(seed_weights)
            seed_w /= seed_w.norm(p=1, dim=1, keepdim=True)
        else:
            seed_w = None

        if args.recon_method == 'centr':
            centroids = []
            for seeds in aspects_ids:
                centroids.append(w_emb_array[seeds].mean(0))
            a_emb = torch.from_numpy(np.array(centroids))
            args.aspects = len(centroids)
            args.num_seeds = len(aspects_ids[0])
        else:
            clouds = []
            for seeds in aspects_ids:
                clouds.append(w_emb_array[seeds])
            a_emb = torch.from_numpy(np.array(clouds))
            args.aspects = len(clouds)
            args.num_seeds = a_emb.size()[1]
    else:
        a_emb = None
        seed_w = None
        args.num_seeds = None


    if not args.quiet:
        print 'Building multitask model..'

    net = MultitaskAutoencoder(vocab_size, emb_size,
            num_aspects=args.aspects, neg_samples=args.negative,
            w_emb=w_emb, a_emb=a_emb, recon_method=args.recon_method, seed_w=seed_w,
            num_seeds=args.num_seeds, attention=args.attention, fix_w_emb=args.fix_w_emb,
            fix_a_emb=args.fix_a_emb, num_domains=num_domains,
            midlayer=args.midlayer, dropout=args.dropout)

    if torch.cuda.is_available():
        net = net.cuda()

    rec_loss = TripletMarginCosineLoss(sum_loss=False)
    if not args.fix_a_emb: # orthogonality loss is only used when training aspect matrix
        orth_loss = OrthogonalityLoss()
    crossent = nn.CrossEntropyLoss()

    params = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    if not args.quiet:
        print 'Starting training...'
        print

    start_all = time()

    if not args.anneal_k:
        k = args.k
    else:
        k = args.start_k

    for epoch in range(args.epochs):
        if not args.quiet:
            print 'Epoch', epoch+1, '(k={0})'.format(k)

        start = time()
        perm = permutation(len(batches))

        for i in range(len(batches)):
            inputs = batches[perm[i]]
            labels = domains[perm[i]]

            if inputs.shape[1] < args.min_len:
                continue

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            domain_mask = Variable((labels.data == args.in_id).long())
            r, d_out, a_probs = net(inputs, domain_mask=domain_mask, batch_num=perm[i])

            positives, negatives = net.get_targets()
            loss_rec = rec_loss(r, positives, negatives)
            loss_cl = crossent(d_out, labels)

            loss = loss_rec + k * loss_cl

            if not args.fix_a_emb:
                aspects = net.get_aspects()
                loss += args.l * orth_loss(aspects)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if args.semeval != '':
            semout = ''

        if args.sumout != '':
            sumout = ''

        net.eval()
        for i in range(len(test_batches)):
            inputs = test_batches[i]
            labels = test_labels[i]
            orig = test_original[i]
            sc = test_scodes[i]
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            _, _, a_probs = net(inputs, batch_num=i)

            for j in range(a_probs.size()[0]):
                if args.semeval != '':
                    semout += 'nan {0}'.format(sc[j])
                    for a in range(a_probs.size()[1]):
                        semout += ' c{0}/{1:.4f}'.format(a, a_probs.data[j][a])
                    semout += '\n'
                if args.sumout != '':
                    sumout += sc[j]
                    for a in range(a_probs.size()[1]):
                        sumout += '\t{0:.6f}'.format(a_probs.data[j][a])
                    sumout += '\t' + orig[j] + '\n'

        if args.semeval != '':
            fsem = open('{0}{1}.key'.format(args.semeval, epoch+1), 'w')
            fsem.write(semout)
            fsem.close()

        if args.sumout != '':
            fsum = open('{0}{1}.sum'.format(args.sumout, epoch+1), 'w')
            fsum.write(sumout)
            fsum.close()

        net.train()
        fsum.close()

        if args.anneal_k:
            k *= args.k_decay

        if not args.quiet:
            print '({0:6.2f}sec)'.format(time() - start)

    if not args.quiet:
        print 'Finished training... ({0:.2f}sec)'.format(time() - start_all)
        print

    if args.savemodel != '':
        if not args.quiet:
            print 'Saving model...'
        torch.save(net.state_dict(), args.savemodel)
