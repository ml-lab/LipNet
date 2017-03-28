'''

LipNet: End-to-End Sentence-level Lipreading. arXiv preprint arXiv:1611.01599 (2016).

Copyright (C) 2017 Yannis M. Assael, Brendan Shillingford, Shimon Whiteson, Nando de Freitas

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

'''

import collections
import h5py
import multiprocessing as mp
import numpy as np
import os
import pickle
import sys

os.environ['OMP_NUM_THREADS'] = '1'

vocab = '- abcdefghijklmnopqrstuvwxyz'


def int_to_char(int_seq):
    return ''.join([vocab[i] for i in int_seq])


def ngram(b, n):
    return set([b[i:i + n] for i in range(len(b) - n + 1)])


def ngram_full(b, n_start, n_stop):
    ret = set()
    for n in range(n_start, n_stop + 1):
        ret = ret | set([b[i:i + n] for i in range(len(b) - n + 1)])
    return ret


class Hyp(object):
    """
    Container class for a single hypothesis
    """

    def __init__(self, pb, pnb, nc):
        self.p_b = pb
        self.p_nb = pnb
        self.n_c = nc


class DecoderBase(object):
    """
    This is the base class for any decoder.
    It does not perform actual decoding (think of it as an abstract class)
    To make your decoder work, extend the class and implement the decode function
    """

    # __metaclass__ = abc.ABCMeta

    # TODO are static methods mixed with cython still fast?
    # @staticmethod
    def combine(self, a, b, c=float('-inf')):
        psum = np.exp(a) + np.exp(b) + np.exp(c)
        if psum == 0.0:
            return float('-inf')
        else:
            return np.log(psum)

    # character mapping objects. Cython requires declaring in advance
    def load_chars(self, charmap_file):
        """
        Loads a mapping of character -> int
        Stores mapping in self.char_int_map
        Stores int -> char mapping in self.int_char_map
        returns True if maps created successfully
        """
        with open(charmap_file) as fid:
            self.char_int_map = dict(
                tuple(l.strip().split()) for l in fid.readlines())

        self.int_char_map = {}
        for k, v in self.char_int_map.items():
            self.char_int_map[k] = int(v)
            self.int_char_map[int(v)] = k

        return True

    # cpdef seq_int_to_char(self, )
    # @abc.abstractmethod
    def decode(self, probs):
        """
        Child classes must implement the decode function
        Minimally the decode function takes a matrix of probabilities
        output by the network (characters vs time)
        returns the best hypothesis in characters
        """
        return None


class ArgmaxDecoder(DecoderBase):
    """
    This is the simple argmax decoder. It doesn't need an LM
    It performs basic collapsing decoding
    """

    def decode(self, probs):
        """
        Takes matrix of probabilities and computes per-frame argmax
        Applies basic blank/duplicate collapsing
        returns the best hypothesis in characters
        Charmap must be loaded 
        """
        maxInd = np.argmax(probs, axis=0)
        pmInd = -1
        hyp = []
        # TODO is this the right way to score argmax decoding?
        hyp_score = 0.0
        for t in range(probs.shape[1]):
            hyp_score = hyp_score + probs[maxInd[t], t]
            if maxInd[t] != pmInd:
                pmInd = maxInd[t]
                if pmInd > 0:
                    hyp.append(self.int_char_map[pmInd])

        # collapsed hypothesis (this is our best guess)
        hyp = ''.join(hyp)
        return hyp, hyp_score


class BeamLMDecoder(DecoderBase):
    """
    Beam-search decoder with character LM
    """

    def load_lm(self, lmfile):
        """
        Loads a language model from lmfile
        returns True if lm loading successful
        """
        self.lm = pickle.load(open("lm/lm_2gram.p", "rb")) | pickle.load(open("lm/lm_3gram.p", "rb")
                                                                         ) | pickle.load(
            open("lm/lm_4gram.p", "rb")) | pickle.load(open("lm/lm_5gram.p", "rb"))
        return True

    def lm_score_final_char(self, prefix, query_char, alpha):
        """
        uses lm to score entire prefix
        returns only the log prob of final char
        """
        if alpha > 0:
            # convert prefix and query to actual text
            full_int = list(prefix) + [query_char]

            full_str = ''.join([self.int_char_map[i] for i in full_int])

            if full_str[0] not in ['b', 'p', 'l', 's']:
                return float('-inf')

            # print(full_str)
            for g in ngram_full(full_str, 2, 5):
                if g not in self.lm:
                    return float('-inf')

        return 0

    def decode(self, probs, beam=40, alpha=1.0, beta=0.0):
        """
        Decoder with an LM
        returns the best hypothesis in characters
        Charmap must be loaded 
        """
        N = probs.shape[0]
        T = probs.shape[1]

        keyFn = lambda x: self.combine(x[1][0], x[1][1]) + beta * x[1][2]
        initFn = lambda: [float('-inf'), float('-inf'), 0]

        # [prefix, [p_nb, p_b, node, |W|]]
        Hcurr = [[(), [float('-inf'), 0.0, 0]]]
        Hold = collections.defaultdict(initFn)

        # loop over time
        for t in range(T):
            Hcurr = dict(Hcurr)
            Hnext = collections.defaultdict(initFn)

            for prefix, (v0, v1, numC) in Hcurr.items():

                valsP = Hnext[prefix]
                valsP[1] = self.combine(
                    v0 + probs[0, t], v1 + probs[0, t], valsP[1])
                valsP[2] = numC
                if len(prefix) > 0:
                    valsP[0] = self.combine(v0 + probs[prefix[-1], t], valsP[0])

                for i in range(1, N):
                    nprefix = tuple(list(prefix) + [i])
                    valsN = Hnext[nprefix]

                    # query the LM_SCORE_FINAL_CHAR for final char score
                    lm_prob = self.lm_score_final_char(prefix, i, alpha)
                    # lm_prob = alpha*lm_placeholder(i,prefix)
                    valsN[2] = numC + 1
                    if len(prefix) == 0 or (len(prefix) > 0 and i != prefix[-1]):
                        valsN[0] = self.combine(
                            v0 + probs[i, t] + lm_prob, v1 + probs[i, t] + lm_prob, valsN[0])
                    else:
                        valsN[0] = self.combine(
                            v1 + probs[i, t] + lm_prob, valsN[0])

                    if nprefix not in Hcurr:
                        v2, v3, _ = Hold[nprefix]
                        valsN[1] = self.combine(
                            v2 + probs[0, t], v3 + probs[0, t], valsN[1])
                        valsN[0] = self.combine(v2 + probs[i, t], valsN[0])

            Hold = Hnext
            Hcurr = sorted(Hnext.items(), key=keyFn, reverse=True)[:beam]

        hyp = ''.join([self.int_char_map[i] for i in Hcurr[0][0]])

        return hyp  # , keyFn(Hcurr[0])
        # return hyp
        # return list(Hcurr[0][0]),keyFn(Hcurr[0])


if __name__ == '__main__':
    if len(sys.argv) > 1:
        beam_width = int(sys.argv[1])
    else:
        beam_width = 100

    if len(sys.argv) > 2:
        filename = sys.argv[2]
    else:
        filename = '/tmp/pctc.h5'

    if len(sys.argv) > 3:
        alpha = float(sys.argv[3])
    else:
        alpha = 1

    if len(sys.argv) > 4:
        beta = float(sys.argv[4])
    else:
        beta = 0

    # filename = 'pctc.h5'

    # Load probs
    pctc = h5py.File(filename, 'r')['/pctc'][()]

    ctc_beam = BeamLMDecoder()

    ctc_beam.int_char_map = {}
    ctc_beam.char_int_map = {}
    for i in range(len(vocab)):
        ctc_beam.int_char_map[i] = vocab[i]
        ctc_beam.char_int_map[vocab[i]] = i

    ctc_beam.load_lm('lm/lm3.klm')


    def ctc_bs(x):
        return ctc_beam.decode(x.T, beam=beam_width, alpha=alpha, beta=beta)


    ctc_bs_in = []
    for b in range(pctc.shape[1]):
        pctc_log = np.log(pctc[:, b, :].squeeze())
        ctc_bs_in.append(pctc_log)

    # print(ctc_bs(ctc_bs_in[0]))

    pool = mp.Pool(processes=100)
    results = []
    imap_res = pool.imap(ctc_bs, ctc_bs_in)
    for r in imap_res:
        results.append(r)

    pool.close()
    pool.join()

    for r in results:
        print(r)
