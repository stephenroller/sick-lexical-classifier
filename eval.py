#!/usr/bin/env python

import sys
import os
import pickle
import os.path
import argparse

import numpy as np
import pandas as pd
import sklearn

from os.path import basename
from itertools import chain, izip, cycle
from collections import defaultdict, Counter
from multiprocessing import Pool
from operator import add
from functools import partial

from sklearn.preprocessing import normalize
from sklearn import feature_extraction, cross_validation, svm, dummy, tree, utils, naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


from nltk.corpus import wordnet

import utdeftvs

# load up spacy and override it to use fixed whitespace.
import spacy
from spacy.tokens import Doc
class WhitespaceTokenizer(object):
    def __init__(self, nlp):
        self.vocab = nlp.vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

english = spacy.en.English()
english.make_doc = WhitespaceTokenizer(english)

# this was the final data set
DATA_FILE = "data/sick_train.txt"
TEST_FILE = "data/sick_test.txt"

POS_MAP = {'n': 'NN', 'v': 'VB', 'a': 'JJ', 'r': 'RB', 's': 'JJ'}

RANDOM_SEED = 31337
NUM_CROSS_VAL = 10

# spaces:
SPACE_ROOT = "/scratch/cluster/roller/spaces/giga+bnc+uk+wiki2015/output/svd"
WINDOW_SPACE = os.path.join(SPACE_ROOT, "window20.svd300.ppmi.top250k.top20k.npz")
DEP_SPACE = os.path.join(SPACE_ROOT, "dependency.svd300.ppmi.top250k.top1m.npz")

# FEATURE SETTINGS
ENABLE_MEMO = False
ENABLE_WF = False
ENABLE_DIST = False
ENABLE_WN = False
ENABLE_BASE = False
ENABLE_GOLD = False
ENABLE_ASYM = False
ENABLE_LEFTVEC = False
ENABLE_RIGHTVEC = False

ENABLE_ALIGNMENT = False
ENABLE_HEAD  = False

ENABLE_DETAILED = False

def percentage(binary_array):
    return float(np.sum(binary_array)) / binary_array.shape[0]

def name_error_class(row):
    w = row['w']
    gsw = row['gsw']
    if w == gsw == 1:
        return 'TRUE_POSITIVE'
    elif w == gsw == 0:
        return 'TRUE_NEGATIVE'
    elif w == 0 and gsw == 1:
        return 'FALSE_NEGATIVE'
    elif w == 1 and gsw == 0:
        return 'FALSE_POSITIVE'
    else:
        return 'OTHER'

def norm(v):
    magn = np.sqrt(v.dot(v))
    if magn == 0.0:
        return v
    return v / magn

def cosine(u, v):
    return norm(u).dot(norm(v))

def cosine_w(space, a, b):
    if a == b:
        return 1.0
    z = np.zeros(space.matrix.shape[1])
    av = bv = z
    if a in space: av = space[a]
    if b in space: bv = space[b]
    return cosine(av, bv)

def parse_islam_token(token):
    brokenup = token.split("-")
    token = "-".join(brokenup[:-2])
    pos = brokenup[-2]
    tokenno = int(brokenup[-1]) - 1

    return token, POS_MAP[pos], tokenno

def parse_islam_sentence(text, original_text):
    parsed = [parse_islam_token(t) for t in text.split(" ")]
    original = original_text.split(" ")

    position_of_tokens = {}
    idx = 0
    for i, t in enumerate(original):
        position_of_tokens[i] = idx
        idx += len(t) + 1

    retval = []
    for t, pos, j in parsed:
        retval.append((t, pos, position_of_tokens[j]))
    return retval

def lemma_pos(token):
    return token.lemma_ + "/" +  token.tag_[:2]

def find_alignments(left_tokens, right_tokens, space):
    left_tokens = set(left_tokens)
    right_tokens = set(right_tokens)

    retval = []
    while left_tokens and right_tokens:
        scores = [(l, r, cosine_w(space, lemma_pos(l), lemma_pos(r)))
                  for l in left_tokens for r in right_tokens]
        l, r, s = max(scores, key=lambda x: x[2])
        retval.append((l, r))
        left_tokens.remove(l)
        right_tokens.remove(r)

    return retval, list(left_tokens), list(right_tokens)

def extract_wordnet_features(corenlp_left, corenlp_right):
    if not ENABLE_WN:
        return {}

    left_synsets = wordnet.synsets(corenlp_left.lemma_)
    right_synsets = wordnet.synsets(corenlp_right.lemma_)

    left_synsets = [l for l in left_synsets]
    right_synsets = [r for r in right_synsets]


    possible_matches = [
        (l.path_similarity(r) or -1, l, r)
        for l in left_synsets
        for r in right_synsets]

    wn_pathsim, wn_left, wn_right = max(
        possible_matches +
        # have a default value in case a word isnt in wordnet
        [(0, None, None)],
        key=lambda x: x[0])

    if not len(left_synsets) or not len(right_synsets):
        pass

    if not wn_left or not wn_right:
        return {
            'wn|out_of_wordnet': True,
            'wn|pathsim': 0.0,
            'wn|is_hyper_strict': False,
            'wn|is_hypo_strict': False,
            'wn|is_syn': False,
            'wn|is_ant': False,
        }

    features = {}
    features['wn|pathsim'] = wn_pathsim
    features = bin_feature(features, 'wn|pathsim', wn_pathsim)

    lowest_common_hyper = wn_left.lowest_common_hypernyms(wn_right)
    if lowest_common_hyper:
        lowest_common_hyper = max(lowest_common_hyper, key=lambda x: x.max_depth())
    else:
        lowest_common_hyper = None

    features['wn|is_hyper_strict'] = (wn_left != wn_right) and (wn_left in wn_right.hypernym_paths()[0])
    features['wn|is_hypo_strict'] = (wn_left != wn_right) and (wn_right in wn_left.hypernym_paths()[0])
    features['wn|is_syn'] = (wn_left == wn_right)

    rhs_lemmas = wn_right.lemmas
    lhs_antos = set()
    for lemma in wn_left.lemmas:
        lhs_antos.update(lemma.antonyms())
    features['wn|is_ant'] = any(r in lhs_antos for r in rhs_lemmas)

    features['wn|out_of_wordnet'] = False

    return features

def bin_feature(hashtable, featurename, value):
    hashtable[featurename + "=0.00"] = (value == 0.0)
    hashtable[featurename + "=1.00"] = (value == 1.0)
    for i in xrange(9):
        asfloat = i / 10.
        upper = asfloat + .1
        n = "%s_%.2f<x<=%.2f" % (featurename, asfloat, upper)
        hashtable[n] = (asfloat < value <= upper)
    return hashtable

def extract_distributional_features(left, right):
    if not ENABLE_DIST:
        return {}

    features = {}
    try:
        bow_left_vector = bow_space[lemma_pos(left)]
        bow_right_vector = bow_space[lemma_pos(right)]
        dep_left_vector = dep_space[lemma_pos(left)]
        dep_right_vector = dep_space[lemma_pos(right)]
    except KeyError:
        return {
            'dist|out_of_dist': True,
        }

    cosine_bow = cosine(bow_left_vector, bow_right_vector)
    cosine_dep = cosine(dep_left_vector, dep_right_vector)

    output = {
        'dist|cosine_bow': cosine_bow,
        'dist|cosine_dep': cosine_dep,
        'dist|out_of_dist': False
    }
    output = bin_feature(output, 'dist|cosine_bow', cosine_bow)
    output = bin_feature(output, 'dist|cosine_dep', cosine_dep)

    return output

def extract_vecraw_features(word, name, space):
    try:
        vector = space[lemma_pos(word)]
    except KeyError:
        return {
            'vector|%s_delta' % name: np.zeros(space.matrix.shape[1]),
        }

    return {
        'vector|%s_delta' % name: vector,
    }

def extract_asym_features(left, right, name, space):
    if not ENABLE_ASYM:
        return {}

    try:
        left_vector = space[lemma_pos(left)]
        right_vector = space[lemma_pos(right)]
    except KeyError:
        return {
            'asym|%s_delta' % name: np.zeros(space.matrix.shape[1]),
            'asym|%s_delta_sq' % name: np.zeros(space.matrix.shape[1]),
        }

    # asym stuff
    delta = norm(left_vector) - norm(right_vector)
    delta_sq = np.multiply(delta, delta)

    return {
        'asym|%s_delta' % name: delta,
        'asym|%s_delta_sq' % name: delta_sq
    }


def extract_word_features(left, right):
    if not ENABLE_WF:
        return {}

    return {
        'wf|same_word': left.orth_ == right.orth_,
        'wf|same_lemma': left.lemma_ == right.lemma_,
        'wf|same_shortpos': left.pos_ == right.pos_,
        'wf|same_pos': left.tag_ == right.tag_,
        'wf|left_noun': left.tag_[:2] == 'NN',
        'wf|left_adj': left.tag_[:2] == 'JJ',
        'wf|left_verb': left.tag_[:2] == 'VB',
        'wf|left_adverb': left.tag_[:2] == 'RB',
        'wf|right_noun': right.tag_[:2] == 'NN',
        'wf|right_adj': right.tag_[:2] == 'JJ',
        'wf|right_verb': right.tag_[:2] == 'VB',
        'wf|right_adverb': right.tag_[:2] == 'RB',
        'wf|left_singular': left.tag_ in ('NN', 'NNP'),
        'wf|left_plural': left.tag_ in ('NNS', 'NNPS'),
        'wf|right_singular': right.tag_ in ('NN', 'NNP'),
        'wf|right_plural': right.tag_ in ('NNS', 'NNPS'),
        'wf|both_singular': (right.tag_ in ('NN', 'NNP')) and (left.tag_ in ('NN', 'NNP')),
        'wf|both_plural': (right.tag_ in ('NNS', 'NNPS')) and (left.tag_ in ('NNS', 'NNP')),
    }

def feature_union(*featuresets):
    return dict(chain(*[d.iteritems() for d in featuresets]))

def feature_merge(list_of_featuresets, name=''):
    super_features = defaultdict(list)
    for featureset in list_of_featuresets:
        for k, v in featureset.iteritems():
            super_features[k].append(v)
    output = {}
    for k, v in super_features.iteritems():
        output['%s_max_%s' % (k, name)] = np.max(v)
        output['%s_min_%s' % (k, name)] = np.min(v)
        output['%s_mean_%s' % (k, name)] = np.mean(v)

    return output

def find_tokens(sentence, islam_tokens):
    retval = []
    for t, s, p in islam_tokens:
        possible = [token for token in sentence if token.orth_ == t]
        if not possible:
            print t
            print sentence
            print "wasnt possible"
        else:
            found = min(possible, key=lambda x: np.square(x.idx - p))
            retval.append(found)
    return retval

def bfs(root):
    # breadth first search
    yield root
    for c in root.children:
        for n in bfs(c):
            yield n

def find_head(sentence, tokens):
    if not tokens:
        raise ValueError("Can't find the headmost of an empty token list!")
    tokens = set(tokens)
    for node in bfs(sentence.root):
        if node in tokens:
            return node
    raise ValueError("None of them were in in the sentence.")


def generate_features(irow, dist_space):
    i, row = irow
    corenlp_left_sentence = row['corenlp_left']
    corenlp_right_sentence = row['corenlp_right']

    left_tokens_raw = parse_islam_sentence(row['lhsText'], row['text'])
    right_tokens_raw = parse_islam_sentence(row['rhsText'], row['hypothesis'])

    left_tokens = find_tokens(corenlp_left_sentence, left_tokens_raw)
    right_tokens = find_tokens(corenlp_right_sentence, right_tokens_raw)

    left_head = find_head(corenlp_left_sentence, left_tokens)
    right_head = find_head(corenlp_right_sentence, right_tokens)

    if not left_tokens or not right_tokens or not left_head or not right_head:
        sys.stderr.write('No token or head (%d:"%s").\n' % (i, irow))
        return

    alignments, hanging_left, hanging_right = find_alignments(left_tokens, right_tokens, dist_space)
    if not alignments:
        sys.stderr.write('No alignments (%d:"%s").\n' % (i, irow))
        return

    final_features = {'null|null': 0}
    gold_feature = {
        'gold|con': row['gsw'] == -1,
        'gold|neu': row['gsw'] == 0,
        'gold|ent': row['gsw'] == 1
    }
    if ENABLE_GOLD:
        final_features = feature_union(final_features, gold_feature)

    base_features = {
        'base|length_left': len(left_tokens),
        'base|length_right': len(right_tokens),
        'base|length_diff': len(left_tokens) - len(right_tokens),
        'base|length_diff_abs': abs(len(left_tokens) - len(right_tokens)),
        'base|num_alignments': len(alignments),
        'base|num_hanging_left': len(hanging_left),
        'base|num_hanging_right': len(hanging_right),
        'base|percent_aligned': len(alignments) / float(len(alignments) + len(hanging_left) + len(hanging_right)),
        'base|percent_hanging_left': len(hanging_left) / float(len(left_tokens)),
        'base|percent_hanging_right': len(hanging_right) / float(len(right_tokens)),
    }
    if ENABLE_BASE:
        final_features = feature_union(final_features, base_features)

    if ENABLE_ALIGNMENT:
        features_from_alignment = []
        for left, right in alignments:
            word_features = extract_word_features(left, right)
            wn_features = extract_wordnet_features(left, right)
            dist_features = extract_distributional_features(left, right)
            all_features = feature_union(word_features, wn_features, dist_features)
            features_from_alignment.append(all_features)
        features_from_alignment = feature_merge(features_from_alignment, 'align')
        final_features = feature_union(final_features, features_from_alignment)

    if ENABLE_HEAD:
        features_from_head = feature_union(
            extract_word_features(left_head, right_head),
            extract_wordnet_features(left_head, right_head),
            extract_distributional_features(left_head, right_head),
            extract_islams_features(row),
            extract_asym_features(left_head, right_head, "dep", dist_space),
        )
        if ENABLE_LEFTVEC:
            final_features = feature_union(final_features, extract_vecraw_features(left_head, "leftdep", dep_space))
        if ENABLE_RIGHTVEC:
            final_features = feature_union(final_features, extract_vecraw_features(right_head, "rightdep", dep_space))
        final_features = feature_union(final_features, features_from_head)

    return final_features

def klassifier_factory():
    clf = LogisticRegression(penalty='l2')
    #clf = tree.DecisionTreeClassifier()
    return clf


def predict_fold(fold, X, Y):
    foldno, (train, test) = fold
    train_X = X[train]
    train_Y = Y[train]
    test_X = X[test]
    test_Y = Y[test]

    clf = klassifier_factory()

    scaler = sklearn.preprocessing.MinMaxScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)
    clf.fit(train_X, train_Y)
    predicted_labels = clf.predict(test_X)
    predicted_proba = clf.predict_proba(test_X)

    fold_info = {
        'test_fold': test,
        'accuracy': percentage(predicted_labels == test_Y),
        'predictions': predicted_labels,
        'predictions_proba': predicted_proba,
        'foldno': foldno,
        '#train': train_X.shape[0],
        '#test': train_Y.shape[0],
        'baseline': percentage(test_Y == 0),
    }
    for k, v in Counter(train_Y).iteritems():
        fold_info['train_#%d' % k] = v
        fold_info['train_%%%d' % k] = percentage(train_Y == k)
    for k, v in Counter(test_Y).iteritems():
        fold_info['test_#%d' % k] = v
        fold_info['test_%%%d' % k] = percentage(test_Y == k)

    return fold_info

def run_crossval(output_filename, X, Y, folds, original_data, data):
    sys.stderr.write("Performing cross validation for %s...\n" % output_filename)
    THE_POOL = Pool()
    try:
        results = THE_POOL.map(partial(predict_fold, X=X, Y=Y), enumerate(folds))
        #results = map(partial(predict_fold, X=X, Y=Y), enumerate(skf))
    except KeyboardInterrupt:
        THE_POOL.terminate()
        sys.exit(1)
    THE_POOL.close()

    sys.stderr.write("Classifier: %s\n" % klassifier_factory())
    baselines = [x['baseline'] for x in results]
    baseline_mean = np.mean(baselines)
    sys.stderr.write("Base: %6.3f +/- %6.3f\n" % (baseline_mean, 2 * np.std(baselines)))
    #sys.stderr.write("Base: %s\n" % (baselines))
    accs = [x['accuracy'] for x in results]
    accs_mean = np.mean(accs)
    sys.stderr.write("Accu: %6.3f +/- %6.3f\n" % (accs_mean, 2 * np.std(accs)))
    #sys.stderr.write("Accu: %s\n" % (accs))

    indices = reduce(add, (list(x['test_fold']) for x in results))
    predictions_reordered = reduce(add, [list(x['predictions']) for x in results])
    predictions = np.zeros(len(indices))
    predictions[indices] = predictions_reordered

    probas = np.zeros((X.shape[0], len(set(Y))))
    for result in results:
        probas[result['test_fold']] = result['predictions_proba']

    sys.stderr.write("Confusion matrix: (true \\ prediction)\n")
    sys.stderr.write(str(confusion_matrix(Y, predictions) / float(len(predictions))))
    sys.stderr.write("\n")

    sys.stderr.write("Outputting stuff for analysis...\n")
    original_data = original_data.copy()
    data = data.copy()
    original_data['w'] = predictions
    original_data['prob_con'] = probas[:,0]
    original_data['prob_neu'] = probas[:,1]
    original_data['prob_ent'] = probas[:,2]
    original_data['prob_pred'] = probas.max(axis=1)
    data['w'] = predictions
    data['correct'] = (predictions == Y)
    data['prob_con'] = probas[:,0]
    data['prob_neu'] = probas[:,1]
    data['prob_ent'] = probas[:,2]
    data['prob_pred'] = probas.max(axis=1)
    if not output_filename.startswith("None"):
        original_data.to_csv(output_filename + ".cv", sep="\t", index=False)
    if not output_filename.startswith("None") and ENABLE_DETAILED:
        del data['corenlp_left']
        del data['corenlp_right']
        if 'asym|dep_delta' in data:
            del data['asym|dep_delta']
            del data['asym|dep_delta_sq']
        data['error_class'] = [name_error_class(row) for i, row in data.iterrows()]
        data.to_csv(output_filename + '.detailedcv', sep="\t", index=False)
    sys.stderr.write("\n")

def predict_test(output_filename, X, Y, Xt, original_test, test_data):
    sys.stderr.write("Predicting test for %s...\n" % output_filename)
    clf = klassifier_factory()
    clf.fit(X, Y)

    predictions = clf.predict(Xt)
    probas = clf.predict_proba(Xt)
    original_test = original_test.copy()
    test_data = test_data.copy()
    original_test['w'] = predictions

    original_test['prob_con'] = probas[:,0]
    original_test['prob_neu'] = probas[:,1]
    original_test['prob_ent'] = probas[:,2]
    original_test['prob_pred'] = probas.max(axis=1)
    test_data['prob_con'] = probas[:,0]
    test_data['prob_neu'] = probas[:,1]
    test_data['prob_ent'] = probas[:,2]
    test_data['prob_pred'] = probas.max(axis=1)

    if not output_filename.startswith("None"):
        original_test.to_csv(output_filename, sep="\t", index=False)
    test_data['w'] =  predictions
    if not output_filename.startswith("None") and ENABLE_DETAILED:
        del test_data['corenlp_left']
        del test_data['corenlp_right']
        if 'asym|delta_dep' in test_data:
            del test_data['asym|delta_dep']
            del test_data['asym|delta_dep_sq']
        test_data.to_csv(output_filename + ".testdetailed.txt", sep="\t", index=False)

    if not output_filename.startswith("None"):
        with open(output_filename + ".weights", "w") as f:
            for i, scores in enumerate(clf.coef_):
                f.write("Class %d\n" % clf.classes_[i])
                with_labels = zip(dv.feature_names_, scores)
                ranked = sorted(with_labels, key=lambda x: x[1])
                f.write("  %5.2f  %s\n" % (clf.intercept_[i], "Intercept"))
                for l, v in ranked:
                    f.write("  %5.2f  %s\n" % (v, l))
                f.write("\n")

def parse_sentences(arr):
    unicode_arr = map(unicode, arr)
    processed = (english.pipe(unicode_arr, entity=False, n_threads=10))
    sentences = (d.sents.next() for d in processed)
    return list(sentences)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Lexical Entailment Prediction')

    # feature sets
    parser.add_argument('--memo', action='store_true')
    parser.add_argument('--wf', action='store_true')
    parser.add_argument('--dist', action='store_true')
    parser.add_argument('--wn', action='store_true')
    parser.add_argument('--base', action='store_true')
    parser.add_argument('--gold', action='store_true')
    parser.add_argument('--asym', action='store_true')
    parser.add_argument('--lhsvec', action='store_true')
    parser.add_argument('--rhsvec', action='store_true')

    parser.add_argument('--align', action='store_true')
    parser.add_argument('--head', action='store_true')

    parser.add_argument('--nophrase', action='store_true')
    parser.add_argument('--nolex', action='store_true')
    parser.add_argument('--solo', action='store_true')

    parser.add_argument('--output')
    parser.add_argument('--detailed', action='store_true')

    args = parser.parse_args()

    ENABLE_MEMO = args.memo
    ENABLE_WF = args.wf
    ENABLE_DIST = args.dist
    ENABLE_WN = args.wn
    ENABLE_BASE = args.base
    ENABLE_GOLD = args.gold
    ENABLE_HEAD = args.head
    ENABLE_ASYM = args.asym
    ENABLE_LEFTVEC = args.lhsvec
    ENABLE_RIGHTVEC = args.rhsvec

    ENABLE_ALIGNMENT = args.align
    ENABLE_DETAILED = args.detailed

    sys.stderr.write("Arguments:\n")
    sys.stderr.write(str(args))
    sys.stderr.write("\n")

    sys.stderr.write("Using data file: %s\n" % DATA_FILE)
    sys.stderr.write("Using test file: %s\n" % TEST_FILE)

    sys.stderr.write("Loading spaces...\n")
    bow_space = utdeftvs.load_numpy(WINDOW_SPACE)
    dep_space = utdeftvs.load_numpy(DEP_SPACE)

    sys.stderr.write("Reading data...\n")
    data = pd.read_table(DATA_FILE)
    test_data = pd.read_table(TEST_FILE)

    sys.stderr.write("Parsing...\n")
    data['corenlp_left'] = parse_sentences(data['text'])
    data['corenlp_right'] = parse_sentences(data['hypothesis'])
    test_data['corenlp_left'] = parse_sentences(test_data['text'])
    test_data['corenlp_right'] = parse_sentences(test_data['hypothesis'])

    if args.solo:
        # only look at items that need a single rule
        pairCounts = Counter(data['pairIndex'])
        data = data[[pairCounts[pi] == 1 for pi in data['pairIndex']]].reset_index()

    if args.nophrase:
        data = data[data['isInWordnet'] != 'Phrasal'].reset_index()
        test_data = test_data[test_data['isInWordnet'] != 'Phrasal'].reset_index()
    if args.nolex:
        data = data[data['isInWordnet'] == 'Phrasal'].reset_index()
        test_data = test_data[test_data['isInWordnet'] == 'Phrasal'].reset_index()

    #data.gsw[data['isInWordnet'] == 'Hypernym'] = 1
    original_data = data.copy()
    del original_data['corenlp_left']
    del original_data['corenlp_right']
    original_test = test_data.copy()
    del original_test['corenlp_left']
    del original_test['corenlp_right']

    sys.stderr.write("Generating features...\n")
    extra_info_list = map(partial(generate_features, dist_space=dep_space), data.iterrows())
    extra_info_test = map(partial(generate_features, dist_space=dep_space), test_data.iterrows())
    sys.stderr.write("Done generating features...\n")
    final_features = extra_info_list[0]

    data = pd.concat([data, pd.DataFrame(extra_info_list)], axis=1)
    test_data = pd.concat([test_data, pd.DataFrame(extra_info_test)], axis=1)

    Y = np.array(data['gsw'])
    FEATURES = set()
    for ei in extra_info_list:
        FEATURES = FEATURES.union(ei.keys())
    FEATURES = list(FEATURES)

    # we need to separate out the vector features here
    VECTOR_FEATURES = []
    for key in FEATURES:
        if type(data[key][0]) == np.ndarray:
            VECTOR_FEATURES.append(key)
    for key in VECTOR_FEATURES:
        FEATURES.remove(key)

    sys.stderr.write("%d normal features\n" % (len(FEATURES)))
    sys.stderr.write("%d vector features\n" % (len(VECTOR_FEATURES)))

    # first all the standard variables
    as_records = data[FEATURES].to_dict('records')

    dv = feature_extraction.DictVectorizer()
    X = dv.fit_transform(as_records)
    X = X.todense()
    for key in VECTOR_FEATURES:
        X = np.concatenate([X, np.array(list(data[key]))], axis=1)
    X[np.isnan(X)] = 0.0

    for f in FEATURES:
        if f not in test_data.columns:
            test_data[f] = 0.0
    test_as_records = test_data[FEATURES].to_dict('records')
    Xt = dv.transform(test_as_records)
    Xt = Xt.todense()
    for key in VECTOR_FEATURES:
        Xt = np.concatenate([Xt, np.array(list(test_data[key]))], axis=1)
    Xt[np.isnan(Xt)] = 0.0

    scaler = sklearn.preprocessing.MinMaxScaler()
    X = scaler.fit_transform(X)
    Xt = scaler.transform(Xt)

    predict_test("%s/%s" % (args.output, basename(TEST_FILE)), X, Y, Xt, original_test, test_data)

    # time for actual machine learning

    # need to make sure we don't repeat sick id's across training/test
    sickIDs = np.array(list(set(data["pairIndex"])))
    # randomize
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(sickIDs)
    # count 'em off like assigning groups in class
    splitLookup = dict(izip(sickIDs, cycle(xrange(NUM_CROSS_VAL))))
    # look up each group number to determine the fold
    skf = cross_validation.LeaveOneLabelOut([splitLookup[sickID] for sickID in data['pairIndex']])

    run_crossval("%s/%s" % (args.output, basename(DATA_FILE)), X, Y, skf, original_data, data)

