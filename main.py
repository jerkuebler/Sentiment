"""Run sentiment analysis on a collection of movie reviews"""

import collections

import itertools
import nltk


def word_feats(words):
    return dict([(i, True) for i in words])


def stopword_filtered_word_feats(words):
    stopset = set(nltk.corpus.stopwords.words('english'))
    return dict([(item, True) for item in words if item not in stopset])


def bigram_word_feats(words, score_fn=nltk.BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = nltk.BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return dict([(item, True) for item in itertools.chain(words, bigrams)])


def evaluate_classifier(featx):
    negids = nltk.corpus.movie_reviews.fileids('neg')
    posids = nltk.corpus.movie_reviews.fileids('pos')

    negfeats = [(featx(nltk.corpus.movie_reviews.words(fileids=[f])), 'neg') for f in negids]
    posfeats = [(featx(nltk.corpus.movie_reviews.words(fileids=[f])), 'pos') for f in posids]

    negcutoff = int(len(negfeats) * 3 / 4)
    poscutoff = int(len(posfeats) * 3 / 4)

    trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
    testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
    print('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))

    classifier = nltk.classify.NaiveBayesClassifier.train(trainfeats)
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(testfeats):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)

    print('pos precision', nltk.precision(refsets['pos'], testsets['pos']))
    print('pos recall', nltk.recall(refsets['pos'], testsets['pos']))
    print('neg precision', nltk.precision(refsets['neg'], testsets['neg']))
    print('neg recall', nltk.recall(refsets['neg'], testsets['neg']))
    print('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
    classifier.show_most_informative_features()


word_fd = nltk.FreqDist()
label_word_fd = nltk.ConditionalFreqDist()

for word in nltk.corpus.movie_reviews.words(categories=['pos']):
    word_fd[word.lower()] += 1
    label_word_fd['pos'][word.lower()] += 1

for word in nltk.corpus.movie_reviews.words(categories=['neg']):
    word_fd[word.lower()] += 1
    label_word_fd['neg'][word.lower()] += 1

# n_ii = label_word_fd[label][word]
# n_ix = word_fd[word]
# n_xi = label_word_fd[label].N()
# n_xx = label_word_fd.N()

pos_word_count = label_word_fd['pos'].N()
neg_word_count = label_word_fd['neg'].N()
total_word_count = pos_word_count + neg_word_count

word_scores = {}

for word, freq in word_fd.items():
    pos_score = nltk.BigramAssocMeasures.chi_sq(label_word_fd['pos'][word],
                                                (freq, pos_word_count), total_word_count)
    neg_score = nltk.BigramAssocMeasures.chi_sq(label_word_fd['neg'][word],
                                                (freq, neg_word_count), total_word_count)
    word_scores[word] = pos_score + neg_score


def keyfunc(x):
    w, s = x
    return s


best = sorted(word_scores.items(), key=keyfunc, reverse=True)[:10000]
bestwords = set([w for w, s in best])


def best_word_feats(words):
    return dict([(item, True) for item in words if item in bestwords])


def best_bigram_word_feats(words, score_fn=nltk.BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = nltk.BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    d = dict([(bigram, True) for bigram in bigrams])
    d.update(best_word_feats(words))
    return d


print('evaluating best word features')
evaluate_classifier(best_word_feats)
