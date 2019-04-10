import os
import numpy as np
from collections import defaultdict
from nltk import bigrams
import string
from collections import Counter
import re

class POSTagger():

    def __init__(self):
        self.pos_dict = None
        self.word_dict = None
        self.initial = None
        self.list_sentence = []
        self.inverted_word_dict = None
        self.inverted_pos_dict = None
        self.init_dictionary = defaultdict(int)
        self.pos_dictionary = defaultdict(int)
        self.dictionary_wordpos = defaultdict(int)
        self.dict_twopos = defaultdict(int)
        self.inverted_word_dict = None
        self.transition = None
        self.emission = None
        self.UNK = '*UNKNOWN*'
        self.poslist = []

    '''
    Trains a supervised hidden Markov model on a training set.
    '''
    def train(self, train_set):
        # iterate over training documents
        trans_tag_dict = defaultdict(int)
        tag_list = []
        pos_bigram = []
        for root, dirs, files in os.walk(train_set):
            for name in files:
                sentence_in_file = []
                with open(os.path.join(root, name)) as f:
                    sentence_in_file = f.readlines()
                    assert isinstance(sentence_in_file, list)
                    self.list_sentence.append(sentence_in_file)

        #print(self.list_sentence)
        tag_list = []
        word_list = []
        wordpos_list = []
        trans_list = []
        # create the dictionary for self.initial[POS]
        for file in self.list_sentence:
            for sent in file:
                sent.strip()
                sent.strip('\n')
                sent.strip('\n\n')
                if len(sent.split()) > 0:
                    # find the initial tag for the dictionary for self.initial
                    init_postag = sent.split()[0].split('/')[-1]
                    if init_postag[:2] == 'fw':
                        init_postag = init_postag[3:]
                    if init_postag[:2] == '--':
                        init_postag = init_postag[:2]
                    if '+' or '-' in init_postag:
                        if len(re.findall("^[^-\+]+", init_postag)) > 0:
                            init_postag = re.findall("^[^-\+]+", init_postag)[0]

                    pattern = r"([^\/])(\*)"
                    init_postag = re.sub(pattern, r"\1", init_postag)

                    self.init_dictionary[init_postag] += 1

                # create the dictionary for self.transition[POS1][POS2]
                pos_in_file = []
                for w_tag in sent.split():
                    word_tag = w_tag.split('/')[-1]
                    word = w_tag.split('/')[0]
                    if word_tag[:2] == 'fw':
                        word_tag = word_tag[3:]
                    if word_tag[:2] == '--':
                        word_tag = word_tag[:2]
                    if '+' or '-' in init_postag:
                        if len(re.findall("^[^-\+]+", word_tag)) > 0:
                            word_tag = re.findall("^[^-\+]+", word_tag)[0]

                    pattern = r"([^\/])(\*)"
                    word_tag = re.sub(pattern, r"\1", word_tag)

                    self.pos_dictionary[word_tag] += 1

                    wordpos = (word, word_tag)
                    wordpos_list.append(wordpos)

                    pos_in_file.append(word_tag)
                    tag_list.append(word_tag)
                    word_list.append(word)

                pos_bigram.extend(list(bigrams(pos_in_file)))


        tag_list = list(set(tag_list))
        self.pos_dict = dict(zip(range(len(tag_list)), tag_list))
        self.inverted_pos_dict = dict(map(reversed, self.pos_dict.items()))
        self.transition = np.zeros((len(tag_list), len(tag_list)))

        self.initial = np.zeros(len(self.pos_dict))

        for i in range(len(self.pos_dict)):
            self.initial[i] = np.log((self.init_dictionary[self.pos_dict[i]]+1) / (len(self.pos_dict) + len(tag_list)))


        # create the dictionary for self.transition[POS1][POS2]
        for pospair in pos_bigram:
            self.dict_twopos[pospair] += 1

        for tag1 in tag_list:
            for tag2 in tag_list:
                self.transition[self.inverted_pos_dict.get(tag1)][self.inverted_pos_dict.get(tag2)] = \
                    np.log((self.dict_twopos[(tag1, tag2)]+1)/(self.pos_dictionary.get(tag1)+len(tag_list)))

        # calculate the self.emission
        # create the dictionary for self.pos_dict and self.word_dict
        for wt in wordpos_list:
            self.dictionary_wordpos[wt] += 1

        word_list = set(word_list)

        self.word_dict = dict(zip(range(len(word_list)), word_list))
        self.word_dict[len(word_list)] = "UNK"
        self.inverted_word_dict = dict(map(reversed, self.word_dict.items()))

        self.emission = np.zeros((len(tag_list), len(word_list)))

        for tag in tag_list:
            for w in word_list:
                wtag = (w, tag)
                self.emission[self.inverted_pos_dict[tag]][self.inverted_word_dict[w]] = np.log((self.dictionary_wordpos[wtag]+1) /
                                               (self.pos_dictionary[tag] + len(word_list)))


    '''
    Implements the Viterbi algorithm.
    Use v and backpointer to find the best_path.
    '''
    def viterbi(self, sentence):
        v = None
        backpointer = None
        # initialization step
        # recursion step
        # termination step

        v = np.zeros((len(self.pos_dict), len(sentence)))
        backpointer = np.zeros((len(self.pos_dict), len(sentence)))
        # initialization step
        v[:, 0] = self.initial + self.emission[:, self.inverted_word_dict.get(sentence[0], 0)]
        backpointer[:, 0] = 0
        # recursion step
        for w_index, wd in enumerate(sentence[1:], 1):
            for t_index in self.pos_dict:
                v[t_index][w_index] = np.max((v[:, w_index - 1] + self.transition[:, t_index] +
                                               self.emission[t_index][self.inverted_word_dict.get(wd, 0)]))

                backpointer[t_index][w_index] = np.argmax((v[:, w_index - 1] + self.transition[:, t_index] +
                                                            self.emission[t_index][self.inverted_word_dict.get(wd, 0)]))

        # termination step
        best_path_pointer = int(np.argmax(v[:, -1]))
        best_path = []


        for i in range(len(sentence) - 1, -1, -1):
            best_path.insert(0, best_path_pointer)
            best_path_pointer = int(backpointer[best_path_pointer][i])

        result = []
        for i in best_path:
            result.append(self.pos_dict[i])

        return result

    '''
    Tests the tagger on a development or test set.
    Returns a dictionary of sentence_ids mapped to their correct and predicted
    '''
    def test(self, dev_set):
        results = defaultdict(dict)
        lst_sent = []
        # iterate over testing documents
        for root, dirs, files in os.walk(dev_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    lines = f.readlines()
                    assert isinstance(lines, list)
                    lst_sent.extend(lines)

        for i in range(len(lst_sent)):
            s = lst_sent[i]
            s.strip()
            s.strip('\n')
            s.strip('\n\n')
            if len(s.split()) > 0:
                correct_tag = []
                predict = []
                for wg in s.split():
                    segment = wg.split('/')[-1]
                    word = wg.split('/')[0]
                    if segment[:2] == 'fw':
                        segment = segment[3:]
                    if segment[:2] == '--':
                        segment = segment[:2]
                    if '+' or '-' in segment:
                        if len(re.findall("^[^-\+]+", segment)) > 0:
                            segment = re.findall("^[^-\+]+", segment)[0]

                    pattern = r"([^\/])(\*)"
                    segment = re.sub(pattern, r"\1", segment)

                    correct_tag.append(segment)
                    predict.append(word)

                results[i] = {'correct': correct_tag, 'predicted': self.viterbi(predict)}

        return results

    '''
    Given results, calculates overall accuracy.
    '''
    def evaluate(self, results):
        correct = 0
        total = 0

        for id in results.keys():
            lst_correct = results[id]['correct']
            lst_predicted = results[id]['predicted']
            for i in range(len(lst_correct)):
                if lst_correct[i] == lst_predicted[i]:
                    correct += 1
                total += 1

        accuracy = correct / total
        return accuracy

if __name__ == '__main__':
    pos = POSTagger()
    pos.train('brown/train')
    results = pos.test('brown/dev_small')
    print('Accuracy:', pos.evaluate(results))
