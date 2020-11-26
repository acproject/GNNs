import collections
UNK_TOKEN = '<UNK>'
UNK_INDEX = 0

class Vocabulary(object):
    def __init__(self, unk_threshold=0):
        '''

        :param unk_threshold: words with <= this many counts will be considered <UNK>.
        '''
        self.unk_threshold = unk_threshold
        self.counts = collections.Counter()
        self.word2index = {UNK_TOKEN: UNK_INDEX}
        self.word_list = [UNK_TOKEN]

    def add_word(self, word, count=1):
        '''
        Add a word (may still map to UNK if it doesn't pass unk_threshold).
        :param word:
        :param count:
        :return:
        '''
        self.counts[word] += count
        if word not in self.word2index and self.counts[word] > self.unk_threshold:
            index = len(self.word_list)
            self.word2index[word] = index
            self.word_list.append(word)

    def add_word(self, words):
        for w in words:
            self.add_word(w)

    def add_sentence(self, sentence):
        self.add_word(sentence.split(' '))

    def add_sentences(self, sentences):
        for s in sentences:
            self.add_sentences(s)

    def add_word_hard(self, word):
        '''
        Add word, make sure it is not UNK.
        :param word:
        :return:
        '''
        self.add_word(word, count=(self.unk_threshold+1))

    def get_word(self, index):
        return self.word_list[index]

    def get_index(self, word):
        if word in self.word2index:
            return self.word2index[word]
        return UNK_INDEX

    def indexify_sentence(self, sentence):
        return [self.get_index(w) for w in sentence.split(' ')]

    def indexify_list(self, elems):
        return [self.get_index(w) for w in elems]

    def recover_sentenc(self, indices):
        return ' '.join(self.get_word(i) for i in indices)

    def has_word(self, word):
        return word in self.word2index

    def __contains__(self, word):
        return word in self.word2index

    def size(self):
        return len(self.word2index)

    def __len__(self):
        return self.size()
    def __iter__(self):
        return iter(self.word_list)

    def save(self, filename):
        '''Save word list.'''
        with open(filename, 'w') as f:
            for w in self.word_list:
                print(w, file=f)

    @classmethod
    def load(cls, filename):
        '''Load word list (does not load counts).'''
        vocab = cls()
        with open(filename) as f:
            for line in f:
                w = line.strip('\n')
                vocab.add_word_hard(w)
        return vocab