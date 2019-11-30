import json
import tqdm
import string
import random
from nltk.stem.porter import *
from collections import defaultdict

punctuation = set(string.punctuation)
with open('data/goodreads_reviews_spoiler.json') as f:
    lines = []
    for i in tqdm.tqdm(range(1378033)):
        lines.append(json.loads(f.readline()))


def clean_review(review):
    return ''.join([c if c not in punctuation else ' ' + c + ' ' for c in review.lower()])


random.shuffle(lines)
test_size = int(len(lines)*0.2)
test_lines = lines[:test_size]
valid_lines = lines[test_size:test_size+10000]
train_lines = lines[test_size+10000:]

stemmer = PorterStemmer()
words = defaultdict(int)
map_stem_words = {}
for line in tqdm.tqdm(train_lines):
    for sentence in line['review_sentences']:
        sentence = clean_review(sentence[1])
        if (sentence != '') and (sentence is not None):
            for word in sentence.split():
                if word in map_stem_words:
                    words[map_stem_words[word]] += 1
                else:
                    map_stem_words[word] = stemmer.stem(word)
                    words[map_stem_words[word]] += 1

counts = [(words[w], w) for w in words]
counts.sort()
counts.reverse()
word_bags = counts[:20000]
popular_words = set([word[1]for word in word_bags])


def write_data(filename, lines):
    with open('data/'+filename+'.txt', 'w+') as file:
        for line in tqdm.tqdm(lines):
            paragraph = ""
            for sentence in line['review_sentences']:
                paragraph = paragraph + clean_review(sentence[1])
            candidate_words = []
            for word in paragraph.split():
                if word in map_stem_words:
                    if map_stem_words[word] in popular_words:
                        candidate_words.append(map_stem_words[word])
                    else:
                        candidate_words.append('MASK')
                else:
                    candidate_words.append('MASK')
            text = " ".join(candidate_words)
            label = "__label__"+str(line['has_spoiler'])
            file.write("%s\t%s\r\n" % (text, label))


write_data('goodread_train', train_lines)
write_data('goodread_valid', valid_lines)
write_data('goodread_test', test_lines)
