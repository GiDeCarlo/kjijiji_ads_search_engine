import re
import time
import linecache
from collections import Counter
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

noise = '[^A-Za-z0-9 ]+'
regex = re.compile(noise)

stop_words = stopwords.words('english')
stopwords_dict = Counter(stop_words)

# I/O files
input_f = 'jobs.txt'
output_f = 'tfidf.txt'

# lower bound of the cosine similiarity s.t.
# a document is considered related to the query
threshold = 0.02

# vectorizer used to transform strings
vectorizer = TfidfVectorizer()


def textPreprocessing(text):
    # remove stop words
    t = ' '.join([word for word in text.split()
                  if word not in stopwords_dict])
    # remove noise
    t = re.sub(regex, '', t.lower().strip())
    return t


print('# '*20)
print('# SEARCH ENGINE by GIANLUCA DE CARLO  #')
print('# '*20)
print()
print()

create_index = input('[?] Do you need to generate the index file? (y/n) > ')
while create_index != 'y' and create_index != 'n':
    print('[Err] Wrong input. Try again!')
    create_index = input(
        '[?] Do you need to generate the index file? (y/n) > ')

if create_index == 'y':

    # Starting time of elaboration
    start = time.time()

    # Reading the file
    print('[+] Reading ', input_f, 'file...', end='', flush=True)

    lines = []
    with open('jobs.txt') as f:
        for line in f.readlines():
            description = line.split('\t')[1]
            lines.append(textPreprocessing(description))

    print('ok')

    # Computing the TFIDF of the lines (each line is intended as a document)
    print('[+] Computing TFIDF...', end='', flush=True)

    vectors = vectorizer.fit_transform(lines)
    feature_names = vectorizer.get_feature_names_out()
    dense = vectors.todense()
    denselist = dense.tolist()

    print('ok')

    # Read the matrix as its transpose to create the inverted index word: (doc, freq)
    # the pair (doc, freq) is saved only if freq > 0, otherwise that word is not in that
    # document
    print('[+] Saving index to', output_f, 'file...', end='', flush=True)
    output = open(output_f, 'w')

    for i in range(len(feature_names)):
        s = feature_names[i] + '\t'
        to_save = 0
        for j in range(len(denselist)):
            freq = denselist[j][i]
            if(freq > 0):
                s += str(j) + ',' + str(freq) + '\t'
                to_save = 1
        if to_save:
            s += '\n'
            output.write(s)

    output.close()
    print('ok')

    # Computation of index terminated, let's see how much time did it take
    end = time.time()
    elsapsed_time = end-start
    print('\n[Info] All the queries have been processed in:',
          round(elsapsed_time, 3), 'second(s)')

    print('[Info] Search engine ready!\n\n')

# Preprocessing part is finished
# Let's start the search engine part!
# TO DO List:
# 1) Take the query in input - ok
# 2) Search all the documents that contain the words in the query - ok
# 3) take the union of the documents - ok
# 4) for each document in the intersection compute the cosine similiarity with the query
# 5) return the documents with cosine similiarity above a certain threshold

# open the file containing the index created before

print('\n[Info] Insert # to end\n')
query = input('Query > ')

while(query != '#'):
    documents = []
    # process the query
    words_in_query = query.split(' ')
    for word in words_in_query:
        with open(output_f, 'r') as index_f:
            for line in index_f.readlines():
                line = line.strip().split('\t')
                if line[0] == word:
                    for i in range(1, len(line)):
                        tuple = line[i].split(',')
                        doc = tuple[0]
                        if doc not in documents:
                            documents.append(doc)

    data = []
    # data.append(query)
    for i in range(len(documents)):
        # we need to add +1 to the document number because
        # the vectorizer starts from line 0 but in linecache
        # the first line has line number 1
        ad = linecache.getline(input_f, int(documents[i])+1)
        description = textPreprocessing(ad.split('\t')[1])
        data.append(description)

    no_res = 1
    if len(data) > 0:
        data_vectors = vectorizer.fit_transform(data)
        query_vec = vectorizer.transform([query])
        cosine_sim = cosine_similarity(data_vectors, query_vec)

        advertisements = []
        for i in range(len(cosine_sim)):
            if cosine_sim[i] >= threshold:
                advertisements.append(data[i])

        if len(advertisements) > 1:
            for i in range(len(advertisements)):
                print('\n[+] Annuncio ', i+1)
                print(advertisements[i])
            no_res = 0

    if no_res:
        print('[-] No advertisement found')

    query = input('Query > ')

print('[GOOD BYE]')
