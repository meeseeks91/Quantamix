import os
import sys, getopt
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


from io import StringIO

#The text is preprocessed to remove stopwords and being stemmed
def preprocess(file):
    text = open(file).read()
    tokens = word_tokenize(text)
    words = [w.lower() for w in tokens]

    port = nltk.PorterStemmer()
    stem_tokens = [port.stem(t) for t in words]

    stop_words = set(stopwords.words('english'))
    filter_tokens = [w for w in stem_tokens if not w in stop_words]

    count = nltk.defaultdict(int)
    for word in filter_tokens:
        count[word] += 1
    return count;

#main cosine similarity formula
def cosine_sim (x,y):
    cos_sim = np.dot(x,y)/ ((np.linalg.norm(x))*(np.linalg.norm(y)))
    return cos_sim

#text are vectorized here and the formula is applied on the vectors.
def similarity(text1, text2):
    total_words =[]
    for key in text1:
        total_words.append(key)
    for key in text2:
        total_words.append(key)
    total_words_size = len(total_words)

    vec1 = np.zeros(total_words_size, dtype=np.int)
    vec2 = np.zeros(total_words_size, dtype=np.int)
    i = 0
    for (key) in total_words:
        vec1[i] = text1.get(key, 0)
        vec2[i] = text2.get(key, 0)
        i = i+1
    return cosine_sim(vec1, vec2);


#The job is to be stored at a job.txt file.

job_path = 'job.txt'
job_text = preprocess(job_path)


text_directory = "Candidates_txt"

score = {} #Dictionary which stores all the scores

directory = os.fsencode(text_directory)

for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if filename.endswith(".pdf"):
         filepath = os.path.join(text_directory, filename)
         #print(filename)
         cv_text = preprocess(filepath)
         textsimilarity = similarity(cv_text, job_text)*100
         #print(textsimilarity)
         score[filename] = round(textsimilarity,2)

     else:
         continue

#Score sorted by value and printed.
match = sorted(score.items(), key=lambda x: x[1], reverse=True)
print(match)
