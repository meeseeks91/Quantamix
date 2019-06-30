import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_excel


import gensim
from gensim import models
from gensim.summarization import keywords


df =pd.read_excel('data.xlsx')
jd = df['Job_Description'].tolist()
companies = df['Company'].tolist()
positions = df['Position'].tolist()


for i in range(len(jd)):
    if type(jd[i]) == float:
        jd[i] = str(jd[i])


docs = []
for i in range(len(jd)):
    sent = gensim.models.doc2vec.LabeledSentence(words = jd[i].split(),tags = ['{}_{}'.format(companies[i], i)])
    docs.append(sent)

print (len(docs))


model = gensim.models.doc2vec.Doc2Vec(vector_size=100, min_count=2, epochs=40)

model.build_vocab(docs)

model.train(docs, total_examples=model.corpus_count, epochs=model.epochs)



with open('resumeconverted.txt','r') as f:
    resume = f.read().split()


data = []
for i in range(len(jd)):
    data.append(model.docvecs[i])

data.append(model.infer_vector(resume))

print ("resumes loaded")



def cosine_sim (x,y):
    cos_sim = np.dot(x,y)/ ((np.linalg.norm(x))*(np.linalg.norm(y)))
    return cos_sim


score = {}
jobtag = [None] * len(data)
for i in range(len(data)-1):
    similarity = cosine_sim(model.infer_vector(resume),data[i])
    jobtag[i] = str(companies[i])+str(positions[i])
    score[jobtag[i]] =  round(similarity,2)



match = sorted(score.items(), key=lambda x: x[1], reverse=True)
print(match[:10])
