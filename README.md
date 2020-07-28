# sentiment_analysis_scikit_learn
sentiment analysis of imdb movie review data using logistic regression 

IMDB movie reviews dataset
http://ai.stanford.edu/~amaas/data/sentiment
Contains 25000 positive and 25000 negative reviews 

Contains at most reviews per movie
At least 7 stars out of 10 →
positive (label = 1)
At most 4 stars out of 10 →
negative (label = 0)
50/50 train/test split
Evaluation accuracy

Features: bag of 1-grams with TF-IDF values:

    Extremely sparse feature matrix - close to 97% are zeros

 Model: Logistic regression

    𝑝(𝑦=1|𝑥)=𝜎(𝑤𝑇𝑥)

Linear classification model
Can handle sparse data
Fast to train
Weights can be interpreted 


we will call the fit_transform method on CountVectorizer. This will construct the vocabulary of the bag-of-words model and transform the following three sentences into sparse feature vectors:

    The sun is shining
    The weather is sweet
    The sun is shining, the weather is sweet, and one and one is two

tf-idf(𝑡,𝑑)=tf (t,d)×idf(𝑡,𝑑)

idf(𝑡,𝑑)=log(𝑛𝑑/1+df(𝑑,𝑡))

where 𝑛𝑑
is the total number of documents, and df(d, t) is the number of documents d that contain the term t.
