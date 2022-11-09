# 1. Setting file directory, test email variable, importing all requirements, and downloads
data_dir = "Machine-Learning-for-Security-Analysts-master"
# Test email
test_email = """
Re: Re: East Asian fonts in Lenny. Thanks for your support.  Installing unifonts did it well for me. ;)
Nima
--
To UNSUBSCRIBE, email to debian-user-REQUEST@lists.debian.org
with a subject of "unsubscribe". Trouble? Contact listmaster@lists.debian.org
"""
# print(test_email)

import re, os, math, string, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import re

# Import Natural Language ToolKit library and download dictionaries, uncomment out the downloads if you need to download the dictionaries
import nltk
# nltk.download('stopwords')
# nltk.download('punkt')

# Import Scikit-learn helper functions
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Import Scikit-learn models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Import Scikit-learn metric functions
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
print("\n### Libraries Imported ###\n")
# 1. End Section



# 2. Define tokenizer and test against our test email then count classes
#   The purpose of a tokenizer is to separate the features from the raw data

def tokenizer(text):
    """Separates feature words from the raw data
    Keyword arguments:
      text ---- The full email body

    :Returns -- The tokenized words; returned as a list
    """

    # Retrieve a list of punctuation characters, a list of stopwords, and a stemmer function
    punctuations = list(string.punctuation)
    stopwords = nltk.corpus.stopwords.words('english')
    stemmer = nltk.stem.PorterStemmer()

    # Set email body to lowercase, separate words and strip out punctuation
    tokens = nltk.word_tokenize(text.lower())
    tokens = [i.strip(''.join(punctuations))
              for i in tokens
              if i not in punctuations]

    # User Porter Stemmer on each token
    tokens = [stemmer.stem(i)
              for i in tokens]
    return [w for w in tokens if w not in stopwords and w != ""]
print("\n### Tokenizer defined ###\n")

# Optional Let's see how our tokenizer changes our email
# print("\n- Test Email Body -\n")
# print(test_email)

# Optional Tokenize test email
# print("\n - Tokenized Output -\n")
# tokenized_email = tokenizer(test_email)
# print(tokenized_email)

# Get counts of each class
ham_count = len(os.listdir(data_dir+"/ham"))
spam_count = len(os.listdir(data_dir+"/spam"))
test_count = len(os.listdir(data_dir+"/test"))
# Optional print of class counts
# print(ham_count)
# print(spam_count)
# print(test_count)
print("\n### Class Counting Complete ###\n")
# 2. End section



# 3. Load data and count classes
# Load the training data
#   "corpus" is used to store all of the email bodies in a list
#   "labels" is used to store all of the labes for those email bodies

corpus = []  # X values
labels = []  # Y values

# Load all of the emails from the "ham" directory
print("- Loading Ham -")

for each in os.listdir(data_dir + '/ham'):
    with open(data_dir + '/ham/' + each, 'r', encoding="utf-8") as f:
        corpus.append(f.read())
        labels.append("ham")

# Load all of the emails from the "spam" directory
print("- Loading Spam -")

for each in os.listdir(data_dir + '/spam'):
    with open(data_dir + '/spam/' + each, 'r', encoding="utf-8") as f:
        corpus.append(f.read())
        labels.append("spam")
print("\n### Loading Complete ###\n")

# Optional Print counts of each class
# count_classes = pd.value_counts(labels)
# print("Ham:", count_classes['ham'])
# print("Spam:", count_classes['spam'])

# Optional graph counts of each class, uncomment for a graph of the class counts
# count_classes.plot(kind="bar", fontsize=16)
# plt.title("Class Count (training)", fontsize=20)
# plt.xticks(rotation="horizontal")
# plt.xlabel("Class", fontsize=20)
# plt.ylabel("Class Count", fontsize=20)
# plt.show()

# Optional Let's see how our corpus looks
# emailID = 5
# email = corpus[emailID]
# print("\n- Email ({}) Body -\n".format(emailID))
# print(email)
# print("\n- Tokenized Output -\n")
# print(tokenizer(email))
# 3. End section



# 4. Vectorizing the data
# Vectorize the training inputs -- Takes about 90 seconds to complete
#   There are two types of vectors:
#     1. Count vectorizer
#     2. Term Frequency-Inverse Document Frequency (TF-IDF)

print("- Training Count Vectorizer -")
cVec = CountVectorizer(tokenizer=tokenizer)
count_X = cVec.fit_transform(corpus)

print("- Training TF-IDF Vectorizer -")
tVec = TfidfVectorizer(tokenizer=tokenizer)
tfidf_X = tVec.fit_transform(corpus)

print("\n### Vectorizing Complete ###\n")

# Optional manually perform term count on test_email
# for i in list(dict.fromkeys(tokenized_email)):
#   print("{} - {}".format(tokenized_email.count(i), i))

# Optional view vectorizers
# example_cVec = CountVectorizer(tokenizer=tokenizer)
# example_X = example_cVec.fit_transform([test_email])
# print("\n- Count Vectorizer (test_email) -\n")
# print(example_X)
# print()
# print("="* 50)
# print()
# example_tVec = TfidfVectorizer(tokenizer=tokenizer)
# example_X = example_tVec.fit_transform([test_email])
# print("\n- TFidf Vectorizer (test_email) -\n")
# print(example_X)
# 4. End section



# 5. Load test data
# List first 10 files in "test" directory
#   The labels for each email are in the file name
os.listdir(data_dir + '/test')[:10]

# Load the testing data
#   "test_corpus" is used to store all of the email bodies in a list
#   "test_labels" is used to store all of the labes for those email bodies
test_corpus = []  # X values
test_labels = []  # Y values

# Load all of the emails from the "test" directory
print("- Loading Test Set -")
for filename in os.listdir(data_dir + '/test'):
    with open(data_dir + '/test/' + filename, 'r', encoding="utf-8") as f:
        test_corpus.append(f.read())
        label = re.split("txt\.", filename)[1]
        test_labels.append(label)
print("\n### Loading Complete ###\n")

# Print counts of each class
count_test_classes = pd.value_counts(test_labels)
print("Ham:", count_test_classes['ham'])
print("Spam:", count_test_classes['spam'])

# Optional graph counts of each class, uncomment for a graph
# count_test_classes.plot(kind="bar", fontsize=16)
# plt.title("Class Count (Testing)", fontsize=20)
# plt.xticks(rotation="horizontal")
# plt.xlabel("Class", fontsize=20)
# plt.ylabel("Class Count", fontsize=20)
# plt.show()

# Vectorize the testing inputs -- Takes about 30 seconds to complete
#   Use 'transform' instead of 'fit_transform' because we've already trained our vectorizer
print("- Count Vectorizer -")
test_count_X = cVec.transform(test_corpus)
print("- Tfidf Vectorizer -")
test_tfidf_X = tVec.transform(test_corpus)
print("\n### Vectorizing Complete ###\n")
# 5. End section



# 6. Report generator
def generate_report(cmatrix, score, creport):
    """Generates and displays graphical reports
    Keyword arguments:
      cmatrix - Confusion matrix generated by the model
      score --- Score generated by the model
      creport - Classification Report generated by the model

    :Returns -- N/A
    """

    # Transform cmatrix because Sklearn has pred as columns and actual as rows.
    cmatrix = cmatrix.T

    # Generate confusion matrix heatmap
    plt.figure(figsize=(5, 5))
    sns.heatmap(cmatrix,
                annot=True,
                fmt="d",
                linewidths=.5,
                square=True,
                cmap='Blues',
                annot_kws={"size": 16},
                xticklabels=['ham', 'spam'],
                yticklabels=['ham', 'spam'])

    plt.xticks(rotation='horizontal', fontsize=16)
    plt.yticks(rotation='horizontal', fontsize=16)
    plt.xlabel('Actual Label', size=20);
    plt.ylabel('Predicted Label', size=20);

    title = 'Accuracy Score: {0:.4f}'.format(score)
    plt.title(title, size=20);

    # Display classification report and confusion matrix
    print(creport)
    # Disable plt.show() if you dont want graphs
    # plt.show()
print("\n### Report Generator Defined ###\n")
# 6. End section



# 7. Multinomial Naive Bayesian with TF-IDF
# Train the model
mnb_tfidf = MultinomialNB()
mnb_tfidf.fit(tfidf_X, labels)

# Test the mode (score, predictions, confusion matrix, classification report)
score_mnb_tfidf = mnb_tfidf.score(test_tfidf_X, test_labels)
predictions_mnb_tfidf = mnb_tfidf.predict(test_tfidf_X)
cmatrix_mnb_tfidf = confusion_matrix(test_labels, predictions_mnb_tfidf)
creport_mnb_tfidf = classification_report(test_labels, predictions_mnb_tfidf)

print("\n### Multinomial Naive Bayesian with TF-IDF Model Built ###\n")
generate_report(cmatrix_mnb_tfidf, score_mnb_tfidf, creport_mnb_tfidf)
# 7. End section



# 8. Multinomial Naive Bayesian with Count Vectorizer
# Train the model
mnb_count = MultinomialNB()
mnb_count.fit(count_X, labels)

# Test the mode (score, predictions, confusion matrix, classification report)
score_mnb_count = mnb_count.score(test_count_X, test_labels)
predictions_mnb_count = mnb_count.predict(test_count_X)
cmatrix_mnb_count = confusion_matrix(test_labels, predictions_mnb_count)
creport_mnb_count = classification_report(test_labels, predictions_mnb_count)

print("\n### Multinomial Naive Bayesian with Count Vectorizer Model Built ###\n")
generate_report(cmatrix_mnb_count, score_mnb_count, creport_mnb_count)
# 8. Section end

# 9. Logistic Regression with TF-IDF
# Train the model
lgs_tfidf = LogisticRegression(solver='lbfgs')
lgs_tfidf.fit(tfidf_X, labels)

# Test the mode (score, predictions, confusion matrix, classification report)
score_lgs_tfidf = lgs_tfidf.score(test_tfidf_X, test_labels)
predictions_lgs_tfidf = lgs_tfidf.predict(test_tfidf_X)
cmatrix_lgs_tfidf = confusion_matrix(test_labels, predictions_lgs_tfidf)
creport_lgs_tfidf = classification_report(test_labels, predictions_lgs_tfidf)

print("\n### Logistic Regression with TF-IDF Model Built ###\n")
generate_report(cmatrix_lgs_tfidf, score_lgs_tfidf, creport_lgs_tfidf)
# 9. Section end


# 10. Logistic Regression with Count Vectorizer
# Train the model
lgs_count = LogisticRegression(solver='lbfgs')
lgs_count.fit(count_X, labels)

# Test the mode (score, predictions, confusion matrix, classification report)
score_lgs_count = lgs_count.score(test_count_X, test_labels)
predictions_lgs_count = lgs_count.predict(test_count_X)
cmatrix_lgs_count = confusion_matrix(test_labels, predictions_lgs_count)
creport_lgs_count = classification_report(test_labels, predictions_lgs_count)

print("\n### Logistic Regression with Count Vectorizer Model Built ###\n")
generate_report(cmatrix_lgs_count, score_lgs_count, creport_lgs_count)
# 10. End section