import pandas as pd
import re
import os
import time
import pickle
import nltk
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud
nltk.download('stopwords')
nltk.download('punkt')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# from scipy.sparse import hstack

from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics
from sklearn.metrics import accuracy_score



def cleanResume(resumeText):
    resumeText = re.sub('httpS+s*', ' ', resumeText)       # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)          # remove RT and cc
    resumeText = re.sub('#S+', '', resumeText)             # remove hashtags
    resumeText = re.sub('@S+', '  ', resumeText)           # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^x00-x7f]',r' ', resumeText)    # Replace non asciii characters
    resumeText = re.sub('s+', ' ', resumeText)             # remove extra whitespace
    return resumeText

df = pd.read_csv('Data/UpdatedResumeDataSet.csv')
print("Data Loaded..........!")

df['cleaned_resume'] = df['Resume'].apply(lambda x: cleanResume(x))
print("\nClean Resume......!")

var_mod = ['Category']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])

category_classes = list(le.classes_)
print("\nLabel Encoded....\nNumber of Category : ",len(category_classes))

with open("Models/category_classes.txt", 'w') as f:
    for s in category_classes:
        f.write(str(s) + '\n')
print("\ncategory_classes of Resume Saved......!")

with open("Models/category_classes.txt", 'r') as f:
    classes = [line.rstrip('\n') for line in f]
print("\ncategory_classes of Resume Loaded......!")
# print(classes)

X = df['cleaned_resume'].values
y = df['Category'].values

word_vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english',)
# word_vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english', max_features=1500)  # max_features ordered by term frequency across the corpus.
word_vectorizer.fit(X)
WordFeatures = word_vectorizer.transform(X)
# We have ‘WordFeatures’ as vectors and ‘requiredTarget’ and target 
print ("\nFeature completed .....")


# We will use 80% data for training and 20% data for validation. Let’s split the data now into training and test set.
X_train,X_test,y_train,y_test = train_test_split(WordFeatures, y, 
                                                 random_state=42, test_size=0.2,
                                                 shuffle=True, stratify = y)
print("\nX_train shape:",X_train.shape)
print("\nX_test shape:",X_test.shape)

clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)

# Results
print('\nAccuracy of KNeighbors Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('\nAccuracy of KNeighbors Classifier on test set:     {:.2f}'.format(clf.score(X_test, y_test)))
acc_score = accuracy_score(prediction, y_test)
print("\nAccuracy_score:",acc_score)
# Dump the file

pickle.dump(word_vectorizer, open("Models/word_vectorizer.pkl", "wb"))
print ("\nVectorizer Dump at Models/word_vectorizer.pkl.....!")

pickle.dump(clf, open("Models/model.pkl", "wb"))
print ("\nModel Dump at Models/model.pkl.....!")

def defaultTrain():
    path = 'Data/UpdatedResumeDataSet.csv'
    df = pd.read_csv(path)
    print("Data Loaded..........!")
    shape = str(df.shape)

    df['cleaned_resume'] = df['Resume'].apply(lambda x: cleanResume(x))
    print("\nClean Resume......!")

    var_mod = ['Category']
    le = LabelEncoder()
    for i in var_mod:
        df[i] = le.fit_transform(df[i])

    category_classes = list(le.classes_)
    cat_len = len(category_classes)
    print("\nLabel Encoded....\nNumber of Category : ",len(category_classes))

    cat_path = "Models/category_classes.txt"
    with open(cat_path, 'w') as f:
        for s in category_classes:
            f.write(str(s) + '\n')
    print("\ncategory_classes of Resume Saved......!")

    with open(cat_path, 'r') as f:
        classes = [line.rstrip('\n') for line in f]
    print("\ncategory_classes of Resume Loaded......!")
    # print(classes)

    X = df['cleaned_resume'].values
    y = df['Category'].values
    
    word_vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english',)
    word_vectorizer.fit(X)
    WordFeatures = word_vectorizer.transform(X)
    # We have ‘WordFeatures’ as vectors and ‘requiredTarget’ and target 
    print ("\nFeature completed .....")


    # We will use 80% data for training and 20% data for validation. Let’s split the data now into training and test set.
    X_train,X_test,y_train,y_test = train_test_split(WordFeatures, y, 
                                                    random_state=42, test_size=0.2,
                                                    shuffle=True, stratify = y)
    print("\nUsing KNeighborsClassifier with OneVsRestClassifier multilabel classification")

    clf = OneVsRestClassifier(KNeighborsClassifier())
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)

    # Results
    print('\nAccuracy of KNeighbors Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
    print('\nAccuracy of KNeighbors Classifier on test set:     {:.2f}'.format(clf.score(X_test, y_test)))
    acc_score = accuracy_score(prediction, y_test)
    print("\nAccuracy_score:",acc_score)
    # Dump the file

    vect_model_path = "Models/word_vectorizer.pkl"
    pickle.dump(word_vectorizer, open(vect_model_path, "wb"))
    print ("\nVectorizer Dump at Models/word_vectorizer.pkl.....!")

    model_path = "Models/model.pkl"
    pickle.dump(clf, open(model_path, "wb"))
    print ("\nModel Dump at Models/model.pkl.....!")

    result = {  
                "Number": 1101,
                "File Name":path,
                "Shape of Dataframe": shape,
                "Number of category classes": cat_len,
                "Category classes file path":cat_path,
                "Used Vectorizer": "TfidfVectorizer",
                "Used Classifier": "KNeighborsClassifier",
                "Model Description": "KNeighborsClassifier with OneVsRestClassifier multilabel classification",
                "Model Accuracy":acc_score,
                "Vectorizer Model file path":vect_model_path,
                "Classifier Model file path":model_path
            }
    return result

def trainNewData():
    path = 'Data/UpdatedResumeDataSet.csv'

    def CreateNewData():
        data0 = pd.DataFrame(columns=['Category', 'Resume'])
        path1 = "predTextFile"
        pred_files = os.listdir(path1)

        for i in pred_files:
            file_path = 'predTextFile/'+i
            resume_txt = open(file_path, 'r').read()
            category = i.split('.')[0].split('_')[-1]
            data = {'Category':category,'Resume': resume_txt}
            data0 = data0.append(data, ignore_index=True)
        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y', t)
        df_len = str(data0.shape[0])
        new_data_file = ("Data/new_data_"+df_len+'_'+ timestamp+'.csv')
        data0.to_csv(new_data_file)
        CreateNewData.new_data_file = new_data_file
        print("\nNew data Saved in csv file name is",new_data_file)

        # create dummy resume dict
        d = {}
        df1 = pd.read_csv(path)
        print("Main Data Loaded..........!")
        for cat in list(df1['Category'].unique()):
            res = df1[df1['Category']==cat].iloc[0,1]  # select first resume as dummy for certain category
            d.update({cat:res})
        new_data = pd.DataFrame(columns=['Category', 'Resume'])
        for i in range(len(data0)):
            cat = (data0.iloc[i,0])
            res = (data0.iloc[i,1])+d[cat] # add dummy resume from dict
            data = {'Category':cat,'Resume': res}
            new_data = new_data.append(data, ignore_index=True)
        new_data11 = df1.append(new_data, ignore_index=True)  # New Data append to df
        return new_data11
    df = CreateNewData()
  
    shape = str(df.shape)

    df['cleaned_resume'] = df['Resume'].apply(lambda x: cleanResume(x))
    print("\nClean Resume......!")

    var_mod = ['Category']
    le = LabelEncoder()
    for i in var_mod:
        df[i] = le.fit_transform(df[i])

    category_classes = list(le.classes_)
    cat_len = len(category_classes)
    print("\nLabel Encoded....\nNumber of Category : ",len(category_classes))

    cat_path = "Models/category_classes.txt"
    with open(cat_path, 'w') as f:
        for s in category_classes:
            f.write(str(s) + '\n')
    print("\ncategory_classes of Resume Saved......!")

    with open(cat_path, 'r') as f:
        classes = [line.rstrip('\n') for line in f]
    print("\ncategory_classes of Resume Loaded......!")
    # print(classes)

    X = df['cleaned_resume'].values
    y = df['Category'].values
    
    word_vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english',)
    word_vectorizer.fit(X)
    WordFeatures = word_vectorizer.transform(X)
    # We have ‘WordFeatures’ as vectors and ‘requiredTarget’ and target 
    print ("\nFeature completed .....")


    # We will use 80% data for training and 20% data for validation. Let’s split the data now into training and test set.
    X_train,X_test,y_train,y_test = train_test_split(WordFeatures, y, 
                                                    random_state=42, test_size=0.2,
                                                    shuffle=True, stratify = y)
    print("\nUsing KNeighborsClassifier with OneVsRestClassifier multilabel classification")

    clf = OneVsRestClassifier(KNeighborsClassifier())
    # clf.fit(X_train, y_train)
    clf.fit(WordFeatures, y)
    prediction = clf.predict(X_test)

    # Results
    print('\nAccuracy of KNeighbors Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
    print('\nAccuracy of KNeighbors Classifier on test set:     {:.2f}'.format(clf.score(X_test, y_test)))
    acc_score = accuracy_score(prediction, y_test)
    print("\nAccuracy_score:",acc_score)
    # Dump the file

    vect_model_path = "Models/word_vectorizer.pkl"
    pickle.dump(word_vectorizer, open(vect_model_path, "wb"))
    print ("\nVectorizer Dump at Models/word_vectorizer.pkl.....!")

    model_path = "Models/model.pkl"
    pickle.dump(clf, open(model_path, "wb"))
    print ("\nModel Dump at Models/model.pkl.....!")

    result = {  
                "Number": 1102,
                "Main File Name":path,
                "New Data file Name": CreateNewData.new_data_file,
                "Shape of Dataframe after append": shape,
                "Number of category classes": cat_len,
                "Category classes file path":cat_path,
                "Used Vectorizer": "TfidfVectorizer",
                "Used Classifier": "KNeighborsClassifier",
                "Model Description": "KNeighborsClassifier with OneVsRestClassifier multilabel classification",
                "Model Accuracy":acc_score,
                "Vectorizer Model file path":vect_model_path,
                "Classifier Model file path":model_path
            }
    return result