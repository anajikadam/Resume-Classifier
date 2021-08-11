from argparse import ArgumentParser,ArgumentTypeError
import os
import pandas as pd
import re
import time
import pickle
from wordcloud import WordCloud
# import matplotlib.pyplot as plt

def validate_file(f):
    if not os.path.exists(f):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        print(ArgumentTypeError("{0} does not exist".format(f)))
    return f
parser = ArgumentParser(description="Read file Path form Command line.")
parser.add_argument('-p', '--path', dest="filename", type=validate_file,
            help="input file path", metavar="FILE PATH")
path = parser.parse_args()
print("FILE Name: ",path.filename)
ab = path.filename

file_path = 555  # Python Developer
df1 = pd.read_csv('Data/UpdatedResumeDataSet.csv')
test1_resume = df1['Resume'][file_path]

wcname = "Default_wordcloud.png"

if ab:
    file_path = path.filename
    fname = file_path.split("_")[0]
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    wcname = fname+"_"+timestamp + '.png'
    with open(file_path, 'r') as in_file:
        test1_resume = in_file.read()

vectorizer = pickle.load(open("Models/word_vectorizer.pkl", 'rb'))
print ("\nVectorizer Loaded.....!")
model = pickle.load(open("Models/model.pkl", 'rb'))
print ("\nModel Loaded.....!")

with open("Models/category_classes.txt", 'r') as f:
    category_classes = [line.rstrip('\n') for line in f]
print("\ncategory_classes of Resume Loaded......!")

def cleanResume(resumeText):
    resumeText = re.sub('httpS+s*', ' ', resumeText)       # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)          # remove RT and cc
    resumeText = re.sub('#S+', '', resumeText)             # remove hashtags
    resumeText = re.sub('@S+', '  ', resumeText)           # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^x00-x7f]',r' ', resumeText)    # Replace non asciii characters
    resumeText = re.sub('s+', ' ', resumeText)             # remove extra whitespace
    return resumeText

tx = test1_resume
clean_resume = cleanResume(tx)

tx_vector = vectorizer.transform([clean_resume])
resume_pred_id = model.predict(tx_vector)
print("\nresume_pred_id: ",resume_pred_id)
resume_category = category_classes[resume_pred_id[0]]

print("Predicted Category for given Resume is {}".format(resume_category))

def WordCloudDrow(txx, wcname):
    wc = WordCloud().generate(txx)
    wc.to_file('wordcloud/'+wcname)
    print("WordCloud Saved at wordcloud/{}".format(wcname))

WordCloudDrow(clean_resume, wcname)
print("+"*30)

def predictCategory(path, text):
    clean_resume = cleanResume(text)
    tx_vector = vectorizer.transform([clean_resume])
    resume_pred_id = model.predict(tx_vector)
    resume_category = category_classes[resume_pred_id[0]]
    a = path.split("_")[-1]
    fname = a.split(".")[0]
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    wcname = fname+"_"+timestamp + '.png'
    WordCloudDrow(clean_resume, wcname)
    return resume_category




# python test.py
# python test.py -p "abcd_Aug-10-2021_1207.txt"