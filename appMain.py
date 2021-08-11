import os
import time
import PyPDF2
import pandas as pd
import re
import time
import pickle
from wordcloud import WordCloud
from io import StringIO

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser



vectorizer = pickle.load(open("Models/word_vectorizer.pkl", 'rb'))
print ("\nVectorizer Loaded.....!")
model = pickle.load(open("Models/model.pkl", 'rb'))
print ("\nModel Loaded.....!")

with open("Models/category_classes.txt", 'r') as f:
    category_classes = [line.rstrip('\n') for line in f]
print("\ncategory_classes of Resume Loaded......!")


class data_func():
    def convert_pdf_to_string(file_path):

        output_string = StringIO()
        with open(file_path, 'rb') as in_file:
            parser = PDFParser(in_file)
            doc = PDFDocument(parser)
            rsrcmgr = PDFResourceManager()
            device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            for page in PDFPage.create_pages(doc):
                interpreter.process_page(page)
        return(output_string.getvalue())

    def convert_title_to_filename(title):
        filename = title.lower()
        filename = filename.replace(' ', '_')
        return filename

    def split_to_title_and_pagenum(table_of_contents_entry):
        title_and_pagenum = table_of_contents_entry.strip()

        title = None
        pagenum = None

        if len(title_and_pagenum) > 0:
            if title_and_pagenum[-1].isdigit():
                i = -2
                while title_and_pagenum[i].isdigit():
                    i -= 1
                title = title_and_pagenum[:i].strip()
                pagenum = int(title_and_pagenum[i:].strip())

        return title, pagenum


def pdfReaderNew(path):
    text = data_func.convert_pdf_to_string(path)
    text = text.replace('\uf09f ','')
    text = text.replace('\x0c','')
    text_en = text.encode("ascii","ignore")
    text = text_en.decode()
    a = path.split("_")[-1]
    fname = a.split(".")[0]
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    fileName = fname+"_"+timestamp + '.txt'
    with open('textFiles/'+fileName, "w") as text_file:
        text_file.write(text)
    return text, category_classes

def saveTxtCategory(text, category):
    fileName = "PredCat_"+category + '.txt'
    with open('predTextFile/'+fileName, "w") as text_file:
        text_file.write(text)


def cleanResume(resumeText):
    resumeText = re.sub('httpS+s*', ' ', resumeText)       # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)          # remove RT and cc
    resumeText = re.sub('#S+', '', resumeText)             # remove hashtags
    resumeText = re.sub('@S+', '  ', resumeText)           # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^x00-x7f]',r' ', resumeText)    # Replace non asciii characters
    resumeText = re.sub('s+', ' ', resumeText)             # remove extra whitespace
    return resumeText

def WordCloudDrow(txx, wcname):
    wc = WordCloud().generate(txx)
    wc.to_file('static/wordcloud/'+wcname)
    # print("WordCloud Saved at wordcloud/{}".format(wcname))

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
    wcname1 = 'wordcloud/'+wcname
    return resume_category, wcname1
