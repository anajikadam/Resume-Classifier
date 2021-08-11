from argparse import ArgumentParser,ArgumentTypeError
import os
import time
import PyPDF2

from io import StringIO

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

def validate_file(f):
    if not os.path.exists(f):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        print(ArgumentTypeError("{0} does not exist".format(f)))
    return f

parser = ArgumentParser(description="Read file Path form Command line.")
parser.add_argument('-p', '--path', dest="filename", required=True, type=validate_file,
            help="input file path", metavar="FILE PATH")
path = parser.parse_args()
print(path.filename)
file_path = path.filename

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

text = data_func.convert_pdf_to_string('abcd.pdf')
# print(text)
text = text.replace('\uf09f ','')
text = text.replace('\x0c','')
text_en = text.encode("ascii","ignore")
text = text_en.decode()
# table_of_contents_raw = text.split('\n')
# table_of_contents_raw
fname = file_path.split(".")[0]
t = time.localtime()
timestamp = time.strftime('%b-%d-%Y_%H%M', t)
fileName = fname+"_"+timestamp + '.txt'
with open(fileName, "w") as text_file:
    text_file.write(text)
    print("Saved text file {}".format(fileName))

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
    return text
# python pdfReader.py --path "abcd.pdf"
