import os
from re import L
import time
from flask import Flask, redirect, url_for, flash, request, render_template, session, jsonify
from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer

from appMain import pdfReaderNew
from appMain import predictCategory
from appMain import saveTxtCategory

# Define a flask app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H%M', t)
        fileName = timestamp + '_' + f.filename
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(  basepath, 'uploads', secure_filename(fileName) )
        f.save(file_path)
        text, category_classes = pdfReaderNew(file_path)
        category, wcname = predictCategory(file_path, text)
        # preds = classify(file_path)
        # saveTxtCategory(text, category)
        session['text'] = text
        session['category'] = category
        session['category_classes'] = category_classes
        # print(category_classes)
        return {'category':category,'wcname':wcname}
    return None

@app.route('/checkCat', methods=['GET', 'POST'])
def checkCat():
    text = session.get('text', None)
    category = session.get('category', None)
    category_classes = session.get('category_classes', None)
    return render_template('index1.html', data=text, category=category, category_classes= category_classes)

@app.route("/GetCat", methods=['POST', 'GET'])
def GetCat():
    category = session.get('category', None)
    if request.method == "POST":     
        text = session.get('text', None)   
        tvalue = request.form['option']
        # print(tvalue)
        msg = "Given Resume saved with Predicted Category"
        if category!=tvalue:
            category = tvalue
            msg = "Sorry, we predicted wrong Resume Category.Given Resume saved with Updated Category that is {}".format(tvalue)
    saveTxtCategory(text, category)
    return render_template('index3.html', msg=msg)

ALLOWED_EXTENSIONS = set(['pdf'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/getFile', methods=['POST'])
def predict():
        # Catch the image file from a POST request
    if 'file' not in request.files:
        return "Please try again. rename KEY name is file"
    if request.method == 'POST':
        f = request.files['file']
        fname = f.filename
        #print(f.filename)
    if f and allowed_file(f.filename):
        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H%M', t)
        fileName = timestamp + '_' + f.filename
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(  basepath, 'uploads', secure_filename(fileName) )
        f.save(file_path)
        text, category_classes = pdfReaderNew(file_path)
        category, wcname = predictCategory(file_path, text)
        wcpath = r"static/"+wcname
        saveTxtCategory(text, category)
    else:
        print("Please Select PDF file Only")
        return "Please Select PDF file Only"

    result = {
                "Number": 101,
                "File Name": fname,
                "File Path": file_path,
                "WordCloud Path": wcpath,
                "Predicted Category":category,
                "category_classes": category_classes
            }
    return jsonify(result)

# http://127.0.0.1:5000/api/train
@app.route('/api/train', methods=['GET'])
def train():
    from train import defaultTrain
    rs1 = defaultTrain()
    result = {
                "Number": 101,
                "Model Train": "Success",
                "Training Details":rs1
            }
    return jsonify(result)

# http://127.0.0.1:5000/api/trainNewData
@app.route('/api/trainNewData', methods=['GET'])
def trainNewData():
    from train import trainNewData
    rs1 = trainNewData()
    result = {
                "Number": 102,
                "New Model Train": "Success",
                "Training Details":rs1
            }
    return jsonify(result)

if __name__ == '__main__':
    app.secret_key = "Drmhze6EPcv0fN_81Bj-nA"
    app.config['JSON_SORT_KEYS'] = False
    app.run(debug=True)





# http://127.0.0.1:5000/api/getFile 
# file AutoTesting.pdf as input
# Output as 
"""{
    "File Name": "AutoTesting.pdf",
    "File Path": "E:\\anaji Python\\HomeProject\\ResumeScreening\\uploads\\Aug-11-2021_1224_AutoTesting.pdf",
    "Number": 101,
    "Predicted Category": "Automation Testing",
    "WordCloud Path": "static/wordcloud/AutoTesting_Aug-11-2021_1224.png",
    "category_classes": [
        "Advocate",
        "Arts",
        "Automation Testing",
        "Blockchain",
        "Business Analyst",
        "Civil Engineer",
        "Data Science",
        "Database",
        "DevOps Engineer",
        "DotNet Developer",
        "ETL Developer",
        "Electrical Engineering",
        "HR",
        "Hadoop",
        "Health and fitness",
        "Java Developer",
        "Mechanical Engineer",
        "Network Security Engineer",
        "Operations Manager",
        "PMO",
        "Python Developer",
        "SAP Developer",
        "Sales",
        "Testing",
        "Web Designing"
    ]
}"""