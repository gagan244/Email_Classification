from flask import Flask,render_template,url_for,request
import os,joblib,re
import datetime as dt

text_conversion_model = joblib.load('./models/count_vectorizer.lb')
model = joblib.load('./models/bernoulliNB.lb')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        useremail = request.form['useremail']
        email = [useremail]

        # performing text cleaning
        # removing unnecessary/unmeaningful data
        # converting msg to lower case
        # and removing all the numeric data and special symbols

        for msg in email:
            lower_msg = msg.lower()
            clean_msg = re.sub('[^a-zA-Z ]','',lower_msg)
            email = clean_msg
        
        data = [email]
        # cleaning done

        # conversion of text data into numeric data
        x_transformed = text_conversion_model.transform(data)
        # converted data into sparse matrix

        x = x_transformed.toarray()  # x-variable
        # sparse matrix converted to array 
        # our model is trained in array only

        PREDICTION = model.predict(x)[0]

        pred_dict = {1:'spam',0:'ham'}

        # inserting data into a text file
        fp = open('./database.txt','a')
        
        time = dt.datetime.now()
        fp.write('Date:- ' + str(time.day) + '/' + str(time.month) + '/' + str(time.year) + '\n')
        fp.write('Time:- ' + str(time.hour) + ':' + str(time.minute) + ':' + str(time.second) + '\n')

        fp.write('\nMsg: ' + useremail)
        fp.write('\n\n')
        fp.write('Email is: '+ str(pred_dict[PREDICTION]))
        fp.write('\n-------------------------------------------------------------------------------------------------------------\n')
        fp.close()

        return render_template('output.html',answer=pred_dict[PREDICTION])


if __name__ == '__main__':
    app.run(debug=True)