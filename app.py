from flask import Flask
from sentiment_classifier import SentimentClassifier
from flask import Flask, render_template, request
import time 

app = Flask(__name__)

print( "Preparing classifier")
start_time = time.time()
classifier = SentimentClassifier()
print( "Classifier is ready")
print( time.time() - start_time, "seconds")

@app.route("/", methods=["POST", "GET"])
def index_page(text="", prediction_message=""):
    if request.method == "POST":
        text = request.form["text"]
        print(text)
        prediction_message = classifier.get_prediction_message(text)
        return render_template('index.html', text=text, prediction_message=prediction_message)
    else:
        return render_template('index.html', text="", prediction_message="Добро пожаловать!")
    


# @app.route("/sentiment-demo", methods=["POST", "GET"])
# def index_page(text="", prediction_message=""):
#     if request.method == "POST":
#         text = request.form["text"]
    
#     print(text)
#     prediction_message = classifier.get_prediction_message(text)
    

#     return render_template('hello.html', text=text, prediction_message=prediction_message)


# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=80, debug=False)


if __name__ == "__main__":
    app.run(debug=True)