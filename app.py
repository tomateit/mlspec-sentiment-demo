from flask import Flask
from sentiment_classifier import SentimentClassifierEN, SentimentClassifierRU
from flask import Flask, render_template, request
import time 

app = Flask(__name__)

print( "Preparing classifiers")
start_time = time.time()
classifier_en = SentimentClassifierEN()
classifier_ru = SentimentClassifierRU()
print( "Classifiers are ready")
print( time.time() - start_time, "seconds")

@app.route("/", methods=["GET"])
def index_page(text="", prediction_message=""):
    return render_template('index.html')
    
@app.route("/sentiment-demo1", methods=["POST", "GET"])
def demo1(text="", prediction_message=""):
    title = "Sentiment demo 1"
    description = "Это - первая демонстрация для финальной недели курса. Модель плохого качества и для английского языка."
    if request.method == "POST":
        text = request.form["text"]
        print(text)
        prediction_message = classifier_en.get_prediction_message(text)
        return render_template('demo.html', 
            text=text, 
            prediction_message=prediction_message,
            title=title,
            description=description,
            demo="demo1"
        )
    else:
        return render_template('demo.html', 
            text="", 
            prediction_message="Добро пожаловать!",
            title=title,
            description=description,
            demo="demo1"
            )

@app.route("/sentiment-demo2", methods=["POST", "GET"])
def demo2(text="", prediction_message=""):
    title = "Sentiment demo 2"
    description = "Это - вторая демонстрация для финальной недели курса. Модель улучшена и поддерживает русский язык."
    if request.method == "POST":
        text = request.form["text"]
        print(text)
        prediction_message = classifier_ru.get_prediction_message(text)
        return render_template('demo.html', 
            text=text, 
            prediction_message=prediction_message,
            title=title,
            description=description,
            demo="demo2"
        )
    else:
        return render_template('demo.html', 
            text="", 
            prediction_message="Добро пожаловать!",
            title=title,
            description=description,
            demo="demo2"
            )
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