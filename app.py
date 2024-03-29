from flask import Flask
from models.model1_en.Classifier import Classifier as ClassifierEN1
from models.model2_ru.Classifier import Classifier as ClassifierRU1
from models.model5tl_en.Classifier import Classifier as ClassifierEN5
from flask import Flask, render_template, request
import time 
import os

app = Flask(__name__)

print( "Preparing classifiers")
start_time = time.time()
classifier_ru1 = ClassifierRU1()
classifier_en1 = ClassifierEN1()
classifier_en5 = ClassifierEN5()
print( f"Classifiers are ready in {time.time() - start_time} seconds")

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
        prediction_message = classifier_en.get_prediction(text)
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
        prediction_message = classifier_ru.get_prediction(text)
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

# @app.route("/sentiment-demo3", methods=["POST", "GET"])
# def demo3(text="", prediction_message=""):
#     title = "Sentiment demo 3"
#     description = "FILL IN"
#     if request.method == "POST":
#         text = request.form["text"]
#         print(text)
#         prediction_message = classifier_ru.get_prediction(text)
#         return render_template('demo.html', 
#             text=text, 
#             prediction_message=prediction_message,
#             title=title,
#             description=description,
#             demo="demo3"
#         )
#     else:
#         return render_template('demo.html', 
#             text="", 
#             prediction_message="Добро пожаловать!",
#             title=title,
#             description=description,
#             demo="demo3"
#             )

@app.route("/sentiment-demo5", methods=["POST", "GET"])
def demo5(text="", prediction_message=""):
    title = "Sentiment demo 5"
    description = "Это - пятая модель из серии. На этот раз - используем transfer learning. Предтренированный distilBERT должен справляться просто отлично"
    if request.method == "POST":
        text = request.form["text"]
        print(text)
        prediction_message = classifier_en5.get_prediction(text)
        return render_template('demo.html', 
            text=text, 
            prediction_message=prediction_message,
            title=title,
            description=description,
            demo="demo5"
        )
    else:
        return render_template('demo.html', 
            text="", 
            prediction_message="Добро пожаловать!",
            title=title,
            description=description,
            demo="demo5"
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
    PORT = os.environ.get("PORT", None)
    if PORT:
        app.run(port=PORT, debug=True)
    else:
        app.run(debug=False)