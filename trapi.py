from flask import Flask, jsonify, request, render_template
from transformers import MarianMTModel, MarianTokenizer

# Carica il modello1 e il tokenizer
model_name1 = 'Helsinki-NLP/opus-mt-tc-big-en-it'
tokenizer = MarianTokenizer.from_pretrained(model_name1)
model1 = MarianMTModel.from_pretrained(model_name1)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def submit():
    source = request.form['source']
    src_text = [source]
    print (f'{source}')
    translated = model1.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
    out_text1 = tokenizer.decode(translated[0], skip_special_tokens=True)
    return f"<b>Source:</b> {source} <BR/<BR/><b>Translation:</b> {out_text1} <BR/>"


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0') 
