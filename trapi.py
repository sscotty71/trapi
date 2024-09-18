from flask import Flask, jsonify, request, render_template
from transformers import MarianMTModel, MarianTokenizer

# Carica il modello1 e il tokenizer
model_name = 'Helsinki-NLP/opus-mt-tc-big-it-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    sentences = data['sentences']
    print (sentences)
    translated = model.generate(**tokenizer(sentences, return_tensors="pt", padding=True))
    translated_sentences = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return jsonify(translated_sentences)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0') 
