from flask import Flask, jsonify, request, render_template
from transformers import MarianMTModel, MarianTokenizer

# Carica il modello e il tokenizer
model_name = 'Helsinki-NLP/opus-mt-tc-big-en-fr'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def submit():
    source = request.form['source']
    src_text = [source]
    print (f'{source}')
    translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
    out_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return f"Source: {source} / Translation: {out_text}"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0') 
