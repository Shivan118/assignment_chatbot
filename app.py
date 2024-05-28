from flask import Flask, request, jsonify, render_template
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os

app = Flask(__name__)

# Initialize the embeddings and FAISS database
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2', model_kwargs={'device': 'cpu'})
faiss_db = r'vector_space/faiss'
db = FAISS.load_local(faiss_db, embeddings)

chat_history = []

@app.route('/')
def index():
    return render_template('index.html', chat_history=chat_history)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    query = user_input
    docs_and_scores = db.similarity_search_with_score(query)
    
    response = ""
    for item in docs_and_scores[:1]:
        document, score = item
        page_content = document.page_content
        response = page_content

    chat_history.append({"user": user_input, "bot": response})
    return jsonify({"response": response, "chat_history": chat_history})

if __name__ == '__main__':
    app.run(debug=True)
