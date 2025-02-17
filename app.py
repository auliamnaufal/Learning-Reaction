from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
	return 'Welcome to the home page!'

@app.route('/post', methods=['POST'])
def post():
	data = request.get_json()
	comment = data.get('comment')
	sentimen = data.get('sentimen')
	
	# Do something with the comment and sentimen
	
	return jsonify({'message': 'Post received successfully'})

if __name__ == '__main__':
	app.run()