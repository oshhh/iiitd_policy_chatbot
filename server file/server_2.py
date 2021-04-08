from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import uuid 
import sys
sys.path.insert(1, 'C:/Users/mohni/Desktop/github_chatbot/iiitd_policy_chatbot/notebooks')
import qa_helper
import qa_model
from dotenv import load_dotenv
import os



load_dotenv()
# NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
# NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

qa_model.init("NEO4J_USERNAME", "NEO4J_PASSWORD")

hostName = "localhost"
serverPort = 8080

sessions = {}

class MyServer(BaseHTTPRequestHandler):
	def do_GET(self):
		self.handle_error("GET request not accepted")

	def do_POST(self):
		data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))

		if self.path == '/get_answer':
			sentences = qa_model.find_answer(data['query'])
			self.send_response(200)
			self.send_headers()
			self.wfile.write(json.dumps({'query':data['query'], 'sentences': sentences}).encode('utf-8'))
		elif self.path == '/get_sentence_details':
			neighbouring_sentences, document = qa_helper.get_sentence_details(data['sentence'])
			self.send_response(200)
			self.send_headers()
			self.wfile.write(json.dumps({'sentence': data['sentence'], 'neighbouring_sentences': neighbouring_sentences, 'document': document}).encode('utf-8'))
		elif self.path == '/get_closest_entities':
			closest_entities = qa_helper.get_closest_entities(data['query'], data['threshold'])
			self.send_response(200)
			self.send_headers()
			self.wfile.write(json.dumps({'query': data['query'], 'closest_entities': closest_entities}).encode('utf-8'))
		elif self.path == '/get_graph_with_neighbours':
			nodes, edges = qa_helper.get_graph_with_neighbours(data['nodes'], data['nbr_of_nodes'])
			self.send_response(200)
			self.send_headers()
			self.wfile.write(json.dumps({'nodes': nodes, 'edges': edges}).encode('utf-8'))
		else:
			self.handle_error('invalid request path')
			return

	def handle_error(self, error):
		self.send_response(400)
		self.send_headers()
		self.wfile.write(('Error: ' + error).encode('utf-8'))  

	def send_headers(self):
		self.send_header('Access-Control-Allow-Origin', '*')
		self.send_header('Access-Control-Allow-Headers', '*')
		self.send_header('Access-Control-Allow-Methods', '*')
		self.send_header("Content-Type", "application/json")
		self.end_headers()


if __name__ == "__main__":        
	webServer = HTTPServer((hostName, serverPort), MyServer)
	print("Server started http://%s:%s" % (hostName, serverPort))

	try:
		webServer.serve_forever()
	except KeyboardInterrupt:
		pass

	webServer.server_close()
	print("Server stopped.")