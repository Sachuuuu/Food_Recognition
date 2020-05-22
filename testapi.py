
from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS, cross_origin
from flask import jsonify
import shutil

from werkzeug import secure_filename
import numpy as np
from PIL import Image

import base64
from PIL import Image
from io import BytesIO
import main as MN

app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)


class food_api(Resource):
    @app.route('/',methods=['POST'])
    def post(self):

        receivedData = request.get_json() # Receieve data]
        #file = request.files['file'].read()
        text = receivedData['text']
        print()
        byte_check = text[0:2]
        if byte_check == "b'":            
            text = text[2:len(text)-1]
            img = base64.b64decode(text)
        else:
            img = base64.b64decode(text)
        


        with open('test2.jpg','wb') as f:
            f.write(img)
        
        file_name = 'test2.jpg'
        try:            
            res = MN.run(file_name)
            returnJson = {
            	'result': res,
            	'status': 200 
	        }
            return jsonify(returnJson) 
        except Exception as e:
        	returnJson = {
            	'msg': e,
            	'status': 500
        	}
        	return jsonify(returnJson)
       

api.add_resource(food_api,'/expression/')

if __name__ == '__main__':
    
    app.run()
    

@app.route('/api')
def welcome():
    return 'Food Identification API'