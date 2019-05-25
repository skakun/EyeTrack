from flask  import Flask,jsonify,make_response
import pickle
import os
from config import BaseConfig
url='/track/'
app=Flask(__name__)
_config=BaseConfig()
app.config.from_object(_config)
centre=(None,None)
def set_point(p):
    centre=p
@app.route(url)
def serve():
#   fp=open('home/krzysztof/reps/EyeTrack/cexch.pkl','rb')
    fp=open(app.config["WORKING_DIR"]+'cexch.pkl','r')
    j=fp.read()
    fp.close()
    return  jsonify(j)


