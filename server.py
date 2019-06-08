from flask  import Flask,jsonify,make_response
import json
import os
import subprocess
from config import BaseConfig
url='/track/'
app=Flask(__name__)
_config=BaseConfig()
app.config.from_object(_config)
centre=(None,None)
def set_point(p):
    centre=p
@app.route(url,methods=['GET'])
def serve():
    fp=open(app.config["WORKING_DIR"]+'cexch.pkl','r')
    j=json.load(fp)
    fp.close()
    fp=open(app.config["WORKING_DIR"]+'alert_data.pkl','r')
    j1=json.load(fp)
#   return  jsonify(j)
    return json.dumps({**j,**j1})
@app.route("/bible")
def bible():
    return subprocess.check_output("randverse")

