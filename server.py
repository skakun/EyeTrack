from flask  import Flask,jsonify,make_response
import json
import os
import random
import subprocess
from gpvars import tmp,j
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

@app.route('/pulse', methods=['POST','GET'])
def postJsonHandler():
    global j
    global tmp
    f = open(app.config["WORKING_DIR"]+'GibonPuls/data.txt', "r")
    f1  =open(app.config["WORKING_DIR"]+'GibonPuls/data2.txt',"r")
    lines = f.readlines()
    lines1 = f1.readlines()
    try:
          gotdata = lines[5]
    except IndexError:
          gotdata = 'null'
    json = {"status": (lines[0]).replace('\n', ''),
            "heart_rate": (lines[1]).replace('\n', ''),
            "head_positions": (lines[2]).replace('\n', ''),
            "forehead_position": (lines[3]).replace('\n', ''),
            "fps": (lines[4]).replace('\n', ''),
            "ImgBase64": (gotdata),
            "breath_rate": int(random.uniform(17, 23)),# (lines1[0]).replace('\n', ''),
            "convolutions:": (lines1[1]).replace('\n', ''),
            "breath": str(tmp[j:j+120])}
    j += 30
    if j >= len(tmp):
        j = 0
    f.close()
    f1.close()
    return jsonify(json)
