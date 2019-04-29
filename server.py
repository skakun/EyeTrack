from flask  import Flask,jsonify
import pickle
url='/track/'
app=Flask(__name__)
centre=(None,None)
def set_point(p):
    centre=p
@app.route(url)
def serve():
    fp=open('cexch.pkl','rb')
    pickle(
    return jsonify(x=centre[0],y=centre[1])


