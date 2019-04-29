from flask  import Flask  
url='/track/'
app=Flask(__name__)
centre=(None,None)
def set_point(p):
    centre=p
@app.route(url)
def serve():
    return jsonify(x=centre[0],y=centre[1])

