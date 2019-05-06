import argparse
import cv2
import model
import sys
import pickle
import json
from flask  import Flask 
centrum=(None,None)

url='/track/'
parser=argparse.ArgumentParser()
parser.add_argument('-c','--capture',help='name of videofile to be analysied or numer of capture device',default=2)
parser.add_argument('-f','--showframe',help='show full frame',action='store_true')
parser.add_argument('-s','--showsnip',help='show frame cropped',action='store_true')
parser.add_argument('--url',help='adress for posting results TODO')
args=parser.parse_args()
detec=model.Retina_detector(args.capture)
detec.set_display_opt(args.showframe,args.showsnip,False)
detec.set_cascade()
detec.set_predictor()
while True:
    jc=detec.detect()
    fp=open('cexch.pkl','w')
    json.dump(jc,fp)
    fp.close()
