import argparse
import cv2
import model
from model import SnipMethod,CenterDetectMethod
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
parser.add_argument('--snip',type=SnipMethod,choices=list(SnipMethod),help='method of detecting eye region',default=SnipMethod.haar)
parser.add_argument('--center',type=CenterDetectMethod,choices=list(CenterDetectMethod),help='method of detecting retina center',default=CenterDetectMethod.blob)
args=parser.parse_args()
detec=model.Retina_detector(args.capture)
detec.snip_method=args.snip
detec.center_detec_method=args.center
detec.set_display_opt(args.showframe,False,args.showsnip)
if detec.snip_method==SnipMethod.haar:
    detec.set_cascade()
if detec.snip_method==SnipMethod.convex:
    detec.set_predictor()
while True:
    jc=detec.detect()
    fp=open('cexch.pkl','w')
    json.dump(jc,fp)
    fp.close()
