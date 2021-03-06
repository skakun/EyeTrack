# EyeTrack
Module of program *Sentiel* enabling oversing over disabled patient. This one proviedes to patient control over cursor and communication with medical asistant via movements of retina, and sends to him medical parameters.
## Usage
	The retina detector is ran from file track.py
```Console
usage: track.py [-h] [-c CAPTURE] [-f] [--showleye] [--showreye]
                [--snip {haar,convex,skip}]
                [--center {CenterDetectMethod.blob,CenterDetectMethod.darkestpoint}]
                [--use_control USE_CONTROL]

optional arguments:
  -h, --help            show this help message and exit
  -c CAPTURE, --capture CAPTURE
                        name of videofile to be analysied or numer of port
                        with capture device.
  -f, --showframe       show full frame
  --showleye, -l        show frame cropped to left eye surroundings
  --showreye, -r        show frame cropped to right eye surroundings
  --snip {haar,convex,skip}
                        method of detecting eye region
  --center {CenterDetectMethod.blob,CenterDetectMethod.darkestpoint}
                        method of detecting retina center
  --use_control USE_CONTROL
                        use eye movements to control cursor

```
Debug calling(showing eye snip from default webcam with marked retina) is:
```Console
	python track.py -sf
```

## Setting up server
 To enable serving response via apache server:
	1. Copy config.py.template as config.py.
	2. In config.py assign url of endpoint where search results will be posted. (Default: "localhost/track/")
	3. In config.py assign absolute path of EyeTrack directory to WORKING_DIR
	4. Add this directory to apache config.

## Server response
Program should serve similar json on assigned endpoint.
<pre>
{  
   "detected":true,
   "alarm":"False",
   "move_mode_open": false,
   "wanna_talk":"False",
   "time_stamp":"2019-05-31 17:07:28.434922",
   "no_eye_contact_since_frames":0,
   "center_x":423,
   "center_y":1069,
   "frame_size_x":1080,
   "frame_size_y":1920,
   "right_eye_winked":"False",
   "left_eye_winked":"False",
   "retina_size":5.8087029457092285,
   "eye_snip_minx":530,
   "eye_snip_maxx":747,
   "eye_snip_miny":426,
   "eye_snip_maxy":496
}
</pre>
### detected
	Was eye detected in current iteration?
### move_mode_open
    Has user opened mode of controling cursor with eye? Info for gui
### alarm
	Did patient called emergency in GUI?
### wanna_talk
	Did patient called assistence in GUI?
### no_conntact_since_frames:
	How many frames, since patient had open eyes? (May be replaced with time)
### center_x 
### center_y
	Position of retina center in frame
### frame_size_x
### frame_size_y
	Dimensions of detected frame
### right_eye_winked
### left_eye_winked
	Did patient closed, or winked ne of his eyes?
### retina_size
	Size of detected retina. (Maybe ratio to snip will be more usefull?)
### eye_snip_minx
### eye_snip_maxx
### eye_snip_miny
### eye_snip_maxy
	Dimensions of area around eye, to which frame was cropped.
### time_stamp
    Time of last detection

The others are described in submodule *GibbonPulse*
