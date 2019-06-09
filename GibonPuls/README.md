# GibonPulse

## Server response
Program should serve similar json on assigned endpoint.

<pre>
/pulse
{  
   "status":stabilized, <>
   "heart_rate":52.439548898620686,
   "head_positions":[437, 24, 114, 147],
   "forehead_position":[479, 39, 28, 22],
   "fps":20.320325198215514
}
</pre>
### status
	Returns the current status of the application 
### heart_rate
	Beats of heart per minute
### head_positions
	Position of head [x, y, w, h]
### forehead_position:
	Position of forehead [x, y, w, h] 
### fps 
  frame per second
