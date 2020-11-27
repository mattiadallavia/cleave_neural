set key noautotitles

set multiplot layout 3, 1 # title 'PID controller'

set xlabel 'Time elapsed (seconds)'
set ylabel 'Angle (decimal degrees)'

plot datafile using 1:($4/pi*180) with lines linecolor 'blue'

set xlabel 'Time elapsed (seconds)'
set ylabel 'Force (Newton)'

plot datafile using 1:10 with lines linecolor 'red'

set xlabel 'Time elapsed (seconds)'
set ylabel 'Period (milliseconds)'

plot datafile using 1:($2*1000) with lines linecolor 'black'

unset multiplot
