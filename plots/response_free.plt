set key noautotitles

set multiplot layout 2, 1

set lmargin at screen 0.11

set xlabel 'Time elapsed (seconds)'
set ylabel 'Angle (degrees)'

plot datafile using 1:($4/pi*180) with lines linecolor 'blue'

set xlabel 'Time elapsed (seconds)'
set ylabel 'Period (milliseconds)'

plot datafile using 1:($2*1000) with lines linecolor 'black'

unset multiplot
