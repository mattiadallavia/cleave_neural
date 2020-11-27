set key noautotitles

set multiplot layout 2, 1

set lmargin at screen 0.13

set xlabel 'Time elapsed (seconds)'
set ylabel 'Angle (decimal degrees)'

plot datafile using 1:($4/pi*180) with lines linecolor 'blue'

set xlabel 'Time elapsed (seconds)'
set ylabel 'Force (Newton)'

plot datafile using 1:10 with lines linecolor 'red'

unset multiplot
