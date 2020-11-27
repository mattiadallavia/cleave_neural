set key noautotitles

set xlabel 'Time elapsed (seconds)'
set ylabel 'Angle (decimal degrees)'

plot datafile using 1:($4/pi*180) with lines linecolor 'blue'
