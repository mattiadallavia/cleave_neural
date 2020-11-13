set key noautotitles

set multiplot layout 2, 1

set xlabel 'Time elapsed (seconds)'
set ylabel 'Angle (decimal degrees)'

plot datafile using ($1/1000.0):3 with lines linecolor 'blue'

set xlabel 'Time elapsed (seconds)'
set ylabel 'Period (milliseconds)'

set yrange [:100]

plot datafile using ($1/1000.0):2 with lines linecolor 'black'

unset multiplot
