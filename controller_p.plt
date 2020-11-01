set key noautotitles

set multiplot layout 2, 1 title 'Proportional controller'

set xlabel 'Time elapsed (seconds)'
set ylabel 'Angle (decimal degrees)'

plot 'controller_p.dat' using ($1/1000.0):2 with lines linecolor 'blue'

set xlabel 'Time elapsed (seconds)'
set ylabel 'Force (Newton)'

plot 'controller_p.dat' using ($1/1000.0):4 with lines linecolor 'red'

unset multiplot

pause 1
reread