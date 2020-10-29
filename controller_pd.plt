set key noautotitles

set multiplot layout 2, 1 title 'Proportional-derivative controller'

set xlabel 'Time elapsed (milliseconds)'
set ylabel 'Angle (decimal degrees)'

plot 'controller_pd.dat' using 1:2 with lines linecolor 'blue'

set xlabel 'Time elapsed (milliseconds)'
set ylabel 'Force (Newton)'

plot 'controller_pd.dat' using 1:4 with lines linecolor 'red'

unset multiplot

pause 1
reread