set key noautotitles

set xlabel 'Sample count'
set ylabel 'Angle (degrees)'

plot 'controller_p.dat' using 1:2 with lines

pause 1
reread