set terminal pdfcairo size 12cm,12cm font "Helvetica,10"
set output 'plot.pdf'

unset key
unset grid
set tics out
set border 11
set grid 

# Shared X range
set xrange [3.0:4.5]

set multiplot layout 2,1 margins 0.12,0.95,0.10,0.95 spacing 0.0,0.03

# -------------------------------------------------
set title 'LHD shot #054167   YAG Thomson scattering   t = 1.756 s'
unset xlabel
unset xtics

set ylabel 'n_e (10^{19} m^{-3})'
set yrange [0:2.5]
set ytics nomirror

plot 'LHD_1756.dat' using 1:3 \
     with points pt 7 ps 0.6 lc rgb '#804008'

# -------------------------------------------------
unset title
set xlabel 'R (m)'
set xtics

set ylabel 'T_e (eV)'
set yrange [0:1500]
set ytics nomirror

plot 'LHD_1756.dat' using 1:2 \
     with points pt 7 ps 0.6 lc rgb '#008080'

unset multiplot
