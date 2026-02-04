set terminal pdfcairo size 12cm,12cm font "Helvetica,10"
set output 'tjii_019002_001171.pdf'

# --- General style ---
unset key
set grid
set tics out
set border 11
set datafile missing ""

set xlabel 'ρ'
set xrange [0:1.0]

set multiplot layout 2,1 margins 0.12,0.95,0.10,0.95 spacing 0.0,0.03

# -------------------------------
set title 'TJ-II shot 18998   t = 1.150 s'
unset xlabel
unset xtics

set ylabel 'n_e (10^{19} m^{-3})'
set yrange [0:3.5]
set ytics nomirror

plot 'tjii_019002_001171.dat' using 1:2 \
     with points pt 7 ps 0.6 lc rgb '#804008'

# -------------------------------
set xlabel 'ρ'
set xtics
unset title
set ylabel 'T_e (keV)'
set yrange [0:0.35]
set ytics nomirror

plot 'tjii_019002_001171.dat' using 1:3 with points pt 7 ps 0.6 lc rgb '#008080'

unset multiplot
