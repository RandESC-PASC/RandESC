
# Size in cm
width_cm  = 8
height_cm = 6

# Convert to pixels (300 dpi, change as needed)
dpi = 600
width_px  = width_cm  / 2.54 * dpi
height_px = height_cm / 2.54 * dpi


set terminal pdf enhanced font "Times,12" size 8 cm, 6 cm
set xlabel "SCF iteration"
set ylabel "Δρ"
set key top right

set grid
set logs y 10
set format y "10^{%L}"

set output "si.pdf"
set title "Si — Density convergence"
plot "si-jd_block.out" using 1:(10**$3) w lp lt 2 pt 7 lw 2 title "Randomized Jacobi-Davidson", \
     "si-lobpcg.out"   using 1:(10**$3) w lp lt 3 pt 5 lw 2 title "LOBPCG"

set output "h3s.pdf"
set title "H_3S — Density convergence"
plot "h3s-jd_block.out" using 1:(10**$3) w lp lt 2 pt 7 lw 2 title "Randomized Jacobi-Davidson", \
     "h3s-lobpcg.out"   using 1:(10**$3) w lp lt 3  pt 5 lw 2 title "LOBPCG"

set output "c2h14s2_medium.pdf"
set title "C_2H_{14}S_2 — Density convergence"
plot "c2h14s2_medium-jd_block.out" using 1:(10**$3) w lp lt 2 pt 7 lw 2 title "Randomized Jacobi-Davidson", \
     "c2h14s2_medium-lobpcg.out"   using 1:(10**$3) w lp lt 3 pt 5 lw 2 title "LOBPCG"

set output
