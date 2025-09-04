# https://lweb.cfa.harvard.edu/~dfabricant/huchra/ay145/mags.html

ly =9.461e+15
R_sun = 6.957e8
h = 6.626e-27 # erg s
c = 3e8    # m/s

d = (408+49.) * ly
r = (640+62.) * R_sun
Vmag = 0.50

VAB = Vmag - 0.044
lam = 5500. *1e-10
f = 10**(-(VAB + 48.6)/2.5) # erg sec^-1 cm^-2 Hz^-1
n = f / (h * c / lam)


A = 3.14*15**2
dGammadn = A * n

sig_t = 40e-12 / numpy.sqrt(2)
T = 3600

sigma_inv = dGammadn * numpy.sqrt(T/sig_t)*(128*3.14)**(-1/4)

rho = 1.22 * lam * d/2/r

print(sigma_inv, rho)