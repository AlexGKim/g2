import numpy
import astropy.units as u
import astropy.cosmology
import matplotlib.pyplot as plt
import matplotlib
import scipy.integrate as integrate
from scipy.fft import fft2, rfft2, fftshift
from scipy.special import jv
import sncosmo
import pandas
import h5py
from astropy.io import fits
import astropy.units as units
from astropy.cosmology import Planck18 as cosmo

matplotlib.use("TkAgg")

cosmo = astropy.cosmology.FlatLambdaCDM(H0=70, Om0=0.3)

tmax=20 # days
vmax=1e4 # km/s

tmax_Ibc = 17
tmax_II =7.5

r = vmax * tmax * 3600 * 24
r_Ibc = vmax * tmax_Ibc * 3600 * 24
r_II = vmax * tmax_II * 3600 * 24

z=1200 # km/s
d_z = z/ 70 # Mpc
d_z = d_z * 3.09e13  # Mpc to km

theta = r/d_z

d = 176 / (theta * 1e6)

r_snIa = 2.43e-5 / u.yr / (u.Mpc)**3 # SNe/yr/Mpc^3/h^3_70
r_cc = 9.1e-5 / u.yr / (u.Mpc)**3 # SNe/yr/Mpc^3/h^3_70
r_II = r_cc* 0.649
r_IIL = r_cc* 0.079
r_IIP = r_cc* 0.57
r_IIb = r_cc * 0.109
r_IIn = r_cc * 0.047
r_Ic = r_cc * 0.075
r_Ib = r_cc * 0.108

M_snIa = (-19.46 + 5 * numpy.log10(70/60))
# M_II= (-18 + 5 * numpy.log10(70/60)) 
M_II =  -15.97
M_IIL =  -18.28
M_IIP =  -16.67
M_IIb =  -16.69
M_IIn =  -17.66
M_Ic =  -17.44
M_Ib =  -18.26

v_snIa = 1e4 / 3.09e19 # Mpc/s
v_cc = 1e4 / 3.09e19 # Mpc/s

t_snIa = 18*3600*24 #[s]
t_cc  = 25*3600*24 #[s]

# r=[r_snIa,r_cc]
# M=[M_snIa, M_cc]
# v=numpy.array([v_snIa, v_cc])
# t=numpy.array([t_snIa, t_cc])

r=[r_snIa,r_II, r_IIL, r_IIP, r_IIb, r_IIn, r_Ic, r_Ib]
M=[M_snIa,M_II, M_IIL, M_IIP, M_IIb, M_IIn, M_Ic, M_Ib]
v=numpy.array([v_snIa, v_cc, v_cc, v_cc, v_cc, v_cc, v_cc, v_cc])
t=numpy.array([t_snIa, t_cc, t_cc, t_cc, t_cc, t_cc, t_cc, t_cc])
type_str = ["Ia","II", "IIL", "IIP", "IIb", "IIn", "Ic", "Ib"]

def dNdm(m, M, r):
	z = astropy.cosmology.z_at_value(cosmo.distmod, (m-M)*u.mag,0.000001,0.5)
	dVdz = cosmo.differential_comoving_volume(z)
	dmdz = 5/numpy.log(10)*(astropy.constants.c.to('km/s')*(1+z)/cosmo.H(z)/cosmo.luminosity_distance(z)+ 1/(1+z))
	return (r*dVdz/dmdz*4*numpy.pi*u.sr/(1+z)).value

def snRate():
	limmag = numpy.linspace(8,12,40)
	rates = []
	zs = []
	index=True
	for _r, _M in zip(r, M):
		if index:
			_rates=[]
			_zs=[]
			for l in limmag:
				_rates.append( integrate.quad(dNdm, 0, l, args=(_M,_r))[0])
				_zs.append(astropy.cosmology.z_at_value(cosmo.distmod, (l-_M)*u.mag,0.0000001,0.5).value)
			rates.append(_rates)
			zs.append(_zs)
			index = False

	fig, ax1 = plt.subplots(constrained_layout = True)
	color = 'tab:red'
	ax1.set_xlabel(r'$m_\text{lim}$')
	ax1.set_ylabel(r'$z_\text{max}$ [dotted]')# , color=color)
	ax1.tick_params(axis='y') #, labelcolor=color)
	for z,ts in zip(zs,type_str):
		ax1.plot(limmag, z, ls='dotted',lw=1.5)#, label=r"${} [z_\text{{max}}]$".format(ts))
		# a1$.lot(limmag, zs[1], ls='-.', label=r'CCSN [$z_\text{max}$]')

	ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

	color = 'tab:blue'
	ax2.set_ylabel(r'$\log{N}_\text{cum}$ [$\text{yr}^{-1}$] [solid]') #, color=color)  # we already handled the x-label with ax1
	for rate, ts in zip(rates,type_str):
		ax2.plot(limmag, numpy.log10(rate), label=r"{0}".format(ts),lw=1.5)
	# ax2.plot(limmag, rates[1], label=r'CCSN [$N_\text{cum}$]')
	ax2.tick_params(axis='y') #, labelcolor=color)

	# fig.tight_layouts()  # otherwise the right y-label is slightly clipped
	# ax2.legend(loc=2)
	fig.tight_layout() 
	plt.savefig('rates.pdf')
	plt.clf()
	# plt.show()

def angularSize():

	zs = numpy.linspace(0.0001,0.004,40)
	dA = cosmo.angular_diameter_distance(zs)


	# theta2=(t*v)[:, None]/dA[None,:] * 206265 * 1e6
	# print(176/theta2)
	theta=2*(t*v)[:, None]/dA[None,:]

	fig, ax1 = plt.subplots(constrained_layout = True)
	color = 'tab:red'
	ax1.set_xlabel(r'$z$')
	ax1.set_ylabel(r'$\log{{\theta}}$ [mas] [dotted]') #, color=color)
	ax1.tick_params(axis='y') #, labelcolor=color)
	ax1.plot(zs, numpy.log10(theta[0].value*180/numpy.pi*3600*1e3),ls=':',lw=1.5) #, ls='-.', label=r'SN Ia [$\theta$]')
	# ax1.plot(zs, theta[1]*1e9) #, ls='-.', label=r'CC [$\theta$]')
	# plt.plot(limmag, zs[1], ls='-.', label=r'CCSN [$z_\text{max}$]')
	ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

	color = 'tab:blue'
	ax2.set_ylabel(r'$d$ [km] [solid]') #, color=color)  # we already handled the x-label with ax1
	# ax2.plot(zs, 176*(550/700)/(theta[0]*1e6), label=r'SN Ia [$r$]')
	# ax2.plot(zs, 176*(550/700)/(theta[1]*1e6), label=r'CCSN [$r$]')
	ax2.plot(zs, 1.22*440e-9/(theta[0])*1e-3, label=r'SN Ia',lw=1.5)
	# ax2.plot(zs, 1.22*440e-9/(theta[1])*1e-3, label=r'CCSN')
	ax2.tick_params(axis='y') # , labelcolor=color)

	# fig.tight_layouts()  # otherwise the right y-label is slightly clipped
	# ax2.legend(loc=2)
	plt.savefig('angle.pdf')
	plt.clf()
	# plt.show()

def gamma():
	def Pz(p):
		rmax = 2.25
		y2 = rmax**2 - p**2
		cos2  = y2/rmax**2
		return (1-cos2)/(1+cos2)


	u = numpy.linspace(-1.5,1.5,101)
	# plt.plot(u,Pz(u))
	# plt.xlabel('p')
	# plt.ylabel('Pz')
	# plt.savefig('Pz.pdf')
	# plt.clf()
	# wfe
	def intensity(t1, t2, disk=False):
		rho = numpy.sqrt(t1**2+t2**2)
		if (rho>1):
			return 0
		else:
			if disk:
				return 1
			theta = numpy.arctan2(t1,t2)
			return 0.5*(1- Pz(rho)) + Pz(rho) * numpy.cos(theta)**2

	nbin=1001
	u = numpy.linspace(-10.,10.,nbin)
	v = u

	I = numpy.zeros((nbin,nbin))
	for i,_u in enumerate(u):
		for j,_v in enumerate(v):
			I[i,j] = intensity(_u,_v)

	I=I/I.sum()

	# plt.plot(I[nbin//2,450:550])
	# plt.plot(I[450:550,nbin//2])
	# plt.show()
	# wef

	nrange = 16
	plt.imshow(I[(nrange//2-1)*nbin//nrange:(nrange//2+1)*nbin//nrange,(nrange//2-1)*nbin//nrange:(nrange//2+1)*nbin//nrange])
	plt.savefig('intensity.pdf')
	plt.clf()

	gamma = fft2(I)
	gamma2 = numpy.abs(gamma)**2
	# print(gamma2)

	I = numpy.zeros((nbin,nbin))
	for i,_u in enumerate(u):
		for j,_v in enumerate(v):
			I[i,j] = intensity(_u,_v,disk=True)

	I = I/I.sum()

	dum = fft2(I)
	dum2 = numpy.abs(dum)**2

	nrange = 40
	# plt.plot(fftshift(gamma2)[(nrange//2-1)*nbin//nrange:(nrange//2+1)*nbin//nrange,nbin//2],label='u',color='blue'); 
	# plt.plot(fftshift(gamma2)[nbin//2,(nrange//2-1)*nbin//nrange:(nrange//2+1)*nbin//nrange],label='y',color='brown'); 
	# plt.plot(fftshift(dum2)[nbin//2,(nrange//2-1)*nbin//nrange:(nrange//2+1)*nbin//nrange],label='Airy',color='red'); 
	plt.plot(numpy.arange(20)/3.14,gamma2[:20,0],label='u',color='blue'); 
	plt.plot(numpy.arange(20)/3.14,gamma2[0,:20],label='y',color='brown'); 
	plt.plot(numpy.arange(20)/3.14,dum2[0,:20],label='Airy',color='red'); 
	plt.xlabel(r"$\zeta$")
	plt.ylabel(r"$\gamma^2$")
	plt.legend()
	plt.savefig('gamma.pdf')
	plt.clf()

	# plt.imshow(fftshift(gamma2)[(nrange//2-1)*nbin//nrange:(nrange//2+1)*nbin//nrange,(nrange//2-1)*nbin//nrange:(nrange//2+1)*nbin//nrange])
	# plt.savefig('gamma_im.pdf')
	# plt.clf()



def snr():
    zeta = numpy.linspace(0.01,6,600)
    plt.plot(zeta,2*numpy.abs((2*jv(1,zeta)/zeta)*(jv(0,zeta)-jv(2,zeta)-2*jv(1,zeta)/zeta)))
    plt.xlabel(r"$\zeta$")
    plt.ylabel(r"$\text{SNR}_\theta$ [$\sigma^{-1}$]")
    plt.savefig("snr.pdf")
    plt.clf()

def sn2011fe():
	with fits.open('11feP027.fit') as hdul:
	    # Access the primary HDU (header data unit)
	    primary_hdu = hdul[0]

	    # Get the image data
	    image_data = primary_hdu.data

	    # Get the header information
	    header = primary_hdu.header

	    crval1 = header["CRVAL1"]  # Starting wavelength
	    cdelt1 = header["CDELT1"]  # Wavelength step size
	    naxis1 = header["NAXIS1"]  # Number of pixels in the wavelength axis

	    # Create the wavelength grid
	    wavelength = crval1 + cdelt1 * numpy.arange(naxis1)
	return wavelength, image_data

def tardis():
	intensity = pandas.read_hdf('SN2011fe_MLE_intensity_maxlight.hdf', key='intensity')
	lambdas = intensity.index.values
	I_nu_p = intensity.values
	p_rays = intensity.columns.values

	# flip the order of wavelengths
	lambdas = numpy.flip(lambdas)
	I_nu_p = numpy.flip(I_nu_p,axis=0)
	I_lam_p = I_nu_p/lambdas[:,None]/lambdas[:,None]
	flux_int = numpy.trapz(I_lam_p*p_rays, x=p_rays, axis=1)
	return lambdas, I_lam_p, flux_int

def sedona():
	wavegrid_S = numpy.flip(numpy.load("WaveGrid.npy"))
	flux_S = numpy.flip(numpy.load("Phase0Flux.npy"),axis=0)
	flux_int = flux_S.sum(axis=(1,2))
	return wavegrid_S, flux_S, flux_int

def normalize_spectrum(lambdas, flux_int):
	spectrum = sncosmo.Spectrum(lambdas, flux_int)
	spectrum_mag = spectrum.bandmag('bessellb', magsys='vega')
	return flux_int * 10**((spectrum_mag-12)/2.5) # now in units of  (erg / s / cm^2 / A) for B=12 mag


def nlam():
	wavelength, image_data = sn2011fe()
	lambdas, I_lam_p, flux_int = tardis()
	wavegrid_S, flux_S, flux_int_S = sedona()

	image_data = normalize_spectrum(wavelength,image_data)
	flux_int = normalize_spectrum(lambdas, flux_int)
	flux_int_S = normalize_spectrum(wavegrid_S, flux_int_S)

	h = 6.626* 10**(-34+7)  # erg s
	c = 3e10 # cm/s

	plt.plot(wavelength,image_data*(wavelength)*(wavelength/1e8)**2/h/c/c, label='SNF +2.7',ls='-', linewidth=1.5)
	plt.plot(lambdas, flux_int*(lambdas)*(lambdas/1e8)**2/h/c/c,label='TARDIS',ls=':', linewidth=1.5)
	plt.plot(wavegrid_S, flux_int_S*(wavegrid_S)*(wavegrid_S/1e8)**2/h/c/c,label="SEDONA",ls='-.', linewidth=1.5)
	plt.ylim((0,0.3e-12))
	plt.xlim((3000,10000))
	plt.xlabel(r'Wavelength [Å]')
	plt.ylabel(r'$n_\nu$[s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]')
	plt.legend()
	# plt.show()

	plt.savefig('nlam.pdf')

if __name__ == "__main__":
	angularSize()
	snRate()
	wef
	nlam()
	wef

	gamma()

	snr()