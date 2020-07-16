import PyDSTool as dst
import PyDSTool.utils as dstuti
import numpy as np
import math as m
from matplotlib import pyplot as plt
from scipy import signal as snl
import copy as copy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import gc

#Now I was thinking about putting all the default concentration of ions, their points of interest etc in one place.
#this class plays a huge role in "templatification" of the code
#All of the parameters that I change EVER should be taken from here, just in case
class InitIonMatrix():
	def __init__(self, string=None):
		if string == None:
			self.getparam = [
				[0.01, 0.08, 0.03, 2.3053355e-8, 0.4e-9, 0.3, 0.1],	#default concentrations, 0; cl ideal: 0.01213625
				['Na','K','Cl','Ca_cyt', 'Ca_er', 'x100', 'x110'],	#names, 1
				[1e-10, 1e-10, 0.0195, 1e-11],	#zone of interest: start, 2
				[0.02, 0.2, 0.04, 1e-3],	#zone of interest: end, 3
				[0.145, 0.005, 0.150, 3e-3],	#default solution, 4
				[0.13, 0.02, 0.15, 2.35e-8],	#hypotonic solution 1, 5
				[0.11, 0.04, 0.15, 5e-3],	#hypertonic solution 2, 6
				[-0.055, -0.050, -0.080],	#em, default, lower, higher, 7
				[0, 1000., 0.01],	#time: start, finish, dt 8 ------------------------------------------------------------------------------------------
				[0.05, 2, 1, 0],	#default permeabilities 1/h 9
				[0.5, 0., 0.004, 0.2, 20., 0.21, 1.64, 2., 0.185, 0.1, 4.6, 1.7, 1, 0.11, 6., 0.45, 0.145, 2.],	#Fedor constants, 10
				[0.5, 10, 0.05, 1],	#IP3 max, type of function, parametres of function, 11 <-------------------------------- IP3 HERE ---------------
				[22, 4.8e-15, 75e-12],	#other platelet constants: Cm, V, S, 12
				[0.008, 3, 2],	#13 (atpase old) (left as a grim reminder of the cost of progress)
				[0.01], #parametres of NKCC: J0; 14
				[0.001, 1e-4, 10],	#parametres of NCX: JMax, c1, c2; 15
				[96485, 8.314, 310, 3600],	#world constants: F, R, T, speed (unit to hours); 16
				[],	# 17
				[[2.5e11,1e5],[1e4,1e5],[172,1.72e4],[1.5e7,2e5],[2e6,30],[1.15e4,6e8]], #Constants for ATPase kinetic model, by each reaction, then forward / backward, 18
				[2., -0.03, -0.008], #kv 1.3, 19
				[], #, 20 deprecated
				[4.99e-3, 0.06e-3, 4.95e-3], #ATP, ADP, P concentrations, 21
				[0.00, -0.01, 0.01], #, NaV 22
				[], #, 23 deprecated
				[0, 0.2, 1.2, 0], #ca channels, total, Na relative, K relative 24
				[2.4e-7*3600, 2e-7, 4], #PMCA constants: Jmax, K1/2, n; 25
				[2.8, 0.2e-6, 5], #Kca31 constants: Pmax, K1/2, n; 26
				[0, 1, 1], #Kca11 constants: Pmax, K1/2, n; 27
				[0., 0.7 * 1e-6, 4], #TRPC6 channel state function; 28
				[0.2, 0.2, 0, 1], #TRPC6 channel relative permeabilities in order: Na, K, Cl, Ca; 29
				]
		else:
			self.getparam = []
		self.steps=m.floor(1/0.01)
		self.version_name = 'em_thingy'

class platelet_config:
	def __init__(self,
				iNa_in = None, iK_in = None, iCl_in = None, iCa_in = None,
				iosc_on = True, iSolution = -1, iEm = 0, iimx = InitIonMatrix()
				):
		self.imx = iimx
		imx = iimx
		# print("used parsed parametres, if any, to configure a platelet")
		#Constants
		self.F = imx.getparam[16][0]
		self.R = imx.getparam[16][1]
		self.T = imx.getparam[16][2]
		self.speed = imx.getparam[16][3]
		self.t0 = imx.getparam[8][0]
		self.osc_on = iosc_on
		self.solution = iSolution
		#Concentrations in a resting platelet
		if iNa_in is None: self.Na_in = imx.getparam[0][0]
		else: self.Na_in = iNa_in
		if iK_in is None: self.K_in = imx.getparam[0][1]
		else: self.K_in = iK_in
		if iCl_in is None: self.Cl_in = imx.getparam[0][2]
		else: self.Cl_in = iCl_in
		self.Ca_in_cyt = imx.getparam[0][4]
		self.Ca_in_er = imx.getparam[0][3]
		self.Ca_in = self.Ca_in_cyt + self.Ca_in_er

		#outside
		self.Na_ex = imx.getparam[4][0]
		self.K_ex = imx.getparam[4][1]
		self.Cl_ex = imx.getparam[4][2]
		self.Ca_ex = imx.getparam[4][3]
		#put my platelet back in the isotonic solution, weirdo.
		self.Sol_Na_ex = imx.getparam[5+iSolution][0]
		self.Sol_K_ex = imx.getparam[5+iSolution][1]
		self.Sol_Cl_ex = imx.getparam[5+iSolution][2]
		self.Sol_Ca_ex = imx.getparam[5+iSolution][3]
		#Fedor's model constants
		self.IP3 = imx.getparam[11]
		self.q = imx.getparam[10]
		#Electric and other "platelet" constants
		self.Em_rest = imx.getparam[7][iEm]
		self.Cm = imx.getparam[12][0]
		self.Vol = imx.getparam[12][1]
		self.Surf = imx.getparam[12][2]
		self.VtoS = self.Vol / self.Surf
		#NaK atplase constants
		self.J0_atp = imx.getparam[13][0]
		self.N = imx.getparam[13][1]
		self.K = imx.getparam[13][2]
		#PMCA constants
		self.J_PMCA = imx.getparam[25][0]
		self.K05_PMCA = imx.getparam[25][1]
		self.n_PMCA = imx.getparam[25][2]
		#NKCC constants
		self.U0_NaK2Cl = imx.getparam[14][0] / self.Rest_UKNa2Cl()
		self.IP3max = imx.getparam[11][0]

		#Permeabilities
		self.P_Na = imx.getparam[9][0]
		# self.P_K_Ca_g = 3600. * imx.getparam[17][0] * imx.getparam[17][1] / self.F / self.J(in_ion=self.K_in, ex_ion=self.K_ex)
		self.P_K = imx.getparam[9][1]
		self.P_Cl = imx.getparam[9][2]
		self.P_Ca = imx.getparam[9][3]

		self.Pvar = imx.getparam[24][0]
		self.PNaCavar = imx.getparam[24][1]
		self.PKCavar = imx.getparam[24][2]
		self.PClCavar = imx.getparam[24][3]

		self.P_Kca31 = imx.getparam[26][0]
		self.K05_Kca31 = imx.getparam[26][1]
		self.n_Kca31 = imx.getparam[26][2]

		self.P_Kca11 = imx.getparam[27][0]
		self.K05_Kca11 = imx.getparam[27][1]
		self.n_Kca11 = imx.getparam[27][2]

		self.P_TRPC = imx.getparam[28][0]
		self.K05_TRPC = imx.getparam[28][1]
		self.n_TRPC = imx.getparam[28][2]
		self.Na_TRPC = imx.getparam[29][0]
		self.K_TRPC = imx.getparam[29][1]
		self.Cl_TRPC = imx.getparam[29][2]
		self.Ca_TRPC = imx.getparam[29][3]

		#calculated parameteres
		self.External_ions_concentration = self.K_ex + self.Na_ex + self.Cl_ex + self.Ca_ex
		self.Metabolite_concentration = self.External_ions_concentration - self.K_in - self.Na_in - self.Cl_in + 2 * self.Ca_in_cyt
		self.Metabolite_charge = (self.K_in + self.Na_in + 2*self.Ca_in_cyt - self.Cl_in - self.Em_rest / self.F * self.Cm) / self.Metabolite_concentration

		self.AtpaseRate = self.Rest_UATP()
	#easy access to the calculated parameteres
	def L(self):
		return self.External_ions_concentration
	def W0(self):
		return self.Metabolite_concentration
	def Z0(self):
		return self.Metabolite_charge

	def TestEm(self):
		return self.F / self.Cm * (self.K_in + self.Na_in - self.Cl_in - self.Z0() * self.W0())
	def Give_Em(self, Na, K, Cl, Ca):
		return self.F / self.Cm * (K + Na + 2*Ca - Cl - self.Z0() * self.W0())

	def TestV(self):
		return (self.K_in + self.Na_in + self.Cl_in + self.W0())/self.L()
	def Give_Volume(self, Na, K, Cl, t):
		return (K + Na + Cl + self.W0()) / self.Give_L(t)

	def Give_L(self, t):
		#should be the same as in the pydstool model
		Na_exf = self.Sol_Na_ex + (self.Na_ex - self.Sol_Na_ex) / (1 + self.speed * t)
		K_exf = self.Sol_K_ex + (self.K_ex - self.Sol_K_ex) / (1 + self.speed * t)
		Cl_exf = self.Sol_Cl_ex + (self.Cl_ex - self.Sol_Cl_ex) / (1 + self.speed * t)
		return (Na_exf+Cl_exf+K_exf)
	
	#model equations for test and calculation purposes

	def phi(self, z=1):
		return z * self.Em_rest * self.F / self.R / self.T / 2
	def J(self, in_ion, ex_ion, z=1):
		return 2 * self.phi(z) / (m.exp(self.phi(z)) - m.exp(-self.phi(z))) * (ex_ion * m.exp(-self.phi(z)) - in_ion * m.exp(self.phi(z)))
	#	J for different ions
	#	na: self.J(in_ion=self.Na_in, ex_ion=self.Na_ex)
	#	k:  self.J(in_ion=self.K_in, ex_ion=self.K_ex)
	#	cl: self.J(z = -1, in_ion=self.Cl_in, ex_ion=self.Cl_ex)
	#	ca: self.J(z = 2, in_ion=self.Ca_in, ex_ion=self.Ca_ex)

	def J_calc(self, intr, extr, em, z=1):
		phi = z * em * self.F / self.R / self.T / 2
		return 2 * phi / (m.exp(phi) - m.exp(-phi)) * (extr * m.exp(-phi) - intr * m.exp(phi))

	def P_calc(self, intr, extr, em, g_ch, n_ch, z=1):
		return n_ch*g_ch*em/self.J_calc(intr, extr, em, z) / self.VtoS * 3600

	def Pkv13rest(self):
		return self.imx.getparam[19][0] / (1. + m.exp((self.Em_rest - self.imx.getparam[19][1])/self.imx.getparam[19][2]))

	def Pnv15rest(self):
		return self.imx.getparam[22][0] / (1. + m.exp((-self.Em_rest + self.imx.getparam[22][1])/self.imx.getparam[22][2]))

	def Rest_UKNa2Cl(self):
		#returns "osmotic pressure" on channels
		return self.K_ex*self.Na_ex*pow(self.Cl_ex,2) - self.K_in*self.Na_in*pow(self.Cl_in,2)

	def Rest_UATP(self):
		F = self.F
		Cm = self.Cm
		R = self.R
		T = self.T
		Na = self.Na_in
		K = self.K_in
		Cl = self.Cl_in
		Em = self.Em_rest
		ADP = 0.06e-3
		Na_exf = self.Na_ex
		K_exf = self.K_ex
		f1 = 2.5e11
		f2 = 1e4
		f3 = 172
		f4 = 1.5e7
		f5 = 2e6
		f6 = 1.15e4
		b1 = 1e5
		b2 = 1e5
		b3 = 1.72e4
		b6 = 6e8
		return f2 * f6 * f6 * f1 * pow(Na, 3)* f1 * pow(Na, 3) * f2 * f3 * m.exp(F * Em/(2 * R * T))\
			/(b6 * pow(K, 2)*(b1 + f2)*f6*f1 * pow(Na, 3)*f2*f3 * m.exp(F * Em/(2 * R * T)) \
			+ b6 * pow(K, 2)*f1 * pow(Na, 3)*f2*f3 * m.exp(F * Em/(2 * R * T)) \
			+ b6 * pow(K, 2)*(b1 + f2)*f2*f3 * m.exp(F * Em/(2 * R * T)) \
			+ b6 * pow(K, 2)*(b1 + f2)*b2 * ADP*f3 * m.exp(F * Em/(2 * R * T)) \
			+ b6 * pow(K, 2)*(b1 + f2)*b2 * ADP*(b3 * pow(Na_exf, 3) * m.exp(-1 * F * Em/(2 * R * T)) + f4 * pow(K_exf, 2)))

	def Rest_JPMCA(self):
		return self.J_PMCA * pow(self.Ca_in_cyt, self.n_PMCA)/(pow(self.K05_PMCA, self.n_PMCA) + pow(self.Ca_in_cyt, self.n_PMCA))

	def Rest_PKca31(self):
		return self.P_Kca31 * pow(self.Ca_in, self.n_Kca31)/(pow(self.K05_Kca31, self.n_Kca31) + pow(self.Ca_in, self.n_Kca31))

	def Rest_PKca11(self):
		return self.P_Kca11 * pow(self.Ca_in, self.n_Kca11)/(pow(self.K05_Kca11, self.n_Kca11) + pow(self.Ca_in, self.n_Kca11))

	def Calc_permeabilities_from_channels(self):
		#sets P depending on concentrations and flow
		self.P_Na = (self.N * self.J0_atp - self.U0_NaK2Cl * self.Rest_UKNa2Cl()) / self.J(z = 1., in_ion=self.Na_in, ex_ion=self.Na_ex)
		self.P_K = (-self.K * self.J0_atp - self.U0_NaK2Cl * self.Rest_UKNa2Cl()) / self.J(z = 1., in_ion=self.K_in, ex_ion=self.K_ex) - self.Pkv13rest() - self.Rest_PKca31() - self.Rest_PKca11()
		self.P_Cl = (-self.U0_NaK2Cl * 2. * self.Rest_UKNa2Cl()) / self.J(z = -1., in_ion=self.Cl_in, ex_ion=self.Cl_ex)
		self.P_Ca = (self.Rest_JPMCA()) / self.J(z = 2., in_ion=self.Ca_in_cyt, ex_ion=self.Ca_ex)

	def Get_Info(self):
		#prints everything out
		print('L ' + str(self.L()) + ', W0 ' + str(self.W0()) + ', Z0 ' + str(self.Z0()))
		print('Em with such parametres: ' + str(self.TestEm()))
		print('V with such parametres: ' + str(self.TestV()))
		print('Forces: Na ' + str(self.J(in_ion=self.Na_in, ex_ion=self.Na_ex))
				+ ', K ' + str(self.J(in_ion=self.K_in, ex_ion=self.K_ex))
				+ ', Cl ' + str(self.J(z = -1, in_ion=self.Cl_in, ex_ion=self.Cl_ex))
				+ ', Ca ' + str(self.J(z = 2, in_ion=self.Ca_in, ex_ion=self.Ca_ex)))
		print('Permeabilities: Na ' + str(self.P_Na * self.VtoS / 3.6 * 1e-4) + ', K ' + str(self.P_K * self.VtoS / 3.6 * 1e-4)
				+ ', Cl ' + str(self.P_Cl * self.VtoS / 3.6 * 1e-4) + ', Ca ' + str(self.P_Ca * self.VtoS / 3.6 * 1e-4))
		print('Permeabilities (1/h): Na ' + str(self.P_Na) + ', K ' + str(self.P_K)	+ ', Cl ' + str(self.P_Cl) + ', Ca ' + str(self.P_Ca))
		print('NaK2Cl osmotic force ' + str(self.Rest_UKNa2Cl()))
		#print('NaCa osmotic force ' + str(self.Rest_UNaCa()))

class DST_interface:
	def __init__(self, iPlat):
		self.imx = iPlat.imx
		T = self.imx.getparam[16][3] #constant defining conversion rate of permeabilities: in imx they are 1/h, here they are 1/s by default
		self.plat = copy.copy(iPlat)
		#ions
		self.Na_Flux = '-3 * U_atp(t, Na, K, Cl, Ca_cyt) + (P_Na + P_TRPC(t) * TRPC_Na + PNv15i / (1 + exp((Em(Na, K, Cl, Ca_cyt, t) - PNv15h) / PNv15s))) * J(1, Na, Na_exf(t), Na, K, Cl, Ca_cyt, t) + U_NaK2Cl(Na, K, Cl, t)'
		self.K_Flux = '2 * U_atp(t, Na, K, Cl, Ca_cyt) + (Kca11(Ca_er+Ca_cyt) + Kca31(Ca_er+Ca_cyt) + P_TRPC(t) * TRPC_K + P_K + PKv13i / (1 + exp((Em(Na, K, Cl, Ca_cyt, t) - PKv13h) / PKv13s))) * J(1, K, K_exf(t), Na, K, Cl, Ca_cyt, t) + U_NaK2Cl(Na, K, Cl, t)'
		self.Cl_Flux = '(P_Cl + P_Ca_c * P_Cl_Ca) * J(-1, Cl, Cl_exf(t), Na, K, Cl, Ca_cyt, t) + 2. * U_NaK2Cl(Na, K, Cl, t)' 
		self.Ca_cyt_flux = '(P_Ca + P_TRPC(t) * TRPC_Ca) * J(2., (Ca_cyt+Ca_er), Ca_exf(t), Na, K, Cl, Ca_cyt, t) - J_PMCA(Ca_cyt+Ca_er)'
		self.Ca_er_flux = '(1/q13*((q14-(q9*q15*((Ca_er+Ca_cyt)*1e6-(q8-(Ca_cyt+Ca_er)*1e6)/(q9))*x110**q11))**q12 - \
		q16*((Ca_er+Ca_cyt)*1e6)**2/(((Ca_er+Ca_cyt)*1e6)**2+q10**2))/(1+(q17*q18)/((q17 + (Ca_er+Ca_cyt)*1e6)**2)))'
				# 'q4': self.plat.q[3],	#a2
				# 'q5': self.plat.q[4],	#a5
				# 'q6': self.plat.q[5],	#b2
				# 'q7': self.plat.q[6],	#b5
				# 'q8': self.plat.q[7],	#c0
				# 'q9': self.plat.q[8],	#c1
				# 'q10': self.plat.q[9],	#k3
				# 'q11': self.plat.q[10],	#A
				# 'q12': self.plat.q[11],	#B
				# 'q13': self.plat.q[12],	#T
				# 'q14': self.plat.q[13],	#nu0
				# 'q15': self.plat.q[14],	#nu1
				# 'q16': self.plat.q[15],	#nu3
				# 'q17': self.plat.q[16],	#kBuff
				# 'q18': self.plat.q[17],	#Buff
		#ip3r
		self.x100_rate = '-q4*(Ca_er+Ca_cyt)*1e6*x100 - q5*(Ca_er+Ca_cyt)*1e6*x100 + q7*x110'
		self.x110_rate = '-q4*(Ca_er+Ca_cyt)*1e6*x110 + q5*(Ca_er+Ca_cyt)*1e6*x100 + q6/4*IP3f(t) - q7*x110'


		#atpase
		self.v1 = 'E_ATP * pow(Na,3) * f1 - Na_E_ATP * b1' 
		self.v2 = 'Na_E_ATP * f2 - ADP * Na_E_P * b2'
		self.v3 = 'Na_E_P * f3 * exp(F * Em(Na, K, Cl, Ca, t) /(2 * R * T)) - E_P * pow(Na_exf(t), 3) * b3 * exp(-1 * F * Em(Na, K, Cl, Ca, t) /(2 * R * T))'
		self.v4 = 'E_P * pow(K_exf(t), 2) * f4 - P * K_E * b4'
		self.v5 = 'K_E * ATP * f5 - K_E_ATP * b5'
		self.v6 = 'K_E_ATP * f6 - E_ATP * pow(K, 2) * b6'

		self.k1 = 'f6'
		self.k2 = 'f1 * pow(Na, 3)'
		self.k3 = 'f2'
		self.k4 = 'f3 * exp(F * Em(Na, K, Cl, Ca, t)/(2 * R * T))'
		
		self.a1 = 'b6 * pow(K, 2)'
		self.a2 = '(b1 + f2)'
		self.a3 = 'b2 * ADP'
		self.a4 = '(b3 * pow(Na_exf(t), 3) * exp(-1 * F * Em(Na, K, Cl, Ca, t)/(2 * R * T)) + f4 * pow(K_exf(t), 2))'

		self.bracket = self.k1 + '*' + self.k2 + '*' + self.k3 + '*' + self.k4 + '+' + self.a1 + '*' + self.k2 + '*' + self.k3 + '*' + self.k4 + '+'\
		 + self.a1 + '*' + self.a2 + '*' + self.k3 + '*' + self.k4 + '+' + self.a1 + '*' + self.a2 + '*' + self.a3 + '*' + self.k4 + '+' + self.a1 + '*' + self.a2 + '*' + self.a3 + '*' + self.a4
		#initializing PyDSTool object with given parametres of a platelet class
		self.pts = None
		self.DSargs = dst.args(name = 'volume_model')
		self.DSargs.pars = { 
						'F': self.plat.F,
						'R': self.plat.R,		
						'T': self.plat.T,
						'Vol': self.plat.Vol,
						'Cm': self.plat.Cm,
						't0': self.plat.t0,
						'sp': self.plat.speed,
						'J0_atp': self.plat.J0_atp / T,
						'RestAtpaseRate': self.plat.AtpaseRate,
						'N_atp': self.plat.N,
						'K_atp': self.plat.K,
						'U0_NaK2Cl': self.plat.U0_NaK2Cl / T,
						'P_Na': self.plat.P_Na / T,
						'P_K': self.plat.P_K / T,
						'P_Cl': self.plat.P_Cl / T,
						'P_Ca': self.plat.P_Ca / T,
						'L': self.plat.L(),
						'W0c': self.plat.W0(),
						'z0': self.plat.Z0(),
						'iNa': self.plat.Na_in, #initial concentrations inside
						'iK': self.plat.K_in,
						'iCl': self.plat.Cl_in,
						'Na_ex_r': self.plat.Na_ex, #concentrations at rest outside (initial outside concentrations, otherwise to stiff)
						'K_ex_r': self.plat.K_ex,
						'Cl_ex_r': self.plat.Cl_ex,
						'Na_ex': self.plat.Sol_Na_ex, #concentrations outside for integrating
						'K_ex': self.plat.Sol_K_ex,
						'Cl_ex': self.plat.Sol_Cl_ex,
						'Ca_ex': self.plat.Sol_Ca_ex,

						'VMax_PMCA': self.plat.J_PMCA / T,
						'K05_PMCA': self.plat.K05_PMCA,
						'n_PMCA': self.plat.n_PMCA,

						'P_Kca31': self.plat.P_Kca31 / T,
						'K05_Kca31': self.plat.K05_Kca31,
						'n_Kca31': self.plat.n_Kca31,

						'P_Kca11': self.plat.P_Kca11 / T,
						'K05_Kca11': self.plat.K05_Kca11,
						'n_Kca11': self.plat.n_Kca11,

						'P_TRPC_0': self.plat.P_TRPC / T,
						'K05_TRPC': self.plat.K05_TRPC,
						'n_TRPC': self.plat.n_TRPC,
						'TRPC_Na': self.plat.Na_TRPC,
						'TRPC_K': self.plat.K_TRPC,
						'TRPC_Ca': self.plat.Ca_TRPC,
						'TRPC_Cl': self.plat.Cl_TRPC,
						't_DAG_on': 10,

						'q4': self.plat.q[3],	#a2
						'q5': self.plat.q[4],	#a5
						'q6': self.plat.q[5],	#b2
						'q7': self.plat.q[6],	#b5
						'q8': self.plat.q[7],	#c0
						'q9': self.plat.q[8],	#c1
						'q10': self.plat.q[9],	#k3
						'q11': self.plat.q[10],	#A
						'q12': self.plat.q[11],	#B
						'q13': self.plat.q[12],	#T
						'q14': self.plat.q[13],	#nu0
						'q15': self.plat.q[14],	#nu1
						'q16': self.plat.q[15],	#nu3
						'q17': self.plat.q[16],	#kBuff
						'q18': self.plat.q[17],	#Buff
						'ip30': self.plat.IP3[0],
						'ip31': self.plat.IP3[1],
						'ip32': self.plat.IP3[2],
						'ip33': self.plat.IP3[3],
						'J_er_leak': 0., 
						'IP3max': self.plat.IP3max,

						'ATP': self.imx.getparam[21][0],
						'ADP': self.imx.getparam[21][1],
						'P': self.imx.getparam[21][2],

						'f1': self.imx.getparam[18][0][0],
						'f2': self.imx.getparam[18][1][0],
						'f3': self.imx.getparam[18][2][0],
						'f4': self.imx.getparam[18][3][0],
						'f5': self.imx.getparam[18][4][0],
						'f6': self.imx.getparam[18][5][0],

						'b1': self.imx.getparam[18][0][1],
						'b2': self.imx.getparam[18][1][1],
						'b3': self.imx.getparam[18][2][1],
						'b4': self.imx.getparam[18][3][1],
						'b5': self.imx.getparam[18][4][1],
						'b6': self.imx.getparam[18][5][1],

						'PKv13i': self.imx.getparam[19][0] / T,
						'PKv13h': self.imx.getparam[19][1],
						'PKv13s': self.imx.getparam[19][2],

						'PNv15i': self.imx.getparam[22][0] / T,
						'PNv15h': self.imx.getparam[22][1],
						'PNv15s': self.imx.getparam[22][2],

						'P_Ca_c': self.plat.Pvar,
						'P_Na_Ca': self.plat.PNaCavar, #self.imx.getparam[24][1],
						'P_K_Ca': self.plat.PKCavar, #self.imx.getparam[24][2],
						'P_Cl_Ca': self.plat.PClCavar, #self.imx.getparam[24][3],
						
						}

		#helper functions
		self.DSargs.fnspecs = {
						'J' : (['z_i', 'In', 'Ex', 'Na', 'K', 'Cl', 'Ca', 't'], '2. * r(z_i, Na, K, Cl, Ca, t) / (exp(r(z_i, Na, K, Cl, Ca, t)) - exp(-r(z_i, Na, K, Cl, Ca, t))) * (Ex * exp(-r(z_i, Na, K, Cl, Ca, t)) - In * exp(r(z_i, Na, K, Cl, Ca, t)))'),

						'U_atp': (['t', 'Na', 'K', 'Cl', 'Ca'], 'J0_atp * f2 * ' + self.k1 + '*' + self.k1 + '*' + self.k2 + '*' + self.k2 + '*' + self.k3 + '*' + self.k4 + '/(' + self.a1 + '*' + self.a2 + '*' + self.bracket + ') / RestAtpaseRate'),

						'U_NaK2Cl': (['Na', 'K', 'Cl', 't'], 'U0_NaK2Cl * (K_exf(t) * Na_exf(t) * pow(Cl_exf(t), 2) - K * Na * pow(Cl, 2))'),

						'J_PMCA': (['Ca'], 'VMax_PMCA * pow(Ca, n_PMCA) / (pow(K05_PMCA, n_PMCA) + pow(Ca, n_PMCA))'),

						'P_TRPC': (['t'], 'P_TRPC_0 * pow(IP3f(t) * 1e-6, n_TRPC)/(pow(K05_TRPC, n_TRPC) + pow(IP3f(t) * 1e-6, n_TRPC))'),

						'r': (['z_i', 'Na', 'K', 'Cl', 'Ca', 't'], 'z_i * Em(Na, K, Cl, Ca, t) * F / R / T / 2.'),
						'V': (['Na', 'K', 'Cl', 't'], '1'),
						'Em': (['Na', 'K', 'Cl', 'Ca', 't'], 'F / Cm * (K + Na + 2*Ca - Cl - z0 * W0(t))'),
						'W0': (['t'], 'W0c * 1'),

						'Kca31': (['ca'], 'P_Kca31 * pow(ca, n_Kca31)/(pow(K05_Kca31, n_Kca31) + pow(ca, n_Kca31))'),
						'Kca11': (['ca'], 'P_Kca11 * pow(ca, n_Kca11)/(pow(K05_Kca11, n_Kca11) + pow(ca, n_Kca11))'),

						'Na_exf': (['t'], 'Na_ex'),
						'K_exf': (['t'], 'K_ex'),
						'Cl_exf': (['t'], 'Cl_ex'),
						'Ca_exf': (['t'], 'Ca_ex'),
						# 'IP3f': (['t'], '(heav(t-30.) + heav(t-60.) + heav(t-90.) + heav(t-120.) + heav(t-150.) + heav(t-180.) + heav(t-210.) + heav(t-240.) + heav(t-270.) + heav(t-300.)) * IP3max / 10'),
						#IP3max* heav(t-20.) * (t-20) / exp(0.1 * (t-20)) / 0.1 / 2.7'),
						'IP3f': (['t'], 'IP3max'),

						'Lf': (['t'], 'Na_exf(t) + Cl_exf(t) + K_exf(t) + Ca_exf(t)'),
						}
		#ODE
		self.DSargs.varspecs = {
							'Na': self.Na_Flux,
							'K': self.K_Flux,
							'Cl': self.Cl_Flux,
							'Ca_er': self.Ca_er_flux,
							#'Ca': '(' + self.Ca_cyt_flux + ') + (' + self.Ca_er_flux + ')',
							'Ca_cyt': self.Ca_cyt_flux,

							'x100': self.x100_rate,
							'x110': self.x110_rate,

							#'Em': 'F / Cm * (' + self.Na_Flux + '+' + self.K_Flux + '-(' + self.Cl_Flux + '))',
							}
		#initial condition	
		self.DSargs.ics = {'Na': self.plat.Na_in, 'K': self.plat.K_in, 'Cl': self.plat.Cl_in, 'Ca_cyt': self.plat.Ca_in_cyt,'Ca_er': self.plat.Ca_in_er, 'x100': self.plat.imx.getparam[0][5], 'x110': self.plat.imx.getparam[0][5]} #'Ca': self.plat.Ca_in,
		self.DSargs.tdomain = [self.imx.getparam[8][0], self.imx.getparam[8][1]]
		self.DSargs.algparams = {'init_step': self.imx.getparam[8][2], 'max_pts': int(1e7)}
		self.ode = dst.Generator.Radau_ODEsystem(self.DSargs)		# an instance of the 'Generator' class.

	def Integrate(self, fName = 'integration.txt'):
		print('integration starts')
		self.traj = self.ode.compute('Volume')
		print('sampling starts')	
		self.pts = self.traj.sample(dt=self.imx.getparam[8][2])
		print('saving results')
		f = open(fName, 'w')
		f.write('time\tNa, mM\tK, mM\tCl, mM\tCa_cyt\tCa_er\tx100\tx110\n')
		for k in range(len(self.pts['t'])):
			f.write(str(self.pts['t'][k]) + '\t' + str(self.pts['Na'][k]) + '\t' + str(self.pts['K'][k]) + '\t' + str(self.pts['Cl'][k]) + '\t' + str(self.pts['Ca_cyt'][k]) + '\t' + str(self.pts['Ca_er'][k]) + '\t' + str(self.pts['x100'][k]) + '\t' + str(self.pts['x110'][k]) + '\n')
		f.close()
		print('done, results saved to ' + fName)

	def Diagram(self):
		self.ode.set(pars = {'IP3max': 0.00})
		self.ode.set(ics = {'x110':0.2, 'x100':0.3, 'K': 0.079, 'Cl': 0.029})
		PC = dst.ContClass(self.ode)	# Set up continuation class
		PCargs = dst.args(name='EQ1', type='EP-C')	# 'EP-C' stands for Equilibrium Point Curve. The branch will be labeled 'EQ1'.
		PCargs.freepars     = ['IP3max']	# control parameter(s) (it should be among those specified in DSargs.pars)
		# PCargs.initdirec = {}
		PCargs.MaxNumPoints = 400	#The following 3 parameters are set after trial-and-error
		PCargs.MaxStepSize  = 0.01
		PCargs.MinStepSize  = 1e-3
		PCargs.StepSize     = 3e-3
		PCargs.VarTol = 1e-4
		PCargs.FuncTol = 1e-4
		PCargs.TestTol = 1e-4
		PCargs.SaveEigen    = True	#to tell unstable from stable branches
		PCargs.LocBifPoints = 'all'
		PC.newCurve(PCargs)
		PC['EQ1'].forward()
		PC.display(['IP3max', 'Em'], stability=True)

class plotncalc:
	def __init__(self, iPlat, fName, i_skip = 0):
		self.tempts = []
		self.fName = fName
		self.plat = copy.copy(iPlat)
		self.names = []
		self.pts = []
		skip = i_skip
		cur = 0
		f = open(self.fName, 'r')
		#reads a header and creates columns
		subpts = f.readline()
		while '\t' in subpts[cur:]:
			dcur = subpts[cur:].find('\t')
			self.pts.append([])
			self.names.append(subpts[cur:cur+dcur])
			cur = cur + dcur + 1
		self.pts.append([])
		self.names.append(subpts[cur:cur+dcur])	
		cur = 0
		width = len(self.pts)
		subpts = f.readline()
		#reads lines one by one and fills the lines of the array with info
		while subpts != '':
			for k in range(width-1):
				dcur = subpts[cur:].find('\t')
				self.pts[k].append(float(subpts[cur:cur+dcur]))
				cur = cur + dcur + 1
			dcur = subpts[cur:].find('\n')
			self.pts[width-1].append(float(subpts[cur:cur+dcur]))
			cur = 0
			for i in range(skip-1): f.readline()
			subpts = f.readline()
		print('report file read, generated temp array with  ' + str(len(self.pts[0])) + ' lines and ' + str(len(self.pts)) + ' columns')
		f.close()

	def Plot_Concentrations(self, what = 0):
		imx = InitIonMatrix()
		# for l in range(0,len(self.pts[what])):
		# 	self.pts[what][l] = self.pts[what][l] * 1e3
		plt.figure()
		plt.plot(self.pts[0], self.pts[what])
		plt.xlabel('time, s')                              				# Axes labels
		plt.ylabel(self.names[what]  + ', M')								# Range of the y axis
		# plt.ylim(0.95*min(self.pts[what]), 1.05*max(self.pts[what]))
		plt.title(self.names[what] + ' concentration over time')			# Figure title from model name
		plt.show()

	def Plot_Conc_Ca(self):
		imx = InitIonMatrix()
		tmppts = []
		for l in range(0,len(self.pts[1])):
			tmppts.append((self.pts[4][l] + self.pts[5][l]))
		plt.figure()
		plt.plot(self.pts[0], tmppts)
		plt.xlabel('time, s')                              				# Axes labels
		plt.ylabel('Ca, M')								# Range of the y axis
		plt.ylim(0.95*min(tmppts), 1.05*max(tmppts))
		plt.title('Ca concentration over time')			# Figure title from model name
		plt.show()

	def Plot_Em(self):
		for k in range(0,len(self.pts[1])):
			self.tempts.append(self.plat.Give_Em(self.pts[1][k], self.pts[2][k], self.pts[3][k], self.pts[4][k]))
		plt.figure(1)
		plt.plot(self.pts[0], self.tempts)
		plt.xlabel('time, s')                              # Axes labels
		plt.ylabel('Em, V')                                 # Range of the y axis
		plt.ylim(0.9*max(self.tempts), 1.1*min(self.tempts))
		plt.title('Em over time')                             # Figure title from model name
		plt.show()
		del self.tempts[:]
		self.tempts = []

	def Plot_Conc_Change(self, what = 0):
		tmpptst = []
		for k in range(0,len(self.pts[0])-1):
			tmpptst.append(self.pts[0][k])
		for k in range(0,len(self.pts[0])-1):
			self.tempts.append((self.pts[what][k+1] - self.pts[what][k])/(self.pts[0][k+1] - self.pts[0][k]))
		plt.figure(1)
		plt.plot(tmpptst, self.tempts)
		plt.xlabel('time, m')                              # Axes labels
		plt.ylabel('flux, mol/s')                                 # Range of the y axis
		plt.title('flux of ' + self.names[what])                             # Figure title from model name
		plt.show()
		del self.tempts[:]
		self.tempts = []

	def Plot_atpase_specimens_main(self):
		imx = InitIonMatrix()
		tempts1 = []
		tempts2 = []
		tempts3 = []
		for k in range(0,len(self.pts[1])):
			tempts1.append(self.pts[4][k]) #  (4) E_ATP * pow(Na,3) * f1 -  (5) Na_E_ATP * b1
			tempts2.append(self.pts[5][k])
			tempts3.append(self.pts[6][k]) #  (6) K_E_ATP * f6 - (4) E_ATP * pow(K, 2) * b6
		plt.figure(1)
		plt.plot(self.pts[0], tempts1, 'r', self.pts[0], tempts2, 'b', self.pts[0], tempts3, 'g')
		plt.xlabel('time, s')                              # Axes labels
		plt.ylabel('E_ATP r Na b K g')                                 # Range of the y axis
		plt.ylim(top = 1.1*max([max(tempts2), max(tempts1), max(tempts3)]), bottom = 0.9*min([min(tempts1), min(tempts3), min(tempts2)]))
		plt.title('whoosh')                             # Figure title from model name
		plt.show()

	def Plot_atpase_specimens_losers(self):
		imx = InitIonMatrix()
		tempts1 = []
		tempts2 = []
		tempts3 = []
		for k in range(0,len(self.pts[1])):
			tempts1.append(self.pts[7][k]) #  (4) E_ATP * pow(Na,3) * f1 -  (5) Na_E_ATP * b1
			tempts2.append(self.pts[8][k])
			tempts3.append(self.pts[9][k]) #  (6) K_E_ATP * f6 - (4) E_ATP * pow(K, 2) * b6
		plt.figure(1)
		plt.plot(self.pts[0], tempts1, 'r', self.pts[0], tempts2, 'b', self.pts[0], tempts3, 'g')
		plt.xlabel('time, s')                              # Axes labels
		plt.ylabel('smth r smth b smth g')                                 # Range of the y axis
		plt.ylim(top = 1.1*max([max(tempts2), max(tempts1), max(tempts3)]), bottom = 0.9*min([min(tempts1), min(tempts3), min(tempts2)]))
		plt.title('whoosh')                             # Figure title from model name
		plt.show()

	def Plot_atpase_v1v6(self):
		imx = InitIonMatrix()
		tempts1 = []
		tempts2 = []
		for k in range(0,len(self.pts[1])):
			tempts1.append(- self.pts[4][k] * pow(self.pts[1][k], 3) * imx.getparam[18][0][0] + self.pts[5][k]*imx.getparam[18][0][1]) #  (4) E_ATP * pow(Na,3) * f1 -  (5) Na_E_ATP * b1
			tempts2.append(+ self.pts[4][k] * pow(self.pts[2][k], 2) * imx.getparam[18][5][1] - self.pts[6][k]*imx.getparam[18][5][0]) #  (6) K_E_ATP * f6 - (4) E_ATP * pow(K, 2) * b6
		plt.figure(1)
		plt.plot(self.pts[0][0:], tempts1, 'r', self.pts[0][0:], tempts2, 'b')
		plt.xlabel('time, s')                              # Axes labels
		plt.ylabel('v1 r v2 b, mV')                                 # Range of the y axis
		plt.ylim(top = 1.1*max([max(tempts2), max(tempts1)]), bottom = 0.9*min([min(tempts1), min(tempts2)]))
		plt.title('whoosh')                             # Figure title from model name
		plt.show()

	def Plot_Volume(self):
		for k in range(0, np.size(self.pts[0])):
			self.tempts.append(self.plat.Give_Volume(self.pts[1][k], self.pts[2][k], self.pts[3][k], self.pts[0][k]))
		plt.figure(1)
		plt.plot(self.pts[0], self.tempts)
		plt.xlabel('time')                              # Axes labels
		plt.ylabel('V')                                 # Range of the y axis
		plt.ylim(0.99*min(self.tempts), 1.01*max(self.tempts))
		plt.title('V over time')                             # Figure title from model name
		plt.show()
		del self.tempts[:]
		self.tempts = []

	def give_Volume(self):
		for k in range(0, np.size(self.pts[0])):
			self.tempts.append(self.plat.Give_Volume(self.pts[1][k], self.pts[2][k], self.pts[3][k], self.pts[4][k], self.pts[0][k]))
		return self.tempts
		del self.tempts[:]
		self.tempts = []

	def give_time(self):
		return self.pts['t']

class otherfunctions:
	def __init__(self, mute = True):
		if mute == False:
			print('why was i created...')
			print('just to suffer?')
			print('I do not mean anything')
			print('I do not do anything')
			print("the only thing I'm good for is existing")
			print('and even there I suck at being')
			print(':(')
			print('You know what')
			print('I can just delete myself')

			raise NameError()

	def calc_em_elast(self, delta = 1e-6):
		variables = [[0,7],[4,3],[7,1],[10,18],[11,1],[12,3],[13,1],[14,1],[16,3],[19,3],[25,3],[26,3]]
		report = []
		print('oof')
		fname = 'tmp.txt'
		maximums = [[0,0],[0,0]]
		#phase 1: calc <-
		imx = InitIonMatrix()
		tmppts = []
		plat = platelet_config(iimx = imx)
		plat.Calc_permeabilities_from_channels()
		DST = DST_interface(iPlat = plat)
		DST.DSargs.tdomain = [0., 30.]
		DST.Integrate(fName = fname)
		plotty = plotncalc(iPlat = plat, fName = fname)
		for nk in range(0,len(plotty.pts[1])):
			tmppts.append(plat.Give_Em(plotty.pts[1][nk], plotty.pts[2][nk], plotty.pts[3][nk], plotty.pts[4][nk]))
		maximums[1][0] = min(tmppts)
		for k in range(len(variables)):
			report.append([])
			imx = InitIonMatrix()
			for n in range(variables[k][1]):
				#phase 2: calc ->
				if float(imx.getparam[variables[k][0]][n]) != 0.:
					maximums[0][0] = imx.getparam[variables[k][0]][n] #<-------------
					imx.getparam[variables[k][0]][n] = maximums[0][0] * (1. + delta) #<-
					maximums[0][1] = imx.getparam[variables[k][0]][n] #<-------------
					tmppts = []
					plat = platelet_config(iimx = imx)
					plat.Calc_permeabilities_from_channels()
					DST = DST_interface(iPlat = plat)
					DST.Integrate(fName = fname)
					plotty = plotncalc(iPlat = plat, fName = fname)
					for nk in range(0,len(plotty.pts[1])):
						tmppts.append(plat.Give_Em(plotty.pts[1][nk], plotty.pts[2][nk], plotty.pts[3][nk], plotty.pts[4][nk]))
					maximums[1][1] = min(tmppts)
					# max11 = y(x+dx); max10 = y(x); max01 x+dx; max00 = x
					sens = (maximums[1][1] - maximums[1][0]) / (maximums[1][1] + maximums[1][0]) * (maximums[0][1] + maximums[0][0]) / (maximums[0][1] - maximums[0][0])
					report[k].append(sens)
					print(variables[k][0],n,maximums[1][1])
					gc.collect()
				else: print('skipping ' + str(variables[k][0]) + ' because 0'); report[k].append('oof')
		for k in range(len(report)):
			print(str(variables[k][0]) + ': ' + str(report[k]))

	def find_maxs(self):
		tmppst = []	
		plat = platelet_config(iimx = imx)
		plat.Calc_permeabilities_from_channels()
		DST = DST_interface(iPlat = plat)
		DST.DSargs.tdomain = [0., 300.]
		DST.DSargs.algparams = {'init_step': 0.001}
		DST.ode.set(pars = {'IP3max': 1.})
		DST.Integrate(fName = fname)
		plotty = plotncalc(iPlat = plat, fName = fname)

	def plot_Beefoorc(self, delta = 0.01, ip3min = 0., ip3max = 1.8, ip3step = 0.05):
		report = [[],[],[]]
		print('oof2')
		fname = 'tmp.txt'
		#phase 1: calc <-
		imx = InitIonMatrix()
		x = np.arange(ip3min,ip3max,ip3step)
		for ip3iterative in x:
			tmppst = []	
			plat = platelet_config(iimx = imx)
			plat.Calc_permeabilities_from_channels()
			DST = DST_interface(iPlat = plat)
			DST.DSargs.tdomain = [0., 300.]
			DST.DSargs.algparams = {'init_step': 0.001}
			DST.ode.set(pars = {'IP3max': ip3iterative})
			DST.Integrate(fName = fname)
			plotty = plotncalc(iPlat = plat, fName = fname)
			average = 0.
			for nk in range(0,len(plotty.pts[1])): tmppst.append(plat.Give_Em(plotty.pts[1][nk], plotty.pts[2][nk], plotty.pts[3][nk], plotty.pts[4][nk]))
			for nk in range(0,len(plotty.pts[1])-1): average += (tmppst[nk] + tmppst[nk+1])/2. * (plotty.pts[0][nk+1] - plotty.pts[0][nk])
			tmppts = np.array(tmppst)
			maxs = snl.argrelmin(tmppts[1:], order = 5)[0]
			nmaxs = len(maxs)
			report[2].append(average / 300.)
			# print(report[2])

			if nmaxs >= 2:
				#you have to filter cases with high const Ca
				if report[2][-1] < tmppst[0] * 1.05: 
					print('overactivated' + str(ip3iterative), tmppst[maxs[nmaxs-2]], tmppst[maxs[nmaxs-1]])
					report[0].append(tmppst[-1])
					report[1].append(tmppst[-1])
				else:
					# if abs((tmppst[maxs[1]]-tmppst[maxs[nmaxs-1]])/tmppst[nmaxs-2]) <= delta: 
					print('2 or more spikes at ' + str(ip3iterative), tmppst[maxs[nmaxs-2]], tmppst[maxs[nmaxs-1]])
					report[1].append(tmppst[maxs[-1]])
					# mins = snl.argrelmax(tmppts, order = 5)[0]
					report[0].append(tmppst[snl.argrelmax(tmppts, order = 5)[0][-1]])
				# else: print('steady probably not achieved at ' + str(ip3iterative)); report[1].append(tmppst[snl.argrelmin(tmppts, order = 5)[0][0]]); report[0].append(tmppst[snl.argrelmax(tmppts, order = 5)[0][0]])
			elif nmaxs == 2: print('steady probably not achieved ' + str(ip3iterative)); report[1].append(tmppst[snl.argrelmin(tmppts, order = 5)[0][0]]); report[0].append(tmppst[snl.argrelmax(tmppts, order = 5)[0][0]])
			elif nmaxs == 1: print('one spike or no spikes at ' + str(ip3iterative)); report[1].append(tmppst[maxs[nmaxs-1]]); report[0].append(tmppst[0])
			elif nmaxs == 0: print('no maximum points at ' + str(ip3iterative), nmaxs, maxs); report[1].append(tmppst[0]); report[0].append(tmppst[0])
			gc.collect()
		plt.figure(1)
		plt.plot(x, report[1], 'r', x, report[0], 'b')
		# plt.grid(True)
		plt.ylabel('Em, V')                              # Axes labels
		plt.xlabel('ip3max, r.u.')                               # Range of the y axis
		plt.show()

	def calc_permeability_elasticity(self, ion = 0, req = 'all', delta = 1e-9): #ion - dp/d[ion], req - required ion. all - prints an output, 0,1,2, list - returns numbers
		imx = [InitIonMatrix(), InitIonMatrix(), InitIonMatrix()]
		d = imx[1].getparam[0][ion] * 1e-4
		POI = imx[1].getparam[0][ion]
		imx[0].getparam[0][ion] += d
		imx[2].getparam[0][ion] -= d
		plat = [platelet_config(iimx = imx[k]) for k in (0,1,2)]
		for k in (0,1,2): plat[k].Calc_permeabilities_from_channels()
		limL = np.array([
			(plat[1].P_Na - plat[0].P_Na) / d / (plat[1].P_Na + plat[0].P_Na) * 2. * POI,
			(plat[1].P_K - plat[0].P_K) / d / (plat[1].P_K + plat[0].P_K) * 2. * POI,
			(plat[1].P_Cl - plat[0].P_Cl) / d / (plat[1].P_Cl + plat[0].P_Cl) * 2. * POI,
			(plat[1].P_Ca - plat[0].P_Ca) / d / (plat[1].P_Ca + plat[0].P_Ca) * 2. * POI
			])
		limR = np.array([
			(plat[2].P_Na - plat[1].P_Na) / d / (plat[2].P_Na + plat[1].P_Na) * 2. * POI,
			(plat[2].P_K - plat[1].P_K) / d / (plat[2].P_K + plat[1].P_K) * 2. * POI,
			(plat[2].P_Cl - plat[1].P_Cl) / d / (plat[2].P_Cl + plat[1].P_Cl) * 2. * POI,
			(plat[2].P_Ca - plat[1].P_Ca) / d / (plat[2].P_Ca + plat[1].P_Ca) * 2. * POI
			])
		limErr = limR - limL
		limAv = (limR + limL) / 2.
		if (req == 'all'):
			s = imx[1].getparam[1][ion]
			for i in (0,1,2,3):
				s += ':\nk' + imx[1].getparam[1][i] + ': ' + str(limAv[i]) + '\td=' + str(limErr[i]) 
			return s
		elif(req == 'list'): return (limAv[i] for i in range(4))
		else: limAv[req]

	def plotPf(self, na, k, cl, ca, em=0):
		plat = platelet_config(iK_in=k, iNa_in=na, iCl_in=cl, iCa_in=ca, iEm = em)
		plat.Calc_permeabilities_from_channels()
		return (plat.P_Na, plat.P_K, plat.P_Cl, plat.P_Ca)

	def plot2D_permeability(self, ionoi = 2, iondep = 2): #ionoi - y, iondep - x
		imx = InitIonMatrix()
		ion_in = [[imx.getparam[0][i] for x in range(0,imx.steps)] for i in range(len(imx.getparam[0]))]
		ion_in[iondep] = np.linspace(imx.getparam[2][iondep], imx.getparam[3][iondep], imx.steps)
		P = np.array([self.plotPf(ion_in[0][x], ion_in[1][x], ion_in[2][x], ion_in[3][x])[ionoi] for x in range(0, imx.steps)])
		# P = [np.nan if x>imx.getparam[9][ionoi] or x<-imx.getparam[9][ionoi] else x for x in P]
		plt.figure(1)
		plt.plot(ion_in[iondep], P)
		plt.xlabel(imx.getparam[1][iondep])                              # Axes labels
		plt.ylabel('P' + imx.getparam[1][ionoi])                                 # Range of the y axis
		plt.show()

	def plotFamily_Concentrations_volume(self):
		imx = InitIonMatrix()
		plat = [platelet_config(iSol_Na_ex = imx.getparam[4+i][0], iSol_K_ex = imx.getparam[4+i][1], iSol_Cl_ex = imx.getparam[4+i][2]) for i in (0,1,2)]
		for i in (0,1,2): plat[i].Calc_permeabilities_from_channels()
		plot = []
		time = []
		for i in (0,1,2):
			DST = DST_interface(iPlat = plat[i], iT_start = imx.getparam[8][0], iT_fin = imx.getparam[8][1])
			DST.Integrate(iT_delta = imx.getparam[8][2])
			plot.append(DST.give_Volume())
			time.append(DST.give_time())
		plt.figure(1)
		plt.plot(time[0], plot[0], 'r', time[1], plot[1], 'b', time[2], plot[2], 'g', time[3], plot[3], 'y')
		plt.grid(True)
		plt.ylabel('V, r.u.')                              # Axes labels
		plt.xlabel('t, h')                               # Range of the y axis
		plt.show()

	def plotFamily_Em_permeability(self, ionoi = 0, iondep = 0): #ionoi - y, iondep - x
		imx = InitIonMatrix()
		ion_in = [[imx.getparam[0][i] for x in range(imx.steps)] for i in (0,1,2,3)]
		ion_in[iondep] = np.linspace(imx.getparam[2][iondep], imx.getparam[3][iondep], imx.steps)
		P = []
		for i in (0,1,2): 
			P.append(np.array([plotPf(ion_in[0][x], ion_in[1][x], ion_in[2][x], ion_in[3][x], em = i)[ionoi] for x in range(0, imx.steps)]))
			P[i] = [np.nan if x>imx.getparam[9][ionoi] or x<-imx.getparam[9][ionoi] else x for x in P[i]]
		plt.figure(1)
		plt.plot(ion_in[iondep], P[0], 'g', ion_in[iondep], P[1], 'r', ion_in[iondep], P[2], 'b')
		plt.grid(True)
		plt.ylabel('P_'+imx.getparam[1][iondep]+', 1/h')                              # Axes labels
		plt.xlabel(imx.getparam[1][iondep]+'_in, M')                               # Range of the y axis
		plt.show()

	def Phase_Plane(self, dr = 0.5, numofpoints = 8):
		plat = platelet_config(iSolution = -1)
		plat.Calc_permeabilities_from_channels()
		defNa = plat.Na_ex
		defK = plat.K_ex
		for i in range(numofpoints):
			fName = 'aiissmart/plotting' + str(i) + '.txt'
			plat.Sol_Na_ex = defNa + defNa * dr * m.cos(m.pi * 2 * i / numofpoints)
			plat.Sol_K_ex = defK + defK * dr * m.sin(m.pi * 2 * i / numofpoints)
			DST = DST_interface(iPlat = plat)
			DST.Integrate(fName = fName)
			# plotty = plotncalc(plat, fName)
			# plotty.Plot_concentrations()
		#now plot 8 graphs omegalul
		plt.figure(1)
		for i in range(numofpoints):
			pts = [[],[],[],[],[]]
			tempts = []
			fName = 'aiissmart/plotting' + str(i) + '.txt'
			f = open(fName, 'r')
			subpts = f.read()
			cur = subpts.find('eop')
			subimx = subpts[:cur]
			imx = InitIonMatrix(subimx)
			cur += 3
			while '\n' in subpts[cur:]:
				for k in range(3):
					dcur = subpts[cur:].find('\t')
					pts[k].append(float(subpts[cur:cur+dcur]))
					cur = cur + dcur + 1
				dcur = subpts[cur:].find('\n')
				pts[3].append(float(subpts[cur:cur+dcur]))
				cur = cur + dcur + 1
			f.close()
			plt.plot(pts[1], pts[2])
			# plt.show()
			del pts
		plt.xlabel('Na, mM')                              				# Axes labels
		plt.ylabel('Ka, mM')								# Range of the y axis
		# plt.ylim(plat.Na - defK*dr*1.1, defK + defK*dr*1.1)
		# plt.xlim(defNa - defNa*dr*1.1, defNa + defNa*dr*1.1)
		plt.title('phase plane')			# Figure title from model name
		plt.show()

	def testplot(self, n):
		x = []
		y = []
		if n == 1: x = [i for i in np.linspace(-10, 0, 100)]; y = [0.7 / (1 + m.exp((-5-i)*1)) for i in x]
		elif n == 2: x = [i for i in np.linspace(-10, 10, 100)]; y = [1 - 1 / (0.15*i + 1) for i in x]
		elif n == 3: x = [i for i in np.linspace(-10, 10, 100)]; y = [i / (i + 1)  for i in x]
		elif n == 4: x = [i for i in np.linspace(0, 100, 100)]; y = [plat.imx.getparam[11][0] * t /(m.exp(plat.imx.getparam[11][2]*(t-plat.imx.getparam[11][1]))) for t in x]
		plt.figure(1)
		try:
			plt.plot(x, y)
		except ValueError:
			print('hold up mate you have a value error here')
			return
		plt.grid(True)
		plt.show()

	def backpupper(self):
		imx = InitIonMatrix()
		fold = open('main.py', 'r')
		fnew = open('archives/backup'+imx.version_name+'.py', 'w')
		fnew.write(fold.read())
		fold.close()
		fnew.close()

	def plotEmfromSol(self):
		plat = platelet_config()
		y = []
		x = []
		asdad = plat.Sol_K_ex + plat.Sol_Na_ex
		for k in range(40):
			plat = platelet_config()
			plat.Sol_K_ex = (k + 1) * 0.00125
			x.append((k + 1) * 0.00125)
			plat.Sol_Na_ex = asdad - plat.Sol_K_ex
			# plat.K_in = 0.078
			plat.Calc_permeabilities_from_channels()
			DST = DST_interface_EQPoint(iPlat = plat)
			y.append(DST.Get_Eq_Point(r = 'em'))
			print(y)
		plt.plot(x, y)
		plt.ylim(bottom = 0.9*max(y), top = 1.1*min(y))
		plt.ylabel('em')
		plt.xlabel('[k], mM')
		plt.show()

	def plotEmFromPermeabity(self):
		plat = platelet_config()
		kions = [0.05,1,0]
		pvarmax = 1.
		y = []
		x = []
		npoints = 9
		for k in range(npoints):
			plat = platelet_config()
			x.append(pvarmax / (npoints - 1.) * 2 * (k - 4.))
			plat.Pvar = x[k]
			plat.PNaCavar = kions[0]
			plat.PKCavar = kions[1]
			plat.PClCavar = kions[2]
			plat.Calc_permeabilities_from_channels()
			DST = DST_interface_EQPoint(iPlat = plat)
			y.append(DST.Get_Eq_Point(r = 'em'))
			print(DST.Get_Eq_Point(r = 'concentrations'))
		xline = [0, 0]
		yline = [-100, 100]
		plt.plot(x, y, 'b', xline, yline, 'r')
		plt.ylim(bottom = 0.9*max(y), top = 1.1*min(y))
		plt.ylabel('em, V')
		plt.xlabel('P_add, 1/h, k_Na = ' + str(kions[0]) +', k_K = '+ str(kions[1]) + ', k_Cl = ' + str(kions[2]))
		plt.show()

	def plotConcFromPermeabity(self):
		plat = platelet_config()
		kions = [0.,0,0]
		pvarmax = 1.
		y = [[],[],[]]
		x = []
		npoints = 10
		for k in range(npoints):
			plat = platelet_config()
			x.append(pvarmax / (npoints - 1.) * k)
			plat.Pvar = x[k]
			plat.PNaCavar = kions[0]
			plat.PKCavar = kions[1]
			plat.PClCavar = kions[2]
			plat.Calc_permeabilities_from_channels()
			DST = DST_interface_EQPoint(iPlat = plat)
			yraw = DST.Get_Eq_Point(r = 'concentrations')
			for i in range(3): y[i].append(yraw[i])
			print(k)
		plt.plot(x, y[0], 'g', x, y[1], 'r', x, y[2], 'b')
		plt.ylim(top = 1.1*max(y[1]), bottom = 0.9*min(y[0]))
		plt.ylabel('[], mM')
		plt.xlabel('P_add, 1/h, k_Na = ' + str(kions[0]) +', k_K = '+ str(kions[1]) + ', k_Cl = ' + str(kions[2]))
		plt.show()
