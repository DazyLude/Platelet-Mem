import PyDSTool as dst
import PyDSTool.utils as dstuti
import numpy as np
import math as m
import random as rng
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal as snl
import copy as copy
import ast
import gc

import ModelParameters as IPA #initially named as InitialParametersArray

#Now I was thinking about putting all the default concentration of ions, their points of interest etc in one place.
#All of the parameters that I change EVER should be taken from here, just in case
class InitIonMatrix():
	def __init__(self, string=None):
		if string == None:
			self.getparam = IPA.TheArray
		else:
			self.getparam = []
		self.steps=m.floor(1/0.01)
		self.version_name = 'em_thingy'

class platelet_config:
	def __init__(self,
				iNa_in = None, iK_in = None, iCl_in = None, iCa_in = None,
				iosc_on = True, iSolution = -1, iEm = 0, iimx = InitIonMatrix(), 
				stationary = False, output = ""
				):
		#the successor of the IMX
		self.params = IPA.TheDictionary
		#params that won't change, like, ever (R, F etc) are still located here
		#also consider storing parametres of other models (fedya's or ATPase here too)
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
		self.Ca_cyt_in = 23e-9
		self.Ca_er_in = 200.

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
		self.Vol = imx.getparam[12][1]
		self.Surf = imx.getparam[12][2]
		self.VtoS = self.Vol / self.Surf
		self.Cm = imx.getparam[12][0] / self.Vol
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

		self.NCXM_Volt_Dep = 0.001
		self.NCXM_Empty_Charge = -2.5

		# self.K_NCX = self.params['NCX'][0]

		self.dPhiInit = -0.0

		#calculated parameteres
		self.External_ions_concentration = self.K_ex + self.Na_ex + self.Cl_ex + self.Ca_ex
		self.Metabolite_concentration = self.External_ions_concentration - self.K_in - self.Na_in - self.Cl_in + 2 * self.Ca_cyt_in
		self.Metabolite_charge = (self.K_in + self.Na_in + 2*self.Ca_cyt_in - self.Cl_in - self.Em_rest / self.F * self.Cm) / self.Metabolite_concentration

		self.AtpaseRate = self.Rest_UATP()

		if (stationary == True): self.MakeStationary()
		if (output == 'Full'): self.Get_Info()

	#easy access to the calculated parameteres
	def L(self):
		return self.External_ions_concentration
	def W0(self):
		return self.Metabolite_concentration
	def Z0(self):
		return self.Metabolite_charge

	def TestEm(self):
		return self.F / self.Cm * (self.K_in + self.Na_in + 2*self.Ca_cyt_in - self.Cl_in - self.Z0() * self.W0())
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
	#	ca: self.J(z = 2, in_ion=self.Ca_cyt_in, ex_ion=self.Ca_ex)

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

	def Rest_JPMCA(self):
		return self.J_PMCA * pow(self.Ca_cyt_in, self.n_PMCA)/(pow(self.K05_PMCA, self.n_PMCA) + pow(self.Ca_cyt_in, self.n_PMCA))

	def Rest_PKca31(self):
		return self.P_Kca31 * pow(self.Ca_cyt_in, self.n_Kca31)/(pow(self.K05_Kca31, self.n_Kca31) + pow(self.Ca_cyt_in, self.n_Kca31))

	def Rest_PKca11(self):
		return self.P_Kca11 * pow(self.Ca_cyt_in, self.n_Kca11)/(pow(self.K05_Kca11, self.n_Kca11) + pow(self.Ca_cyt_in, self.n_Kca11))

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

	def Rest_NCX_Na(self):
		return self.params['NCXM']['Speed_Mod'] * (self.params['NCXM_init']['Cin_Na'] * self.params['NCXM']['x'] - self.params['NCXM']['e'] * self.params['NCXM_init']['Cin_E'] * pow(self.Na_in * 1e3, 3))

	def Rest_NCX_Ca(self):
		return self.params['NCXM']['Speed_Mod'] * (self.params['NCXM_init']['Cin_Ca'] * self.params['NCXM']['y'] - self.params['NCXM']['f'] * self.params['NCXM_init']['Cin_E'] * self.Ca_cyt_in * 1e3)


	def MakeStationary(self):
		#sets P depending on concentrations and flow
		self.P_Na = (self.N * self.J0_atp - self.U0_NaK2Cl * self.Rest_UKNa2Cl() + 3*self.Rest_NCX_Na()) / self.J(z = 1., in_ion=self.Na_in, ex_ion=self.Na_ex)
		self.P_K = (-self.K * self.J0_atp - self.U0_NaK2Cl * self.Rest_UKNa2Cl()) / self.J(z = 1., in_ion=self.K_in, ex_ion=self.K_ex) - self.Pkv13rest() - self.Rest_PKca31()
		self.P_Cl = (-self.U0_NaK2Cl * 2. * self.Rest_UKNa2Cl()) / self.J(z = -1., in_ion=self.Cl_in, ex_ion=self.Cl_ex)
		self.P_Ca = (self.Rest_JPMCA()+self.Rest_NCX_Ca()) / self.J(z = 2., in_ion=self.Ca_cyt_in, ex_ion=self.Ca_ex)
		# self.P_Ca = 1e-18

	def Get_Info(self):
		#prints everything out
		print('L ' + str(self.L()) + ', W0 ' + str(self.W0()) + ', Z0 ' + str(self.Z0()))
		print('Em with such parametres: ' + str(self.TestEm()))
		print('V with such parametres: ' + str(self.TestV()))
		print('Forces: Na ' + str(self.J(in_ion=self.Na_in, ex_ion=self.Na_ex))
				+ ', K ' + str(self.J(in_ion=self.K_in, ex_ion=self.K_ex))
				+ ', Cl ' + str(self.J(z = -1, in_ion=self.Cl_in, ex_ion=self.Cl_ex))
				+ ', Ca ' + str(self.J(z = 2, in_ion=self.Ca_cyt_in, ex_ion=self.Ca_ex)))
		#print('Permeabilities: Na ' + str(self.P_Na * self.VtoS / 3.6 * 1e-4) + ', K ' + str(self.P_K * self.VtoS / 3.6 * 1e-4)
		#		+ ', Cl ' + str(self.P_Cl * self.VtoS / 3.6 * 1e-4) + ', Ca ' + str(self.P_Ca * self.VtoS / 3.6 * 1e-4))
		print('Permeabilities (1/h): Na ' + str(self.P_Na) + ', K ' + str(self.P_K)	+ ', Cl ' + str(self.P_Cl) + ', Ca ' + str(self.P_Ca))
		print('Permeabilities (total/h): Na ' + str(self.P_Na) + ', K ' + str(self.P_K + self.Pkv13rest() + self.Rest_PKca31())	+ ', Cl ' + str(self.P_Cl) + ', Ca ' + str(self.P_Ca))
		print('NaK2Cl osmotic force ' + str(self.Rest_UKNa2Cl()))
		print('ATPase rest activity mod' + str(self.Rest_UATP()))
		#print('NaCa osmotic force ' + str(self.Rest_UNaCa()))

class DST_interface:
	def __init__(self, iPlat):
		VoltClamp = False
		self.imx = iPlat.imx
		T = 3600 #constant defining conversion rate of permeabilities: in imx they are 1/h, here they are 1/s by default
		self.plat = copy.copy(iPlat)
		#ions		
		self.Ca_cyt_Flux = '((P_Ca + P_TRPC(t) * TRPC_Ca) * J(2., (Ca*1e-6), Ca_exf(t), Na, K, Cl, Ca_cyt, t, dPhi) - J_PMCA(Ca*1e-6) - J_NCX_Ca(Ca, NCXM_Cin_Ca, NCXM_Cin_E))*0'
		self.Ca_er_Flux = '-20*((fv0*(CaE-Ca)-fv1*(Ca-CaE)*x110**fA)**fB-fv3*Ca**2/(Ca**2+fk3**2)-0*fv4*Ca**2/(Ca**2+fk4**2))'

		self.Na_Flux = '-3 * U_atp(t, Na, K, Cl, Ca_cyt, dPhi) + (P_Na + P_TRPC(t) * TRPC_Na) * J(1, Na, Na_exf(t), Na, K, Cl, Ca_cyt, t, dPhi) + U_NaK2Cl(Na, K, Cl, t) - 3 * J_NCX_Na(Na, NCXM_Cin_Na, NCXM_Cin_E)'
		self.K_Flux = '2 * U_atp(t, Na, K, Cl, Ca_cyt, dPhi) + (Kca31(Ca*1e-6) + P_TRPC(t) * TRPC_K + P_K + PKv13i / (1 + exp((Em(Na, K, Cl, Ca_cyt, t, dPhi) - PKv13h) / PKv13s))) * J(1, K, K_exf(t), Na, K, Cl, Ca_cyt, t, dPhi) + U_NaK2Cl(Na, K, Cl, t)'
		self.Cl_Flux = '(P_Cl + P_Ca_c * P_Cl_Ca) * J(-1, Cl, Cl_exf(t), Na, K, Cl, Ca_cyt, t, dPhi) + 2. * U_NaK2Cl(Na, K, Cl, t)' 
		self.Ca_Flux = '(((fv0*(CaE-Ca)-fv1*(Ca-CaE)*x110**fA)**fB-fv3*Ca**2/(Ca**2+fk3**2)-0*fv4*Ca**2/(Ca**2+fk4**2) +' + self.Ca_cyt_Flux + '*1e6' + ')/ftu)/(1+(fkBuff*fBuff)/((fkBuff + Ca)**2) + (fkBuff2*fBuff2)/((fkBuff2 + Ca)**2))'

		self.x100_rate = '-fa2*Ca*x100-fa5*Ca*x100+fb5*x110'
		self.x110_rate = '-fa2*Ca*x110+fa5*Ca*x100-fb5*x110+0.25*fb2*(IP3f(t))'
		self.Ip3Ch = '0'

		self.PipetteCurrent = '0*F*Vol*(' + '0*U_atp(t, Na, K, Cl, Ca_cyt, dPhi)+' + \
			'+ (P_Na * 0.01) * JPip(1, Na, Na_exf(t), Na, K, Cl, Ca_cyt, t, dPhi)' + \
			'+ (HypoChN * HypoChP + P_K * 0.01) * JPip(1, K, K_exf(t), Na, K, Cl, Ca_cyt, t, dPhi)' + \
			'- (P_Cl* 0.01) * JPip(-1, Cl, Cl_exf(t), Na, K, Cl, Ca_cyt, t, dPhi)' + \
			'+ 2. * ((P_Ca* 0.01) * JPip(2., (Ca*1e-6), Ca_exf(t), Na, K, Cl, Ca_cyt, t, dPhi))' + \
			')'
		self.PipettePotential = '0*(' + self.PipetteCurrent + ')/Vol'
		
		self.NCX_Model = {
			'NCXM_Cin_E': '- NCXM_Cin_E * (NCXM_e * pow(Na * 1000, 3) + NCXM_f * Ca / 1000) + NCXM_x * NCXM_Cin_Na + NCXM_y * NCXM_Cin_Ca',
			#'NCXM_Cout_E': '- NCXM_Cout_E * (NCXM_c * pow(Na_exf(t) * 1000, 3) + NCXM_d * Ca_exf(t) * 1000) + NCXM_b * NCXM_Cout_Na + NCXM_a * NCXM_Cout_Ca',
			'NCXM_Cin_Na': '- NCXM_Cin_Na * (NCXM_x + NCXM_h * exp(NCXM_Volt_Dependancy * (NCXM_Charge_Empty + 3.)*F*Em(Na, K, Cl, Ca, t, dPhi)/R/T)) + NCXM_e * NCXM_Cin_E * pow(Na * 1000, 3) + NCXM_g * exp( - NCXM_Volt_Dependancy * (NCXM_Charge_Empty + 3.)*F*Em(Na, K, Cl, Ca, t, dPhi)/R/T)* NCXM_Cout_Na',
			'NCXM_Cout_Na': '- NCXM_Cout_Na * (NCXM_b + NCXM_g * exp( - NCXM_Volt_Dependancy * (NCXM_Charge_Empty + 3.)*F*Em(Na, K, Cl, Ca, t, dPhi)/R/T)) + NCXM_c * NCXM_Cout_E(NCXM_Cin_E, NCXM_Cin_Na, NCXM_Cout_Na, NCXM_Cin_Ca, NCXM_Cout_Ca) * pow(Na_exf(t) * 1000, 3) + NCXM_h * exp(NCXM_Volt_Dependancy * (NCXM_Charge_Empty + 3.)*F*Em(Na, K, Cl, Ca, t, dPhi)/R/T) * NCXM_Cin_Na',
			'NCXM_Cin_Ca': '- NCXM_Cin_Ca * (NCXM_y + NCXM_k * exp(NCXM_Volt_Dependancy * (NCXM_Charge_Empty + 2.)*F*Em(Na, K, Cl, Ca, t, dPhi)/R/T)) + NCXM_f * NCXM_Cin_E * Ca / 1000 + NCXM_j * exp( - NCXM_Volt_Dependancy * (NCXM_Charge_Empty + 2.)*F*Em(Na, K, Cl, Ca, t, dPhi)/R/T) * NCXM_Cout_Ca',
			'NCXM_Cout_Ca': ' - NCXM_Cout_Ca * (NCXM_a + NCXM_j * exp( - NCXM_Volt_Dependancy * (NCXM_Charge_Empty + 2.)*F*Em(Na, K, Cl, Ca, t, dPhi)/R/T)) + NCXM_d * NCXM_Cout_E(NCXM_Cin_E, NCXM_Cin_Na, NCXM_Cout_Na, NCXM_Cin_Ca, NCXM_Cout_Ca) * Ca_exf(t) * 1000 + NCXM_k * exp(NCXM_Volt_Dependancy * (NCXM_Charge_Empty + 2.)*F*Em(Na, K, Cl, Ca, t, dPhi)/R/T) * NCXM_Cin_Ca ',
		}
		

		#atpase
		v1 = 'E_ATP * pow(Na,3) * f1 - Na_E_ATP * b1' 
		v2 = 'Na_E_ATP * f2 - ADP * Na_E_P * b2'	
		v3 = 'Na_E_P * f3 * exp(F * Em(Na, K, Cl, Ca_cyt, t, dPhi) /(2 * R * T)) - E_P * pow(Na_exf(t), 3) * b3 * exp(-1 * F * Em(Na, K, Cl, Ca_cyt, t, dPhi) / (2 * R * T))'
		v4 = 'E_P * pow(K_exf(t), 2) * f4 - P * K_E * b4'
		v5 = 'K_E * ATP * f5 - K_E_ATP * b5'
		v6 = 'K_E_ATP * f6 - E_ATP * pow(K, 2) * b6'

		self.k1 = 'f6'
		self.k2 = 'f1 * pow(Na, 3)'
		self.k3 = 'f2'
		self.k4 = 'f3 * exp(F * Em(Na, K, Cl, Ca_cyt, t, dPhi)/(2 * R * T))'
		
		self.a1 = 'b6 * pow(K, 2)'
		self.a2 = '(b1 + f2)'
		self.a3 = 'b2 * ADP'
		self.a4 = '(b3 * pow(Na_exf(t), 3) * exp(-1 * F * Em(Na, K, Cl, Ca_cyt, t, dPhi)/(2 * R * T)) + f4 * pow(K_exf(t), 2))'

		self.bracket = self.k1 + '*' + self.k2 + '*' + self.k3 + '*' + self.k4 + '+' + self.a1 + '*' + self.k2 + '*' + self.k3 + '*' + self.k4 + '+' + self.a1 + '*' + self.a2 + '*' + self.k3 + '*' + self.k4 + '+' + self.a1 + '*' + self.a2 + '*' + self.a3 + '*' + self.k4 + '+' + self.a1 + '*' + self.a2 + '*' + self.a3 + '*' + self.a4
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
						#Concentrations in the basin solution
						'Na_ex': self.plat.Sol_Na_ex, #concentrations outside for integrating
						'K_ex': self.plat.Sol_K_ex,
						'Cl_ex': self.plat.Sol_Cl_ex,
						'Ca_ex': self.plat.Sol_Ca_ex,
						#Concentrations in the pipette solution
						'NaPip_ex': 0.,
						'KPip_ex': 0.,
						'ClPip_ex': 0.,
						'CaPip_ex': 0.,

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
						't_DAG_on': 10.,

						# 'k_NCX': self.plat.K_NCX / T,

						'fa2': 0.6,
		                'fa5': 400.,
		                'fb2': 0.6*1., #
		                'fb5': 400.*0.08234, #
		                'fv0': 0.001,
		                'fv1': 0.65, # was 0.5
		                'fv3': 1.8, # was 10
		                'fv4': 27.,
		                'ftu': 1.,
		                'fk3': 0.12, # was 0.26
		                'fk4': 1.1,
		                'fA': 4.,
		                'fB': 1.5,
		                'fc0': 2., #
		                'fc1': 0.05,
		                'fkBuff': 1.,
		                'fBuff':10.,
		                'fkBuff2': 1.,
		                'fBuff2':2.,

						'ip30': self.plat.IP3[0],
						'ip31': self.plat.IP3[1],
						'ip32': self.plat.IP3[2],
						'ip33': self.plat.IP3[3],
						'IP3max': self.plat.params['ip3'][0],

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
						'PKv13iPip': self.imx.getparam[19][0] / T / 20.,

						'PNv15i': self.imx.getparam[22][0] / T,
						'PNv15h': self.imx.getparam[22][1],
						'PNv15s': self.imx.getparam[22][2],

						'NCXM_e': self.plat.params['NCXM']['e'],
						'NCXM_f': self.plat.params['NCXM']['f'],
						'NCXM_x': self.plat.params['NCXM']['x'],
						'NCXM_y': self.plat.params['NCXM']['y'],
						'NCXM_Speed_Mod': self.plat.params['NCXM']['Speed_Mod'] / T,

						'NCXM_Charge_Empty': self.plat.NCXM_Empty_Charge,
						'NCXM_Volt_Dependancy': self.plat.NCXM_Volt_Dep,	
						'NCXM_a': self.plat.params['NCXM']['a'],
						'NCXM_b': self.plat.params['NCXM']['b'],
						'NCXM_c': self.plat.params['NCXM']['c'],
						'NCXM_d': self.plat.params['NCXM']['d'],
						'NCXM_g': self.plat.params['NCXM']['g'] / m.exp(-self.plat.NCXM_Volt_Dep * (self.plat.NCXM_Empty_Charge + 3.)*self.plat.F*self.plat.Em_rest/self.plat.R/self.plat.T),
						'NCXM_h': self.plat.params['NCXM']['h'] / m.exp(self.plat.NCXM_Volt_Dep * (self.plat.NCXM_Empty_Charge + 3.)*self.plat.F*self.plat.Em_rest/self.plat.R/self.plat.T),
						'NCXM_j': self.plat.params['NCXM']['j'] / m.exp(-self.plat.NCXM_Volt_Dep * (self.plat.NCXM_Empty_Charge + 2.)*self.plat.F*self.plat.Em_rest/self.plat.R/self.plat.T),
						'NCXM_k': self.plat.params['NCXM']['k'] / m.exp(self.plat.NCXM_Volt_Dep * (self.plat.NCXM_Empty_Charge + 2.)*self.plat.F*self.plat.Em_rest/self.plat.R/self.plat.T),

						'P_Ca_c': self.plat.Pvar,
						'P_Na_Ca': self.plat.PNaCavar, #self.imx.getparam[24][1],
						'P_K_Ca': self.plat.PKCavar, #self.imx.getparam[24][2],
						'P_Cl_Ca': self.plat.PClCavar, #self.imx.getparam[24][3],

						'Spip': 0.05,
						'Smem': 1.,
						'HypoChN': 0,
						'HypoChP': 0.01,
						}

		#helper functions
		self.DSargs.fnspecs = {
						'J' : (['z_i', 'In', 'Ex', 'Na', 'K', 'Cl', 'Ca', 't', 'dPhi'], '2. * r(z_i, Na, K, Cl, Ca, t, dPhi) / (exp(r(z_i, Na, K, Cl, Ca, t, dPhi)) - exp(-r(z_i, Na, K, Cl, Ca, t, dPhi))) * (Ex * exp(-r(z_i, Na, K, Cl, Ca, t, dPhi)) - In * exp(r(z_i, Na, K, Cl, Ca, t, dPhi)))'),
						'JPip' : (['z_i', 'In', 'Ex', 'Na', 'K', 'Cl', 'Ca', 't', 'dPhi'], '2. * rPip(z_i, Na, K, Cl, Ca, t, dPhi) / (exp(rPip(z_i, Na, K, Cl, Ca, t, dPhi)) - exp(-rPip(z_i, Na, K, Cl, Ca, t, dPhi))) * (Ex * exp(-rPip(z_i, Na, K, Cl, Ca, t, dPhi)) - In * exp(rPip(z_i, Na, K, Cl, Ca, t, dPhi)))'),

						'U_atp': (['t', 'Na', 'K', 'Cl', 'Ca_cyt', 'dPhi'], 'J0_atp * f2 * ' + self.k1 + '*' + self.k1 + '*' + self.k2 + '*' + self.k2 + '*' + self.k3 + '*' + self.k4 + '/(' + self.a1 + '*' + self.a2 + '*' + self.bracket + ') / RestAtpaseRate'),

						'U_NaK2Cl': (['Na', 'K', 'Cl', 't'], 'U0_NaK2Cl * (K_exf(t) * Na_exf(t) * pow(Cl_exf(t), 2) - K * Na * pow(Cl, 2))'),

						'J_PMCA': (['Ca'], 'VMax_PMCA * pow(Ca, n_PMCA) / (pow(K05_PMCA, n_PMCA) + pow(Ca, n_PMCA))'),

						'P_TRPC': (['t'], 'P_TRPC_0 * pow(IP3f(t) * 1e-6, n_TRPC)/(pow(K05_TRPC, n_TRPC) + pow(IP3f(t) * 1e-6, n_TRPC))'),

						'NCXM_Cout_E': (['NCXM_Cin_E', 'NCXM_Cin_Na', 'NCXM_Cout_Na', 'NCXM_Cin_Ca', 'NCXM_Cout_Ca'], '1 - NCXM_Cin_E - NCXM_Cin_Na - NCXM_Cout_Na - NCXM_Cin_Ca - NCXM_Cout_Ca'),
						'J_NCX_Na': (['Na', 'NCXM_Cin_Na', 'NCXM_Cin_E'], 'NCXM_Speed_Mod * (NCXM_Cin_Na * NCXM_x - NCXM_e * NCXM_Cin_E * pow(Na * 1e3, 3))'),
						'J_NCX_Ca': (['Ca', 'NCXM_Cin_Ca', 'NCXM_Cin_E'], 'NCXM_Speed_Mod * (NCXM_Cin_Ca * NCXM_y - NCXM_f * NCXM_Cin_E * Ca * 1e-3)'),

						'r': (['z_i', 'Na', 'K', 'Cl', 'Ca', 't', 'dPhi'], 'z_i * Em(Na, K, Cl, Ca, t, dPhi) * F / R / T / 2.'),
						'rPip': (['z_i', 'Na', 'K', 'Cl', 'Ca', 't', 'dPhi'], 'z_i * (Em(Na, K, Cl, Ca, t, dPhi) + dPhi) * F / R / T / 2.'),

						'Em': (['Na', 'K', 'Cl', 'Ca_cyt', 't', 'dPhi'], 'F / Cm * (K + Na + 2*(Ca_cyt) - Cl - z0 * W0(t)) - 0*Spip/Smem*dPhi'),

						'W0': (['t'], 'W0c * 1'),
						'V': (['Na', 'K', 'Cl', 't'], '1'),

						'Kca31': (['ca'], 'P_Kca31 * pow(ca, n_Kca31)/(pow(K05_Kca31, n_Kca31) + pow(ca, n_Kca31))'),
						'Kca11': (['ca'], 'P_Kca11 * pow(ca, n_Kca11)/(pow(K05_Kca11, n_Kca11) + pow(ca, n_Kca11))'),

						'Na_exf': (['t'], 'Na_ex'),
						'K_exf': (['t'], 'K_ex'),
						'Cl_exf': (['t'], 'Cl_ex'),
						'Ca_exf': (['t'], 'Ca_ex'),
						# 'IP3f': (['t'], '(heav(t-30.) + heav(t-60.) + heav(t-90.) + heav(t-120.) + heav(t-150.) + heav(t-180.) + heav(t-210.) + heav(t-240.) + heav(t-270.) + heav(t-300.)) * IP3max / 10'),
						# 'IP3f': (['t'], 'IP3max * heav(t-20.) * (t-20) / exp(0.1 * (t-20)) / 0.1 / 2.7'),
						'IP3f': (['t'], 'IP3max'),

						'Lf': (['t'], 'Na_exf(t) + Cl_exf(t) + K_exf(t) + Ca_exf(t)'),
						}
		#ODE
		if (VoltClamp): self.PipettePotential = '0'
		self.DSargs.varspecs = {
							#currents through the cell membrane
							'Na': self.Na_Flux,
							'K': self.K_Flux,
							'Cl': self.Cl_Flux,
							'Ca': self.Ca_Flux,
							'CaE': self.Ca_er_Flux,
							'Ca_cyt': self.Ca_cyt_Flux,
							#IP3rState
							'x100': self.x100_rate,
							'x110': self.x110_rate,
							#Currents through the pipette part of the membrane

							'Ip3Ch': self.Ip3Ch,

							'PipetteCurrent': self.PipetteCurrent,
							'dPhi': self.PipettePotential,

							'NCXM_Cin_E': self.NCX_Model['NCXM_Cin_E'],
							'NCXM_Cin_Na': self.NCX_Model['NCXM_Cin_Na'],
							'NCXM_Cin_Ca': self.NCX_Model['NCXM_Cin_Ca'],
							# 'NCXM_Cout_E': self.NCX_Model['NCXM_Cout_E'],
							'NCXM_Cout_Na': self.NCX_Model['NCXM_Cout_Na'],
							'NCXM_Cout_Ca': self.NCX_Model['NCXM_Cout_Ca'],
							}
		#initial condition	
		self.DSargs.ics = {'Na': self.plat.Na_in, 'K': self.plat.K_in, 'Cl': self.plat.Cl_in, 'Ca_cyt': self.plat.Ca_cyt_in, 
							'x100': 0.02, 'x110': 0.02, 'Ca': 0.023, 'CaE': 200,
							'Ip3Ch': 0.18, 'PipetteCurrent': 0., 'dPhi': 0., 
							'NCXM_Cout_Na': self.plat.params['NCXM_init']['Cout_Na'], 'NCXM_Cout_Ca': self.plat.params['NCXM_init']['Cout_Ca'],
							'NCXM_Cin_Ca': self.plat.params['NCXM_init']['Cin_Ca'], 'NCXM_Cin_Na': self.plat.params['NCXM_init']['Cin_Na'], 
							'NCXM_Cin_E': self.plat.params['NCXM_init']['Cin_E']}
		self.DSargs.tdomain = [self.plat.params['time'][0], self.plat.params['time'][1]]
		self.DSargs.algparams = {'init_step': self.plat.params['time'][2], 'max_pts': int(1e7)}
		# self.ode = dst.Generator.Radau_ODEsystem(self.DSargs)		# an instance of the 'Generator' class.

	def Integrate(self, fName = 'integration.txt'):
		self.ode = dst.Generator.Radau_ODEsystem(self.DSargs)
		print('integration starts')
		self.traj = self.ode.compute('Volume')
		print('sampling starts')	
		self.pts = self.traj.sample()
		print('saving results')
		f = open(fName, 'w')
		# f.write('time\tNa, mM\tK, mM\tCl, mM\tCa, mM\tCa_cyt\tCa_er\tx100\tx110\tPipCurrent\tdPhi\n')

		coordNames = self.pts.coordnames
		coordNames.insert(0, 't') #because Rob thinks that t, being an independent variable, should not be in the coordnames list
		# print(coordNames)

		for k in coordNames:
			f.write(k + '\t')
		f.write('\n')

		for k in range(len(self.pts['t'])-1):
			for name in coordNames:
				f.write(str(self.pts[name][k]) + '\t')
			f.write('\n')
			# f.write(str(self.pts['t'][k]) + '\t' + str(self.pts['Na'][k]) + '\t' + str(self.pts['K'][k]) + '\t' + str(self.pts['Cl'][k]) + '\t' + str(self.pts['Ca'][k]) + '\t' + str(self.pts['Ca_cyt'][k]) + '\t' + 
			# 	str(self.pts['CaE'][k]) + '\t' + str(self.pts['x100'][k]) + '\t' + str(self.pts['x110'][k]) + '\t' + str(self.pts['PipetteCurrent'][k]) + '\t' + str(self.pts['dPhi'][k]) + '\n')
		f.close()
		print('done, results saved to ' + fName)
		gc.collect()

		fp = len(self.pts['t']) - 1

		f2 = open('TmpChAndT.dat', 'w')
		f2.write(str([self.pts['t'][fp], self.DSargs.pars['HypoChN']]))
		f2.close()

		f1 = open('TmpIcs.dat', 'w')
		f1.write(str({'Na': self.pts['Na'][fp], 'K': self.pts['K'][fp], 'Cl': self.pts['Cl'][fp], 'Ca_cyt': self.pts['Ca_cyt'][fp], 'x100': self.pts['x100'][fp], 'x110': self.pts['x110'][fp], 'Ca': self.pts['Ca'][fp], 'CaE': self.pts['CaE'][fp], 'Ip3Ch': self.pts['Ip3Ch'][fp], 'PipetteCurrent': self.pts['PipetteCurrent'][fp], 'dPhi': self.pts['dPhi'][fp]}))
		f1.close()

		return {'Na': self.pts['Na'][fp], 'K': self.pts['K'][fp], 'Cl': self.pts['Cl'][fp], 'Ca_cyt': self.pts['Ca_cyt'][fp], 'x100': self.pts['x100'][fp], 'x110': self.pts['x110'][fp], 'Ca': self.pts['Ca'][fp], 'CaE': self.pts['CaE'][fp], 'Ip3Ch': self.pts['Ip3Ch'][fp], 'PipetteCurrent': self.pts['PipetteCurrent'][fp], 'dPhi': self.pts['dPhi'][fp]}


	def IntegrateGetLastTwo(self, var_names = ['t']):
		self.ode = dst.Generator.Radau_ODEsystem(self.DSargs)
		print('integration starts')
		self.traj = self.ode.compute('Volume')
		print('sampling starts')	
		self.pts = self.traj.sample()
		temp = []
		length = len(self.pts['t'])
		print('returning the last values at ' + str(self.pts['t'][length-1]) + ' s') 
		for var in var_names:
			temp.append([self.pts[var][length-2], self.pts[var][length-1]])
		gc.collect()
		return temp

	def CreateNewSaveFile(self, fName = 'temp.txt'):
		f = open(fName, 'w')
		f.write('time\tNa, mM\tK, mM\tCl, mM\tCa, mM\tCa_cyt\tCa_er\tx100\tx110\tPipCurrent\tdPhi\n')
		f.close()


	def IntegrateAppendFile(self, ics, t0, t1, fName = 'temp.txt'):
		print('setting the parametres and ics')
		self.DSargs.ics = ics #{'Na':  'K': 'Cl': 'Ca_cyt': 'x100': 'x110': 'Ca': 'CaE': 'Ip3Ch': 'PipetteCurrent': 'dPhi': }
		self.DSargs.tdomain = [t0, t1]
		# self.DSargs.pars.update(channels)
		self.ode = dst.Generator.Radau_ODEsystem(self.DSargs)
		print('integrating from ' + str(self.DSargs.tdomain[0]) + ' to ' + str(self.DSargs.tdomain[1]))
		self.traj = self.ode.compute('Volume')
		print('sampling starts')	
		self.pts = self.traj.sample()
		f = open(fName, 'a')
		for k in range(len(self.pts['t'])-1):
			f.write(str(self.pts['t'][k]) + '\t' + str(self.pts['Na'][k]) + '\t' + str(self.pts['K'][k]) + '\t' + str(self.pts['Cl'][k]) + '\t' + str(self.pts['Ca'][k]) + '\t' + str(self.pts['Ca_cyt'][k]) + '\t' + 
				str(self.pts['CaE'][k]) + '\t' + str(self.pts['x100'][k]) + '\t' + str(self.pts['x110'][k]) + '\t' + str(self.pts['PipetteCurrent'][k]) + '\t' + str(self.pts['dPhi'][k]) + '\n')
		f.close()
		gc.collect()

		fp = len(self.pts['t']) - 1

		f2 = open('TmpChAndT.dat', 'w')
		f2.write(str([self.pts['t'][fp], self.DSargs.pars['HypoChN']]))
		f2.close()

		f1 = open('TmpIcs.dat', 'w')
		f1.write(str({'Na': self.pts['Na'][fp], 'K': self.pts['K'][fp], 'Cl': self.pts['Cl'][fp], 'Ca_cyt': self.pts['Ca_cyt'][fp], 'x100': self.pts['x100'][fp], 'x110': self.pts['x110'][fp], 'Ca': self.pts['Ca'][fp], 'CaE': self.pts['CaE'][fp], 'Ip3Ch': self.pts['Ip3Ch'][fp], 'PipetteCurrent': self.pts['PipetteCurrent'][fp], 'dPhi': self.pts['dPhi'][fp]}))
		f1.close()

		return {'Na': self.pts['Na'][fp], 'K': self.pts['K'][fp], 'Cl': self.pts['Cl'][fp], 'Ca_cyt': self.pts['Ca_cyt'][fp], 'x100': self.pts['x100'][fp], 'x110': self.pts['x110'][fp], 'Ca': self.pts['Ca'][fp], 'CaE': self.pts['CaE'][fp], 'Ip3Ch': self.pts['Ip3Ch'][fp], 'PipetteCurrent': self.pts['PipetteCurrent'][fp], 'dPhi': self.pts['dPhi'][fp]}


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
		PCargs.VarTol  = 1e-4
		PCargs.FuncTol = 1e-4
		PCargs.TestTol = 1e-4
		PCargs.SaveEigen    = True	#to tell unstable from stable branches
		PCargs.LocBifPoints = 'all'
		PC.newCurve(PCargs)
		PC['EQ1'].forward()
		PC.display(['IP3max', 'Em'], stability=True)

class notstoch:
	def __init__(self):
		self.t0 = 0.
		self.t1 = 30.
		self.dT = 0.001

		self.channels = 100.
		self.open_channels = 0.

		self.pts = []
		self.DST = []

	def p_open(self, stationar, tau):
		return stationar / tau * self.dT

	def p_close(self, stationar, tau):
		return (1 - stationar) / tau * self.dT

	def binomial_probability(self,n,k,p):
		if (n>=k) and (p<=1):
			return float(m.factorial(n))/float(m.factorial(n-k))/float(m.factorial(k))*pow(p, k)*pow(1-p, n-k)
		else:
			raise Exception('text')

	def delta_channels(self, nch, prob):
		if (nch == 0.):
			return 0.
		rgn = rng.random()
		# print(rgn)
		probsum = 0.
		for i in range(round(nch+1)):
			probsum += self.binomial_probability(nch, i, prob)
			# print('ps = ' + str(probsum))
			if (probsum >= rgn):
				return i
		
	
	def DoTheTest(self, tau, stat):
		temp_X = []
		temp_Y = []
		for t_step in np.arange(self.t0, self.t1, self.dT):
			self.open_channels = self.open_channels + self.delta_channels(self.channels - self.open_channels, self.p_open(stat, tau)) - self.delta_channels(self.open_channels, self.p_close(stat, tau))
			# print(self.delta_channels(self.channels - self.open_channels, self.p_open(0.7, 1.)), self.delta_channels(self.open_channels, self.p_close(0.7, 1.)), self.p_open(0.7, 1.), self.p_close(0.7, 1.))
			temp_X.append(t_step)
			temp_Y.append(self.open_channels)

		plt.figure()
		plt.plot(temp_X, temp_Y, 'b', temp_X, np.full(np.shape(temp_X), np.average(temp_Y)), 'r')
		plt.xlabel('time, s')                              		# Axes labels
		plt.ylabel('open_channels')								# Range of the y axis
		plt.ylim(-0.01, self.channels+0.01)
		plt.title('test')										# Figure title from model name
		plt.show()

	def HypoStat(self, Ca):
		return pow(Ca*1e-6, 5)/(pow(0.2e-6, 5) + pow(Ca*1e-6, 5))

	def StartTheThing(self):
		#First iteration
		HypoCh = 0
		HypoChTotal = 1

		plat = platelet_config(iSolution = -1)
		plat.MakeStationary()
		DST = DST_interface(iPlat = plat)
		DST.DSargs.tdomain = [self.t0, self.t0 + self.dT]
		ics = DST.Integrate(fName = 'temp.txt')
		#Starting the cycle with given 
	
	def ContinueTheThing(self):
		f2 = open('Temp/TmpChAndT.dat', 'r')
		thing = ast.literal_eval(f2.readline())
		HypoChTotal = 1
		HypoCh = thing[1]
		f2.close()

		f1 = open('Temp/TmpIcs.dat', 'r')
		ics = ast.literal_eval(f1.readline())
		f1.close()

		plat = platelet_config(iSolution = -1)
		plat.MakeStationary()
		DST = DST_interface(iPlat = plat)

		HypoCh = HypoCh + self.delta_channels(HypoChTotal - HypoCh, self.p_open(self.HypoStat(ics['Ca']), 0.01)) - self.delta_channels(HypoCh, self.p_close(self.HypoStat(ics['Ca']), 0.01))
		DST.DSargs.pars['HypoChN'] = HypoCh

		DST.IntegrateAppendFile(ics = ics, fName = 'temp.txt', t0 = thing[0], t1 = thing[0] + self.dT)

	def Do500Continues(self):
		for i in range(500):
			self.ContinueTheThing()

	def DoTheThing(self):
		#First iteration
		HypoCh = 0
		HypoChTotal = 1

		plat = platelet_config(iSolution = -1)
		plat.MakeStationary()
		DST = DST_interface(iPlat = plat)
		DST.DSargs.tdomain = [self.t0, self.t0 + self.dT]
		ics = DST.Integrate(fName = 'temp.txt')
		#Starting the cycle with given 
		for t_step in np.arange(self.t0+self.dT, self.t1, self.dT):
			# gc.collect()
			# DST = DST_interface(iPlat = plat)
			# DST.DSargs.tdomain = [t_step, t_step + dT]
			HypoCh = HypoCh + self.delta_channels(HypoChTotal - HypoCh, self.p_open(self.HypoStat(ics['Ca']), 0.01)) - self.delta_channels(HypoCh, self.p_close(self.HypoStat(ics['Ca']), 0.01))
			DST.DSargs.pars['HypoChN'] = HypoCh
			ics = DST.IntegrateAppendFile(ics = ics, fName = 'temp.txt', t0 = t_step, t1 = t_step + self.dT)

# For plotting a few things at once

class comparative_plotter:
	def __init__(self, iPlat, fName1, fName2, i_skip = 0):

		self.tempts = []
		self.fName = fName1
		self.plat = iPlat
		self.names = []
		self.pts1 = {}
		skip = i_skip
		cur = 0
		f = open(self.fName, 'r')
		#reads a header and creates columns
		subpts = f.readline()
		while '\t' in subpts[cur:]:
			dcur = subpts[cur:].find('\t') #how far is \t from current postion on the line
			self.pts1[subpts[cur:cur+dcur]] = []
			# self.names.append(subpts[cur:cur+dcur])
			cur = cur + dcur + 1
		cur = 0
		width = len(self.pts1)
		subpts = f.readline() #do-while is not supported sadge
		#reads lines one by one and fills the lines of the array with info
		while subpts != '':
			for k in iter(self.pts1):
				dcur = subpts[cur:].find('\t')
				self.pts1[k].append(float(subpts[cur:cur+dcur]))
				cur = cur + dcur + 1
			cur = 0
			if skip != 0: 
				for i in range(skip-1): f.readline()
			subpts = f.readline()
		print('report file 1 read, generated temp array with  ' + str(len(self.pts1['t'])) + ' lines and ' + str(len(self.pts1)) + ' columns')
		f.close()

		self.tempts = []
		self.fName = fName2
		self.names = []
		self.pts2 = {}
		skip = i_skip
		cur = 0
		f = open(self.fName, 'r')
		#reads a header and creates columns
		subpts = f.readline()
		while '\t' in subpts[cur:]:
			dcur = subpts[cur:].find('\t') #how far is \t from current postion on the line
			self.pts2[subpts[cur:cur+dcur]] = []
			# self.names.append(subpts[cur:cur+dcur])
			cur = cur + dcur + 1
		cur = 0
		width = len(self.pts2)
		subpts = f.readline() #do-while is not supported sadge
		#reads lines one by one and fills the lines of the array with info
		while subpts != '':
			for k in iter(self.pts2):
				dcur = subpts[cur:].find('\t')
				self.pts2[k].append(float(subpts[cur:cur+dcur]))
				cur = cur + dcur + 1
			cur = 0
			if skip != 0: 
				for i in range(skip-1): f.readline()
			subpts = f.readline()
		print('report file 2 read, generated temp array with  ' + str(len(self.pts2['t'])) + ' lines and ' + str(len(self.pts2)) + ' columns')
		f.close()

	def Plot_Em(self):
		tempts1 = []
		tempts2 = []

		for k in range(0,len(self.pts1['t'])):
			tempts1.append(self.plat.Give_Em(self.pts1['Na'][k], self.pts1['K'][k], self.pts1['Cl'][k], self.pts1['Ca_cyt'][k]))
		
		for k in range(0,len(self.pts2['t'])):
			tempts2.append(self.plat.Give_Em(self.pts2['Na'][k], self.pts2['K'][k], self.pts2['Cl'][k], self.pts2['Ca_cyt'][k]))

		plt.figure(1)
		plt.plot(self.pts1['t'], tempts1, 'r', self.pts2['t'], tempts2, 'g')
		plt.xlabel('time, s')                              # Axes labels
		plt.ylabel('Em, V')                                 # Range of the y axis
		plt.ylim(0.9*max(tempts1), 1.1*min(tempts1))
		plt.title('Em over time')                             # Figure title from model name
		plt.show()
		del self.tempts[:]
		self.tempts = []


	def Plot_Ca(self):
		plt.figure()
		plt.plot(self.pts1['t'], self.pts1['Ca'], 'r', self.pts2['t'], self.pts2['Ca'], 'g')
		plt.xlabel('time, s')                              				# Axes labels
		plt.ylabel('Ca, $\mu$M')								# Range of the y axis
		plt.ylim(0.95*min(self.pts1['Ca']), 1.05*max(self.pts1['Ca']))
		plt.title('Ca concentration over time')			# Figure title from model name
		plt.show()

#For plotting th results of one integration

class plotter:
	def __init__(self, iPlat, fName, i_skip = 0):

		self.tempts = []
		self.fName = fName
		self.plat = iPlat
		self.names = []
		self.pts = {}
		skip = i_skip
		cur = 0
		f = open(self.fName, 'r')
		#reads a header and creates columns
		subpts = f.readline()
		while '\t' in subpts[cur:]:
			dcur = subpts[cur:].find('\t') #how far is \t from current postion on the line
			self.pts[subpts[cur:cur+dcur]] = []
			# self.names.append(subpts[cur:cur+dcur])
			cur = cur + dcur + 1
		cur = 0
		width = len(self.pts)
		subpts = f.readline() #do-while is not supported sadge
		#reads lines one by one and fills the lines of the array with info
		while subpts != '':
			for k in iter(self.pts):
				dcur = subpts[cur:].find('\t')
				self.pts[k].append(float(subpts[cur:cur+dcur]))
				cur = cur + dcur + 1
			cur = 0
			if skip != 0: 
				for i in range(skip-1): f.readline()
			subpts = f.readline()
		print('report file read, generated temp array with  ' + str(len(self.pts['t'])) + ' lines and ' + str(len(self.pts)) + ' columns')

		f.close()

	def Plot_Concentrations(self, what):
		plt.figure()
		plt.plot(self.pts['t'], self.pts[what])
		plt.xlabel('time, s')                              				# Axes labels
		plt.ylabel(what  + ', M?')								# Range of the y axis
		plt.ylim(1.05*min(self.pts[what]), 0.95*max(self.pts[what]))
		plt.title(what + ' concentration over time')			# Figure title from model name
		plt.show()

	def Plot_Em(self):
		for k in range(0,len(self.pts['t'])):
			self.tempts.append(self.plat.Give_Em(self.pts['Na'][k], self.pts['K'][k], self.pts['Cl'][k], self.pts['Ca_cyt'][k]))
		plt.figure(1)
		plt.plot(self.pts['t'], self.tempts)
		plt.xlabel('time, s')                              # Axes labels
		plt.ylabel('Em, V')                                 # Range of the y axis
		plt.ylim(0.9*max(self.tempts), 1.1*min(self.tempts))
		plt.title('Em over time')                             # Figure title from model name
		plt.show()
		del self.tempts[:]
		self.tempts = []

	def Plot_NaKCl(self):
		plt.figure()
		plt.plot(self.pts['t'], self.pts['Na'], 'r', self.pts['t'], self.pts['K'], 'b', self.pts['t'], self.pts['Cl'], 'g')
		plt.xlabel('time, s')                              				# Axes labels
		plt.ylabel('[Na] [K] and [Cl], M')								# Range of the y axis
		plt.ylim(0, 1.05*max(self.pts['K']))
		plt.title('concentration over time')			# Figure title from model name
		plt.show()

	def Plot_Ca(self):
		tmppts = []
		for l in range(0,len(self.pts['t'])):
			tmppts.append((self.pts['Ca'][l]))
		plt.figure()
		plt.plot(self.pts['t'], tmppts)
		plt.xlabel('time, s')                              				# Axes labels
		plt.ylabel('Ca, $\mu$M')								# Range of the y axis
		plt.ylim(0.95*min(tmppts), 1.05*max(tmppts))
		plt.title('Ca concentration over time')			# Figure title from model name
		plt.show()

	def Plot_Concentrations_Change(self, what = 'Ca'):
		tmpptst = []
		for k in range(0,len(self.pts['t'])-1):
			tmpptst.append(self.pts['t'][k])
		for k in range(0,len(self.pts['t'])-1):
			if (self.pts['t'][k+1] - self.pts['t'][k]) != 0:
				self.tempts.append((self.pts[what][k+1] - self.pts[what][k])/(self.pts['t'][k+1] - self.pts['t'][k]))
			else: self.tempts.append(self.tempts[k-1])
		plt.figure(1)
		plt.plot(tmpptst, self.tempts)
		plt.xlabel('time, m')                              # Axes labels
		plt.ylabel('flux, mol/s')                                 # Range of the y axis
		plt.title('flux of ' + self.names[what])                             # Figure title from model name
		plt.show()
		del self.tempts[:]
		self.tempts = []

	def Plot_NCX_Species(self):
		fig, axs = plt.subplots(ncols=1, nrows=2,constrained_layout=True, figsize=(5.5, 3.5))
		self.pts['NCXM_Cout_E'] = []
		for k in range(0,len(self.pts['t'])):
			self.pts['NCXM_Cout_E'].append(1 - self.pts['NCXM_Cout_Na'][k] - self.pts['NCXM_Cout_Ca'][k] - self.pts['NCXM_Cin_E'][k] - self.pts['NCXM_Cin_Ca'][k] - self.pts['NCXM_Cin_Na'][k])

		axs[0].plot(self.pts['t'], self.pts['NCXM_Cin_E'], 'r', self.pts['t'], self.pts['NCXM_Cin_Ca'], 'g',self.pts['t'], self.pts['NCXM_Cin_Na'], 'b')
		axs[1].plot(self.pts['t'], self.pts['NCXM_Cout_E'], 'r', self.pts['t'], self.pts['NCXM_Cout_Ca'], 'g',self.pts['t'], self.pts['NCXM_Cout_Na'], 'b')
		plt.title('NCX species distribution')
		plt.show()

	def Plot_NCX_Currents(self):
		fig, axs = plt.subplots(ncols=1, nrows=1,constrained_layout=True, figsize=(5.5,3.5))
		tmppts = {'CaCur': [], 'NaCur': []}
		for k in range(0,len(self.pts['t'])):
			tmppts['NaCur'].append(self.plat.params['NCXM']['Speed_Mod'] * (self.pts['NCXM_Cin_Na'][k] * self.plat.params['NCXM']['x'] - self.plat.params['NCXM']['e'] * self.pts['NCXM_Cin_E'][k] * pow(self.pts['Na'][k] * 1e3, 3)))
			tmppts['CaCur'].append(self.plat.params['NCXM']['Speed_Mod'] * (self.pts['NCXM_Cin_Ca'][k] * self.plat.params['NCXM']['y'] - self.plat.params['NCXM']['f'] * self.pts['NCXM_Cin_E'][k] * self.pts['Ca'][k] * 1e-3))
		axs.plot(self.pts['t'], tmppts['CaCur'], 'g', self.pts['t'], tmppts['NaCur'], 'r')
		plt.title('NCX Fluxes')
		plt.show()

	def calc_Kd(self):
		print(self.pts['Ca'][20] * self.pts['NCXM_Cin_E'][20] / self.pts['NCXM_Cin_Ca'][20])

### DEPRECATED ###
### MOVING TO ANOTHER CLASS ###
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
			dcur = subpts[cur:].find('\t') #how far is \t from current postion on the line
			self.pts.append([])
			self.names.append(subpts[cur:cur+dcur])
			cur = cur + dcur + 1
		cur = 0
		width = len(self.pts)
		subpts = f.readline() #do-while is not supported sadge
		#reads lines one by one and fills the lines of the array with info
		while subpts != '':
			for k in range(width):
				dcur = subpts[cur:].find('\t')
				self.pts[k].append(float(subpts[cur:cur+dcur]))
				cur = cur + dcur + 1
			cur = 0
			if skip != 0: 
				for i in range(skip-1): f.readline()
			subpts = f.readline()
		print('report file read, generated temp array with  ' + str(len(self.pts[0])) + ' lines and ' + str(len(self.pts)) + ' columns')
		f.close()

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

	def Plot_Conc_Change(self, what = 0):
		tmpptst = []
		for k in range(0,len(self.pts[0])-1):
			tmpptst.append(self.pts[0][k])
		for k in range(0,len(self.pts[0])-1):
			if (self.pts[0][k+1] - self.pts[0][k]) != 0:
				self.tempts.append((self.pts[what][k+1] - self.pts[what][k])/(self.pts[0][k+1] - self.pts[0][k]))
			else: self.tempts.append(self.tempts[k-1])
		plt.figure(1)
		plt.plot(tmpptst, self.tempts)
		plt.xlabel('time, m')                              # Axes labels
		plt.ylabel('flux, mol/s')                                 # Range of the y axis
		plt.title('flux of ' + self.names[what])                             # Figure title from model name
		plt.show()
		del self.tempts[:]
		self.tempts = []

	def Plot_Conc_Change_Histogram(self, what = 0):
		tmpptst = []
		for k in range(0,len(self.pts[0])-1):
			tmpptst.append(self.pts[0][k])
		for k in range(0,len(self.pts[0])-1):
			if (self.pts[0][k+1] - self.pts[0][k]) != 0:
				self.tempts.append((self.pts[what][k+1] - self.pts[what][k])/(self.pts[0][k+1] - self.pts[0][k]))
			else: self.tempts.append(self.tempts[k-1])
		values = []
		cumulative = []
		for k in range(1,len(self.tempts)-2):
			deriv = self.tempts[k]
			dr = round(deriv, 16)
			if dr in values:
				cumulative[values.index(dr)] += (self.pts[0][k+1] - self.pts[0][k-1])/2
				# print('+1')
			else:
				values.append(dr)
				cumulative.append((self.pts[0][k-1] + self.pts[0][k+1])/2)
		print(values)
		plt.figure(1)
		plt.plot(values, cumulative, '.b')
		plt.xlabel('current')                              # Axes labels
		plt.ylabel('appearances')                                 # Range of the y axis
		plt.title('histogram of ' + self.names[what])                             # Figure title from model name
		plt.show()
		del self.tempts[:]
		self.tempts = []

	def Plot_Conc_Change_Histogram(self, what = 0):
		tmpptst = []
		for k in range(0,len(self.pts[0])-1):
			tmpptst.append(self.pts[0][k])
		for k in range(0,len(self.pts[0])-1):
			if (self.pts[0][k+1] - self.pts[0][k]) != 0:
				self.tempts.append((self.pts[what][k+1] - self.pts[what][k])/(self.pts[0][k+1] - self.pts[0][k]))
			else: self.tempts.append(self.tempts[k-1])
		xvals = np.linspace(self.pts[0][0], self.pts[0][len(self.pts[0])-1], len(self.pts[0])*5)
		print(len(self.pts[0]))
		print(len(self.tempts))

		yinterp = np.interp(xvals, self.pts[0][1:], self.tempts)
		yinterp = yinterp / 0.05
		weights = np.ones_like(yinterp) / len(yinterp)

		plt.hist(x=yinterp, bins=100, weights=weights)
		plt.xlabel('проводимости, пСм')                              # Axes labels
		plt.ylabel('количество точек')                       
		# plt.yscale('log')          # Range of the y axis
		plt.title('histogram of ' + self.names[what])                             # Figure title from model name
		plt.show()

		# values = []
		# cumulative = []
		# for k in range(1,len(yinterp)-1):
		# 	deriv = yinterp[k]
		# 	dr = round(deriv, 16)
		# 	if dr in values:
		# 		cumulative[values.index(dr)] +=1
		# 		# print('+1')
		# 	else:
		# 		values.append(dr)
		# 		cumulative.append(1)
		# print(values)
		# plt.figure(1)
		# plt.plot(values, cumulative, '.b')
		# plt.xlabel('current')                              # Axes labels
		# plt.ylabel('appearances')                                 # Range of the y axis
		# plt.title('histogram of ' + self.names[what])                             # Figure title from model name
		# plt.show()
		del self.tempts[:]
		self.tempts = []

	def Plot_Ca_Em(self):
		fig, ax1 = plt.subplots()

		for k in range(0,len(self.pts[0])):
			self.tempts.append(self.plat.Give_Em(self.pts[1][k], self.pts[2][k], self.pts[3][k], self.pts[5][k]))

		color = 'tab:red'
		ax1.set_xlabel('время, с')
		ax1.set_ylabel('Em, В')
		ax1.plot(self.pts[0], self.tempts, color=color)
		ax1.set_ylim(0.98*max(self.tempts), 1.02*min(self.tempts))

		ax1.tick_params(axis='y', labelcolor=color)


		imx = InitIonMatrix()
		tmppts2 = []
		for l in range(0,len(self.pts[1])):
			tmppts2.append((self.pts[4][l] + self.pts[5][l]))

		ax2 = ax1.twinx()
		color = 'tab:blue'
		ax2.set_ylabel('[Ca], мкМ')
		ax2.plot(self.pts[0], tmppts2, color=color)
		ax2.tick_params(axis='y', labelcolor=color)
		ax2.set_ylim(0.9*min(tmppts2), 1.1*max(tmppts2))

		fig.tight_layout()
		plt.show()	


	def Plot_Flux_and_Ca(self):
		fig, ax1 = plt.subplots()

		color = 'tab:red'

		tmpptst = []
		what = 9
		for k in range(0,len(self.pts[0])-1):
			tmpptst.append(self.pts[0][k])
		for k in range(0,len(self.pts[0])-1):
			self.tempts.append(3e12*(self.pts[what][k+1] - self.pts[what][k])/(self.pts[0][k+1] - self.pts[0][k]) + (self.pts[what+1][k+1] - self.pts[what+1][k])/(self.pts[0][k+1] - self.pts[0][k]))
		plt.figure(1)
		ax1.plot(tmpptst, self.tempts)
		ax1.set_xlabel('время, c')                              # Axes labels
		ax1.set_ylabel('ток, пА')                                 # Range of the y axis
		ax1.tick_params(axis='y', labelcolor=color)
		ax1.plot(self.pts[0][1:], self.tempts, color=color)
		ax1.set_ylim(0.9*min(self.tempts), 1.1*max(self.tempts))

		imx = InitIonMatrix()
		tmppts2 = []
		for l in range(0,len(self.pts[1])-1):
			tmppts2.append((self.pts[4][l] + self.pts[5][l]))

		ax2 = ax1.twinx()
		color = 'tab:blue'
		ax2.set_ylabel('[Ca], мкМ')
		ax2.plot(self.pts[0][1:], tmppts2, color=color)
		ax2.tick_params(axis='y', labelcolor=color)
		ax2.set_ylim(0.9*min(tmppts2), 1.1*max(tmppts2))

		fig.tight_layout()
		plt.show()

	def Plot_Pip_Current(self):
		tmpptst = []
		what = 9
		for k in range(0,len(self.pts[0])-1):
			tmpptst.append(self.pts[0][k])
		for k in range(0,len(self.pts[0])-1):
			self.tempts.append(3e12*(self.pts[what][k+1] - self.pts[what][k])/(self.pts[0][k+1] - self.pts[0][k]) + (self.pts[what+1][k+1] - self.pts[what+1][k])/(self.pts[0][k+1] - self.pts[0][k]))
		plt.figure(1)
		plt.plot(tmpptst, self.tempts)
		plt.xlabel('время, c')                              # Axes labels
		plt.ylabel('ток, пА')                                 # Range of the y axis
		plt.title('Ток через мембрану под пипеткой')                             # Figure title from model name
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


# this one is a hell incarnate
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

	def plotVAC(self):
		temp = []
		Amperes = []
		Volts = []
		plat = platelet_config(iSolution = -1)
		for d in np.linspace(-0.010, 0.090, 100):
			plat.dPhiInit = d
			DST = DST_interface(iPlat = plat)
			temp.append(DST.IntegrateGetLastTwo(var_names = ['t', 'dPhi', 'PipetteCurrent']))
		for i in range(len(temp)):
			Amperes.append((temp[i][2][0] - temp[i][2][1])/(temp[i][0][1] - temp[i][0][0])*1e12)
			Volts.append((temp[i][1][0]+plat.params['em default'][0])*1e3)
		plt.figure(1)
		plt.plot(Volts, Amperes)
		plt.xlabel('membrane voltage, mV')                              # Axes labels
		plt.ylabel('patch current, pA')                                 # Range of the y axis
		plt.show()

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
		plat.MakeStationary()
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
					plat.MakeStationary()
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
		plat.MakeStationary()
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
			plat.MakeStationary()
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
		for k in (0,1,2): plat[k].MakeStationary()
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
		plat.MakeStationary()
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
		for i in (0,1,2): plat[i].MakeStationary()
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
		plat.MakeStationary()
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
			plat.MakeStationary()
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
			plat.MakeStationary()
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
			plat.MakeStationary()
			DST = DST_interface_EQPoint(iPlat = plat)
			yraw = DST.Get_Eq_Point(r = 'concentrations')
			for i in range(3): y[i].append(yraw[i])
			print(k)
		plt.plot(x, y[0], 'g', x, y[1], 'r', x, y[2], 'b')
		plt.ylim(top = 1.1*max(y[1]), bottom = 0.9*min(y[0]))
		plt.ylabel('[], mM')
		plt.xlabel('P_add, 1/h, k_Na = ' + str(kions[0]) +', k_K = '+ str(kions[1]) + ', k_Cl = ' + str(kions[2]))
		plt.show()

