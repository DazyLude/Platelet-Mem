TheArray = [
	[0.005, 0.125, 0.03, 2.3053355e-8, 0.4e-9, 0.3, 0.1],	#default concentrations, 0; cl ideal: 0.01213625
	['Na','K','Cl','Ca_cyt', 'Ca_er', 'x100', 'x110'],	#names, 1
	[1e-10, 1e-10, 0.0195, 1e-11],	#zone of interest: start, 2
	[0.02, 0.2, 0.04, 1e-3],	#zone of interest: end, 3
	[0.145, 0.005, 0.150, 3e-3],	#default solution, 4
	[0.13, 0.02, 0.15, 2.35e-8],	#hypotonic solution 1, 5
	[0.11, 0.04, 0.15, 5e-3],	#hypertonic solution 2, 6
	[-0.055, -0.050, -0.080],	#em, default, lower, higher, 7
	[0, 500., 0.01],	#time: start, finish, dt 8 ------------------------------------------------------------------------------------------
	[0.05, 2, 1, 0],	#default permeabilities 1/h 9
	[0.5, 0., 0.004, 0.2, 20., 0.21, 1.64, 2., 0.185, 0.1, 4.6, 1.7, 1, 0.11, 6., 0.45, 0.145, 2.],	#Fedor constants, 10
	[0., 10, 0.05, 1],	#IP3 max, type of function, parametres of function, 11 <-------------------------------- IP3 HERE ---------------
	[128e-15, 5e-15, 100e-12],	#other platelet constants: Cm, V, S, 12
	[0.005, 3, 2],	#13 (atpase old) (left as a grim reminder of the cost of progress)
	[0.005], #parametres of NKCC: J0; 14
	[0.005, 1e-4, 10],	#parametres of NCX: JMax, c1, c2; 15
	[96485, 8.314, 310, 3600],	#world constants: F, R, T, speed (unit to hours); 16
	[],	# 17
	[[2.5e11,1e5],[1e4,1e5],[172,1.72e4],[1.5e7,2e5],[2e6,30],[1.15e4,6e8]], #Constants for ATPase kinetic model, by each reaction, then forward / backward, 18
	[.5, -0.03, -0.008], #kv 1.3, 19
	[], #, 20 deprecated
	[4.99e-3, 0.06e-3, 4.95e-3], #ATP, ADP, P concentrations, 21
	[0.00, -0.01, 0.01], #, NaV 22
	[], #, 23 deprecated
	[0, 0.2, 1.2, 0], #ca channels, total, Na relative, K relative 24
	[2.4e-7*500, 2e-7, 4], #PMCA constants: Jmax, K1/2, n; 25
	[1.3, 0.2e-6, 5], #Kca31 constants: Pmax 2.8, K1/2, n; 26
	[0, 1, 1], #Kca11 constants: Pmax, K1/2, n; 27
	[.8, 0.7 * 1e-6, 4], #TRPC6 channel state function; 28
	[0.2, 0.2, 0, 1], #TRPC6 channel relative permeabilities in order: Na, K, Cl, Ca; 29
	]

TheDictionary = {
	'names': ['Na','K','Cl','Ca_cyt', 'Ca_er', 'x100', 'x110'],
	'solution default': [0.145, 0.005, 0.150, 3e-3], #na k cl ca
	'em default': [-0.055],
	'time': [0, 600., 0.01], #start, end, step
	'constants': [96485, 8.314, 310, 3600], #F, R, k, time units in hours
	'ca model': [0.5, 0., 0.004, 0.2, 20., 0.21, 1.64, 2., 0.185, 0.1, 4.6, 1.7, 1, 0.11, 6., 0.45, 0.145, 2.], #tbd
	'ip3': [.13], #constant
	'NCXM': {'e': 1.67e-2, 'f': 9.34e+4, 'x': 4.81e+1, 'y': 1.25e+0, 'Speed_Mod': 1e-23*1e15*1e4, 
					'a': 6.17e+0, 'b': 8.68e+1, 'c': 3.67e-6, 'd': 1.66e+0,
					'g': 2.85e-2, 'h': 5.82e-1, 'j': 9.49e-1, 'k': 7.93e-1,},
	'NCXM_init': {'Cin_E': 0.161, 'Cin_Na': 0.00696, 'Cin_Ca': 0.279, 'Cout_Ca': 0.24, 'Cout_Na': 0.027, 'Cout_E': 0.286}
	}


def test_kinetics(the_dict, dict_names_forward, dict_names_backward):
	k_b = 1
	k_f = 1
	for name in dict_names_forward:
		k_f *= the_dict[name]
	k_f = k_f

	for name in dict_names_backward:
		k_b *= the_dict[name]

	return (k_f/k_b)



print(test_kinetics(TheDictionary['NCXM'],['a','c','g','x','f','k'],['b','d','j','y','e','h']))