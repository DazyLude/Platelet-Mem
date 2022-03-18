import model as m

p = m.otherfunctions()
plat = m.platelet_config(iSolution = -1, stationary = True, output = 'Full')

# print calc_permeability_elasticity(ion = 1)
# p.plot2D_permeability(3,3)
# p.plotFamily_Em_permeability(3,3)

fName = 'temp.txt'

# fName = 'highip3.txt'
# fName = 'somewhereinbetween.txt'
# fName = 'medip3.txt'
# fName = 'smolip3.txt'
# fName = 'lowEm.txt'

DST = m.DST_interface(iPlat = plat)
DST.Integrate(fName = fName)
# DST.Diagram()
# DST.Get_Eq_Point()1

# c_plotter = m.comparative_plotter(plat, 'no_fv4_less_fv3.txt', 'fv3_18.txt')
# c_plotter.Plot_Em()
# c_plotter.Plot_Ca()


# p.plotVAC()

plotty = m.plotter(plat, fName = fName)
# plotty.Plot_Em()
# plotty.Plot_Concentrations('Ca')
# for i in range(9, 10): plotty.Plot_Conc_Change(what = i)
# plotty.Plot_Concentrations(what = 'Na')
# plotty.Plot_NaKCl()
# plotty.Plot_Ca()
# plotty.calc_Kd()
# plotty.Plot_NCX_Species()
# plotty.Plot_NCX_Currents()
# p.calc_em_elast()
# p.plot_Beefoorc()
# p.backpupper()
# p.plot2D_permeability()
# p.plotEmfromSol()
# p.plotEmFromPermeabity()
# p.plotConcFromPermeabity()
# p.Phase_Plane(numofpoints = 12)
# plotty.Plot_Pip_Current()

# print plat.J_calc(1e-9, 2e-3, -0.055, 2)

print('finished')
