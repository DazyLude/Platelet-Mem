import model as m

p = m.otherfunctions()
plat = m.platelet_config(iSolution = -1)
# print plat.Rest_UATP()
# p.testplot(4)

# print plat.Pnv15rest()
# print plat.P_calc(0.005, 0.08, -0.05, 5000e-12, 1)
plat.Calc_permeabilities_from_channels()
# plat.Get_Info()

# print calc_permeability_elasticity(ion = 1)
# p.plot2D_permeability(3,3)
# p.plotFamily_Em_permeability(3,3)

fName = 'tmp.txt'

# fName = 'highip3.txt'
# fName = 'somewhereinbetween.txt'
# fName = 'medip3.txt'
# fName = 'smolip3.txt'
# fName = 'lowEm.txt'

DST = m.DST_interface(iPlat = plat)
DST.Integrate(fName = fName)
# DST.Diagram()
# DST.Get_Eq_Point()

plotty = m.plotncalc(plat, fName = fName)
# plotty.Plot_Em()	
# for i in range(1,4): plotty.Plot_Conc_Change(what = i)
for i in range(4,8): plotty.Plot_Concentrations(what = i)
plotty.Plot_Conc_Ca()
# p.calc_em_elast()
# p.plot_Beefoorc()
# p.backpupper()
# p.plot2D_permeability()
# p.plotEmfromSol()
# p.plotEmFromPermeabity()
# p.plotConcFromPermeabity()
# p.Phase_Plane(numofpoints = 12)

# print plat.J_calc(1e-9, 2e-3, -0.055, 2)

print('finished')
