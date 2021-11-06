#170590  nizke pomer 30L/30Ra pritom stejne V!!


from numpy import genfromtxt
col =  ['T30L', 'T30R','T330Le','T330Re', 'T210L','T210R', 'V330L', 'V330R','V330Le', 'V330Re','S30L','S30R']
my_data = genfromtxt('beams_corrections_int.txt', delimiter=';')
shot_int = my_data[:,0]
scale_int = my_data[:,1:-1:3]
volt_int = my_data[:,2:-1:3]
time_int = my_data[:,3:-1:3]
my_data = genfromtxt('beams_corrections_impcon.txt', delimiter=';')
shot = my_data[:,0]
scale = my_data[:,1:-1:3]
volt = my_data[:,2:-1:3]
time = my_data[:,3:-1:3]

my_data = genfromtxt('beams_corrections_int_noG.txt', delimiter=';')
shot_int_noG = my_data[:,0]
scale_int_noG = my_data[:,1:-1:3]
volt_int_noG = my_data[:,2:-1:3]
time_int_noG = my_data[:,3:-1:3]
 

ind = slice(None,None)
#ind = volt_int[:,1]>73
plot(shot_int[ind],1/scale_int[ind,1], 'o' ,label = 'new')#,label='(n_30R*G_R)/(n_30L*G_L)')
#ind = volt[:,1]>73
plot(shot[ind],1/scale[ind,1], '.' ,label = 'old')#label='n_30R/n_30L')
ylim(0,3)
axvline(162000,c='k')
axvline(168615,c='k')
axvline(178000,c='k')
axvline(181111,c='k')
axhline(y=1)
xlabel('discharge')
ylabel('ratio')
ylabel('n30R/n30L')

ylim(0.7,2.)
legend(loc='best')
title('beam 30R; energy > 73keV')
show()



ind = slice(None,None)
ind = volt_int_noG[:,1]>73
plot(shot_int_noG[ind],1/scale_int_noG[ind,1], 'o' ,label='(n_30R*G_R)/(n_30L*G_L)')
ind = volt_int[:,1]>73
plot(shot_int[ind],1/scale_int[ind,1], '.' ,label='n_30R/n_30L')
ylim(0,3)
axvline(162000,c='k')
axvline(168615,c='k')
axvline(178000,c='k')
axvline(181111,c='k')
axhline(y=1)
xlabel('discharge')
ylabel('ratio')
ylabel('n30R/n30L')

ylim(0.7,2.)
legend(loc='best')
title('beam 30R; energy > 73keV')
show()








#plot( shot, scale[:,1],'o')
cm = plt.cm.get_cmap('brg')
axvline(162000,c='k')
axvline(168615,c='k')
axvline(178000,c='k')
axvline(181111,c='k')
sc=scatter( shot_int, 1/scale_int[:,1],c= volt_int[:,1]-volt_int[:,0], s=10,alpha=.5, cmap = cm)
c=plt.colorbar(sc)
c.set_label('Volts (30R-30L)')
xlabel('discharge')
ylabel('n30R/n30L')
axhline(1,c='k')
ylim(0.8,2.5)
grid()
show()

sc=scatter( arange(len(scale_int)),  1/scale_int[:,1],c= volt_int[:,1]-volt_int[:,0], s=10,alpha=.5, cmap = cm)
c=plt.colorbar(sc)
c.set_label('Volts (30R-30L)')
xlabel('discharge')
ylabel('n30R/n30L')
axhline(1,c='k')
ylim(0.8,2.5)
grid()
show()


sc=scatter( shot, 1/scale[:,1],c= volt[:,1]-volt[:,0], s=10,alpha=.5, cmap = cm)
c=plt.colorbar(sc)
c.set_label('Volts (30R-30L)')
xlabel('discharge')
ylabel('n30R/n30L')
axhline(1,c='k')
ylim(0.8,2.5)
grid()
show()

sc=scatter( arange(len(scale)),  1/scale[:,1],c= volt[:,1]-volt[:,0], s=10,alpha=.5, cmap = cm)
c=plt.colorbar(sc)
c.set_label('Volts (30R-30L)')
xlabel('discharge')
ylabel('n30R/n30L')
axhline(1,c='k')
ylim(0.8,2.5)
grid()
show()





 

ind_int =  (scale_int[:,1]>0)&isfinite(volt_int[:,1])&isfinite(volt_int[:,0])&(scale_int[:,1]<10)
ind1  = np.zeros(len(scale_int))
ind1[shot_int < 162200] = 1
ind2  = np.zeros(len(scale_int))
ind2[(shot_int > 162200)&(shot_int < 168300)] = 1
ind3  = np.zeros(len(scale_int))
ind3[(shot_int > 168300)&(shot_int < 177777)] = 1
ind4  = np.zeros(len(scale_int))
ind4[(shot_int > 177777)&(shot_int < 181200)] = 1
ind5  = np.zeros(len(scale_int))
ind5[(shot_int > 181200) ] = 1

dV_int = volt_int[:,1]-volt_int[:,0]
dV_int-=median(dV_int[ind_int])


A_int = np.vstack((ind1,ind2,ind3,ind4,ind5, dV_int, dV_int**2)).T[ind_int]
c_int = linalg.lstsq(A_int,log(scale_int[ind_int,1]),rcond=None)[0]
scale_ = exp(-dot(A_int[:,4:],c_int[4:]))*scale_int[ind_int,1]
 

ind =  (scale[:,1]>0)&isfinite(volt[:,1])&isfinite(volt[:,0])&(scale[:,1]<10)
ind1  = np.zeros(len(scale))
ind1[shot < 162200] = 1
ind2  = np.zeros(len(scale))
ind2[(shot > 162200)&(shot < 168300)] = 1
ind3  = np.zeros(len(scale))
ind3[(shot > 168300)&(shot < 177777)] = 1
ind4  = np.zeros(len(scale))
ind4[(shot > 177777)&(shot < 181200)] = 1
ind5  = np.zeros(len(scale))
ind5[(shot > 181200) ] = 1

dV = volt[:,1]-volt[:,0]
dV-=median(dV[ind])
A = np.vstack((ind1,ind2,ind3,ind4, dV, dV**2)).T[ind]
c = linalg.lstsq(A,log(scale[ind,1]),rcond=None)[0]
#scale_ = exp(-dot(A[:,4:],c[4:]))*scale[ind,1]

plot(volt_int[ind_int,1]-volt_int[ind_int,0], 1/(exp(-dot(A_int[:,:4],c[:4]))*scale_int[ind_int,1]),'o',label='new CX rates')
plot(volt[ind,1]-volt[ind,0], 1/(exp(-dot(A[:,:4],c[:4]))*scale[ind,1]),'.',label='old CX rates')
axhline(y=1,c='k')
xlabel('Voltage (30R-30L)')
ylabel('n30R/n30L')
legend()
ylim(.5,3)
show()

 

f,ax=subplots(3,1,sharex=True,sharey=True)

sc1=ax[2].scatter( shot_int[ind_int] ,  1/scale_,c= (volt_int[:,1]-volt_int[:,0])[ind_int], s=10,alpha=.5, cmap = cm)
sc2=ax[1].scatter( shot_int ,  1/scale_int[:,1],c= volt_int[:,1]-volt_int[:,0], s=10,alpha=.5, cmap = cm)
sc3=ax[0].scatter( shot ,  1/scale[:,1],c= volt[:,1]-volt[:,0], s=10,alpha=.5, cmap = cm)

ax[0].axhline(y=1,c='k')
ax[1].axhline(y=1,c='k')
ax[2].axhline(y=1,c='k')
ax[-1].set_xlabel('discharge')
c1=f.colorbar(sc1,ax=ax[0])
c2=f.colorbar(sc2,ax=ax[1])
c3=f.colorbar(sc3,ax=ax[2])
c1.set_label('Volts (30R-30L)')
c2.set_label('Volts (30R-30L)')
c3.set_label('Volts (30R-30L)')
ax[0].set_ylabel('n30R/n30L')
ax[1].set_ylabel('n30R/n30L')
ax[2].set_ylabel('n30R/n30L')
ax[1].set_ylim(0.5,2.5)
show()






figure()




f,ax=subplots(2,1,sharex=True,sharey=True)
sc=ax[1].scatter( shot_int[ind],  1/scale_*0.95,c= (volt_int[:,1]-volt_int[:,0])[ind], s=10,alpha=.5, cmap = cm)
sc=ax[0].scatter( shot_int,  1/scale_int[:,1],c= volt_int[:,1]-volt_int[:,0], s=10,alpha=.5, cmap = cm)

#show()
figure()

#plot(volt_int[ind,1]-volt_int[ind,0], exp(-dot(A[:,:4],c[:4]))*scale_int[ind,1],'.')
scatter(volt_int[ind,1]-volt_int[ind,0], 1/(exp(-dot(A[:,:4],c[:4]))*scale_int[ind,1]) ,c= (volt_int[:,1]-volt_int[:,0])[ind], s=10,alpha=.5, cmap = cm)
show()



plot( scale,'o')
plot( scale_,'.')

A = np.vstack(( np.ones_like(volt[ind,1]), volt[ind,1]-volt[ind,0] )).T
c = linalg.lstsq(A,scale[ind,1]-median(scale[ind,1]),rcond=None)[0]

A_ = np.vstack(( np.ones_like(volt[:,1]), volt[:,1]-volt[:,0] )).T
scale_ = -dot(A_,c)+scale[:,1]+(median(scale[ind,1])-mean(scale[ind,1]))/2
plot(shot,  1/scale[:,1],'.',label='n30R/n30L')
plot(shot, 1/scale_,'.',label='n30R/n30L E_beam corr')
axvline(162000)
axvline(168615)
axvline(178000)
axvline(181111)
legend(loc='best')
axhline(1,c='k')
show()


plot(volt[ind,1]-volt[ind,0], 1/scale[ind,1],'.',label='n30R/n30L')
plot(volt[ind,1]-volt[ind,0], 1/scale_[ind],'.',label='n30R/n30L E_beam corr')
plot(linspace(-33,0),1/(dot(np.vstack((ones(50),linspace(-30,0))).T,c)+median(scale[ind,1])),'b')
legend(loc='best')
xlabel('voltage (30R-30L)')
axhline(1,c='k')
show()



plot(volt[ind,1]/10-8)
plot(volt[ind,0]/10-8)


plot(volt[ind,1],scale[ind,1] ,'o')
#plot(volt[ind,1],scale_ ,'o')
xlabel('30R_volt')
ylabel('n30R/n30L')
figure()

plot(volt[ind,1]-volt[ind,0],scale[ind,1] ,'o')
xlabel('30L_volt-30R_volt')
ylabel('n30R/n30L')
show()



plot(shot, scale[:,2],'o')
show()

plot(scale[:,3],'ro')
plot(scale[:,2],'bo')
plot((volt[:,3]-volt[:,0])/10,'r')
plot((volt[:,2]-volt[:,0])/10,'b')
show()

plot(volt[:,3]-volt[:,2], scale[:,2]/scale[:,3],'o')

show()





plot(volt[:,2]-volt[:,0], scale[:,2],'o')
plot(volt[:,1]-volt[:,0], scale[:,1],'o')

show()


plot(shot,scale[:,-1]/scale[:,-2],'ro')
plot(shot,scale[:,1],'bo')
show()

##################################   SPRED  #################################

f,ax = subplots(3)
sca(ax[0])
valid = (minimum(scale_int[:,-1],scale_int[:,-2]) > .1) &(maximum(scale_int[:,-1],scale_int[:,-2]) < 10) &((shot_int < 174000)|(shot_int > 178000))
semilogy(shot_int[valid],1/scale_int[valid,-1],'.',label='SPRED 30R')
plot(shot_int[valid],1/scale_int[valid,-2],'.',label='SPRED 30L')
axhline(1,c='k')
grid()
legend(loc='best')
#xlabel('shot')
ylabel('n_SPRED/n_30L')
sca(ax[1])
valid = (minimum(scale_int[:,-1],scale_int[:,-2]) > .1) &(maximum(scale_int[:,-1],scale_int[:,-2]) < 10) &((shot_int < 174000)|(shot_int > 178000))
plot(shot_int[valid],scale_int[valid,-2]/scale_int[valid,-1],'o',label='SPRED 30R/SPRED 30L')
plot(shot_int[valid],1/scale_int[valid,1],'.',label='n_30R/n_30L')
axhline(1,c='k')
grid()
legend(loc='best')
xlabel('shot')
ylabel('30R/30L')
ylim(0.6,1.3)

sca(ax[2])
plot( volt_int[valid,1]-volt_int[valid,0]+80  ,scale_int[valid,-2]/scale_int[valid,-1],'.',label='SPRED 30R/SPRED 30L')
xlabel('30R_volt-30L_volt+80')
ylabel('SPRED 30R/SPRED 30L')
axhline(1,c='k')
show()


 
#########################################  330 beams


from numpy import genfromtxt
col =  ['T30L', 'T30R','T330Le','T330Re', 'T210L','T210R', 'V330L', 'V330R','V330Le', 'V330Re','S30L','S30R']
my_data = genfromtxt('beams_corrections_int_330.txt', delimiter=';')
shot = my_data[:,0]
scale = my_data[:,1:-1:3]
volt = my_data[:,2:-1:3]
time = my_data[:,3:-1:3]

col =  ['T30L', 'T30R','T330Le','T330Re', 'T210L','T210R', 'V330L', 'V330R','V330Le', 'V330Re','S30L','S30R']


scale_int[(shot_int > 163500)&(shot_int < 163550),2] = nan
scale_int[(shot_int > 163500)&(shot_int < 163550),3] = nan

f,ax=subplots(2,2,sharex=True,sharey=True)

sc2=ax[1,0].scatter( shot_int ,  1/scale_int[:,2],c= volt_int[:,2]-volt_int[:,0], s=10,alpha=.5, cmap = cm)
sc3=ax[0,0].scatter( shot ,  1/scale[:,2],c= volt[:,2]-volt[:,0], s=10,alpha=.5, cmap = cm)
sc4=ax[1,1].scatter( shot_int ,  1/scale_int[:,3],c= volt_int[:,3]-volt_int[:,0], s=10,alpha=.5, cmap = cm)
sc5=ax[0,1].scatter( shot ,  1/scale[:,3],c= volt[:,3]-volt[:,0], s=10,alpha=.5, cmap = cm)


for a in ax.flatten():
    a.axvline(162000)
    a.axvline(168615)
    a.axvline(178000)
    a.axvline(181111)
ax[0,0].axhline(y=1,c='k')
ax[1,0].axhline(y=1,c='k')
ax[0,1].axhline(y=1,c='k')
ax[1,1].axhline(y=1,c='k')

ax[-1,0].set_xlabel('discharge')
#c1=f.colorbar(sc1,ax=ax[0])
c2=f.colorbar(sc2,ax=ax[1,1])
c3=f.colorbar(sc3,ax=ax[0,1])
c4=f.colorbar(sc4,ax=ax[0,0])
c5=f.colorbar(sc5,ax=ax[1,0])

c3.set_label('Volts (330R-30L)')
c2.set_label('Volts (330R-30L)')
c4.set_label('Volts (330R-30L)')
c5.set_label('Volts (330R-30L)')

#ax[0,0].set_ylabel('n30R/n30L')
#ax[1,0].set_ylabel('n30R/n30L')
ax[1,0].set_ylim(0.5,3.5)
show()




sc=scatter( shot_int_noG, scale_int_noG[:,2]/scale_int_noG[:,3],c= volt_int_noG[:,3]-volt_int_noG[:,2], s=10,alpha=.5, cmap = cm)
c=plt.colorbar(sc)
c.set_label('Volts (330R-330L)')
xlabel('discharge')
ylabel('(n_330R*G_R)/(n_330L*G_L)')
axvline(162000)
axvline(168615)
axvline(178000)
axvline(181111)
axhline(1,c='k')
ylim(0.5,2)
grid()
show()


ind = slice(None,None)
plot(shot_int[ind],1/scale_int[ind,2], 'o' ,label = 'new')#,label='(n_30R*G_R)/(n_30L*G_L)')
plot(shot[ind],1/scale[ind,2], '.' ,label = 'old')#label='n_30R/n_30L')
ylim(0,3)
axvline(162000,c='k')
axvline(168615,c='k')
axvline(178000,c='k')
axvline(181111,c='k')
axhline(y=1)
xlabel('discharge')
ylabel('ratio')
ylabel('n30R/n30L')

ylim(0.7,2.)
legend(loc='best')
title('beam 30R; energy > 73keV')
show()





#semilogy(shot[:],scale[:,3]/scale[:,2],'rx',label='T330Le')
semilogy(shot[:],scale_int[:,3]/scale_int[:,2],'bx',label='T330Le')

semilogy(shot[:],scale[:,3],'bx',label='T330Re')

#semilogy(shot[:],scale[:,8],'gx',label='V330Le')#poor
semilogy(shot[:],scale[:,9],'yx',label='V330Re')
legend()
show()


semilogy(shot[:],scale[:,9]/scale[:,3],'bx',label='T330Re/V330Re')

#semilogy(shot[:],scale[:,8],'gx',label='V330Le')#poor
#semilogy(shot[:],scale[:,9],'yx',label='V330Re')
legend()
show()





 #########################################  210 beams
 


semilogy(shot[:],scale[:,4],'bx',label='T210L')
semilogy(shot[:],scale[:,5],'yx',label='T210R')
legend()
show()


semilogy(shot[:],scale[:,4]/scale[:,5],'bx',label='T210L/210R')
#semilogy(shot[:],scale[:,5],'yx',label='T210R')
legend()
show()



