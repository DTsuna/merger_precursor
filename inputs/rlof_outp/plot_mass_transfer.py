import numpy as np
from matplotlib import pyplot as plt
import glob

# plot a_bin and Mdot evolution for the draft
plt.rcParams['font.size'] = 14
color_array = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig, [ax0, ax1] = plt.subplots(1, 2, figsize=(11,4.5))
for ax in [ax0, ax1]:
    ax.set_xlabel('time before merger [years]')
    ax.grid(linestyle=':')
    ax.set_yscale('log')
    ax.set_xlim(8, -0.2)
ax0.set_ylim(1e0, 3e2)
ax1.set_ylim(1e-3, 1e2)
ax0.set_ylabel('separation [$R_\odot$]')
ax1.set_ylabel(r'$|\dot{M}_*|$ [$M_\odot$ yr$^{-1}$]')

for i, mf in enumerate(sorted(glob.glob('RLOF*txt'))):
    file_id = '_'.join(mf.split('_')[1:]).split('.txt')[0]
    print(file_id)
    rlof_data = np.loadtxt(mf)
    rlof_data[:,0] /= 3.156e7 # convert to year
    rlof_data[:,1] /= 6.96e10 # convert a_bin to Rsun
    rlof_data[:,3] /= (1.98847e33/3.156e7) # convert Mdot to Msun/yr 
    t_merge = rlof_data[-1,0]
    print(t_merge)
    if '2.754' in mf:
    	linestyle='dashed'
    else:
        linestyle='solid'
    ax0.plot(-rlof_data[:,0]+t_merge, rlof_data[:,1], label=file_id, linestyle=linestyle, color=color_array[i])
    ax1.plot(-rlof_data[:,0]+t_merge, rlof_data[:,3], label='_nolegend_', linestyle=linestyle, color=color_array[i]) 
ax0.legend(loc='lower left', ncol=2, fontsize=12.)
plt.tight_layout()
plt.savefig('abin_mdot_evolution.pdf')
