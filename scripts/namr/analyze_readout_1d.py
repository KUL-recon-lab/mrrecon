import json
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

res_dir = 'results/221217_SquaredL2Norm'
noise_level = '0.2'
n = 64

#---------------------------------------------------------------------------

res_files = sorted(list(Path(res_dir).glob(f'nl{noise_level}*.h5')))

df = pd.DataFrame()

# read meta data from all files
for ifile, res_file in enumerate(res_files):
    with h5py.File(res_file, 'r') as f:
        header = json.loads(f.attrs['header'])
        if ifile == 0:
            betas = f['betas'][:]
            true_signal = f['signal'][:]

        df = pd.concat((df, pd.DataFrame(header, index=[ifile])))

df = df.astype({
    'noise_level': 'category',
    'gradient_factor': 'category',
    'seed': 'category'
})

num_seeds = len(df.seed.cat.categories)
num_gf = len(df.gradient_factor.cat.categories)
num_betas = betas.shape[0]

all_recons = np.zeros((num_seeds, num_gf, num_betas, n), dtype=complex)

for ifile, res_file in enumerate(res_files):
    print(f'{(ifile+1):05}/{len(res_files):05} {res_file.name}')
    with h5py.File(res_file, 'r') as f:
        header = json.loads(f.attrs['header'])

        i_seed = df.seed.cat.categories.get_loc(header['seed'])
        i_gf = df.gradient_factor.cat.categories.get_loc(
            header['gradient_factor'])

        all_recons[i_seed, i_gf, ...] = f['recons'][:]

mean_recons = all_recons.mean(0)
std_recons = all_recons.std(0)

rois = {
    'center': np.where(np.abs(true_signal) == 0.65),
    'GM': np.where(np.abs(true_signal) == 0.5),
    'WM': np.where(np.abs(true_signal) == 0.45),
    'CSF': np.where(np.abs(true_signal) == 1)
}

regional_bias = np.zeros((len(rois), num_gf, num_betas))
regional_std = np.zeros((len(rois), num_gf, num_betas))

for ir, (roi_name, roi_inds) in enumerate(rois.items()):
    for i_gf, gf in enumerate(df.gradient_factor.cat.categories):
        for i_b, beta in enumerate(betas):
            regional_bias[ir, i_gf,
                          i_b] = (np.abs(mean_recons[i_gf, i_b, :]) -
                                  np.abs(true_signal))[roi_inds].mean()
            regional_std[ir, i_gf, i_b] = std_recons[i_gf,
                                                     i_b, :][roi_inds].mean()
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
# plot the mean recons
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------

lw = 1
ncols = mean_recons.shape[1]
nrows = mean_recons.shape[0]

fig, ax = plt.subplots(nrows,
                       ncols + 1,
                       figsize=(ncols + 1, nrows),
                       sharex=True,
                       sharey=True)
for i_gf, gf in enumerate(df.gradient_factor.cat.categories):
    for i_b, beta in enumerate(betas):
        ax[i_gf, i_b].plot(np.abs(mean_recons[i_gf, i_b, :]), lw=lw)
for i_b, beta in enumerate(betas):
    ax[0, i_b].set_title(f'b {beta:.1e}', fontsize='small')
for i_gf, gf in enumerate(df.gradient_factor.cat.categories):
    ax[i_gf, 0].set_ylabel(f'gf {gf:.1e}', fontsize='small')
    ax[i_gf, -1].plot(np.abs(true_signal), 'k-', lw=lw)
for axx in ax.ravel():
    axx.grid(ls=':')
fig.suptitle(
    f'mean of {num_seeds} noise realizations - base noise level {noise_level}',
    fontsize='medium')
fig.tight_layout()
fig.show()

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
# plot the std recons
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------

fig2, ax2 = plt.subplots(nrows,
                         ncols,
                         figsize=(ncols, nrows),
                         sharex=True,
                         sharey=True)
for i_gf, gf in enumerate(df.gradient_factor.cat.categories):
    for i_b, beta in enumerate(betas):
        ax2[i_gf, i_b].plot(np.abs(std_recons[i_gf, i_b, :]), lw=lw)
for i_b, beta in enumerate(betas):
    ax2[0, i_b].set_title(f'b {beta:.1e}', fontsize='small')
for i_gf, gf in enumerate(df.gradient_factor.cat.categories):
    ax2[i_gf, 0].set_ylabel(f'gf {gf:.1e}', fontsize='small')
for axx in ax2.ravel():
    axx.grid(ls=':')
fig2.suptitle(
    f'std.dev. of {num_seeds} noise realizations - base noise level {noise_level}',
    fontsize='medium')
fig2.tight_layout()
fig2.show()

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
# plot the mean recons
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------

fig3, ax3 = plt.subplots(nrows,
                         ncols + 1,
                         figsize=(ncols + 1, nrows),
                         sharex=True,
                         sharey=True)
for i_gf, gf in enumerate(df.gradient_factor.cat.categories):
    for i_b, beta in enumerate(betas):
        ax3[i_gf, i_b].plot(np.abs(all_recons[0, i_gf, i_b, :]), lw=lw)
for i_b, beta in enumerate(betas):
    ax3[0, i_b].set_title(f'b {beta:.1e}', fontsize='small')
for i_gf, gf in enumerate(df.gradient_factor.cat.categories):
    ax3[i_gf, 0].set_ylabel(f'gf {gf:.1e}', fontsize='small')
    ax3[i_gf, -1].plot(np.abs(true_signal), 'k-', lw=lw)
for axx in ax3.ravel():
    axx.grid(ls=':')
fig3.suptitle(f'1st noise realization - base noise level {noise_level}',
              fontsize='medium')
fig3.tight_layout()
fig3.show()

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
# plot regional bias vs noise
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
num_col = len(rois)
fig4, ax4 = plt.subplots(1, num_col, figsize=(4 * num_col, 4))

for ir, (roi_name, roi_inds) in enumerate(rois.items()):
    for i_gf, gf in enumerate(df.gradient_factor.cat.categories):
        ax4[ir].plot(regional_std[ir, i_gf],
                     regional_bias[ir, i_gf],
                     '.-',
                     label=f'gf {gf}')
    ax4[ir].set_title(roi_name)
    ax4[ir].set_ylabel('regional bias')
    ax4[ir].set_xlabel('regional std.dev.')

for axx in ax4.ravel():
    axx.grid(ls=':')

ax4[0].legend(ncol=2, fontsize='small')
fig4.tight_layout()
fig4.show()
