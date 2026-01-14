
# ### Region masking GAINS emissions for HTAP3OPNS

import xarray as xr
import glob


# ### Import data and regions file
xr.set_options(keep_attrs=True)

#file locations
file_loc_mask = '/div/no-backup-nac/users/zofias/GAINS'
file_loc_gains = '/div/no-backup-nac/users/zofias/GAINS'

file_outdir = '/div/no-backup-nac/users/zofias/GAINS/regions/'

#filenames
filelist = glob.glob(f'{file_loc_gains}/*CTM*.nc')


#import region mask data
regions_data = xr.open_dataset(f'{file_loc_mask}/HTAP3_Regions_NC01x01_v3.nc')

# make boolean region masks for each region
EAS_mask = regions_data.EAS > 0.9
EMEP_mask = regions_data.EMEP > 0.9
SAS_mask = regions_data.SAS > 0.9
SMD_mask = regions_data.SMD > 0.9
NAM_mask = regions_data.NAM > 0.9

# ### Apply region masks to all species

for file in filelist:
    #import GAINS data, get label
    filename = file.split(sep='/')[-1]
    file_lab = filename[:-3]
    print(f'Processing {file_lab}')

    #open file
    data = xr.open_dataset(file)

    #region masking and save to netcdf
    EAS = data.where(EAS_mask)
    EAS.to_netcdf(f'{file_outdir}{file_lab}_EAS.nc')
    print('EAS done')

    NAM = data.where(NAM_mask)
    NAM.to_netcdf(f'{file_outdir}{file_lab}_NAM.nc')
    print('NAM done')

    SMD = data.where(SMD_mask)
    SMD.to_netcdf(f'{file_outdir}{file_lab}_SMD.nc')
    print('SMD done')

    EMEP = data.where(EMEP_mask)
    EMEP.to_netcdf(f'{file_outdir}{file_lab}_EMEP.nc')
    print('EMEP done')

    SAS = data.where(SAS_mask)
    SAS.to_netcdf(f'{file_outdir}{file_lab}_SAS.nc')
    print('SAS done')



