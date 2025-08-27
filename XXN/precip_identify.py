# %% [markdown]
# ## Identifying precip extremes: test
# ### CESM2 large ensemble data 2015-2050

#env VIRTUAL_ENV='/div/qbo/users/py3Env/venv_basic'

# %%
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import numpy as np
import xarray as xr
from matplotlib import rcParams
import pandas as pd
from os import listdir
import matplotlib as mpl
import cftime
import glob
import math
import sys

# %%
rcParams['font.size'] = 20

# %%
#define file locations
file_loc = '/div/no-backup/Large_Ensemble_data/CESM2-LENS'

#experiment name (ssp370 or ssp370-126aer)
exp_name = sys.argv[1] #'BSSP370smbb' # #'ssp370-126aer'

#change variable file label name, sets variable to read in
var_filelab = 'PRECT'

#index variable name
var_name = 'PRECT'

sigma_level = float(sys.argv[2]) #4. # #4.
sigma_lab = int(sigma_level)

lower_level = sigma_level -1
lower_sigma_lab = int(lower_level)

#output file location
output_write = '/div/nac/users/zofias/XXN/extreme_loc_tables'
out_plots = '/div/nac/users/zofias/plots/XXN'

# %%
# select region by changing region_label
region_label = 'NO' #sys.argv[3] #'EU' # 'SA' 'EA' 'NA' 'EU'

regions = {'SA': [5,35,65,95],
           'EA' : [20,53,95,133],
           'NA' : [25,70,-150,-45],
           'EU' : [35,70,-20,45],
           'NO' : [53,72,0,32]}

#selects lat lon bounds for region
lat_min, lat_max, lon_min, lon_max = regions[region_label]

# %%
var_filepath = f"{file_loc}/{exp_name}/{var_filelab}"

#file_years = '20150101-20541231'

#def make_filename(ens, years):
#    filename = f"b.e21.BSSP370smbb.f09_g17.LE2-{ens}.cam.h1.PRECT.{years}.nc"
#    return filename

#r1011001_files = f"{var_filepath}/{make_filename('1011.001',file_years)}"
#r1031002_files = f"{var_filepath}/{make_filename('1031.002',file_years)}"
#r1051003_files = f"{var_filepath}/{make_filename('1051.003',file_years)}"
#r1071004_files = f"{var_filepath}/{make_filename('1071.004',file_years)}"
#r1091005_files = f"{var_filepath}/{make_filename('1091.005',file_years)}"
#r1111006_files = f"{var_filepath}/{make_filename('1111.006',file_years)}"
#r1131007_files = f"{var_filepath}/{make_filename('1131.007',file_years)}"
##r1151008_files = f"{var_filepath}/{make_filename('1151.008',file_years)}"
#r1171009_files = f"{var_filepath}/{make_filename('1171.009',file_years)}"
#r1191010_files = f"{var_filepath}/{make_filename('1191.010',file_years)}"

# %%
#define filelist
filelist = sorted(glob.glob(f'{var_filepath}/*h1*2100*.nc'))

print(var_filepath)
print(filelist)

# check filename label and adjust if incorrect
# should be e.g. 1011.001
#for file in filelist:
#    filelab = file.split('/')[-1][30:-34]
#    print(filelab)

#experiment labels from filelist
ens_labs = [file.split('/')[-1][30:-34] for file in filelist]

# %%
ensemble_array = xr.open_mfdataset(filelist, decode_times=True, combine="nested", concat_dim="member",data_vars=['PRECT'])

#make member coord labels correct
ensemble_array.coords['member'] = ens_labs

# %%
ensemble_array

# %%
ensemble_array['PRECT'] = ensemble_array['PRECT']*8.64e7
ensemble_array["PRECT"] = ensemble_array.PRECT.assign_attrs(units='mm/day')

# %%
#select lats and lons, switch lon to -180 to 180
ensemble_array = ensemble_array.assign_coords(lon=(((ensemble_array.lon + 180) % 360) - 180)).sortby('lon')
ensemble_array = ensemble_array.sel(lon=slice(lon_min,lon_max),lat=slice(lat_min,lat_max))

# %%
print('data read in')

# %% [markdown]
# ### Land mask

# %%
land_mask_fileloc = '/div/nac/users/zofias/CESM_output/land_mask/Norway_mask_192_288_CESM2-LENS.nc'
land_percent_data = xr.open_dataset(land_mask_fileloc)#.rename({'lon': 'longitude','lat': 'latitude'})

land_percent_data = land_percent_data.assign_coords(lon=(((land_percent_data.lon + 180) % 360) - 180)).sortby('lon')
#land_percent_regional = land_percent_data.sel(lon=slice(lon_min,lon_max),lat=slice(lat_min,lat_max))

# %%
#create boolean array of land mask (True = land)
land_mask = land_percent_data.mask

# %%
#Check land mask on map plot
fig, axis = plt.subplots(1, 1, dpi=150, subplot_kw=dict(projection=ccrs.PlateCarree()))
fg = land_mask.plot(cmap='Greens',
    cbar_kwargs={'label': f'Masked','orientation':'horizontal','pad':0.05},
    xlim=(lon_min-5,lon_max+5),
    ylim=(lat_min-5,lat_max+5),
    )

plt.title(f"Check land mask")
axis.coastlines(linewidth=0.5)  # cartopy function
#axis.add_feature(cartopy.feature.OCEAN,facecolor=("aliceblue"))

# %%
#mask array, anything not land has data = nan
ensemble_array_m = ensemble_array.where(land_mask)
print('land mask done')

# %%
#Check land mask on map plot
fig, axis = plt.subplots(1, 1, dpi=150, subplot_kw=dict(projection=ccrs.PlateCarree()))
fg = ensemble_array_m['PRECT'].isel(time=0).isel(member=0).plot(cmap='Reds',
    cbar_kwargs={'label': f'Temp (t=0)','orientation':'horizontal','pad':0.05},
    xlim=(lon_min-5,lon_max+5),
    ylim=(lat_min-5,lat_max+5),
    )

plt.title(f"Check Norway mask")
axis.coastlines(linewidth=0.5)  # cartopy function
axis.add_feature(cartopy.feature.OCEAN,facecolor=("aliceblue"))

# %% [markdown]
# ### Detrend data

# %%
#calculate temp trend over time to subtract. Mean over ensemble members and annually
ensemble_array_m['temp_trend'] = ensemble_array_m[var_name].mean(dim='member',skipna=True).rolling(time=30,min_periods=1,center=True).mean(skipna=True)


# %%
ensemble_array_m['detrended_temp'] = ensemble_array_m[var_name] - ensemble_array_m['temp_trend']
print('detrending done')

# %%
#calculate standard deviation
ensemble_array_m['std'] = ensemble_array_m['detrended_temp'].std(dim=('member','time'))
ensemble_array_m['sigma_level'] = sigma_level * ensemble_array_m['std']

# %%
# ### Count how many times 3sigma and 5sigma are exceeded in each gridbox

#subtract sigma level from all data (positive values = above sigma level chosen)
ensemble_array_m['exceeding_sigma'] = ensemble_array_m.detrended_temp - ensemble_array_m.sigma_level

# %%
#sigma test sets all non outliers to nan
sigma3_test = ensemble_array_m['exceeding_sigma'].where((ensemble_array_m['exceeding_sigma']>0).compute())

# %%
#count number of days exceeding threshold
ensemble_array_m['number_of_3sigma_days'] = sigma3_test.count(dim='time').sum(dim='member')

# %%
#max_days = ensemble_array_m['number_of_3sigma_days'].max().values.item()

# %% [markdown]
# ### Check whether the extreme persists for >3 days

# %%
#set threshold for consecutive days/lats/lons
consecutive_days_threshold = 3
grid_threshold = 3
total_sum_threshold = consecutive_days_threshold + grid_threshold*2
threshold_label = f'{consecutive_days_threshold}_{grid_threshold}'

# %%
exceed_sigma_boolean = ensemble_array_m['exceeding_sigma'] > 0

# %%
# find where >3days >3lats >3lons condition is met
more_3_days = exceed_sigma_boolean.rolling(time=consecutive_days_threshold,center=True,min_periods=consecutive_days_threshold).sum()
more_3_lons = exceed_sigma_boolean.rolling(lon=grid_threshold,center=True,min_periods=grid_threshold).sum()
more_3_lats = exceed_sigma_boolean.rolling(lat=grid_threshold,center=True,min_periods=grid_threshold).sum()

#more_3_lons_lats_times = exceed_sigma_boolean.rolling({'longitude':grid_threshold,'latitude':grid_threshold,'t':consecutive_days_threshold}, center=True,min_periods=3).sum()

# %%
all_lats_lons_times = more_3_days + more_3_lons + more_3_lats

consecutive_exceed_sigma_all = all_lats_lons_times == total_sum_threshold


# %%
consecutive_exceed_sigma_all

# %%
# ### Selecting lat/lon/time/member

#where gridpoints meet condition
member,time,lat,lon = np.where(consecutive_exceed_sigma_all==True)

# %%
#make array of extreme locations
where_extreme_array = pd.DataFrame(data = {
    'member_ind': member,
    'time_ind': time,
    'lat_ind': lat,
    'lon_ind':lon,
    'member': [ensemble_array_m.member[member[i]].item() for i in np.arange(len(member))],
    'time': [ensemble_array_m.time[time[i]].item() for i in np.arange(len(member))],
    'lat': [ensemble_array_m.lat[lat[i]].item() for i in np.arange(len(member))],
    'lon': [ensemble_array_m.lon[lon[i]].item() for i in np.arange(len(member))]
})

# %%
#save array of lats/lons/members/times
where_extreme_array.to_csv(f'{output_write}/{region_label}_{exp_name}_{var_name}_{sigma_lab}_sigma_{consecutive_days_threshold}days_{grid_threshold}box_locs.csv')


# %%
where_extreme_array_saved = pd.read_csv(f'{output_write}/{region_label}_{exp_name}_{var_name}_{sigma_lab}_sigma_{consecutive_days_threshold}days_{grid_threshold}box_locs.csv',index_col=0)

# %%
where_extreme_array_saved

# %%
#function to convert time from array into pandas time in new column
def convert_pandas_time(dataframe):
    dataframe['pd_time'] = pd.to_datetime(dataframe['time'], format='%Y-%m-%d %H:%M:%S',errors='coerce')

#add column with datetime
convert_pandas_time(where_extreme_array_saved)

# %%
#function to group days into events (by time and mean over lat lon)
#selects relevant columns in first line
def group_event(extreme_locs_data):
    time_member_grouped = extreme_locs_data.iloc[:, np.r_[0,1,2,3,-3,-2,-1]].groupby(by=['member_ind','time_ind'],as_index=False).mean(numeric_only=False)
    #boolean array of whether time index is one above the previous one
    diff_to_previous =  time_member_grouped.time_ind !=  (time_member_grouped.time_ind+1).shift(1)
    #convert boolean to cumulative sum
    cumul_sum = diff_to_previous.cumsum()
    events = time_member_grouped.groupby(cumul_sum,as_index=False).mean(numeric_only=False)
    return events, len(events) #returns array, with time index meaned over consecutive days, and number of events

#group days/gridboxes into events
events_array, events_number = group_event(where_extreme_array_saved)

number_of_events_summary = f'{region_label} {exp_name} {var_name} {sigma_lab} sigma {consecutive_days_threshold} days {grid_threshold} boxes: {events_number}'
print(number_of_events_summary)

with open(f'{output_write}/number_of_events.txt', 'a') as file:
    file.write(f'{number_of_events_summary}\n')

#round time, lat, lon to nearest whole number for indexing
events_array.round({'time_ind': 0, 'lat_ind': 0, 'lon_ind': 0,'lat':2,'lon':2})

#convert to integer for indexing
events_array['time_ind'] = events_array['time_ind'].astype(int)
events_array['lat_ind'] = events_array['lat_ind'].astype(int)
events_array['lon_ind'] = events_array['lon_ind'].astype(int)
events_array['member_ind'] = events_array['member_ind'].astype(int)

#save events array of lats/lons/members/times
events_array.to_csv(f'{output_write}/{region_label}_{exp_name}_{var_name}_{sigma_lab}_sigma_{consecutive_days_threshold}days_{grid_threshold}box_events.csv')

print('events array saved')

# %%
for n,time_ind in enumerate(events_array.time_ind):
    member_ind = events_array.member_ind[n]
    time_lab = str(events_array.pd_time[n])
    lat_lab=events_array.lat[n].round(2)
    lon_lab=events_array.lon[n].round(2)


    fig, axis = plt.subplots(1, 1, dpi=150, subplot_kw=dict(projection=ccrs.PlateCarree()))
    fg = ensemble_array['PRECT'].isel(time=slice(time_ind-2,time_ind+2),member=member_ind).mean(dim='time').plot(cmap='Blues',
                                                                            vmax=40,
                                                                            vmin=0,
                                                                            add_colorbar=False,
                                                                  #cbar_kwargs={'label': f'Precip (mm/day)','orientation':'horizontal','pad':0.05},
                                                                  xlim=(lon_min,lon_max),
                                                                  ylim=(lat_min,lat_max),
    )
    circle2 = plt.Circle((lon_lab, lat_lab), 1, color='red', fill=False)
    axis.add_patch(circle2)
    plt.title('')
    #plt.title(f'r{member_ind+1} {time_lab}',fontsize=16)
    axis.coastlines(linewidth=0.5)  # cartopy function
    fig.savefig(f'{out_plots}/precip_{sigma_lab}_r{member_ind+1}_{time_lab[:10]}_{lat_lab}_{lon_lab}.png', format='png', bbox_inches='tight')
    plt.cla()

# %%
nrows = math.ceil((events_array.shape[0]/3))
#plot above 3 sigma region
# Define the figure and each axis for the 3 rows and 3 columns
fig, axs = plt.subplots(nrows=nrows,ncols=3,
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        figsize=(15,5*nrows)
                       )

# axs is a 2 dimensional array of `GeoAxes`.  We will flatten it into a 1-D array
axs=axs.flatten()

#Loop over all of the models
for n,time_ind in enumerate(events_array.time_ind):
    member_ind = events_array.member_ind[n]
    time_lab = str(events_array.pd_time[n])
    lat_lab=events_array.lat[n]
    lon_lab=events_array.lon[n]

    circle2 = plt.Circle((lon_lab, lat_lab), 2, color='blue', fill=False)

    fg = (ensemble_array['PRECT'].isel(time=time_ind,member=member_ind)).where(ensemble_array['detrended_temp'].isel(time=time_ind,member=member_ind) > lower_level * ensemble_array['std']).plot(ax=axs[n],cmap='Reds',add_colorbar=False,#,norm=norm, #
    #vmax=320, vmin=275, 
        #cbar_kwargs={'label': f'Temperature (K)','orientation':'horizontal','pad':0.05},
        xlim=(lon_min,lon_max),
        ylim=(lat_min,lat_max),
        )
    
    axs[n].add_patch(circle2)
    axs[n].set_title(f'r{member_ind+1} {time_lab}',fontsize=16)
    axs[n].coastlines(linewidth=0.5)  # cartopy function

cbar_ax = fig.add_axes([0.2, 0.47, 0.6, 0.015])
cbar=fig.colorbar(fg,cax=cbar_ax,orientation='horizontal',label='Precip (mm/day)')
plt.suptitle(f'Areas over {lower_sigma_lab} sigma',y=0.93)
fig.subplots_adjust(bottom=0.5, top=0.9, left=0.1, right=0.9,wspace=0.02, hspace=0.17)
fig.savefig(f'{out_plots}{region_label}_{exp_name}_precip_maps_all_{lower_sigma_lab}sigma_area.pdf', format='pdf', bbox_inches='tight')

# %%



