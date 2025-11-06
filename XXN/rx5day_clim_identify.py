# %% [markdown]
# ## Identifying precip extremes with climatology baseline
# ### CESM2 large ensemble data 2015-2050

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
import nc_time_axis
#import xcdat
import datetime

# %%
rcParams['font.size'] = 20

# %%
#define file locations
file_loc = '/div/no-backup/Large_Ensemble_data/CESM2-LENS'

#experiment name (ssp370 or ssp370-126aer)
exp_name = 'BSSP370smbb' #sys.argv[1] #'ssp370-126aer'

#change variable file label name, sets variable to read in
var_filelab = 'PRECT'

#index variable name
var_name = 'PRECT'

sigma_level = float(sys.argv[2]) #4.
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
filelist = sorted(glob.glob(f'{var_filepath}/*h1*.nc'))

# check filename label and adjust if incorrect
# should be e.g. 1011.001
#for file in filelist:
#    filelab = file.split('/')[-1][30:-34]
#    print(filelab)

#experiment labels from filelist
ens_labs = [file.split('/')[-1][30:-34] for file in filelist]

# %%
ensemble_array = xr.open_mfdataset(filelist, decode_times=True, combine="nested", concat_dim="member",data_vars=[var_name])

#make member coord labels correct
ensemble_array.coords['member'] = ens_labs

# %%
ensemble_array

# %%
#select lats and lons, switch lon to -180 to 180
ensemble_array = ensemble_array.assign_coords(lon=(((ensemble_array.lon + 180) % 360) - 180)).sortby('lon')
ensemble_array = ensemble_array.sel(lon=slice(lon_min,lon_max),lat=slice(lat_min,lat_max))

# %%
ensemble_array['PRECT'] = ensemble_array['PRECT']*8.64e7
ensemble_array["PRECT"] = ensemble_array.PRECT.assign_attrs(units='mm/day')

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
#mask array, anything not land has data = nan
ensemble_array_m = ensemble_array.where(land_mask)
print('land mask done')


# ### Calculate 5day sum and climatology

# %%
#calculate 5 day sum
ensemble_array_m['rx5day'] = ensemble_array_m[var_name].rolling(time=5,min_periods=5,center=True).sum(skipna=True)#.mean(dim='member',skipna=True)

# %%
ds_climo = ensemble_array_m['rx5day'].sel(time=slice('2015-01-01','2034-12-31')).groupby('time.month').mean('time').mean(dim='member',skipna=True).to_dataset()
ds_anoms = (ensemble_array_m['rx5day'].groupby('time.month') - ds_climo)

# ### Calculate sigma levels

# %%
#calculate standard deviation
#ds_climo['std_daily'] = ensemble_array_m['rx5day'].groupby('time.dayofyear').mean('time').std('member')
ds_climo['std_monthly'] = ensemble_array_m['rx5day'].sel(time=slice('2015-01-01','2034-12-31')).groupby('time.month').std(('member','time'))

#ds_anoms['std'] = ds_anoms['rx5day'].std(dim=('member','time'))
ds_climo['sigma_level'] = sigma_level * ds_climo['std_monthly']


# %%
# ### Count how many times 3sigma and 5sigma are exceeded in each gridbox

#subtract sigma level from all data (positive values = above sigma level chosen)
ds_anoms['exceeding_sigma'] = ds_anoms['rx5day'].groupby('time.month') - ds_climo['sigma_level']

# %% [markdown]
# ### Check whether the extreme persists for >3 days

# %%
#set threshold for consecutive days/lats/lons
consecutive_days_threshold = 3
grid_threshold = 3
total_sum_threshold = consecutive_days_threshold + grid_threshold*2
threshold_label = f'{consecutive_days_threshold}_{grid_threshold}'

# %%
exceed_sigma_boolean = ds_anoms['exceeding_sigma'] > 0

# %%
# find where >3days >3lats >3lons condition is met
more_3_days = exceed_sigma_boolean.rolling(time=consecutive_days_threshold,center=True,min_periods=consecutive_days_threshold).sum()
more_3_lons = exceed_sigma_boolean.rolling(lon=grid_threshold,center=True,min_periods=grid_threshold).sum()
more_3_lats = exceed_sigma_boolean.rolling(lat=grid_threshold,center=True,min_periods=grid_threshold).sum()

#more_3_lons_lats_times = exceed_sigma_boolean.rolling({'longitude':grid_threshold,'latitude':grid_threshold,'t':consecutive_days_threshold}, center=True,min_periods=3).sum()

# %%
#combine all conditions
all_lats_lons_times = more_3_days + more_3_lons + more_3_lats

#array for where all conditions are met
consecutive_exceed_sigma_all = all_lats_lons_times == total_sum_threshold



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
where_extreme_array.to_csv(f'{output_write}/{region_label}_{exp_name}_{var_name}_{sigma_lab}_sigma_{consecutive_days_threshold}days_{grid_threshold}box_locs_rx5day.csv')


# %%
where_extreme_array_saved = pd.read_csv(f'{output_write}/{region_label}_{exp_name}_{var_name}_{sigma_lab}_sigma_{consecutive_days_threshold}days_{grid_threshold}box_locs_rx5day.csv',index_col=0)

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
    #group by member and time, mean over lat lon (for events that take place in same member at same time)
    time_member_grouped = extreme_locs_data.iloc[:, np.r_[0,1,2,3,-3,-2,-1]].groupby(by=['member_ind','time_ind'],as_index=False).mean(numeric_only=False)
    
    #boolean array of whether time index is one above the previous one (cumul = False)
    diff_to_previous =  time_member_grouped.time_ind !=  (time_member_grouped.time_ind+1).shift(1)
    #convert boolean to cumulative sum
    cumul_sum = diff_to_previous.cumsum()
    events = time_member_grouped.groupby(cumul_sum,as_index=False).mean(numeric_only=False)
    return events, len(events) #returns array, with time index meaned over consecutive days, and number of events

#group days/gridboxes into events
events_array, events_number = group_event(where_extreme_array_saved)

number_of_events_summary = f'{region_label} {exp_name} {var_name} {sigma_lab} sigma {consecutive_days_threshold} days {grid_threshold} boxes, climatology rx5day: {events_number}'
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
    fg = ensemble_array['PRECT'].isel(time=time_ind,member=member_ind).plot(cmap='Blues',
                                                                            vmax=40,
                                                                            vmin=0,
                                                                               #add_colorbar=False,
                                                                  cbar_kwargs={'label': f'Precip (mm/day)','orientation':'horizontal','pad':0.05},
                                                                  xlim=(lon_min,lon_max),
                                                                  ylim=(lat_min,lat_max),
    )
    circle2 = plt.Circle((lon_lab, lat_lab), 1, color='blue', fill=False)
    axis.add_patch(circle2)
    plt.title('')
    #plt.title(f'r{member_ind+1} {time_lab}',fontsize=16)
    axis.coastlines(linewidth=0.5)  # cartopy function
    fig.savefig(f'{out_plots}/PRECT5day_{sigma_lab}_r{member_ind+1}_{time_lab[:10]}_{lat_lab}_{lon_lab}_single.png', format='png', bbox_inches='tight')

# %%
for n,time_ind in enumerate(events_array.time_ind):
    member_ind = events_array.member_ind[n]
    time_lab = str(events_array.pd_time[n])
    lat_lab=events_array.lat[n].round(2)
    lon_lab=events_array.lon[n].round(2)


    fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(20,6), subplot_kw=dict(projection=ccrs.PlateCarree()))
    fg = ensemble_array['PRECT'].isel(time=time_ind - 3,member=member_ind).plot(cmap='Blues',
                                                                               ax=ax1,
                                                                               vmin=0,
                                                                               vmax=40,
                                                                               #add_colorbar=False,
                                                                  cbar_kwargs={'label': f'Precip (mm/day)','orientation':'horizontal','pad':0.05},
                                                                  xlim=(lon_min,lon_max),
                                                                  ylim=(lat_min,lat_max),
    )
    ax1.coastlines(linewidth=0.5)
    ax1.set_title(f'- 3 days')
    fg = ensemble_array['PRECT'].isel(time=time_ind,member=member_ind).plot(cmap='Blues',
                                                                               ax=ax2,
                                                                               vmin=0,
                                                                               vmax=40,
                                                                               #add_colorbar=False,
                                                                  cbar_kwargs={'label': f'Precip (mm/day)','orientation':'horizontal','pad':0.05},
                                                                  xlim=(lon_min,lon_max),
                                                                  ylim=(lat_min,lat_max),
    )
    ax2.coastlines(linewidth=0.5)
    ax2.set_title(f'{time_lab[:10]}')
    circle2 = plt.Circle((lon_lab, lat_lab), 1, color='blue', fill=False)
    ax2.add_patch(circle2)

    fg = ensemble_array['PRECT'].isel(time=time_ind+3,member=member_ind).plot(cmap='Blues',
                                                                               ax=ax3,
                                                                               vmin=0,
                                                                               vmax=40,
                                                                               #add_colorbar=False,
                                                                  cbar_kwargs={'label': f'Precip (mm/day)','orientation':'horizontal','pad':0.05},
                                                                  xlim=(lon_min,lon_max),
                                                                  ylim=(lat_min,lat_max),
    )
    ax3.coastlines(linewidth=0.5)
    ax3.set_title(f'+ 3 days')


    #plt.title('')
    #plt.title(f'r{member_ind+1} {time_lab}',fontsize=16)
    #axis.coastlines(linewidth=0.5)  # cartopy function
    fig.savefig(f'{out_plots}/PRECT5day_{sigma_lab}_r{member_ind+1}_{time_lab[:10]}_{lat_lab}_{lon_lab}_plusmin3.png', format='png', bbox_inches='tight')

# %%
for n,time_ind in enumerate(events_array.time_ind):
    member_ind = events_array.member_ind[n]
    time_lab = str(events_array.pd_time[n])
    lat_lab=events_array.lat[n].round(2)
    lon_lab=events_array.lon[n].round(2)


    fig, axis = plt.subplots(1, 1, dpi=150, subplot_kw=dict(projection=ccrs.PlateCarree()))
    
    fg = ensemble_array['PRECT'].isel(time=time_ind,member=member_ind).plot.contour(ax=axis,cmap='winter',add_colorbar=False,levels=8,#,norm=norm, #
                                                vmax=40, vmin=0, 
                                                zorder=4,linewidths=0.5,
        #cbar_kwargs={'label': f'Temperature (K)','orientation':'horizontal','pad':0.05},
        xlim=(lon_min,lon_max),
        ylim=(lat_min,lat_max),
        )
    argmax_vals = ensemble_array['PRECT'].isel(time=time_ind,member=member_ind).argmax(dim=['lat','lon'])
    axis.scatter(ensemble_array['PRECT'].lon.isel(lon=int(argmax_vals['lon'].values)),ensemble_array['PRECT'].lat.isel(lat=int(argmax_vals['lat'].values)),s=70,marker ='x',color='blue')

    fg = ensemble_array['PRECT'].isel(time=time_ind-2,member=member_ind).plot.contour(ax=axis,cmap='spring',add_colorbar=False,levels=8,#,norm=norm, #
                                                vmax=40, vmin=0, 
                                                zorder=4,linewidths=0.5,
        #cbar_kwargs={'label': f'Temperature (K)','orientation':'horizontal','pad':0.05},
        xlim=(lon_min,lon_max),
        ylim=(lat_min,lat_max),
        )
    argmax_vals_2 = ensemble_array['PRECT'].isel(time=time_ind-2,member=member_ind).argmax(dim=['lat','lon'])
    axis.scatter(ensemble_array['PRECT'].lon.isel(lon=int(argmax_vals_2['lon'].values)),ensemble_array['PRECT'].lat.isel(lat=int(argmax_vals_2['lat'].values)),s=70,marker ='+',color='hotpink')


    circle2 = plt.Circle((lon_lab, lat_lab), 1, color='blue', fill=False)
    axis.add_patch(circle2)
    plt.title('')
    #plt.title(f'r{member_ind+1} {time_lab}',fontsize=16)
    axis.coastlines(linewidth=0.5)  # cartopy function
    fig.savefig(f'{out_plots}/PRECT5day_{sigma_lab}_r{member_ind+1}_{time_lab[:10]}_{lat_lab}_{lon_lab}_contour.png', format='png', bbox_inches='tight')

# %%



