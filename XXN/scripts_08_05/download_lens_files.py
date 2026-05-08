#!/usr/bin/env python
""" 
Python script to download selected files from rda.ucar.edu.
After you save the file, don't forget to make it executable
i.e. - "chmod 755 <name_of_script>"
"""
import sys, os
from urllib.request import build_opener
import numpy as np

opener = build_opener()

save_dir = '/div/no-backup-nac/users/zofias/CESM2-LENS/weather_data/westNO_events_05_05/'

filelist = np.load('/div/no-backup-nac/users/zofias/CESM2-LENS/weather_data/filelist_to_read_cesm2lens.npy').tolist()[6:7]

for file in filelist:
    ofile = os.path.basename(file)
    sys.stdout.write("downloading " + ofile + " ... ")
    sys.stdout.flush()
    infile = opener.open(file)
    outfile = open(f'{save_dir}/{ofile}', "wb")
    outfile.write(infile.read())
    outfile.close()
    sys.stdout.write("done\n")
