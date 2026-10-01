import numpy as num
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import argparse

from Pilatus_Calibration_setup import *

def read_TIFF(file):
    print("Reading TIFF file here...")
    try:
        im = open(file, 'rb')
        im.seek(4096)   # skip the first 4096 bytes of header info for TIFF images
        arr = num.frombuffer(im.read(), dtype='int32')
        im.close()
        arr.shape = (195, 487)
        #arr = num.fliplr(arr)  #for the way mounted at BL2-1
        print(num.shape(arr))
        print(len(arr))
        return arr
    except:
        print("Error reading file: %s" % file)
        return None

def read_RAW(file):
    print("Reading RAW file here...")
    try:
        im = open(file, 'rb')
        arr = num.frombuffer(im.read(), dtype='int32')
        im.close()
        arr.shape = (195, 487)
        #arr = num.fliplr(arr)  #for the way mounted at BL2-1
        return arr
    except:
        print("Error reading file: %s" % file)
        return None

def csvread(filename):  
    print("Reading CSV file here..."))
    csv = open(filename)
    line = csv.readline()
    temp = line.split(',')
    xi = 1
    yi = 4
    i0i = 3
    x = []
    y = []
    i0 = []
    line = csv.readline()
    while line:
        temp = line.split(",")
        x = num.append(x, float(temp[xi]))
        y = num.append(y, float(temp[yi]))       
        i0 = num.append(i0, float(temp[i0i]))
        line = csv.readline()
    csv.close()
    return x, y, i0

def gauss_linbkg(x, m, b, x0, intint, fwhm):
    return m*x + b + intint*(2./fwhm)*num.sqrt(num.log(2.)/num.pi)*num.exp(-4.*num.log(2.)*((x-x0)/fwhm)**2)
    
def Gauss_fit(x, y):
    pguess = [0, 0, num.argmax(y), num.max(y), 5.0]  # linear background (2), pos, intensity, fwhm
    try:
        popt, pcov = curve_fit(gauss_linbkg, x, y, p0=pguess)
        return popt
    except:
        return pguess
    
def simple_line(x, m, b):
    return m*x + b

# Read CSV file, get step size and number of points in scan
x, y, i0 = csvread(csv_path + csv_name)
num_points = len(x)
calib_tth_steps = abs(x[1] - x[0])
x = []
y = []
i0 = []

# Read images, take line cut, fit peak for Al2O3 calibration scan
pks = []
plt.figure()
for i in range(0, num_points):
    print(i)
    filename = data_path + calib_name + str(i).zfill(4) + ".raw"
    data = read_RAW(filename)
    x = num.arange(0, num.shape(data)[1])
    y = data[db_pixel[1], :]
    y += data[db_pixel[1] + 1, :]
    y += data[db_pixel[1] - 1, :]
    y += data[db_pixel[1] + 2, :]
    y += data[db_pixel[1] - 2, :]
    popt = Gauss_fit(x, y)
    plt.cla()
    plt.plot(x,y, 'b.')
    plt.plot(x, gauss_linbkg(x, *popt), 'r-')
    pks = num.append(pks, popt[2])

# Fit line to the extracted peak positions and determine the sample to detector distance
x = num.arange(num_points)
lin_fit, pcov = curve_fit(simple_line, pks, x*calib_tth_steps + 0.00)
det_R = 1.0/num.tan(abs(lin_fit[0])*num.pi/180.0)     # sample to detector distance in pixels
print("Sample to detector distance in pixels = " + str(det_R))
plt.figure()
plt.plot(pks, x*calib_tth_steps, 'b.')
plt.plot(pks, lin_fit[0]*pks + lin_fit[1], 'r-')

outname = csv_path + csv_name[:-4] + "_calib.cal"
outfile = open(outname, "w")
outfile.write("direct_beam_x \t %i\n"  % db_pixel[0])
outfile.write("direct_beam_y \t %i\n" % db_pixel[1])
outfile.write("Sample_Detector_distance_pixels \t %15.6G\n" % det_R)
outfile.write("Sample_Detector_distance_mm \t %15.6G" % (det_R * pix_size / 1000.0))
outfile.close()

plt.show()
