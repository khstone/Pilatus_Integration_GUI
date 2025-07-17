from PyQt5.QtCore import QThread, pyqtSignal  # may not need this
import numpy as np
from scipy.optimize import curve_fit
import scipy.interpolate as interpolate
import math
import os
import time  # used only for benchmarking purposes in how fast variance calculations are
from typing import Optional, Callable  # For type hints on progress_callback


class RunningStats:  # used to calculate the variance using Welford's algorithm
    def __init__(self, index):
        self.index = index
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squares of differences from the current mean

    def add(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def add_list(self, values):
        for x in values:
            self.add(x)

    def variance(self):
        if self.n < 2:
            return float('nan')  # Variance is undefined for n < 2
        return self.M2 / (self.n - 1)

    def population_variance(self):
        if self.n < 2:
            return float('nan')  # Variance is undefined for n < 2
        return self.M2 / self.n
    
def read_RAW(file, minx, maxx, mask = True):
    ##print "Reading RAW file here..."
    try:
        im = open(file, 'rb')
        arr = np.fromstring(im.read(), dtype='int32')
        im.close()
        arr.shape = (195, 487)
        #arr = np.fliplr(arr)               # for the way mounted at BL2-1
        if mask:
            for i in range(0, minx):
                arr[:,i] = -2.0
            for i in range(maxx, 487):
                arr[:,i] = -2.0
        return arr
    except:
        print("Error reading file: %s" % file)
        return None

def SPECread(filename, scan_number):
    #print "Reading SPEC file here..."
    tth = []
    i0 = []
    spec = open(filename)
    for line in spec:
        if "#O" in line and "tth" in line:  #find which line has the 2theta position
            temp = line.split()
            tth_line = temp[0][2]
            for i in range(0, len(temp)):
                if temp[i] == "tth":	#find where in that line the 2theta position is listed
                    tth_pos = i
                    break
            break
    for line in spec:
        if "#S" in line:
            temp = line.split()
            if int(temp[1]) == scan_number:
                break
    for line in spec:
        if "#P" + str(tth_line) in line:
            temp = line.split()
            tth_start = float(temp[tth_pos])
            break
    for line in spec:
        if "#L" in line:
            motors = line.split()[1:]
            if "tth" not in line:
                tth_motor_bool = False
                #print "2theta is not scanned..."
            else:
                tth_motor_bool = True
                tth_motor = motors.index("tth")
            i0_motor = motors.index("Monitor")
            break
    for line in spec:
        try:
            temp = line.split()
            if tth_motor_bool:
                tth = np.append(tth, float(temp[tth_motor]))
            else:
                tth = np.append(tth, tth_start)
            i0 = np.append(i0, float(temp[i0_motor]))
        except:
            break
    spec.close()
    return tth, i0

def Read_Cal(filename):
    cal = open(filename)
    line = cal.readline()
    db_x = int(line.split()[-1])
    line = cal.readline()
    db_y = int(line.split()[-1])
    line = cal.readline()
    det_R = float(line.split()[-1])
    cal.close()
    db_pixel = [db_x, db_y]
    #db_pixel = [487-db_x, db_y]  #This line is important for the way the detector is mounted at BL2-1
    return db_pixel, det_R

def make_map(db_pixel, det_R):
    # Map each pixel into cartesian coordinates (x,y,z) in number of pixels from sample for direct beam conditions (2-theta = 0)
    # We only need to do this once, so we can be inefficient about it
    #filename = image_path + user + spec_name + "_scan" + str(scan_number) + "_" + str(0).zfill(4) + ".raw"
    data = np.zeros((195, 487))
    
    tup = np.unravel_index(np.arange(len(data.flatten())), data.shape)
    xyz_map = np.vstack((tup[0], tup[1], det_R*np.ones(tup[0].shape)))
    xyz_map -= [[db_pixel[1]], 
                [db_pixel[0]], 
                [0]]
    #print np.shape(xyz_map)
    return xyz_map

def rotate_operation(map, tth):
    # Apply a rotation operator to map this into cartesian coordinates (x',y',z') when 2-theta != 0
    # We should be efficient about how this is implemented
    xyz_map_prime = np.empty_like(map)
    tth *= np.pi/180.0
    rot_op = np.array([[1.0, 0.0, 0.0], 
                       [0.0, np.cos(tth), np.sin(tth)], 
                       [0.0, -1.0*np.sin(tth), np.cos(tth)]])
    ##print(np.shape(data))
    xyz_map_prime = np.matmul(rot_op, map)
    return xyz_map_prime
        
def cart2sphere(map):
    # Convert the rotated cartesian coordinate map to spherical coordinates
    # This should also be efficiently implemented
    data = np.zeros((195, 487))
    tth_map = np.empty_like(data, dtype=float).flatten()
    _r = np.sqrt((map[:2,:]**2).sum(axis=0))
    #print _r.shape
    tth_map = np.arctan(_r/map[2, :])*180.0/np.pi
    tth_map = tth_map.reshape(data.shape)
    return tth_map%180.0

class IntegrationEngine:
    def __init__(self):
        self.progress_callback = None  # Callback for progress updates
    
    def set_progress_callback(self, callback):
        self.progress_callback = callback
        
    def integrate(self, specfile, scan_num, image_path, user, xyz_map, settings):
        start_time = time.time()
        stepsize = float(settings['stepsize'])
        lowclip = int(settings['img_clip_low'])
        highclip = int(settings['img_clip_high'])
        spec_path, spec_name = os.path.split(specfile)
        spec_path = spec_path + "/"
        image_path = image_path + "/"
        tth, i0 = SPECread(spec_path + spec_name, scan_num)
        mult = float(i0[0])   # multiplier to put everything back onto a rough scale of counts/pixel
        x = []
        y = []
        xmax_global = 0.0  #set this to some small number so that it gets reset with the first image
        xmin_global = 180.0
        bins = np.arange(0.0, 180.0, stepsize)  # create all of the bins from 0-180 with the specified stepsize
        digit_y = np.zeros_like(bins)   # this will hold the intensities for each bin
        digit_norm = np.zeros_like(bins)    # this will hold the normalization value (monitor counts) for each bin
        for k in range(0, len(tth)):        # loop through images at every 2-theta value
            x = []
            y = []
            filename = image_path + user + "_" + spec_name + "_scan" + str(scan_num) + "_" + str(k).zfill(4) + ".raw"
            data = read_RAW(filename, lowclip, highclip)
            xyz_map_prime = rotate_operation(xyz_map, tth[k])
            tth_map = cart2sphere(xyz_map_prime)
            x = tth_map.flatten()  # flatten into a list of all 2-theta values
            y = data.flatten()/i0[k]    # flatten into a list of all intensity values (normalized by I0)
            y_0 = np.where(y < 0, np.zeros_like(y), y)  # create a list of intensities where any negative numbers are set to 0, this is for every pixel in the current image
            y_1 = np.where(y < 0, np.zeros_like(y), np.ones_like(y))    # create a map of which intensities are to be used (0 if masked out, 1 if included), for every pixel in the current image
    
            digit_y += np.histogram(x + stepsize, weights=y_0, range=(0,180), bins=int(math.ceil(180.0/stepsize)))[0]
            digit_norm += np.histogram(x + stepsize, weights=y_1, range=(0,180), bins=int(math.ceil(180.0/stepsize)))[0]
            
            
            # Report progress if callback exists
            if self.progress_callback:
                self.progress_callback(k/len(tth))
            
        nonzeros = np.nonzero(digit_norm)
        interp = interpolate.InterpolatedUnivariateSpline(bins[nonzeros], digit_y[nonzeros]/digit_norm[nonzeros])
        
        interpbins = np.arange(min(bins[nonzeros]), max(bins[nonzeros]), stepsize)
        interpbins = np.around(interpbins, decimals=3)
        interpy = interp(interpbins)
        
        outname = spec_name + "_scan" + str(scan_num) + ".xye"
        
        good_data = np.where(np.logical_and(interpbins>=settings['min_tth'], interpbins<=settings['max_tth']))  # only take data above a certain 2-theta value
            
        end_time = time.time()  # Record the ending time
        elapsed_time = end_time - start_time
        
        
        print(f"Elapsed time Poisson: {elapsed_time:.4f} seconds")
        
        # Report progress if callback exists
        if self.progress_callback:
            self.progress_callback(step * 100 / total_steps)
        
        return outname, interpbins[good_data], mult * interpy[good_data], np.sqrt(np.abs(mult * interpy[good_data]))

    def integrate_var(self, specfile, scan_num, image_path, user, xyz_map, settings):  # Integrates data using variance for esd values
        start_time = time.time()
        stepsize = float(settings['stepsize'])
        lowclip = int(settings['img_clip_low'])
        highclip = int(settings['img_clip_high'])
        spec_path, spec_name = os.path.split(specfile)
        spec_path = spec_path + "/"
        image_path = image_path + "/"
        tth, i0 = SPECread(spec_path + spec_name, scan_num)
        mult = float(i0[0])   # multiplier to put everything back onto a rough scale of counts/pixel
        x = []
        y = []
        xmax_global = 0.0  #set this to some small number so that it gets reset with the first image
        xmin_global = 180.0
        bins = np.arange(0.0, 180.0, stepsize)  # create all of the bins from 0-180 with the specified stepsize
        y_list = [RunningStats(index=i) for i in range(0, len(bins))]  # create a list of RunningStats for every 2-theta bin
        for k in range(0, len(tth)):        # loop through images at every 2-theta value
            x = []
            y = []
            filename = image_path + user + "_" + spec_name + "_scan" + str(scan_num) + "_" + str(k).zfill(4) + ".raw"
            data = read_RAW(filename, lowclip, highclip)
            xyz_map_prime = rotate_operation(xyz_map, tth[k])
            tth_map = cart2sphere(xyz_map_prime)
            x = tth_map.flatten()  # flatten into a list of all 2-theta values
            y = data.flatten()/i0[k]    # flatten into a list of all intensity values (normalized by I0)
            bin_indices = np.digitize(x, bins)  # array of indices mapping x into the correct bins (index of bins for each x value)
            for i in range(0, len(x)):
                if y[i] >= 0:
                    y_list[bin_indices[i]].add(y[i])
            # Report progress if callback exists
            if self.progress_callback:
                self.progress_callback(k/len(tth))
                    
        digit_norm = [obj.mean for obj in y_list]
        variance = [obj.variance() for obj in y_list]
        y_array = np.array(digit_norm)
        var_array = np.array(variance)
        nonzeros = np.nonzero(digit_norm)
        interpbins = np.arange(min(bins[nonzeros]), max(bins[nonzeros]), stepsize)
        interpbins = np.around(interpbins, decimals=3)
        
        outname = spec_name + "_scan" + str(scan_num) + ".xye"
        
        good_data = np.where(np.logical_and(interpbins>=settings['min_tth'], interpbins<=settings['max_tth']))  # only take data above a certain 2-theta value
        
        end_time = time.time()  # Record the ending time
        elapsed_time = end_time - start_time
        
        print(f"Elapsed time variance: {elapsed_time:.4f} seconds")
        
        return outname, bins[good_data], mult * y_array[good_data], mult * var_array[good_data]

def write_data(output_path, filename, x, y, e):
    outname = output_path + filename
    outfile = open(outname, "w")
    for i in range(0, len(x)):
        outfile.write(f"{x[i]:{6}.{6}} {y[i]:{12}.{9}} {e[i]:{12}.{9}} \n")
    outfile.close()