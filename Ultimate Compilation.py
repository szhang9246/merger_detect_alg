import pandas as pd
import numpy as np
import scipy.ndimage
from scipy.interpolate import interp1d

'''
From the datafile entered, this function will group all the halos by the roots 
of their merger trees. It will then return a list of dataframes of the now 
organized halos.
'''

def create_halo_list(filename):
    data = pd.read_hdf(filename, 'table')
    
    halo_list = []
    for key, group in data.groupby('Tree_root_ID'):
        halo_list.append(group)
        
    return halo_list
    
'''
Using the halo_list created from 'create_halo_list', the halo number, and a 
list of properties, this function returns a 2D list (Scale factor evolution is 
always Row 1; other property evolutions are in remaining rows) and a list of 
scale factors where major mergers are located. These are based off the specific
halo provided by 'num'.
'''

def get_prop_evolution(halo_list, num, prop):
    scale = halo_list[num].scale.values #an array of scale factor
    prop_array = halo_list[num][prop].values #a 2D array - diff prop per col
    
    evolution_list = []
    warning_counter = 0
    for i in range(len(scale)):
        if scale[i] < 0.25: #allows one scale factor value below 0.25 
            warning_counter+=1
            if (warning_counter == 2):
                break
        arr = []
        arr.append(scale[i])
        for j in range(len(prop)):
            arr.append(prop_array[i][j])
        evolution_list.append(arr) #scale and corresponding prop value added
    evolution_list = np.vstack(evolution_list) 
        
    new_scale_list = np.linspace(0.24, 1, 77) #for even scaling
    
    interp_evolution = []
    for i in range(1, evolution_list.shape[1]): #interpolate points based off new scaling
        interp_evolution.append(np.interp(new_scale_list, evolution_list[:,0][::-1], evolution_list[:,i][::-1]))
    
    filtered_evolution = np.empty((1 + len(prop), len(new_scale_list)))
    filtered_evolution[0] = (new_scale_list[::-1]) 
    for i in range(len(interp_evolution)): #use Gaussian filter on evolution
        filtered_evolution[i+1] = (scipy.ndimage.filters.gaussian_filter1d(interp_evolution[i][::-1], 1))

    mm = halo_list[num].scale_of_last_MM.values    
    mm_list = []
    change = -1
    for val in mm: #gets scale factor of major merger events
        if (not val == change) and (val >= 0.25) and (val <= 1): 
            mm_list.append(val)
        change = val
        
    return filtered_evolution, mm_list
        
'''
Given the evolution of scale factor/properties and its index, this function 
calculates the slope, or percentage difference. It returns a list of slopes
respective to the properties looked at.
'''

def get_prop_slope(evolution, num):
    prop_slope = []
    for i in range(1, evolution.shape[0]): #finds percentage difference
        prop_slope.append((evolution[i][num] - evolution[i][num+1])/evolution[i][num+1])
    return prop_slope

'''
Given the halo_list and list of properties, this function returns a list of 
the average of slope (percentage difference) for each property. 
'''

def get_MM_prop_average(halo_list, prop):
    slope_list = []
    
    for num in range(len(halo_list)):
        evolution, mm_list = get_prop_evolution(halo_list, num, prop)
        for x in range(len(mm_list)):
            for i in range(len(evolution[0])-1):
                if (evolution[0][i] >= mm_list[x] and evolution[0][i+1] <= mm_list[x]):
                    prop_slope = get_prop_slope(evolution, i)
                    slope_list.append(prop_slope)
                    break
                
    slope_list = np.vstack(slope_list)
            
    mm_averages = []
    for i in range(len(prop)):
        mm_averages.append(np.mean(slope_list[:,i]))

    return mm_averages
    
'''
This function will pick a random halo at a random scale factor 'num_rand' 
number of times. At that scale factor, it calculates the slope for all the 
properties. In the end, it will return an array of the averages and the 
standard deviations for each property. 
'''

def get_rand_prop_average_std(halo_list, prop, num_rand):
    from random import randint
    
    slope_list = []
    
    for iter in range(num_rand):
        num = randint(0, len(halo_list)-1)       
        evolution, _ = get_prop_evolution(halo_list, num, prop)
        
        rand = randint(0, len(evolution[0])-2)
        slope_list.append(get_prop_slope(evolution, rand))
        
    slope_list = np.vstack(slope_list)
        
    rand_avgstd = np.empty((len(prop), 2))

    for i in range(len(prop)):
        rand_avgstd[i, 0] = (np.mean(slope_list[:, i]))
        rand_avgstd[i, 1] = (np.std(slope_list[:, i]))
        
    return rand_avgstd

'''
This function combines 'get_MM_prop_average' and 'get_rand_prop_average_std'
together to calculate the ratio (emphasis) for every property. The ratio is 
calculated from (MM_Average - Rand_Average)/Rand_STD. The function returns a 
list of these ratios and also returns the results from 
'get_rand_prop_average_std' for later use.
'''

def get_prop_ratios(halo_list, prop, num_rand):
    mm_averages = get_MM_prop_average(halo_list, prop)
    rand_avgstd = get_rand_prop_average_std(halo_list, prop, num_rand)
    
    std_ratios = []
    for i in range(len(prop)):
        std_ratios.append((mm_averages[i] - rand_avgstd[i][0])/rand_avgstd[i][1])
        
    return std_ratios, rand_avgstd

'''
The goal of this function is to form a band where the algorithm predicts to 
locate a major merger. By taking the scale factors and establishing a threshold
value to find all the slopes that are over that certain value. Additionally, it 
interpolates between the data to find the exact location where the slopes are 
indicators of a MM. The function returns a list that has starting and ending 
places where the band forms.
'''

def get_bands(scale_list, slope_list, threshold):
    starting = []
    ending = []
    
    start = True  
    for i in range(len(scale_list)-2):
        if slope_list[i] >= threshold:
            if (i == 0) and start:
                starting.append(scale_list[i])
                
            elif start:
                change_x = ((threshold - slope_list[i-1])/(slope_list[i] - slope_list[i-1])) * (scale_list[i] - scale_list[i-1])
                starting.append((scale_list[i-1] + change_x))

            if slope_list[i+1] < threshold:
                change_x = ((threshold - slope_list[i+1])/(slope_list[i] - slope_list[i+1])) * (scale_list[i+1] - scale_list[i])
                ending.append((scale_list[i+1] - change_x))

            start = False
            
        else:
            start = True
            
    if (not len(starting) == len(ending)):
        ending.append(scale_list[-2])
                
    for x in range(len(starting)-1, -1, -1):
        if (starting[x] - ending[x]) < 0.02:
            starting.pop(x)
            ending.pop(x)
            
    return list(zip(starting, ending))

'''
In this function, it takes in the halo_list, the property ratios, the random
mean/std, and properties and tries to find the optimal threshold where the 
difference between Agreement and Disagreement is largest. When it goes through
all the halos, it combines the slopes of all the properties through this 
formula: (prop_slope) 
'''

def get_optimal_threshold(halo_list, std_ratios, rand_avgstd, prop):
    thresh_list = np.linspace(0.5, 1.5, 1000)
    
    agree_bin = [0] * len(thresh_list)
    total_mergers_bin = [0] * len(thresh_list)
    disagree_bin = [0] * len(thresh_list)
    total_bands_bin = [0] * len(thresh_list)

    for num in range(len(halo_list)):
        evolution, mm_list = get_prop_evolution(halo_list, num, prop)
      
        slope_list = []
        for i in range(len(evolution[0])-1):
            prop_slope = get_prop_slope(evolution, i)
            
            overall_slope = 0
            for j in range(len(prop)):
                overall_slope += ((prop_slope[j] - rand_avgstd[j][0])/rand_avgstd[j][1]) * (std_ratios[j]/sum(std_ratios))
            
            slope_list.append(overall_slope)
            
        for t in range(len(thresh_list)):
            threshold = thresh_list[t]
            
            bands = get_bands(evolution[0], slope_list, threshold)
            
            #for agreements
            for i in range(len(mm_list)):
                for j in range(len(bands)):
                    if (mm_list[i] <= bands[j][0] and mm_list[i] >= bands[j][1]):
                        agree_bin[t] += 1
                        break
                total_mergers_bin[t] += 1
                
            #for disagreements
            has_match = False
            
            for i in range(len(bands)):
                if (len(mm_list) != 0):
                    for j in range(len(mm_list)):
                        if (mm_list[j] <= bands[i][0] and mm_list[j] >= bands[i][1]):
                            has_match = True
                            
                        if (j == (len(mm_list)-1)):
                            if (not has_match):
                                disagree_bin[t] += 1
                            has_match = False
                else:
                    disagree_bin[t] += 1
                total_bands_bin[t] += 1
                
    averages = np.empty((len(thresh_list), 2))
    for i in range(len(thresh_list)):
        averages[i][0] = agree_bin[i]/total_mergers_bin[i]
        averages[i][1] = disagree_bin[i]/total_bands_bin[i]
        
    agree = scipy.ndimage.filters.gaussian_filter1d(averages[:, 0], 1)
    disagree = scipy.ndimage.filters.gaussian_filter1d(averages[:, 1], 1)
    
    diff = agree - disagree
    f = interp1d(diff, thresh_list)
    f_a = interp1d(thresh_list, agree)
    f_d = interp1d(thresh_list, disagree)
    
    print ("Agreement%:", f_a(f(diff.max())))
    print ("Disagreement%:", f_d(f(diff.max())))
    print ("Threshold:", f(diff.max()))
    
    return f(diff.max())      
    
'''
This functions takes the integrals between the bands and then categorizes them
into bins to construct a histogram. It then compares the derived integrals from
the Agreement and Disagreement bands and finds the point of intersection. That
integral then becomes a indicator where the algorithm will classify a band
as a minor merger or a major merger. The function takes in parameters: the 
optimal threshold, the halo_list, the property ratios, and the random
statistics. It returns the integral value for MM classification.
'''

def get_MM_classifier(threshold, halo_list, std_ratios, rand_avgstd):
    import scipy.stats as stats
    from scipy.optimize import fsolve
    
    agree_bin = []
    disagree_bin = []
    
    for num in range(len(halo_list)):
        evolution, mm_list = get_prop_evolution(halo_list, num, prop)

        slope_list = []
        
        for i in range(len(evolution[0])-1):
            prop_slope = get_prop_slope(evolution, i)
            
            overall_slope = 0
            for j in range(len(prop)):
                overall_slope += ((prop_slope[j] - rand_avgstd[j][0])/rand_avgstd[j][1]) * (std_ratios[j]/sum(std_ratios))
            
            slope_list.append(overall_slope)
            
        bands = get_bands(evolution[0], slope_list, threshold)
        
        fstd = interp1d(evolution[0][:-1], slope_list)
        
        has_agreed = False
        for j in range(len(bands)):
            std_y = []
            for x in np.arange(bands[j][1], bands[j][0], 0.01):
                std_y.append(fstd(x))
            area = np.trapz(std_y, dx = 0.01)
            for k in range(len(mm_list)):
                if (mm_list[k] <= bands[j][0] and mm_list[k] >= bands[j][1]):
                    has_agreed = True
                    break
            if (has_agreed):
                agree_bin.append(area)
            else:
                disagree_bin.append(area)
    
    agree_line = stats.gaussian_kde(agree_bin)
    disagree_line = stats.gaussian_kde(disagree_bin)
    
    n, x = np.histogram(disagree_bin, bins=100, normed = False)
    fds = interp1d(x, disagree_line(x))
    n, x = np.histogram(agree_bin, bins=100, normed = False)
    fas = interp1d(x, agree_line(x))

    mm_classifier = (fsolve(lambda x : fas(x) - fds(x), 0.05))

    return mm_classifier
    
'''
This is the final step of the algorithm. This function will add columns to the 
original datafile of the algorithm's detected mM an MM. Additionally, it has 
the feature to decrease the optimal threshold by 'thresh_decrese'%. Note, 
'thresh_decrease' must be a float between 0 and 1. As parameters, the function 
requires the filename, halo_list, optimal threshold, mm_classifier value, 
property list, property ratios, random statistics, and the threshold percentage
decrease. 
'''

def add_file(filename, halo_list, threshold, mm_classifier, prop, std_ratios, rand_avgstd, thresh_decrease, filename_export):
    import math    
    
    data = pd.read_hdf(filename, 'table')
    new_cols = []
    for id, halo in data.groupby('Tree_root_ID'):
        new_cols.append(np.empty((len(halo.scale), (2 * math.ceil(1.0/thresh_decrease)))))
        
    header_list = []
            
    multiplier = 0.0
    while (1.0 - (multiplier * thresh_decrease) > 0.0):
        for num in range(len(halo_list)):
            last_MM = []
            last_mM = []
            
            scale = halo_list[num].scale.values
            evolution, mm_list = get_prop_evolution(halo_list, num, prop)
            
            slope_list = []
            
            for i in range(len(evolution[0])-1):
                prop_slope = get_prop_slope(evolution, i)
                
                overall_slope = 0
                for j in range(len(prop)):
                    overall_slope += ((prop_slope[j] - rand_avgstd[j][0])/rand_avgstd[j][1]) * (std_ratios[j]/sum(std_ratios))
            
                slope_list.append(overall_slope)
                
            bands = get_bands(evolution[0], slope_list, threshold)
            
            minorM_list_starting = []
            minorM_list_ending = []
            majorM_list_starting = []
            majorM_list_ending = []
    
            fstd = interp1d(evolution[0][:-1], slope_list)
            
            for i in range(len(bands)):
                std_y = []
                for x in np.arange(bands[i][1], bands[i][0], 0.01):
                    std_y.append(fstd(x))
                area = np.trapz(std_y, dx = 0.01)
                
                if (area < mm_classifier):
                    minorM_list_starting.append(bands[i][0])
                    minorM_list_ending.append(bands[i][1])
                else:
                    majorM_list_starting.append(bands[i][0])
                    majorM_list_ending.append(bands[i][1])
                    
                
            major_scale = [(a + b)/2 for a, b in zip(majorM_list_starting, majorM_list_ending)]
            minor_scale = [(a + b)/2 for a, b in zip(minorM_list_starting, minorM_list_ending)]
            
            for i in range(len(scale)):
                if (len(minor_scale) > 0):
                    for x in minor_scale:
                        if (scale[i] < min(minor_scale)):
                            last_mM.append(scale[-1])
                            break
                        elif (scale[i] >= x):
                            last_mM.append(x)
                            break
                else:
                   last_mM.append(scale[-1]) 
                   
                if (len(major_scale) > 0):
                    for x in major_scale:
                        if (scale[i] < min(major_scale)):
                            last_MM.append(scale[-1])
                            break
                        elif (scale[i] >= x):
                            last_MM.append(x)
                            break
                else:
                    last_MM.append(scale[-1])          
            
            new_cols[num][:, multiplier*2] = np.asarray(last_mM)
            new_cols[num][:, (multiplier*2)+1] = np.asarray(last_MM)
            
        header_list.append('my_last_mM_' + str(round(1 - (thresh_decrease * multiplier), 2 )))
        header_list.append('my_last_MM_' + str(round(1 - (thresh_decrease * multiplier), 2 )))
            
        multiplier += 1.0
        threshold = threshold * (thresh_decrease * multiplier)
                
        if (thresh_decrease == 0.0):
            break
                
        
    df = pd.concat(halo_list)
    df2 = pd.DataFrame(np.concatenate(new_cols),index=df.index, columns = header_list) # etc, for how many new columns you have
    df = df.join(df2)
        
    df.to_hdf(filename_export, 'table')

'''
A compilation of all the steps in one. This is the algorithm in its entirety.
'''

def merger_detect_alg(filename, prop, num_rand, thresh_decrease, filename_export):
    halo_list = create_halo_list(filename)
    std_ratios, rand_avgstd = get_prop_ratios(halo_list, prop, num_rand)    
    threshold = get_optimal_threshold(halo_list, std_ratios, rand_avgstd, prop)
    mm_classifier = get_MM_classifier(threshold, halo_list, std_ratios, rand_avgstd)
    add_file(filename, halo_list, threshold, mm_classifier, prop, std_ratios, rand_avgstd, thresh_decrease, filename_export)

###############################################################################

prop = ['mvir', 'vmax', 'Xoff', 'Spin_Bullock', 'virial_ratio']
merger_detect_alg('/Users/shawn/Dev/Shawn.git/SIP2017/halo_tracks_mass_11.95_more.h5', 
                  prop, 200000, 0.25, '/Users/shawn/Dev/Shawn.git/SIP2017/modified_halo_tracks_mass_11.95.h5')

    



