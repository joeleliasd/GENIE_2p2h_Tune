import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import pandas as pd
import csv
from scipy.optimize import minimize,least_squares
import re

def flatten_jagged(jagged_array):
    # Flatten the jagged array and return as a NumPy array
    return np.concatenate(jagged_array)

numberOfDivisions=9
columnsOriginal=3
rowsOriginal=3
histX=11

# Define the structure of the jagged array
binStructuredpt = [13, 12, 13, 13, 13, 13, 12, 11, 11, 8, 5]

totalBins=124

#function reads in data file, isolates non zero entries and prints them to an array. 
def import_data(filename):
    file =np.empty(len(np.array(pd.read_csv(filename, header=None, sep='\t'))))
    for i in range(len(file)):
        file[i]=np.array(pd.read_csv(filename, header=None, sep='\t'))[i][1]
    # this data has 273 bins, 205 of which are nonzero, we must delete all zero bins.
    nonzero=[]
    for i in range(len(file)):
        if file[i]>0:
            nonzero.append(file[i])
    return file,nonzero

def read_to_jagged_array(filename):
    # Initialize an empty dictionary to hold lists with their row numbers
    data = {}

    # Compile a regular expression pattern for extracting numbers
    pattern = re.compile(r"\((\d+),(\d+)\) ([\de\.-]+)")

    with open(filename, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                # Extract row, column, and value from the line
                row, col, value = match.groups()
                row, col = int(row), int(col)  # Convert row and col to integers
                value = float(value)  # Convert the value to a float
                
                # Add the value to the corresponding row in our data dictionary
                if row not in data:
                    data[row] = []
                data[row].append(value)

    # Convert the dictionary to a jagged 2D list (array)
    jagged_array = [data[key] for key in sorted(data)]
    
    return jagged_array

def scale(into):
    # Apply scale factors
    temp=into
    for i in range(histX):
        for f in range(binStructuredpt[i]):
            temp[i][f]=temp[i][f]*1000000000000000000000000000000000000000000
    return temp


def read_Data_File(file_path, structure):
    # Initialize the jagged array based on the given structure
    jagged_Array = [[] for _ in structure]

    # Keep track of the current index for the first dimension
    current_Index = 0

    # Open the file and read line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Extract the value after the parenthesis
            value = float(line.split(')')[1].strip())
            
            # Append the value to the current list in the jagged array
            jagged_Array[current_Index].append(value)
            
            # Check if the current list is filled; if so, move to the next list
            if len(jagged_Array[current_Index]) == structure[current_Index]:
                current_Index += 1
            
            # Stop if we have filled all the lists
            if current_Index >= len(structure):
                break

    return jagged_Array

# takes output of NUISANCE_flat() and converts list into matrix for plotting.
# this function imports the covariance matrix and returns the error bars and reshaped covariance
def cov_prune(filename):
    covfile=np.array(pd.read_csv(filename, header=None, sep=','))[:,2]
    mtx=np.split(covfile,np.sqrt(len(covfile)))
    diag_cov=np.diag(mtx)
    indexlist_cov=[]
    j=0
    for i in range(len(diag_cov)):
        if diag_cov[i]>10**-30:
            indexlist_cov.append(i)
            j+=1
        #print(i,j)
        else:
            indexlist_cov.append(0)
    nonzerocov=np.empty(len(indexlist_cov))
    #delete zeros
    covmtx=[]

    for pos in range(len(indexlist_cov)):
        for pos2 in range(len(indexlist_cov)):
        #print(pos,pos2)
            if indexlist_cov[pos]>0 and indexlist_cov[pos2]>0:
                covmtx.append(mtx[pos][pos2])
    covariance=np.split(np.array(covmtx),np.sqrt(len(covmtx)))
    errors =np.sqrt(np.diag(covariance))
    return errors,covariance

# this function 'partitions' the data into the binning scheme
dpt_width=5
dpt_width=.05
BinNorm=[]
def BinInScheme_Norm(lst,binlst,boundaries,binsize):
    for i in range(len(boundaries)):
        Array=np.empty(len(boundaries[i]))
        for j in range(0,len(boundaries[i])-1):
            Array[j]=(boundaries[i][j+1]-boundaries[i][j])/binsize
        BinNorm.append(Array[0:-1])
    offset=0
    TH1DBinned=[]
    for i in range(len(binlst)):
        #print(i)
        h=[]
        for j in range(int(binlst[i])):
        #print(nonzero[j+offset],j+offset)
            h.append(lst[j+offset])
        TH1DBinned.append(h)
        offset+=int(binlst[i])
    return TH1DBinned

def xbin_maker(nBins,boundaries_file):
    boundaries=eval(pd.read_csv(boundaries_file, header=None, sep='\t')[0][0])
    bin_center=[]
    xerror=[]
    for i in range(11):
        x=[]
        xerr=[]
        for j in range(int(nBins[i])):
            #print(j)
            if j>len(boundaries[i]):
                continue;
            x.append((boundaries[i][j+1]-boundaries[i][j])/2+boundaries[i][j])
            xerr.append((boundaries[i][j+1]-boundaries[i][j])/2)
        bin_center.append(x)
        xerror.append(xerr)
        
    return boundaries,bin_center,xerror

dpt_nBins=np.array(eval(pd.read_csv("h_dpt_ptmu_binning.tsv", header=None, sep='\t')[0][0]))
xbins_dpt_boundaries,xbins_dpt,xbins_dpt_xerr=xbin_maker(dpt_nBins,"h_dpt_ptmu_boundaries.tsv")

dpt_ptmu_all,dpt_ptmu=import_data("h_dpt_ptmu_data_result.tsv")
MnvData=BinInScheme_Norm(dpt_ptmu,dpt_nBins,xbins_dpt_boundaries,5)
MnvMC=scale(read_to_jagged_array('ptmu_dpt_MC.tsv'))#mc of all diff xs

del dpt_ptmu[-1]

print('Dpt length: ')
print(len(dpt_ptmu))
print(dpt_nBins)
print()

print('MnvMC:', MnvMC)
print()


# =============================================================================
# #  Print out the size and contents of MnvMC
# print("MnvMC (Size: {} rows):".format(len(MnvMC)))
# for index, row in enumerate(MnvMC):
#     print("Row {}: Size = {}, Contents: {}".format(index + 1, len(row), row))
#  
# #  Print an empty line for better readability
# print("\n")
#  
#Print out the size and contents of MnvData
print("MnvData (Size: {} rows):".format(len(MnvData)))
for index, row in enumerate(MnvData):
    print("Row {}: Size = {}, Contents: {}".format(index + 1, len(row), row))



from scipy.linalg import pinvh
error,covmtx=cov_prune('dpt_ptmu_cov.tsv')
del covmtx[-1] #overflow bin
error=error[0:-1]
for i in range(len(covmtx)):
    covmtx[i]=covmtx[i][0:-1]
covmax=np.amax(covmtx)
invcovmtx=pinvh(covmtx)
invcovmax=np.amax(invcovmtx)

fig = plt.figure(figsize=(12,12),dpi=80)
ax = plt.axes()

# plotting

ax.imshow(invcovmtx,origin='lower',cmap='coolwarm')
#ax.set_xlim3d(0,0.4
#x.set_xlabel('∆pt', fontsize = 10)
#x.set_ylabel('∆pt', fontsize = 10)
ax.set_title('∆dpt:∆pt Covariance')
ax.set_aspect('equal')
plt.pcolormesh(invcovmtx, cmap='coolwarm')
plt.colorbar(location='bottom')
plt.show()

invcovmax


    

division_rows=rowsOriginal
division_columns=columnsOriginal

div1 = scale(read_to_jagged_array('h_ptmu_dpt_2p2h_Div1MQ_UR.tsv'))
div2 = scale(read_to_jagged_array('h_ptmu_dpt_2p2h_Div2MQ_UR.tsv'))
div3 = scale(read_to_jagged_array('h_ptmu_dpt_2p2h_Div3MQ_UR.tsv'))
div4 = scale(read_to_jagged_array('h_ptmu_dpt_2p2h_Div4MQ_UR.tsv'))
div5 = scale(read_to_jagged_array('h_ptmu_dpt_2p2h_Div5MQ_UR.tsv'))
div6 = scale(read_to_jagged_array('h_ptmu_dpt_2p2h_Div6MQ_UR.tsv'))
div7 = scale(read_to_jagged_array('h_ptmu_dpt_2p2h_Div7MQ_UR.tsv'))
div8 = scale(read_to_jagged_array('h_ptmu_dpt_2p2h_Div8MQ_UR.tsv'))
div9 = scale(read_to_jagged_array('h_ptmu_dpt_2p2h_Div9MQ_UR.tsv'))

#print(division_data.values())

combinedArrayb = np.array([div1, div2, div3, div4, div5, div6, div7, div8, div9])

min_valb,max_valb=np.amin(combinedArrayb),np.amax(combinedArrayb)

# Get the data in a list
#division_values = list(div1, div2, div3, div4, div5, div6, div7, div8, div9)
#
# Create the subplots
#fig, axsb = plt.subplots(rowsOriginal, columnsOriginal, figsize=(18, 9))
#
# 
# # Iterate over the list of divisions
# index = 0
# for i in range(rows):
#     for j in range(cols):
#         # Plot the data of the current division
#         print()
#         axsb[i, j].imshow(division_values[index], origin='lower', cmap="coolwarm", vmin=min_valb, vmax=max_valb)
#         axsb[i, j].set_title(f'Axis [{i}, {j}]')
#         
#         # Move to the next division
#         index += 1
# 
# for ax in axsb.flat:
#     ax.set(xlabel='∆pt', ylabel='∆pt')
# 
# # Hide x labels and tick labels for top plots and y ticks for right plots.
# for ax in axsb.flat:
#     ax.label_outer()
#     
# # This next part makes only the color bar for the division plots from a plot of combinedArray which is not printed
# plt.figure(figsize=(18,1.2))
# img = plt.imshow(np.split(np.ndarray.flatten_jagged(combinedArrayb),36),cmap='coolwarm')
# plt.gca().set_visible(False)
# cax = plt.axes([0.1, 0.2, 0.8, 0.6])
# plt.colorbar(orientation="horizontal", cax=cax)
# plt.savefig("colorbar.pdf")
# 
# Allplot = sum(division_data.values())
# 
# fig = plt.figure(figsize=(10, 10), dpi=80)
#  
# # syntax for 3-D projection
# ax = plt.axes()
#  
# # plotting
# 
# ax.imshow(Allplot,origin='lower',cmap='coolwarm')
# #ax.set_xlim3d(0,0.4
# ax.set_xlabel('∆pt', fontsize = 10)
# ax.set_ylabel('∆pt', fontsize = 10)
# ax.set_title('All double diff xs')
# plt.pcolormesh(Allplot, cmap='coolwarm')
# plt.colorbar(location='bottom')
# plt.show()

Mnvpion=scale(read_to_jagged_array('ptmu_dpt_mc_pion.tsv'))#pion component of mc
Mnvqe=scale(read_to_jagged_array('ptmu_dpt_mc_qe.tsv'))#qe component of mc
Mnv2p2h=scale(read_to_jagged_array('ptmu_dpt_mc_2p2h.tsv')) #2p2h component of mc
Mnv2p2h_2=Mnv2p2h
# Loop through each entry in the first dimension
for i in range(len(binStructuredpt)):
    # Loop through each entry in the second dimension
    for j in range(binStructuredpt[i]):
        if MnvMC[i][j]==0:
            Mnv2p2h[i][j] = 0
        elif (Mnv2p2h[i][j] / MnvMC[i][j]) < .2:
            Mnv2p2h[i][j] = 0

print('\nMnv2p2h', Mnv2p2h,'\n')

# Initialize the result array as a jagged array
Mnvdatano2p2h = [ [0]*length for length in binStructuredpt]

# Loop through each entry in the first dimension
for i in range(len(binStructuredpt)):
    # Loop through each entry in the second dimension
    for j in range(binStructuredpt[i]):
        # Directly subtract and assign the result
        Mnvdatano2p2h[i][j] = MnvData[i][j] - Mnv2p2h[i][j]

Mnvmcno2p2h=[ [0]*length for length in binStructuredpt]

# Loop through each entry in the first dimension
for i in range(len(binStructuredpt)):
    # Loop through each entry in the second dimension
    for j in range(binStructuredpt[i]):
        # Directly subtract and assign the result
        Mnvmcno2p2h[i][j] = MnvMC[i][j] - Mnv2p2h[i][j]
        
Mnv2p2hdata=[ [0]*length for length in binStructuredpt]

# Loop through each entry in the first dimension
for i in range(len(binStructuredpt)):
    # Loop through each entry in the second dimension
    for j in range(binStructuredpt[i]):
        # Directly subtract and assign the result
        Mnv2p2hdata[i][j] = MnvData[i][j] - Mnvmcno2p2h[i][j]
        if Mnv2p2hdata[i][j] < 0:
            Mnv2p2hdata[i][j]=0
        if (Mnv2p2hdata[i][j] / MnvData[i][j]) < .2:
            Mnv2p2hdata[i][j] = 0

NUISANCE2p2hb=[ [0]*length for length in binStructuredpt]

for i in range(numberOfDivisions):
    NUISANCE2p2hb = (div1 + div2 + div3 + div4 + div5 + div6 + div7 + div8 + div9)



#again these are defined so that we have the same rande across the plots
mcdatamerged=np.array([Mnv2p2hdata,Mnv2p2h,NUISANCE2p2hb])
min_val,max_val=np.amin(mcdatamerged),np.amax(mcdatamerged)

# fig, axs = plt.subplots(3, 1,figsize=(18, 27))
# axs[ 0].imshow(Mnv2p2h,origin='lower', cmap="coolwarm", vmin=min_val, vmax=max_val)
# axs[ 0].set_title('Mnv2p2hMC')
# axs[ 1].imshow(Mnv2p2hdata,origin='lower', cmap="coolwarm", vmin=min_val, vmax=max_val)
# axs[ 1].set_title('Mnv2p2hDATA')
# axs[ 2].imshow(NUISANCE2p2hb,origin='lower', cmap="coolwarm", vmin=min_val, vmax=max_val)
# axs[ 2].set_title('NUISANCE2p2h')
# 
# for ax in axs.flat:
#     ax.set(xlabel='∆pt', ylabel='∆pt')
# 
# # Hide x labels and tick labels for top plots and y ticks for right plots.
# for ax in axs.flat:
#     ax.label_outer()
#        
# plt.figure(figsize=(18,1.2))
# img = plt.imshow(np.split(np.ndarray.flatten(mcdatamerged),36),cmap='coolwarm')
# plt.gca().set_visible(False)
# cax = plt.axes([0.1, 0.2, 0.8, 0.6])
# plt.colorbar(orientation="horizontal", cax=cax)
# plt.savefig("colorbar.pdf")

MnvDataflat = flatten_jagged(MnvData)
Mnv2p2hflat = flatten_jagged(MnvData)

# # Tikhonov Regularization:

def create_matrix(rows, columns):
    # Initialize result matrix
    result_matrix = [[0 for _ in range(columns)] for _ in range(rows)]
    
    short_changes = (division_columns - 1) * division_rows
    long_changes = (division_rows - 1) * division_columns
    
    counter1 = 0
    short_per = (division_columns - 1)
    
    # First loop
    for i in range(short_changes):
        if i + counter1 < columns:
            result_matrix[i][i + counter1] = 1
        if short_per != 0:
            if (i+1) % short_per == 0:
                counter1 += 1
        if short_per == 0:
            counter1 += 1
    
    # Second loop
    counter1 = 0  # Reset counter1
    for i in range(short_changes):
        if i + counter1 + 1 < columns:
            result_matrix[i][i + counter1 + 1] = -1
        if short_per != 0:
            if (i+1) % short_per == 0:
                counter1 += 1            
        if short_per == 0:
            counter1 += 1
            
    # Third loop
    counter1 = 0 # Reset counter1
    for i in range(long_changes):
        if i < columns:
            result_matrix[i + short_changes][i] = 1
    
    # Fourth loop
    counter1 = 0 # Reset counter1
    for i in range(long_changes):
        if i + 1 < columns:
            result_matrix[i + short_changes][i + division_columns] = -1

     # Flatten the matrix to a 1D list of lists
    flat_matrix = [row for row in result_matrix]

    return flat_matrix

# Define the dimensions of the matrix
number_of_rows = ((division_columns - 1) * division_rows) + (division_columns * (division_rows - 1))
number_of_columns = (division_rows * division_columns)

# Create the matrix
regmtxb = create_matrix(number_of_rows, number_of_columns)

# # Chi^2 Fit:
print('Mnvdatano2p2h',Mnvdatano2p2h)
print()
Mnv2p2hdataflat=np.array(flatten_jagged(Mnv2p2hdata))
Mnvdatano2p2hflat=np.array(flatten_jagged(Mnvdatano2p2h))
MnvDataflat=np.array(flatten_jagged(MnvData))

div1flat = flatten_jagged(div1)
div2flat = flatten_jagged(div2)
div3flat = flatten_jagged(div3)
div4flat = flatten_jagged(div4)
div5flat = flatten_jagged(div5)
div6flat = flatten_jagged(div6)
div7flat = flatten_jagged(div7)
div8flat = flatten_jagged(div8)
div9flat = flatten_jagged(div9)

merged2p2hlistb = np.transpose([div1flat, div2flat, div3flat, div4flat, div5flat, div6flat, div7flat, div8flat, div9flat])

print('Merged List: ')
print(merged2p2hlistb)
print()

sigsquared=np.diagonal(covmtx)



#Chi2 Fit

#Two ways of going around negative bin weights:

def Square_fit_pramsb(vals):
    n = len(vals)
    mtx = np.zeros((n, n))
    np.fill_diagonal(mtx, vals)
    vec = np.array(vals)
    return mtx @ vec

def exp_fit_pramsb(vals):
    return np.exp(vals)

#Chi_squared as a function of the weights
def Chi_squaredb(lst):
    chi2b=((Mnv2p2hdataflat-(merged2p2hlistb@lst))@invcovmtx@(Mnv2p2hdataflat-(merged2p2hlistb@lst)))
    return chi2b
#mnvdatano2p2h should be the data with all non 2p2h components subtracted, compared to merged2p2hlist
# Regularization parameter
def regb(regparm, binWeights_lst):
    binWeights_arr = np.array(exp_fit_pramsb(binWeights_lst))
    regmtxb_arr = np.array(regmtxb)
    return (regparm * regmtxb_arr @ binWeights_arr) @ (regparm * regmtxb_arr @ binWeights_arr)

def Chi2_Regb(binWeights_lst, regparm):
    chi_squared_term = Chi_squaredb(binWeights_lst)
    regularization_term = regb(regparm, binWeights_lst)
    return chi_squared_term + regularization_term

#Lambda scan
scan=False

if scan==True:
    # Arrays to store the penalty chi^2 and data-model chi^2 values
    penalty_chi2 = np.zeros(100)
    data_model_chi2 = np.zeros(100)
    lambdas = np.zeros(100)
    
    # Lambda scan
    for i in range(100):
        lambda_val = i / 10 * .5/9.9  # Lambda ranging from 0 to 2.0 with the same number of intervals
        lambdas[i] = lambda_val
        
        # Recompute binWeightsb for the current lambda value
        binWeightsb = np.ones(numberOfDivisions)
        FitSuccess = False
        while not FitSuccess:
            bestfitpramsb = least_squares(Chi2_Regb, binWeightsb, args=(lambda_val,), method='trf')
            FitSuccess = bestfitpramsb.success
            binWeightsb = bestfitpramsb.x
        
        # Calculate the data-model chi^2
        data_model_chi2[i] = Chi_squaredb(binWeightsb)
        
        # Calculate the penalty chi^2
        penalty_chi2[i] = regb(lambda_val, binWeightsb)
        
    # Plotting penalty chi^2 against data-model chi^2
    # The 'c' parameter represents the lambda values, and 'cmap' specifies the colormap.
    sc = plt.scatter(penalty_chi2, data_model_chi2, c=lambdas, cmap='viridis') # Flipped axes
    plt.xlabel('Penalty Chi^2') # Updated label
    plt.ylabel('Data-Model Chi^2') # Updated label
    plt.title('Data-Model Chi^2 vs Penalty Chi^2')
    
    # Adding a colorbar to show the lambda values
    plt.colorbar(sc, label='Lambda Value')
    plt.show()

#Minimizer for regularized
binWeightsb = np.zeros(numberOfDivisions)
lmda = 0  # Regularization parameter
if lmda != 0:
    FitSuccess = False
    
    while not FitSuccess:
        # Use Chi2_Regb function
        bestfitpramsb = least_squares(Chi2_Regb, binWeightsb, args=(lmda,), method='trf')
        FitSuccess = bestfitpramsb.success
        binWeightsb = bestfitpramsb.x    

    print(Chi_squaredb(binWeightsb))
    weightsb=exp_fit_pramsb(binWeightsb)
    
    ones = np.ones(numberOfDivisions)
    fitted2p2hb=np.array_split(merged2p2hlistb@weightsb,histX)
    NUISANCE2p2hb=np.array_split(merged2p2hlistb@ones,histX)
    
else:
    
    #Minimizer for non regularized
    binWeightsb = np.zeros(numberOfDivisions)
    FitSuccess = True
    while not FitSuccess:
        # Use Chi_sqauredb function
        bestfitpramsb = least_squares(Chi_squaredb, binWeightsb, method='trf')
        FitSuccess = bestfitpramsb.success
        binWeightsb = bestfitpramsb.x
    ones = np.ones(numberOfDivisions)
    
    print('Chi: ', Chi_squaredb(binWeightsb))
    print()
    print('Untuned Chi: ', Chi_squaredb(ones))
    print('Bin weights: ')
    weightsb=binWeightsb
    print(weightsb)
    
    # Make fitted2p2hb
    fitted2p2hb = [ [0]*length for length in binStructuredpt]
    for i in range(len(binStructuredpt)):
        row_length = binStructuredpt[i] - 1
        row = [0 for _ in range(histX)] if row_length > 0 else []
        fitted2p2hb.append(row)

    weightedmerged2p2hlistb=merged2p2hlistb@weightsb
    for i in range(histX):
        for f in range(binStructuredpt[i]):
            fitted2p2hb[i][f]=weightedmerged2p2hlistb[i+f]
    NUISANCE2p2hb=np.array_split(merged2p2hlistb@ones,histX)

noTuneBins=[ [0]*length for length in binStructuredpt]

print('BinNorm: ', BinNorm)

for f in range(histX):
    for i in range(binStructuredpt[f]):
        if Mnv2p2h[f][i] == 0:
            noTuneBins[f][i] = 0
        elif (Mnv2p2h[f][i] / MnvMC[f][i]) > .2:
            noTuneBins[f][i] = 1
        else:
            noTuneBins[f][i] = 0

#Normalize components

newFit=[ [0]*length for length in binStructuredpt]
for f in range(histX):
    for i in range(binStructuredpt[f]):
        newFit[f][i] = (fitted2p2hb[f][i] - Mnv2p2h[f][i] + MnvMC[f][i])/BinNorm[f][i]
        
for f in range(histX):
    for i in range(binStructuredpt[f]):
        MnvData[f][i] = MnvData[f][i]/BinNorm[f][i]
        
for f in range(histX):
    for i in range(binStructuredpt[f]):
        Mnvqe[f][i] = Mnvqe[f][i]/BinNorm[f][i]
        
for f in range(histX):
    for i in range(binStructuredpt[f]):
        Mnvpion[f][i] = Mnvpion[f][i]/BinNorm[f][i]
        
for f in range(histX):
    for i in range(binStructuredpt[f]):
        Mnv2p2h_2[f][i] = Mnv2p2h_2[f][i]/BinNorm[f][i]        

for f in range(histX):
    for i in range(binStructuredpt[f]):
        MnvMC[f][i] = MnvMC[f][i]/BinNorm[f][i]

x11=np.zeros(histX)

for i in range(histX):
    x11[i]=i+1

xdpt = []

for f in range(histX):
    inner_list = []
    for i in range(binStructuredpt[f]):
        inner_list.append(i + 1)
    xdpt.append(inner_list)

for i in range(histX):
    # Plot each point in newFit[i] with color based on noTuneBins[i]
    for f in range(len(newFit[i])):
        color = 'red' if noTuneBins[i][f] == 1 else 'darkgreen'
        plt.plot(xdpt[i][f], 0, marker='o', color=color)

    # Plot data
    p1, = plt.plot(xdpt[i], (newFit)[i], color='r', label="Tune")
    p2, = plt.plot(xdpt[i], MnvMC[i], color='r', linestyle='dashed', label="Untuned Total Prediction")
    p3 = plt.errorbar(xdpt[i], MnvData[i], yerr=error[i], fmt="o", color="k", capsize=5, markersize=2.9, label="Data")
    p4, = plt.plot(xdpt[i], (Mnv2p2h_2)[i], color='b', label="2p2h")
    p5, = plt.plot(xdpt[i], (Mnvqe)[i], color='purple', label="QE")
    p6, = plt.plot(xdpt[i], (Mnvpion)[i], color='cyan', label="Pion")
    
    tuned_patch = mlines.Line2D([], [], color='red', marker='o', markersize=5, label='Tuned')
    untuned_patch = mlines.Line2D([], [], color='darkgreen', marker='o', markersize=5, label='Untuned')

    plt.title(f"ptmu vs dpt - ptmu bin {i+1}")
    plt.xlabel(r'$\Delta p_{T}$')
    plt.ylabel("Differential Cross Section")
    plt.legend(handles=[tuned_patch, untuned_patch, p1, p2, p3, p4, p5, p6], labels=["Normal Bin", "Low 2p2h Bin", "Tuned Total", "Untuned Total", "Data", "2p2h", "QE", "π"], fontsize=8)
    plt.show()