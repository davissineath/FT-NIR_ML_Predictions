import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import glob
import seaborn as sns
import tqdm as tqdm
from BaselineRemoval import BaselineRemoval 
import numpy as np

#FTNIR-csv input
def processing(fp):
    # 1. SETUP & DATA LOADING
    # Update path to look for .csv files
    file_paths = glob.glob(fp)

    # Lists to store data
    intensities_list = []
    wavenumber_col = None

    # Loop through each file to build the dataset
    for i, filename in enumerate(file_paths):
        # Read CSV. standard CSVs use comma delimiter. 
        # header=None assumes no text headers (row 0 is data). 
        # If your files have headers like "Wave, Int", change to header=0
        df = pd.read_csv(filename, header=None)
        
        # Capture Wavenumber from the very first file only (Column 0)
        if i == 0:
            wavenumber_col = df.iloc[:, 0]
        
        # Capture Intensity from every file (Column 1)
        intensities_list.append(df.iloc[:, 1])

    # Create the master DataFrame: Wavenumber + All Intensities
    # axis=1 stacks them side-by-side
    rslt_df = pd.concat([wavenumber_col] + intensities_list, axis=1)

    # Generate column names (A, B, C...) to match the logic your processing steps expect
    # 'A' will be Wavenumber, 'B' onwards will be intensities
    rslt_df.columns = [chr(i + ord('A')) for i in range(rslt_df.shape[1])]
    rslt_df.rename(columns={'A': 'Wavenumber'}, inplace=True)


    # 2. COSMIC RAY REMOVAL
    def modified_z_score(intensity):
        median_int = np.median(intensity)
        mad_int = np.median(np.abs(intensity - median_int))
        # Avoid division by zero if MAD is 0
        if mad_int == 0:
            return np.zeros_like(intensity)
        modified_z_scores = 0.6745 * (intensity - median_int) / mad_int
        return modified_z_scores

    def fixer(y, m):
        threshold = 5 # binarization threshold. 
        y = np.array(y)
        
        # --- FIX START ---
        # np.diff(y) returns an array shorter by 1. 
        # We append a 0 to make 'diff' the same length as 'y' so indices match.
        diff_y = np.diff(y)
        diff_y = np.append(diff_y, 0) 
        # --- FIX END ---
        
        spikes = abs(np.array(modified_z_score(diff_y))) > threshold
        y_out = y.copy() 
        
        for i in np.arange(len(spikes)):
            if spikes[i] != 0: 
                w = np.arange(i-m, i+1+m) 
                # Ensure neighbors are within valid bounds of the array
                w = w[(w >= 0) & (w < len(y))]
                
                # Now 'spikes' and 'y' have the same length, so this indexing is safe
                w2 = w[spikes[w] == 0] 
                
                if len(w2) > 0:
                    y_out[i] = np.mean(y[w2])  
        return y_out

    # Updated Processing Loop (Skips Wavenumber)
    MASTER = []
    # We verify we aren't processing the Wavenumber column
    for col_name in rslt_df.columns:
        # Skip the Wavenumber column (Column 'A')
        if col_name == 'A' or col_name == 'Wavenumber':
            # Just append the raw wavenumber without processing
            MASTER.append(rslt_df[col_name].values)
            continue
            
        intensity = rslt_df[col_name].values
        # No need for tolist(), numpy handles arrays faster
        fixed_intensity = fixer(intensity, m=7)  
        MASTER.append(fixed_intensity)

    # Rebuild DataFrame
    df = pd.DataFrame(MASTER)
    result = df.transpose()
    result.columns = [chr(i + ord('A')) for i in range(result.shape[1])]


    # 3. BASELINE REMOVAL
    # Drop Column 'A' (Wavenumber) so we only process intensities
    data1 = result.drop(columns = ['A'])

    cols = list(data1.columns.values)
    Master2 = []

    for i in data1.columns:
        temp = data1[i].values.tolist()
        baseObj = BaselineRemoval(temp)
        Zhangfit_output = baseObj.ZhangFit()
        Master2.append(Zhangfit_output)
        
    output = pd.DataFrame(Master2)
    reshape = np.transpose(output)


    # 4. NORMALIZATION
    x = reshape.values 
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    # Create dataframe with correct column names
    data_normalized = pd.DataFrame(x_scaled, columns=cols)


    # 5. OUTPUT
    # Retrieve Wavenumber column from the processed result (Column 'A')
    # Note: Since 'result' included the Wavenumber column processed by 'fixer', we use that.
    wavenumber_final = result['A']

    # Concatenate Wavenumber + Normalized Data
    frame = pd.concat([wavenumber_final, data_normalized], axis=1)
    frame.rename(columns={'A': 'Wavenumber'}, inplace=True)
    frame.set_index('Wavenumber', inplace=True)
    
    # Save to Excel
    # output_path = "/Users/Nikki/Desktop/FTNIR/Experiment1/0cm/0cm-processed.xlsx"
    # frame.to_excel(output_path, index=False)
    # print(f"Processing complete. File saved to: {output_path}")

    return frame

def no_processing(data_paths, labels, delim=","):
    data = pd.DataFrame()
    for data_path, label in zip(data_paths, labels):    
        l = [pd.read_csv(filename,header=None,index_col=0, names=[label], delimiter = delim) for filename in glob.glob(data_path)]
        new_data = pd.concat(l, ignore_index=False, axis=1)
        data = pd.concat([data, new_data], ignore_index=False, axis=1)
    return data.T

def processing_pipeline(data_paths, labels): 
    data = pd.DataFrame()
    for data_path, label in zip(data_paths, labels):    
        print(f"Processing {data_path} with label {label}...")
        new_data = processing(data_path)
        new_data = new_data.T
        new_data.index = [label for _ in range(new_data.shape[0])]
        data = pd.concat([data, new_data], ignore_index=False, axis=0)
    return data