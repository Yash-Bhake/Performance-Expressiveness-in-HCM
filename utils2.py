import librosa
import numpy as np
from scipy.signal import butter, filtfilt, convolve, lfilter
import pandas as pd
from music21 import converter, note, chord, tempo, meter, key # for loading scores
from praatio import tgio
import tgt
import soundfile as sf
from IPython.display import display
import matplotlib.pyplot as plt
import scipy
from scipy.signal import butter, lfilter, freqz
import os
import sys
import regex as re
import chardet




#----------------------------------------------------------LOADING STUFF---------------------------------------------------------

def load_audio(audio_file):
    '''
    takes in the song path, returns the sampled audio array and original sampling rate
    '''
    sr = sf.read(audio_file)[1]
    y, _ = librosa.load(audio_file, sr=sr)
    return y, sr


def load_audio2(audio_path, show=[]):
    x, sr = load_audio(audio_path)
    audio_dur = len(x)/sr
    if show:
        plt.figure(figsize=(12, 1))
        plt.plot(np.arange(len(x))/sr, x)
        plt.title(f'{audio_path} at {sr} hz; duration: {audio_dur:.2f} sec', size=10)
        plt.xlabel('Time (s)')
        plt.xlim(show)
        plt.show()
        
    return x, sr, audio_dur

#-----------------------------------------------------Band Pass the signal for speech info-------------------------------------

# Bandpass filter
def bandpass_filter(data, low_f, high_f, sample_rate, order=5):
    '''
    Apply band pass filter to a given signal
    input: audio signal, lower bound, higher bound, sr, order
    output: filtered audio
    '''
    nyquist = 0.5 * sample_rate
    low = low_f / nyquist
    high = high_f / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = lfilter(b, a, data)
    return filtered_data


def plot_filter_response(low_f, high_f, sample_rate, order=5):
    nyquist = 0.5 * sample_rate
    low = low_f / nyquist
    high = high_f / nyquist
    b, a = butter(order, [low, high], btype='band')

    # Frequency response
    w, h = freqz(b, a, worN=8000)
    
    # Plot the frequency response
    plt.plot(0.5 * sample_rate * w / np.pi, np.abs(h), 'b')
    plt.title("Bandpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Gain')
    plt.xlim(0, 8000)
    plt.grid()
    plt.show()

#---------------------------------------------------------Load labels------------------------------------------------------------


def load_label(label_path):
    '''
    Load the labels (Praat .TextGrid or Audacity .txt)
    '''
    assert label_path.split('.')[-1] in ['txt', 'TextGrid'], 'txt/TextGrid file support only'
    
    data = []
    
    if label_path.split('.')[-1] == 'TextGrid':
        time_stamps, texts = textgrid_to_onsets(label_path)
        for timestamp, text in zip(time_stamps, texts):
            data.append(timestamp, text)

    elif label_path.split('.')[-1] == 'txt':
        with open(label_path, 'r', encoding = 'utf-8') as f:
            data = f.read()
        data = [d.split('\t') for d in data.split('\n')[:-1]]
        
    label_df = pd.DataFrame(data)
    label_df.rename(columns={0:'Start'}, inplace=True)
    label_df['Start'] = label_df['Start'].apply(lambda x: float(x))
    # label_df['start_idx'] = label_df['start'].apply(lambda x: int(np.round(x/0.01))) # this is for generating the time_idx for feature rate os 100 hz

    return label_df


#---------------------------------------------------------Textgrid to time stamps------------------------------------------------

def textgrid_to_onsets(tg_path, tier_number = 0):
    '''
    input: path to the .TextGrid file
    output: a tuple of 2 lists, first is the list of time stamps, 2nd is the list of correcponding text marks
    '''
    # Load the TextGrid file
    tg = tgio.openTextgrid(tg_path)

    # Extract intervals from the T
    # TextGrid within the 10-second section
    tier_name = tg.tierNameList[tier_number]  # Assuming you want the first tier
    tier = tg.tierDict[tier_name]
    intervals = tier.entryList

    # Extract time stamps and texts for the segment
    time_stamps = [entry.start for entry in intervals]
    texts = [entry.label for entry in intervals]
    return time_stamps, texts

#-------------------------------------------------------------------------------------------------------------------------------

def onsets_to_textgrid(original_tg_path, detected_onsets, output_tg_path):
    '''
    input: 
        original_tg_path: path to the original .TextGrid file
        detected_onsets: array of detected onset timestamps
        output_tg_path: path where the new .TextGrid file will be saved
    output: 
        None (saves the new TextGrid file to the specified path)
    '''
    # Try different encodings to ensure that it doesnt give an error when the syllable marks are not english
    encodings = ['utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'latin-1', 'ascii']
    
    for encoding in encodings:
        try:
            # Load the original TextGrid file
            original_tg = tgt.read_textgrid(original_tg_path, encoding=encoding)
            break  # If successful, break the loop
        except UnicodeDecodeError:
            continue  # If unsuccessful, try the next encoding
    else:
        raise ValueError(f"Unable to read the TextGrid file with any of the attempted encodings: {encodings}")

    # Get the first tier (assuming you want to modify the first tier)
    tier = original_tg.tiers[0]

    # Create a new tier with the same name and type
    new_tier = tgt.IntervalTier(name=tier.name, start_time=tier.start_time, end_time=tier.end_time)

    # Add intervals based on detected_onsets
    for i in range(len(detected_onsets)):
        start_time = detected_onsets[i]
        
        # If it's not the last onset, use the next onset as the end time
        # Otherwise, use the end time of the original tier
        if i < len(detected_onsets) - 1:
            end_time = detected_onsets[i+1]
        else:
            end_time = tier.end_time
        
        # Create a new interval with an empty label
        new_interval = tgt.Interval(start_time=start_time, end_time=end_time, text="")
        new_tier.add_interval(new_interval)

    # Create a new TextGrid with the modified tier
    new_tg = tgt.TextGrid()
    new_tg.add_tier(new_tier)

    # Write the new TextGrid to file
    tgt.write_to_file(new_tg, output_tg_path, format='short', encoding='utf-8')



def textgrid_to_onsets(tg_path, tier_number = 0):
    '''
    input: path to the .TextGrid file
    output: a tuple of 2 lists, first is the list of time stamps, 2nd is the list of correcponding text marks
    '''
    # Load the TextGrid file
    tg = tgio.openTextgrid(tg_path)

    # Extract intervals from the T
    # TextGrid within the 10-second section
    tier_name = tg.tierNameList[tier_number]  # Assuming you want the first tier
    tier = tg.tierDict[tier_name]
    intervals = tier.entryList

    # Extract time stamps and texts for the segment
    time_stamps = [entry.start for entry in intervals]
    texts = [entry.label for entry in intervals]
    return time_stamps, texts



def textgrid_to_onsets2(tg_path, tier_number=0):
    with open(tg_path, "rb") as f:
        rawData = f.read()
        detectedEncoding = chardet.detect(rawData)["encoding"]
        print(detectedEncoding)

    nHeaderLines = 14
    with open(file=tg_path, mode="r", encoding=detectedEncoding, errors="ignore") as f:
        lines = f.readlines()[nHeaderLines:]

    lines = [line.strip() for line in lines]

    intervalPattern = re.compile(r"intervals \[(\d+)\]")  
    xminPattern = re.compile(r"xmin\s*=\s*([\d\.]+)") 
    xmaxPattern = re.compile(r"xmax\s*=\s*([\d\.]+)")  
    textPattern = re.compile(r'text\s*=\s*"(.*?)"') 

    intervals = [] # [(interval, xmin, xmax, text), (), ...]

    for i, line in enumerate(lines):

        dump = False

        if i % 4 == 0:
            intervalPattern.match(line)
            if intervalPattern:
                intervalVal = int(intervalPattern.match(line).group(1))
            else:
                sys.exit(f"Something went wrong, please check interval at line number: {i+nHeaderLines}")
        if i % 4 == 1:
            xminPattern.match(line)
            if xminPattern:
                xminVal = float(xminPattern.match(line).group(1))
            else:
                sys.exit(f"Something went wrong, please check xmin at line number: {i+nHeaderLines}")

        if i % 4 == 2:
            xmaxPattern.match(line)
            if xmaxPattern:
                xmaxVal = float(xmaxPattern.match(line).group(1))
            else:
                sys.exit(f"Something went wrong, please check xmax at line number: {i+nHeaderLines}")
        if i % 4 == 3:
            textPattern.match(line)
            if textPattern:
                textVal = str(textPattern.match(line).group(1))
                dump = True
            else:
                sys.exit(f"Something went wrong, please check text at line number: {i+nHeaderLines}")
        if dump:
            intervals.append([intervalVal, xminVal, xmaxVal, textVal])
    time_stamps = [i[1] for i in intervals]
    texts = [i[3] for i in intervals]
    return time_stamps, texts

#-------------------------------------------------------.TextGrid to audacity .txt files-----------------------------------------

def TextGrid_to_txt(tg_path, output_file_path, tier_number = 0):
    manual_onsets, texts = textgrid_to_onsets(tg_path, tier_number)
    
    # Open the file in write mode
    with open(output_file_path, 'w', encoding = 'utf-8') as f:
        # Iterate through the arrays
        for onset, text in zip(manual_onsets, texts):
            # Write each line with tab as delimiter
            f.write(f"{onset}\t{text}\n")
            

def TextGrid_to_txt2(tg_path, output_file_path):
    with open(tg_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Extract relevant data
    data = []
    xmin, xmax, text = None, None, None

    for line in lines:
        line = line.strip()
        if line.startswith("xmin ="):
            xmin = float(line.split("=")[1].strip())
        elif line.startswith("xmax ="):
            xmax = float(line.split("=")[1].strip())
        elif line.startswith('text ='):
            text = line.split("=")[1].strip().strip('"')
            if text == "":  # Replace empty text with "Use"
                text = "Use"
            data.append((xmin, xmax, text))

    # Write to a new text file
    with open(output_file_path, "w", encoding="utf-8") as file:
        for entry in data:
            file.write(f"{entry[0]}\t{entry[2]}\n")

    print(f"Processed file saved as {output_file_path}")
 
#--------------------------------------------------------------------------------------------------------------------------------

def usable_intervals_from_tg(textgrid_path):
    use_intervals = []
    with open(textgrid_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
        xmin, xmax, label = None, None, None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("xmin ="):
                xmin = float(line.split('=')[1].strip())
            elif line.startswith("xmax ="):
                xmax = float(line.split('=')[1].strip())
            elif line.startswith("text ="):
                label = line.split('=')[1].strip().strip('"')
                
                # Check if label is "Use" and save the interval
                if label == "Use" and xmin is not None and xmax is not None:
                    use_intervals.append((xmin, xmax))
                    xmin, xmax, label = None, None, None
                    
    return use_intervals


#-------------------------------------------------------predicted onsets to audacity .txt files----------------------------------

def pred_onsets_to_txt(predicted_onsets, output_file_path):
    """
    Create a tab-delimited text file from two NumPy arrays.
    
    :param predicted_onsets: NumPy array of predicted onset times
    :param texts: NumPy array of corresponding texts
    :param output_file_path: Path where the output file will be saved
    """
    # creating an array of null values for the texts of predicted onsets
    texts = np.array(['']*len(predicted_onsets))
    # Open the file in write mode
    with open(output_file_path, 'w', encoding = 'utf-8') as f:
        # Iterate through the arrays
        for onset, text in zip(predicted_onsets, texts):
            # Write each line with tab as delimiter
            f.write(f"{onset}\t{text}\n")


#-------------------------------------------------------Error Metrics-----------------------------------------------------------

def error(onset_times, manual_onsets, tolerance = 0.05):
    '''
    input: predicted_onsets, manual_onsets, margin (tolerance)
    output: tuple of precision value[0, 1], recall value[0, 1], f_score[0, 1], number of manual onsets, number of predicted onsets
    and the list of (matched predicted onset, corresponding manual onset and their difference)
    '''
    TP = 0
    FP = 0
    if onset_times.size == 0:
        return 0, 0, 0, len(manual_onsets), len(onset_times)
    else:
        manual_onsets_copy = np.array(manual_onsets)
        pred_onsets = np.array(np.copy(onset_times))

        matched_onsets = []

        for manual_onset in manual_onsets_copy:
            diff = abs(pred_onsets - manual_onset)

            min_val = np.min(diff)
            if min_val <= tolerance:
                TP+=1
                matched_onsets.append((pred_onsets[np.argmin(diff)], manual_onset, manual_onset - pred_onsets[np.argmin(diff)])) #taking
                pred_onsets = np.delete(pred_onsets, np.argmin(diff))
            FP = len(onset_times) - TP
            precision_value = TP / (len(onset_times)) if (len(onset_times)) > 0 else 0
            recall_value = TP/ (len(manual_onsets)) if (len(manual_onsets)) > 0 else 0
            if recall_value !=0 and precision_value !=0:
                f_score = (2*precision_value*recall_value)/(precision_value+recall_value)        


    return TP, precision_value, recall_value, f_score, len(manual_onsets), len(onset_times), matched_onsets

#--------------------------------------------------------Histogram for showing distribution-----------------------------------

def plot_histogram(data, bins=10, title='Histogram', xlabel='Values', ylabel='Frequency'):
    '''input: data: 1d array of numbers
              bins: number of bins
       output: histogram for the given values'''
    plt.hist(data, bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


#-------------------------------------------------------------Tabla Error------------------------------------------------------

def tabla_error(onset_times, manual_onsets, tolerance = 0.02):
    '''
    input: predicted_onsets, manual_onsets, margin (tolerance)
    output: number of true positives, number of manual onsets, number of predicted onsets, 
    the list of (matched predicted onset, corresponding manual onset and their difference),
    rms of the differences bw true onsets and corresponding manual onsets, and
    std dev of differences bw true onsets and corresponding manual onsets
    '''
    TP = 0
    if onset_times.size == 0:
        return 0, 0, 0, len(manual_onsets), len(onset_times)
    else:
        manual_onsets_copy = np.array(manual_onsets)
        pred_onsets = np.array(np.copy(onset_times))

        matched_onsets = []

        for manual_onset in manual_onsets_copy:
            diff = abs(pred_onsets - manual_onset)

            min_val = np.min(diff)
            if min_val <= tolerance:
                TP+=1
                matched_onsets.append((pred_onsets[np.argmin(diff)], manual_onset, manual_onset - pred_onsets[np.argmin(diff)])) #taking
                pred_onsets = np.delete(pred_onsets, np.argmin(diff))

        matched_onsets = np.array(matched_onsets)
        third_column = matched_onsets[:, 2]
        rms = np.sqrt(np.mean(np.square(third_column)))
        std = np.std(third_column)


    return TP, len(manual_onsets), len(onset_times), matched_onsets, rms, std

#-------------------------------------------------------------Adaptive filtering (biphasic filter)-----------------------------


def adaptation_filter(nov, sr_nov, filter_length):
    
    tau1 = int(0.015 * sr_nov)
    d1 = int(0.020 * sr_nov)
    tau2 = int(0.015 * sr_nov)
    d2 = int(0.020 * sr_nov)
    

    kernel = np.zeros(2 * filter_length)
    t = np.arange(-filter_length, +filter_length+1) 
    kernel = (1/(tau1*np.sqrt(2*np.pi))) * np.exp(-(t-d1)**2/(2*tau1**2)) - (1/(tau2*np.sqrt(2*np.pi))) * np.exp(-(t+d2)**2/(2*tau2**2))
    kernel =  np.exp(-(t-d1)**2/(2*tau1**2)) - np.exp(-(t+d2)**2/(2*tau2**2))
    

    # Apply the biphasic filter using convolution
    der = scipy.signal.convolve(nov, np.array(list(reversed(kernel))), mode='same') # reversed to perform convolution in the right orientation

    der[der < 0] = 0

    return der, kernel

#-----------------------------------------------------------Get voiceless intervals-----------------------------------------------

def get_voiceless_intervals(voiced_flag, times, start_offset=0.05, stop_offset=3):
    unvoiced_regions = []
    for i in voiced_flag:
        if i:
            unvoiced_regions.append(0)
        else:
            unvoiced_regions.append(1)

    unvoiced_timestamps = np.array(times)*np.array(unvoiced_regions)

    def extract_non_zero_sublists(arr, stop_offset):
        sublists = []
        current_sublist = []

        for value in arr:
            if value != 0:
                current_sublist.append(value)
            else:
                if current_sublist:
                    sublists.append(current_sublist)
                    current_sublist = []
        
        # Append the last sublist if not empty
        if current_sublist:
            sublists.append(current_sublist)
        
        for i in sublists:
            i = i[:(-1*stop_offset)]

        return np.array(sublists)

    return extract_non_zero_sublists(unvoiced_timestamps, stop_offset)

#-----------------------------------------------------------Remove voiceless_onsets-----------------------------------------------

def remove_timestamps_in_intervals(detected_onsets, voiceless_intervals):

    for i in detected_onsets:
        if i in voiceless_intervals.flatten():
            detected_onsets = np.delete(detected_onsets, np.where(detected_onsets==i))
    return detected_onsets


#-----------------------------------------------------------Bandpass filter-------------------------------------------------------

def lowpass_filter(data, cutoff, sr, order=5):
    '''
    Apply band pass filter to a given signal
    '''
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_signal = filtfilt(b, a, data)
    return filtered_signal

#-----------------------------------------------------------Remove small stretches------------------------------------------------

def remove_small_stretches(values, a):
    values = np.array(values)
    result = np.zeros_like(values)  # Initialize an array of zeros with the same shape as values
    
    i = 0
    while i < len(values):
        if values[i] != 0:
            start = i
            while i < len(values) and values[i] != 0:
                i += 1
            end = i
            
            # If the length of the stretch is greater than or equal to 'a', copy it to the result
            if end - start >= a:
                result[start:end] = values[start:end]
        else:
            i += 1
    
    return result

#----------------------------------------------------------Evaluate and get FNR, FPR for det curve--------------------------------

def evaluate(pred_onsets, gt_onsets, tolerance_samp=5):
    TP = 0
    FP = 0
    FN = 0
    correct_onsets = []
    incorrect_onsets = []
    used_pred_indices = set()

    if len(pred_onsets) > 0:
        for g in gt_onsets:
            min_val = np.min(np.abs(pred_onsets - g))
            min_index = np.argmin(np.abs(pred_onsets - g)) 
            
            if min_val <= tolerance_samp and min_index not in used_pred_indices:
                TP += 1
                correct_onsets.append(g)
                used_pred_indices.add(min_index)
            else:
                FN += 1
                incorrect_onsets.append(g) # all the gt onsets that were missed
        # Calculate FP for predictions that were not matched to any ground truth
        FP = len(pred_onsets) - TP
    
    P = TP / len(pred_onsets) if len(pred_onsets) > 0 else 0
    R = TP / len(gt_onsets) if len(gt_onsets) > 0 else 0
    
    F1 = (2 * P * R) / (P + R) if (P+R) > 0 else 0

    
    correct_onsets = np.array(correct_onsets)
    incorrect_onsets = np.array(incorrect_onsets)
    
    return TP, FP, FN, P, R, F1, correct_onsets, incorrect_onsets

#-----------------------------------------------------Evalutate 2------------------------------------------------------------------

def evaluate2(pred_onsets, gt_onsets, tolerance_samp=5):
    TP = 0
    FP = 0
    FN = 0
    correct_onsets = []
    incorrect_onsets = []
    used_pred_indices = set()
    matched_onsets = []

    if len(pred_onsets) > 0:
        for g in gt_onsets:
            min_val = np.min(np.abs(pred_onsets - g))
            min_index = np.argmin(np.abs(pred_onsets - g)) 
            
            if min_val <= tolerance_samp and min_index not in used_pred_indices:
                TP += 1
                correct_onsets.append(g)
                used_pred_indices.add(min_index)
                matched_onsets.append((pred_onsets[min_index], g, g - pred_onsets[min_index]))
            else:
                FN += 1
                incorrect_onsets.append(g) # all the gt onsets that were missed
        # Calculate FP for predictions that were not matched to any ground truth
        FP = len(pred_onsets) - TP
    
    P = TP / len(pred_onsets) if len(pred_onsets) > 0 else 0
    R = TP / len(gt_onsets) if len(gt_onsets) > 0 else 0
    
    F1 = (2 * P * R) / (P + R) if (P+R) > 0 else 0

    
    correct_onsets = np.array(correct_onsets)
    incorrect_onsets = np.array(incorrect_onsets)
    
    return TP, FP, FN, P, R, F1, len(gt_onsets), len(pred_onsets), matched_onsets

#------------------------------------------------------DET params-------------------------------------------------------------

def compute_det_params(total_n_gt_onsets, FN, FP, total_n_gt_not_onsets):
    actual_positive_onsets = total_n_gt_onsets
    actual_negative_onsets = total_n_gt_not_onsets - actual_positive_onsets
    FNR = FN/(actual_positive_onsets)
    FPR = FP/(actual_negative_onsets)
    return FNR, FPR

#------------------------------------------------------------------------------------------------------------------------------


def generate_log_spaced_array(central, left_limit, right_limit, n):
    '''
    Generates an array of `n` numbers with concentration near `central`
    and logarithmically spaced as they approach the left and right limits.
    '''
    # Generate a set of log-spaced points between 0 and 1
    log_space = np.logspace(-2, 0, n // 2)

    # Scale these points to fit between left_limit and central
    left_part = central - (central - left_limit) * log_space

    # Scale these points to fit between central and right_limit
    right_part = central + (right_limit - central) * log_space[::-1]

    # Combine both parts and add the central number in between
    full_array = np.concatenate([left_part, [central], right_part])

    return full_array




#----------------------------------------------------------------------------------------------------------------------

import csv

# def convert_ctm_to_audacity_labels(ctm_file, audacity_file):
#     """
#     Convert a .ctm file to an Audacity text label file.
    
#     Parameters:
#     - ctm_file: Path to the input .ctm file.
#     - audacity_file: Path to the output Audacity text label file.
#     """
#     with open(ctm_file, 'r') as infile, open(audacity_file, 'w') as outfile:
#         reader = csv.reader(infile, delimiter=' ')
#         for row in reader:
#             if len(row) < 5:
#                 continue  # Skip rows that do not have enough columns
#             onset = float(row[2])  # 3rd column: onset timestamp
#             duration = float(row[3])  # 4th column: phoneme duration
#             phoneme = row[4]  # 5th column: phoneme
#             offset = onset + duration  # Calculate the end time
#             # Write to the Audacity label file
#             outfile.write(f"{onset:.3f}\t{offset:.3f}\t{phoneme}\n")

# # Example usage
# ctm_file_path = 'asr/yash_phones.ctm'  # Replace with your .ctm file path
# audacity_file_path = 'yash_audacity_phones.txt'  # Replace with desired output file path
# convert_ctm_to_audacity_labels(ctm_file_path, audacity_file_path)

# # print(f"Converted {ctm_file_path} to {audacity_file_path}.")

#---------------------------------------------------------------------------------------------------------------------

def get_file_paths(base_folder, folder: str, artist: str):
    """Returns a list of file paths for a given folder and artist."""
    artist_folder = os.path.join(base_folder, folder, artist)
    
    if not os.path.exists(artist_folder):
        print(f"Error: The folder '{artist_folder}' does not exist.")
        return []
    
    prefix = f"{'ja_jare' if 'ja_jare' in folder else 'yeri_aali'}_{artist}"
    files = [
        os.path.join(artist_folder, f).replace("\\", "/") 
        for f in os.listdir(artist_folder) if f.startswith(prefix)
    ]
    
    return files

