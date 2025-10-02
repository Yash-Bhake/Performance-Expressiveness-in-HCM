
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd
import os
import utils2
import pandas as pd
from matplotlib.font_manager import FontProperties
import regex as re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.lines import Line2D
from scipy.interpolate import make_interp_spline
from scipy.interpolate import Akima1DInterpolator
from itertools import groupby
from collections import defaultdict
from pydub.generators import Sine
from scipy.io.wavfile import write
import parselmouth

from pydub import AudioSegment, silence
from pydub.playback import play
import soundfile as sf
from praatio import tgio
import scipy
import copy

import ipywidgets as widgets
import IPython.display as ipd
from IPython.display import display, clear_output

import expressiveness_measure_new
import Levenshtein
from Levenshtein import distance as levenshtein_distance  # pip install python-Levenshtein
from typing import List
from constants import *
from sklearn.mixture import GaussianMixture


def get_tonic_from_excel(bandish_db, file_name, sheet_name):
    df = pd.read_excel(bandish_db, sheet_name=sheet_name, skiprows=5)
    match = df[df["Recording name"] == file_name]

    if match.empty or "Tonic" not in df.columns:
        raise ValueError(f"Tonic not found for '{file_name}' or missing 'tonic' column.")
    return match["Tonic"].iloc[0]

def get_frequency(tonic):
    return librosa.note_to_hz(tonic)

def get_segments(times, quantized):
    segments = []
    start = times[0]
    prev = quantized[0]

    for t, q in zip(times[1:], quantized[1:]):
        if q != prev:
            if prev != -np.inf:
                segments.append((start, t, prev))
            start = t
            prev = q

    if prev != -np.inf:
        segments.append((start, times[-1], prev))
    return segments

def quantize_pitch_contour(pitch_cents, bins_cents):
    """
    Quantizes a pitch contour (in cents) to the nearest value from a list of bins.

    Parameters:
        pitch_cents (np.ndarray): Array of pitch values in cents. Can contain -np.inf for unvoiced.
        bins_cents (list or np.ndarray): List of bin center values in cents.

    Returns:
        np.ndarray: Quantized pitch contour with -inf preserved for unvoiced.
    """
    bins = np.array(bins_cents)
    quantized = np.copy(pitch_cents)

    # Mask for voiced regions
    voiced_mask = (pitch_cents != -np.inf)

    # For each voiced pitch, find the nearest bin
    voiced_vals = pitch_cents[voiced_mask]
    nearest_bins = bins[np.argmin(np.abs(voiced_vals[:, np.newaxis] - bins), axis=1)]

    quantized[voiced_mask] = nearest_bins

    return quantized

def interpolate_short_unvoiced_gaps(f0, max_gap_frames=7):
    f0 = np.array(f0, dtype=float)
    voiced_mask = (f0 > 0) & ~np.isnan(f0)
    unvoiced_mask = ~voiced_mask

    f0_interp = f0.copy()
    start = 0

    while start < len(f0):
        if unvoiced_mask[start]:
            end = start
            while end < len(f0) and unvoiced_mask[end]:
                end += 1
            gap_length = end - start

            # Interpolate only if short gap and surrounded by voiced
            if start > 0 and end < len(f0) and gap_length <= max_gap_frames:
                x = [start - 1, end]
                y = [f0[start - 1], f0[end]]
                interp = np.interp(np.arange(start, end), x, y)
                f0_interp[start:end] = interp

            start = end
        else:
            start += 1

    return f0_interp

def pitch_contour_extrn(y, sr, tonic_freq, controls, note_cents_eq):
    snd = parselmouth.Sound(values=y, sampling_frequency=sr)
    # create pitch object
    pitch = snd.to_pitch_ac(
            time_step=controls["time_step"],
            pitch_floor=controls["pitch_floor"],
            max_number_of_candidates=controls["max_number_of_candidates"],
            very_accurate=controls["very_accurate"],
            silence_threshold=controls["silence_threshold"],
            voicing_threshold=controls["voicing_threshold"],
            octave_cost=controls["octave_cost"],
            octave_jump_cost=controls["octave_jump_cost"],
            voiced_unvoiced_cost=controls["voicing_threshold"],
            pitch_ceiling=controls["pitch_ceiling"]
            )
    # Get pitch contour values
    f0_praat = pitch.selected_array['frequency']
    # Get corresponding timestamps (in seconds)
    times_praat = pitch.xs()

    #default hop of praat which is 10ms is taken
    voiced_regions = np.sign(f0_praat)
    unvoiced_regions = np.logical_not(voiced_regions).astype(int)

    times_praat = [i/SR_NOV for i in range(0, len(f0_praat))]

    f0_praat = interpolate_short_unvoiced_gaps(f0_praat, max_gap_frames=7)

    f0_cents = 1200 * np.log2(f0_praat / tonic_freq) 
    f0_cents[f0_praat < 20] = np.nan
    times_crepe = times_praat
    times_crepe = np.array(times_crepe)
    def elementwise_product(list1, list2):
        # Pad the shorter list with zeros
        max_len = max(len(list1), len(list2))
        list1 = np.pad(list1, (0, max_len - len(list1)))
        list2 = np.pad(list2, (0, max_len - len(list2)))
        # Element-wise product
        return np.multiply(list1, list2).tolist()

    f0_praat = elementwise_product(voiced_regions, f0_praat)
    # f0_praat = nanning(f0_praat)

    # cents
    f0_praat_cents = 1200 * np.log2(f0_praat / tonic_freq)
    # Find start/end indices of unvoiced regions
    diff = np.diff(np.pad(unvoiced_regions, (1, 1), constant_values=0))
    # Find start/end indices of unvoiced regions
    starts = np.where(diff == -1)[0]
    ends = np.where(diff == 1)[0] - 1
    # Clip ends if needed
    ends = ends[ends < len(times_crepe)]
    starts = starts[starts < len(times_crepe)]
    # Align lengths just in case
    min_len = min(len(starts), len(ends))
    unvoiced_intervals = np.column_stack((times_crepe[starts[:min_len]],
                                        times_crepe[ends[:min_len]]))
    
    return f0_praat_cents, times_crepe, unvoiced_intervals

def pitch_contour_extrn1(y, sr, tonic_freq, controls, note_cents_eq):
    snd = parselmouth.Sound(values=y, sampling_frequency=sr)

    # Extract pitch contour with Praat AC method
    pitch = snd.to_pitch_ac(
        time_step=controls["time_step"],
        pitch_floor=controls["pitch_floor"],
        max_number_of_candidates=controls["max_number_of_candidates"],
        very_accurate=controls["very_accurate"],
        silence_threshold=controls["silence_threshold"],
        voicing_threshold=controls["voicing_threshold"],
        octave_cost=controls["octave_cost"],
        octave_jump_cost=controls["octave_jump_cost"],
        voiced_unvoiced_cost=controls["voicing_threshold"],
        pitch_ceiling=controls["pitch_ceiling"]
    )

    f0_praat = pitch.selected_array['frequency']
    times_praat = pitch.xs()   # use real times!

    # Interpolate short gaps
    f0_praat = interpolate_short_unvoiced_gaps(f0_praat, max_gap_frames=7)

    # Mark unvoiced regions explicitly
    voiced_mask = (f0_praat > 20)
    f0_praat_cents = np.full_like(f0_praat, fill_value=float('-inf'), dtype=float)
    f0_praat_cents[voiced_mask] = 1200 * np.log2(f0_praat[voiced_mask] / tonic_freq)

    # Unvoiced intervals (in seconds)
    unvoiced = ~voiced_mask
    diff = np.diff(np.pad(unvoiced.astype(int), (1,1)))
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0] - 1
    unvoiced_intervals = np.column_stack((times_praat[starts], times_praat[ends]))

    return f0_praat_cents, times_praat, unvoiced_intervals

def resample_to_bins(label_seq: List[str], n_bins: int) -> List[str]:
    total_len = len(label_seq)
    bin_size = total_len / n_bins
    resampled = []
    for i in range(n_bins):
        start = int(i * bin_size)
        end = int((i + 1) * bin_size)
        bin_labels = list(label_seq[start:end]) if end > start else [label_seq[start]]
        most_common = max(set(bin_labels), key=bin_labels.count)
        resampled.append(most_common)   
    return resampled

def remove_trailing_neg_inf(arr):
    # Ensure the array is a NumPy array with dtype float for comparison
    arr = np.asarray(arr, dtype=float)
    # Identify the indices where the value is not -inf
    not_neg_inf = arr != -np.inf
    # Find the position of the last non -inf value
    if np.any(not_neg_inf):
        last_valid_index = np.max(np.where(not_neg_inf[:-4])) # Exclude the last 4 indices
        # Slice the array up to the last valid index
        return arr[:last_valid_index + 1].astype("str")
    else:
        # If all values are -inf, return an empty array
        return np.array([])
    
# Get syllable PC segments, start and stop times from manual onsets and syllable id
def get_syllable_PCs(manual_onsets_plot, manual_dict, f0_praat_cents, syl_id_dict, syllable, SR_NOV):
    syllable_segments = []
    start_stamps = []
    stop_stamps = []

    for index, onset in enumerate(manual_onsets_plot):
        if manual_dict[onset] == syl_id_dict[syllable]:
            if index == len(manual_onsets_plot) - 1:
                break
            start_stamps.append(onset)

            next_onset = manual_onsets_plot[index + 1]
            start_idx = int(onset * SR_NOV)
            end_idx = int(next_onset * SR_NOV)
            segment = f0_praat_cents[start_idx:end_idx]

            # Detect silence of >200ms = 20 consecutive -inf
            silence_len = 20
            break_point = None
            count = 0

            for i, val in enumerate(segment):
                if val == float('-inf'):
                    count += 1
                    if count >= silence_len:
                        break_point = i - silence_len + 1  # first index of silent region
                        break
                else:
                    count = 0

            if break_point is not None:
                stop_stamps.append(onset + break_point / SR_NOV)
            else:
                stop_stamps.append(next_onset)



    for start, stop in zip(start_stamps, stop_stamps):
        segment = f0_praat_cents[int(start*SR_NOV): int(stop*SR_NOV)]
        if len(segment)==0:
            continue
        syllable_segments.append(segment)
    
    return syllable_segments, start_stamps, stop_stamps


def NLSS_matrix(resampled_segments):
    # === EDIT DISTANCE MATRIX USING LEVENSHTEIN ===
    N = len(resampled_segments)
    print(resampled_segments)
    if len(resampled_segments)==1:
        resampled_segments.append(resampled_segments[0])
        N = 2

    
    distMat = np.zeros((N, N))
    subsMat = np.zeros((N, N))
    # insMat = np.zeros((N, N))
    # delMat = np.zeros((N, N))
    ed_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(i + 1, N):
            seg_i = np.array(resampled_segments[i])
            seg_j = np.array(resampled_segments[j])

            valid_i = seg_i != "-inf"
            valid_j = seg_j != "-inf"
            mutual_valid = valid_i & valid_j

            if np.sum(mutual_valid) == 0:
                # Fallback: use only valid segments separately
                filtered_i = seg_i[seg_i != "-inf"]
                filtered_j = seg_j[seg_j != "-inf"]

                min_len = min(len(filtered_i), len(filtered_j))
                if min_len == 0:
                    # Still empty – assign large distance
                    ed_matrix[i, j] = ed_matrix[j, i] = 1.0
                    subsMat[i, j] = subsMat[j, i] = 1.0
                    continue

                filtered_i = filtered_i[:min_len]
                filtered_j = filtered_j[:min_len]

            else:
                filtered_i = seg_i[mutual_valid]
                filtered_j = seg_j[mutual_valid]

            str_i = ''.join(filtered_i)
            str_j = ''.join(filtered_j)

            dist = Levenshtein.distance(str_i, str_j)
            edit_ops = Levenshtein.editops(str_i, str_j)
            substitutions = sum(1 for op in edit_ops if op[0] == "replace")

            effective_len = len(str_i)
            ed_matrix[i, j] = ed_matrix[j, i] = dist / effective_len
            subsMat[i, j] = subsMat[j, i] = substitutions / effective_len

    # === MAKE SUBSMAT SYMMETRIC AND SAFE FOR CLUSTERING ===
    subsMat = (subsMat + subsMat.T) / 2  # Enforce symmetry
    subsMat = np.nan_to_num(subsMat, nan=np.nanmax(subsMat))  # Replace any NaNs (just in case)
    mean_NLSS = subsMat[np.triu_indices_from(subsMat, k=1)].mean()

    return subsMat, mean_NLSS

def generate_pitch_contour(cents_vector, tonic_freq=261.63, duration=1, sr=16000):
    """
    Generate continuous pitch contour from cent values.
    """
    n_notes = len(cents_vector)
    n_samples = int(duration * sr)
    t = np.linspace(0, duration, n_samples)

    # Cents -> frequency
    freqs = tonic_freq * (2 ** (np.array(cents_vector) / 1200.0))

    # Note times (even spacing)
    note_times = np.linspace(0, duration, n_notes)

    # Interpolate to get continuous frequency contour
    contour = np.interp(t, note_times, freqs)
    return contour, t

def synthesize_from_contour(contour, sr=16000):
    """
    Synthesize waveform from frequency contour using oscillator.
    """
    phase = 2 * np.pi * np.cumsum(contour) / sr
    waveform = np.sin(phase)
    return waveform

    
#-------------------------------------------------------------------------------------------------------

# Merge by syllable first key on observations_per_id to get all instances of a syllable across the bandish together
def merge_by_syllable_first_key(og_data, syl_list, label_list):
    data = copy.deepcopy(og_data)

    syl_lab_dict = dict(zip(label_list, syl_list))
    merged_dict = {key: [] for key in syl_list}

    label_map_no_NA = [line for lines in data.values() for line in lines]
    # sorted_label_map
    sorted_label_map = sorted(label_map_no_NA, key=lambda x: x[0][0])
    for line in sorted_label_map:
        syllable = line[0][1]
        if syllable in syl_lab_dict:
            id = syl_lab_dict[syllable]
            merged_dict[id].append(line)

    return merged_dict

# Compute metrics per syllable
def metrics(observations_per_id):
    syllable_count = []
    syllables_list = []
    avg_dev_frn = []
    avg_dev = []
    avg_matra_durn = []
    frn_dev_list = []
    taal_marks = []
    id_list = []
    for id in observations_per_id: #to get complete bandish plot, replace observations_per_id with bandish_order
        lines = observations_per_id[id]
        id_list.append(id)
        syllables_list.append(lines[0][0][1])
        syllable_count.append(len(lines))
        taal_marks.append(lines[0][1][1])
        sam_dev_frn = 0
        sam_dev = 0
        sam_matra = 0
        frn_devs = []
        for i in range(len(lines)):
            sam_dev_frn+=lines[i][0][3]
            sam_dev+=lines[i][0][2]
            sam_matra+=lines[i][1][4]
            frn_devs.append(lines[i][0][3])
        frn_dev_list.append(frn_devs)
        avg_dev_frn.append(sam_dev_frn/len(lines))
        avg_dev.append(sam_dev/len(lines))
        avg_matra_durn.append(sam_matra/len(lines))
    return syllable_count, syllables_list, avg_dev_frn, avg_dev, avg_matra_durn, frn_dev_list, taal_marks, id_list

# Get list of frn_devns for a syllable
def get_syl_devns(observations_per_id, syllable_id):
    syl_devns = []
    for item in observations_per_id[syllable_id]:
        syl_devns.append(item[0][3])
    return syl_devns

# replace -inf with nearest non -inf value in the resampled list
def replace_infs(lst):
    lst = [float('-inf') if x == '-inf' else x for x in lst]  # unify type
    
    # left to right
    for i in range(1, len(lst)):
        if lst[i] == float('-inf'):
            lst[i] = lst[i-1]
    
    # right to left
    for i in range(len(lst)-2, -1, -1):
        if lst[i] == float('-inf'):
            lst[i] = lst[i+1]
    
    return lst


#-------------------------------------------------------------------------------------------------------

# Get bandish specific notes, cents, syllables and ids
def get_bandish_notes(bandish):
    if "ja_jare" in bandish:
        note_indices = [0, 2, 3, 5, 7, 9, 10, 12, 14, 15, 17, 19, 21, 22, 24, 26, 27, 29, 31]
        note_cents_eq = [-1200, -1000, -900, -700, -500, -300, -200, 0, 200, 300, 500, 700, 900, 1000, 1200, 1400, 1500, 1700, 1900]
        syl_list = ["जा1", "जा2", "रे", "अ", "प", "ने", "मं", "दि1", "र", "वा"]
        id_list = ["C1L1B7", "C1L2B9", "C1L2B11", "C1L2B13", "C1L2B14", "C1L2B15", "C2L1B1", "C2L1B3", "C2L1B4", "C2L1B5"]
        syl_id_dict = dict(zip(id_list, syl_list))
    elif "yeri_aali" in bandish:
        note_indices = [0, 2, 4, 6, 7, 9, 11, 12, 14, 16, 18, 19, 21, 23, 24, 26, 28, 30, 31]
        note_cents_eq = [-1200, -1100, -900, -700, -600, -400, -200, 0, 200, 400, 600, 700, 900, 1100, 1200, 1400, 1500, 1700, 1900]
        syl_list = ["स", "खी", "ए", "री1", "आ", "ली", "पि1", "या1", "बि", "न1"]
        id_list = ["C2L1B7", "C2L1B8", "C1L2B9", "C1L2B11", "C1L2B13", "C1L2B15", "C2L1B1", "C2L1B2", "C2L1B3", "C2L1B4"]
        syl_id_dict = dict(zip(id_list, syl_list))
    else:
        raise ValueError("Bandish not recognized. Please use 'ja_jare' or 'yeri_aali'.")
    return note_indices, note_cents_eq, syl_list, id_list, syl_id_dict  

#-------------------------------------------------------------------------------------------------------

# bootstrap sampling with noise for sampling new deviation from the distribution
def sample_with_jitter(data, jitter_std=0.01):
    """
    Bootstrap resampling + Gaussian jitter.
    
    Args:
        data (list or np.array): observed deviations
        jitter_std (float): standard deviation of Gaussian noise
    
    Returns:
        float: new sampled deviation
    """
    data = np.array(data)
    
    # bootstrap resample: pick one of the observed points
    base = np.random.choice(data)
    
    # add jitter
    jitter = np.random.normal(0, jitter_std)
    
    return base + jitter

# gmm sampling from the deviation distribution, bic for getting no. of gaussians to fit
def sample_from_gmm(deviations, max_components=5):
    data = np.array(deviations).reshape(-1, 1)

    # Fit GMM with BIC to choose best number of components
    lowest_bic, best_gmm = np.inf, None
    for k in range(1, min(max_components, len(data)) + 1):
        gmm = GaussianMixture(n_components=k, random_state=None).fit(data)
        bic = gmm.bic(data)
        if bic < lowest_bic:
            lowest_bic, best_gmm = bic, gmm

    # Sample a new deviation (random each time)
    new_dev = best_gmm.sample(1)[0].item()
    return new_dev