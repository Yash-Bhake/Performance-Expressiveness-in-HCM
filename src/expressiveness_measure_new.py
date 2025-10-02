# -*- coding: utf-8 -*-
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd
import os

import utils2
import pandas as pd
import warnings
from matplotlib.font_manager import FontProperties
import regex as re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')



# Ignore all warnings
warnings.filterwarnings("ignore")


#----------------------------------------------------------------------------------------------------------------------------------------

def expressive_timing(folder, artist, prom=0.12, inter_onset_threshold=40, manual=True):

    parent_dir = os.path.dirname(os.getcwd())
    base_folder = os.path.join(parent_dir, 'All_audio_files_16kHz')

    def get_file_paths(folder: str, artist: str):
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

    files = get_file_paths(folder, artist)


    file_name = f"{'_'.join(folder.split('_')[1:])}_{artist}"
    print(file_name)
    for i in files:
        name = i.split("/")[-1]
        if name == file_name+"_trimmed.wav": audio_file = i
        if name == file_name+"_segments.TextGrid": segments_tg = i
        if name == file_name+"_segments.txt": segments_txt = i
        if name == file_name+"_line_labels.txt": line_labels = i
        if name == file_name+"_labels.TextGrid" or name == file_name+"_syllables_corrected.TextGrid": labels_tg = i
        if name == file_name+"_raw_syllables.TextGrid": raw_labels_tg = i
        if name == file_name+"_trimmed_gaudiolab_other.wav": audio_ss_other = i
        if name == file_name+"_trimmed_gaudiolab_vocal.wav": audio_ss_vocal = i
        if name == file_name+"_tabla.TextGrid": tabla_tg = i
        if name == file_name+"_crepe_f0.pkl": f0_crepe_pkl = i

    y, sr = librosa.load(audio_ss_vocal, sr = 16000)
    duration = librosa.get_duration(y=y, sr=sr)
    start_time = 0
    end_time = 20

#----------------------------------------------------------------------------------------------------------------------------------------
    #manual syllable onsets

    tg_path = labels_tg
    manual_onsets = np.array(utils2.textgrid_to_onsets(tg_path, tier_number = 0)[0]) 
    manual_labels = np.array(utils2.textgrid_to_onsets(tg_path, tier_number = 0)[1])
    manual_onsets = np.round(manual_onsets, 2) 

    def replace_syllables1(syllables):
        result, i = [], 0
        
        while i < len(syllables):
            seq = syllables[i:i + 5]

            # Handle all replacement rules with pattern matching
            replacements = {
                ("जा", "जा"): ["जा1", "जा2"],
                # ("न", "न"): ["न2", "न3"],
                ("सु", "न", "पा"): ["सु", "न1", "पा"],
                ("सु", "न", "हो", "स", "दा"): ["सु", "न4", "हो", "स", "दा"],
                ("सु", "न", "स", "दा"): ["सु", "न4", "स", "दा"],
                # ("छ", "ग", "न"): ["छ", "ग", "न5"],
                ("मं", "दि", "र"): ["मं", "दि1", "र"],
                ("मं", "दि"): ["मं", "दि1"],
                ("दि", "र"): ["दि1", "र"],
                ("न", "न", "दि", "या"): ["न2","न3", "दि2", "या"],
                ("न", "न", "दि"): ["न2", "न3", "दि2"],
                ("ग", "न", "दि", "या"): ["ग", "न5", "दि3", "या"],
                ("ग", "न", "दि"): ["ग", "न5", "दि3"],
            }
            matched = False
            for pattern, replacement in replacements.items():
                if seq[:len(pattern)] == list(pattern):
                    result.extend(replacement)
                    i += len(pattern)
                    matched = True
                    break

            if matched:
                continue

            # Handle contextual "जा" replacements
            if syllables[i] == "जा":
                prev, next = result[-1] if result else None, syllables[i + 1] if i + 1 < len(syllables) else None
                result.append("जा1" if prev in ["या", "वा"] and next != "रे" else "जा2" if next == "रे" else "जा")
            else:
                result.append(syllables[i])

            i += 1

        return result
    
    def replace_syllables2(syllables):
        result, i = [], 0
        
        while i < len(syllables):
            seq = syllables[i:i + 5 ]

            # Handle all replacement rules with pattern matching
            replacements = {
                ("ये", "री"): ["ए", "री1"],
                ("ए", "री"): ["ए", "री1"],
                ("पि", "या", "बि", "न"): ["पि1", "या1", "बि", "न1"],
                ("री", "आ"): ["री1", "आ"],
                ("छि", "न"): ["जि", "न"],
                ("छि"):["जि"],
                ("दि"): ["दी"],
                ("तें"):["से"],
                ("दे", "स"): ["दे", "श"],
                ("क", "ल", "न"): ["क", "ल", "ना"],
            }
            matched = False
            for pattern, replacement in replacements.items():
                if seq[:len(pattern)] == list(pattern):
                    result.extend(replacement)
                    i += len(pattern)
                    matched = True
                    break

            if matched:
                continue

            if syllables[i:i + 3] == ["ली", "पि", "या"]:
                if syllables[i + 3:i + 5] != ["बि", "न"]:
                    result.extend(["ली", "पि1", "या1"])
                    i += 3
                    continue

            result.append(syllables[i])

            i += 1

        return result

    if "ja_jare" in file_name:
        manual_labels = replace_syllables1(list(manual_labels))
    elif "yeri_aali" in file_name:
        manual_labels = replace_syllables2(list(manual_labels))


#----------------------------------------------------------------------------------------------------------------------------------------
    #manual tabla onsets

    tg_path2 = tabla_tg
    manual_tabla_onsets = np.array(utils2.textgrid_to_onsets(tg_path2, tier_number=0)[0]) 
    manual_tabla_labels = np.array(utils2.textgrid_to_onsets(tg_path2, tier_number=0)[1]) 
    manual_tabla_onsets = np.round(manual_tabla_onsets, 2) # to be according to the hop size 0.01s resolution

    #calculating interval between the consecutive sams

    inter_sam_interval = []
    for i in range(0, len(manual_tabla_onsets)-2, 2):
        inter_sam_interval.append(manual_tabla_onsets[i+2]-manual_tabla_onsets[i])
        
    inter_sam_interval = np.array(inter_sam_interval)

    mean_sam_interval = np.mean(inter_sam_interval)
    one_matra_interval = mean_sam_interval/16
    mpm = 1/(one_matra_interval/60)
    print("duration of one taal cycle: ", round(mean_sam_interval, 2), "s")
    print("duration of one matra: ", round(one_matra_interval, 2), "s")
    print("mpm: ", round(mpm, 2))

    manual_tabla_onsets = list(manual_tabla_onsets)
    manual_tabla_onsets.append(round((manual_tabla_onsets[0] - one_matra_interval*8),2))
    manual_tabla_onsets.sort()
    manual_tabla_onsets = np.array(manual_tabla_onsets)

    if manual_tabla_labels[0]=='x':
        manual_tabla_labels = list(manual_tabla_labels)
        manual_tabla_labels.insert(0, 'o')
    else:
        manual_tabla_labels = list(manual_tabla_labels)
        manual_tabla_labels.insert(0, 'x')

    # print(manual_tabla_onsets)

    std_sam_interval = np.std(inter_sam_interval)
    print("standard deviation of the duration of taal cycles ", round(std_sam_interval, 2), "s")

    #add manual onsets at each beat in the cycle:

    beat_manual_onsets = manual_tabla_onsets


    for i in range(0, len(manual_tabla_onsets)-1):
        diff = manual_tabla_onsets[i+1] - manual_tabla_onsets[i]
        beat = diff/8
        for j in range(1, 8):
            beat_manual_onsets = np.append(beat_manual_onsets, manual_tabla_onsets[i] + beat*j)
    # for j in range(1, 8):
    #     beat_manual_onsets = np.append(beat_manual_onsets, manual_tabla_onsets[0] - one_matra_interval*j)
    beat_manual_onsets.sort()
    len(beat_manual_onsets)
    # beat_manual_onset
    beat_manual_onsets = np.round(beat_manual_onsets, 2)


#----------------------------------------------------------------------------------------------------------------------------------------

#getting the lyrics lines

    if "ja_jare" in file_name:
        df = pd.read_csv(base_folder + "/bhimpalasi_ja_jare/ja_jare_bhimpalasi.csv")
    elif "yeri_aali" in file_name:
        df = pd.read_csv(base_folder + "/yaman_yeri_aali/yeri_aali_yaman.csv")

    df1 = df['Label']=='Lyrics'
    lines = []
    for i in range(1, df.shape[0]):
        if df1.iloc[i]:
            lines.append(list(df.iloc[i].to_numpy()[1:]))

    lines = [[str(x).strip() if str(x) != 'nan' else '' for x in sublist] for sublist in lines]

#----------------------------------------------------------------------------------------------------------------------------------------\
    #breaking down into 8 beat parts, and assigning them line ids

    lines_div = []
    for i in range(len(lines)):
        l1 = lines[i][:8]
        l2 = lines[i][8:]
        lines_div.append([l1, l2])
    flattened_lines_div = [line for item in lines_div for line in item ]
    flattened_lines_div = [line for line in flattened_lines_div if line]

    flattened_lines_div

    flattened_lines_div_indices = {}
    for i in range(len(flattened_lines_div)):
        flattened_lines_div_indices[f'C{(i//2)+1}L{(i%2)+1}'] = flattened_lines_div[i]

    def transform_dict(input_dict):
        transformed_dict = {}

        for key, values in input_dict.items():
            transformed_values = []
            for i, val in enumerate(values):
                if val == '':
                    transformed_values.append(0)
                elif val == 's':
                    transformed_values.append(0)
                elif i < len(values) - 1 and values[i + 1] == 's':
                    transformed_values.append(2)
                else:
                    transformed_values.append(1)
            
            transformed_dict[key] = transformed_values

        return transformed_dict
    
    def create_block_dict(transformed_dict):
        block_dict = {}

        for key, values in transformed_dict.items():
            line = key[-2:]  # Extract L1 or L2
            for i, val in enumerate(values):
                block_num = i + 1 if line == 'L1' else i + 9
                block_key = f"{key}B{block_num}"
                block_dict[block_key] = val
        
        return block_dict
    alotted_beats =  create_block_dict(transform_dict(flattened_lines_div_indices))

#----------------------------------------------------------------------------------------------------------------------------------------

    #for splitting multiple syllables in a single block acc to the notation

    def split_grapheme_clusters(s):
        # Match Devanagari grapheme clusters, keeping syllables with trailing numbers intact
        grapheme_clusters = re.findall(r'\X(?:\d+)?', s)
        return grapheme_clusters


    def get_line_from_list(line):
        canonical_half_line = [char for string in line for char in split_grapheme_clusters(string.replace('s', '')) if set(string)!={'s'}]
        canonical_half_line = ['']*8 if not canonical_half_line else canonical_half_line
        return canonical_half_line

    #gets just the syllables in the canonical line\

#----------------------------------------------------------------------------------------------------------------------------------------

    #Vibhags

    vibhag_onsets = [i[0] for i in [beat_manual_onsets[i:i+4] for i in range(0, len(beat_manual_onsets), 4)]]
    #getting labels for 2nd and 3rd vibhags
    vibhag_labels = []
    for ctr, i in enumerate(manual_tabla_labels):
        vibhag_labels.append(i)
        if ctr==len(manual_tabla_labels)-1:
            break
        if vibhag_labels[-1]=='x':
            vibhag_labels.append('२')
        else:
            vibhag_labels.append('३')

#----------------------------------------------------------------------------------------------------------------------------------------
    #more lists

    manual_label_dict = dict(zip(manual_onsets, manual_labels))
    manual_label_list = [[i, j] for i, j in zip(manual_onsets, manual_labels)]

    manual_tabla_labels_dict = dict(zip(manual_tabla_onsets, manual_tabla_labels))
    manual_tabla_labels_list = [[i, j] for i, j in zip(manual_tabla_onsets, manual_tabla_labels)]

    vibhag_labels_dict = dict(zip(vibhag_onsets, vibhag_labels))
    vibhag_labels_list = [[i, j] for i, j in zip(vibhag_onsets, vibhag_labels)]


    tabla_beats_list = []
    for onset in beat_manual_onsets:
        if onset in vibhag_labels_dict:
            tabla_beats_list.append([onset, vibhag_labels_dict[onset], ""])
        else:
            tabla_beats_list.append([onset, "", ""])
    tabla_beats_list
    tabla_intervals = [tabla_beats_list[i:i+8] for i in range(0, len(tabla_beats_list), 8)]

    sam_to_khali_intervals = [tabla_intervals[i] for i in range(len(tabla_intervals)) if tabla_intervals[i][0][1]=='x' ]
    khali_to_sam_intervals = [tabla_intervals[i] for i in range(len(tabla_intervals)) if tabla_intervals[i][0][1]=='o' ]

    intervals = []
    for i in range(len(tabla_intervals)-1):
        a = tabla_intervals[i][0][0]-(one_matra_interval*2).round(2) # change this number 4 accordingly
        b = tabla_intervals[i+1][0][0]+(one_matra_interval*2).round(2)
        intervals.append([a.round(3), b.round(3)])

    flattened_intervals = [i for item in intervals for i in item]

#----------------------------------------------------------------------------------------------------------------------------------------
    #manual lines in each half cycle

    lines_manual = []
    for i in intervals:
        start, end = i
        onsets1= [manual_onsets[j] for j in range(len(manual_onsets)) if manual_onsets[j]>start and manual_onsets[j]<end]
        onsets1_labels = [manual_label_dict[a] for a in onsets1]
        lines_manual.append([[p, q] for p, q  in zip(onsets1, onsets1_labels)])

    lines_manual_labels = [[item[1] if item else item for item in sublist] for sublist in lines_manual]

#----------------------------------------------------------------------------------------------------------------------------------------

    #mapping lines

    # Function to get sliding windows from a list of syllables
    def get_sliding_windows(syllables, window_size):
        if len(syllables)<window_size:
            return [syllables]
        else:
            return [syllables[i:i + window_size] for i in range(len(syllables) - window_size + 1)] # hop = 1

    # Function to calculate similarity using sliding windows
    def calculate_similarity(instance_list, reference_lists, window_size=3):
        # Convert the input lists to a single list of strings
        phrase_syllables = instance_list
        reference_syllables_lists = [get_line_from_list(i) for i in reference_lists]

        
        # Initialize similarity scores
        similarity_scores = []

        # For each reference sentence
        for ref_syllables in reference_syllables_lists:
            # Generate sliding windows for the phrase and the reference
            phrase_windows = get_sliding_windows(phrase_syllables, window_size)
            ref_windows = get_sliding_windows(ref_syllables, window_size)
            
            # Vectorize the sliding windows
            
            all_windows = [' '.join(window) for window in (phrase_windows + ref_windows)]
            vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b").fit_transform(all_windows)
            vectors = vectorizer.toarray()
            
            # Calculate cosine similarity between each sliding window pair
            # as in all windows, all the instance and reference windows are concatenated, vectors[:len(phrase_windows)] 
            # gives vectors for all instance windows and vectors[len(phrase_windows):] gives vectors for all reference windows.
            phrase_vector = vectors[:len(phrase_windows)]
            ref_vector = vectors[len(phrase_windows):]
            # print(phrase_vector)
            # print(ref_vector)
            
            # Compute the average similarity for the windows
            window_similarities = cosine_similarity(phrase_vector, ref_vector) #cosine similarity for all the windows. 
            # gives a 2d matrix with (i, j) as the cosine similarity betweenthe ith window in phrase_vector with the jth window in ref_vector
            max_similarities = window_similarities.max(axis=1)  # Best matching ref window for each phrase window. (take max across columns for each row(each instance window))
            avg_similarity = max_similarities.mean()  # Average across all windows - gives the final score for an instance wrt a reference phrase.
            
            # Append the similarity score
            similarity_scores.append(avg_similarity)
        
        return similarity_scores

    lines_manual_labels_canonical = []

    for phrase in lines_manual_labels:
        if not phrase:
            lines_manual_labels_canonical.append(['']*8)
        else:
            similarity_scores = calculate_similarity(phrase, flattened_lines_div)
            # print(similarity_scores)
            lines_manual_labels_canonical.append(flattened_lines_div[np.argmax(similarity_scores)])

    lines_manual_labels_canonical_flat = [string for list in lines_manual_labels_canonical for string in list]

    canonical_onsets_labels = tabla_intervals[:-1]
    canonical_onsets_labels[-3:]

    canonical_onsets_labels = [[row[:2] + [canon[i]] for i, row in enumerate(halfcycle)] for halfcycle, canon in zip(canonical_onsets_labels, lines_manual_labels_canonical)]

    # # canonical_onsets_labels
    # for a, b in zip(lines_manual_labels, lines_manual_labels_canonical):
    #     print(a, b)


#----------------------------------------------------------------------------------------------------------------------------------------

    canonical_onsets_labels_split = []
    for ctr, cycle in enumerate(canonical_onsets_labels):
        new_cycle = []
        for i, line in enumerate(cycle):
            start = line[0]
            if i==len(cycle)-1:
                if ctr==len(canonical_onsets_labels)-1:
                    diff = cycle[i][0] - cycle[i-1][0]
                    next = start+diff
                else:
                    next_cycle = canonical_onsets_labels[ctr+1]
                next = next_cycle[0][0]
            else:
                next = cycle[i+1][0]
            graphemes = split_grapheme_clusters(line[-1])
            
            if len(graphemes)>1:
                for ctr1 in range(len(graphemes)):
                    if ctr1==0:
                        new_cycle.append([start, line[1], graphemes[ctr1]])
                    else:
                        new_cycle.append([(start+ctr1*((next-start)/len(graphemes))).round(3), '', graphemes[ctr1]])

            else:
                new_cycle.append(line)
        canonical_onsets_labels_split.append(new_cycle)



    canonical_onsets_labels_split
    flat_canonical_onsets_labels_split = [line for sublist in canonical_onsets_labels_split for line in sublist]
    split_onsets = [i[0] for i in flat_canonical_onsets_labels_split]
    split_onsets_labels = [i[2] for i in flat_canonical_onsets_labels_split]

#----------------------------------------------------------------------------------------------------------------------------------------

    for i in flattened_lines_div_indices:
        line = []
        for j in flattened_lines_div_indices[i]:
            graphemesss = split_grapheme_clusters(j)
            if not graphemesss:
                line.append('')
            else:
                for k in graphemesss:
                    line.append(k)
        if not line:
            flattened_lines_div_indices[i] = ['']*8
        else:
            flattened_lines_div_indices[i] = line

    for sent in canonical_onsets_labels_split:
        line = [i[2] for i in sent]
        label = [key for key, value in flattened_lines_div_indices.items() if value==line][0]
        # print(label)
        for j, item in enumerate(sent):
            beat_length = sent[1][0]-sent[0][0]
            if int(label[3])==1:
                item.append(str(label)+f'B{j+1}')
            if int(label[3])==2:
                item.append(str(label)+f'B{j+9}')
            item.append(round(beat_length,3))


#----------------------------------------------------------------------------------------------------------------------------------------

    #mapping the syllables

    def find_match_indices(lst, ref_lst):
        n, m = len(lst), len(ref_lst)
        
        max_len = 0
        start_lst_idx = -1
        start_ref_idx = -1
        longest_match_indices = [] 
        
        # Try to find the longest common sublist with possible skips in lst
        i = 0
        while i < n:
            j = 0
            while j < m:
                # Start checking for contiguous matches
                if lst[i] == ref_lst[j]:
                    lst_temp_idx = i
                    ref_temp_idx = j
                    k = 0
                    matched_indices = []
                    
                    # Iterate while lst and ref_lst are matching, allowing skips in lst
                    while ref_temp_idx < m:
                        if lst_temp_idx < n and lst[lst_temp_idx] == ref_lst[ref_temp_idx]:
                            matched_indices.append((lst_temp_idx, ref_temp_idx))
                            lst_temp_idx += 1
                        else:
                            # If lst element is missing, map the ref element to 'NA'
                            matched_indices.append(('NA', ref_temp_idx))
                        ref_temp_idx += 1
                        k += 1
                    
                    # Update if this match is the longest
                    if k > max_len:
                        max_len = k
                        start_lst_idx = i
                        start_ref_idx = j
                        longest_match_indices = matched_indices
                j += 1
            i += 1
        
        # Create a result list for indices, adding 'NA' where no match was found
        result_indices = []
        
        # For lst items before the match (if any)
        for i in range(start_lst_idx):
            result_indices.append((i, 'NA'))
        
        # Append the longest match with possible skips
        if start_lst_idx != -1:
            result_indices.extend(longest_match_indices)
        
        # For lst items after the match (if any)
        for i in range(start_lst_idx + len(longest_match_indices), n):
            result_indices.append((i, 'NA'))
        
        # For ref_lst items before and after the match (if any)
        for j in range(start_ref_idx):
            result_indices.append(('NA', j))
        
        for j in range(start_ref_idx + len(longest_match_indices), m):
            result_indices.append(('NA', j))
        
        return result_indices



    split_onsets_lines_labels = [[line[2] for line in cycle] for cycle in canonical_onsets_labels_split]
    split_onsets_lines = [[line[0] for line in cycle] for cycle in canonical_onsets_labels_split]

    label_map = []


    for ctr in range(len(lines_manual)):
        manual_line = [line[1] for line in lines_manual[ctr]]
        canonical_line = [line[2] for line in canonical_onsets_labels_split[ctr] if line[2]!='s']
        canonical_line_full = [line for line in canonical_onsets_labels_split[ctr] if line[2]!='s']
        if manual_line == [] or canonical_line == ['']*8:
            continue
        else:
            cycle = []
            match_indices = find_match_indices(manual_line, canonical_line)
            for i, j in match_indices:
                if i=='NA':
                    cycle.append(['NA', canonical_line_full[j]])
                elif j=='NA':
                    cycle.append([lines_manual[ctr][i], 'NA'])
                else:
                    cycle.append([lines_manual[ctr][i], canonical_line_full[j]])
            label_map.append(cycle)

    flat_label_map = [line for sublist in label_map for line in sublist]



#----------------------------------------------------------------------------------------------------------------------------------------


    for manual, canon in flat_label_map:
        if manual!='NA' and canon!='NA':
            deviation = round(manual[0] - canon[0], 2)
            manual.append(deviation)
            manual.append(round(deviation/canon[4], 2))

    label_map_no_NA = [line for line in flat_label_map if line[0] != 'NA' and line[1] != 'NA']
    
    #to get unique mappings for each manual onset
    unique_dict = {}
    for manual, canon in label_map_no_NA:
        timestamp = manual[0]
        devn = manual[3]
        if timestamp not in unique_dict:
            unique_dict[timestamp] = [manual, canon]
        else:
            if abs(devn) < unique_dict[timestamp][0][3]:
                unique_dict[timestamp] = [manual, canon]
    label_map_no_NA = list(unique_dict.values())

    id_set = []

    for manual, canon in label_map_no_NA:
        id = canon[3]
        if not id in id_set:
            id_set.append(id)

    def sort_custom(strings):
        def custom_key(s):
            # Extracting the values of C, L, and B
            c = int(s[s.index('C') + 1:s.index('L')])  # Value after C and before L
            l = int(s[s.index('L') + 1:s.index('B')])  # Value after L and before B
            b = int(s[s.index('B') + 1:])              # Value after B
            return c * 100 + l * 10 + b
        return sorted(strings, key=custom_key)
    id_set = sort_custom(id_set)


#----------------------------------------------------------------------------------------------------------------------------------------

    observations_per_id = {}

    for id in id_set:
        for line in label_map_no_NA:
            if line[1][3] == id:
                if id in observations_per_id:
                    observations_per_id[id].append(line)
                else:
                    observations_per_id[id] = [line]


#----------------------------------------------------------------------------------------------------------------------------------------

    return observations_per_id, manual_onsets, manual_labels, len(manual_onsets), duration, split_onsets, split_onsets_labels, manual_tabla_onsets, manual_tabla_labels, vibhag_labels_list, mean_sam_interval, one_matra_interval, mpm, alotted_beats







































