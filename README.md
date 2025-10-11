# Melodic and Metrical Elements of Expressiveness in Hindustani Vocal Music

This repository contains the code, data (or links to data), and supplementary materials for the paper "Melodic and Metrical Elements of Expressiveness in Hindustani Vocal Music," accepted at the 26th International Society for Music Information Retrieval Conference (ISMIR 2025) in Daejeon, South Korea.

**Authors:** Yash Bhake, Ankit Anand, Preeti Rao
**Affiliation:** Digital Audio Processing Lab, Department of Electrical Engineering, Indian Institute of Technology Bombay, India

## Table of Contents

1.  [Introduction](#1-introduction)
2.  [Paper and Presentation](#2-paper-and-presentation)
3.  [Dataset](#3-dataset)
4.  [Project Structure](#4-project-structure)
5.  [Setup and Installation](#5-setup-and-installation)
    * [Prerequisites](#prerequisites)
    * [Environment Setup](#environment-setup)
    * [External Tools](#external-tools)
6.  [Usage and Reproducibility](#6-usage-and-reproducibility)
    * [Generative Experiment](#generative-experiment)
7.  [Supplementary Materials](#7-supplementary-materials)
8.  [Acknowledgments](#8-acknowledgments)
9.  [Citation](#9-citation)
10. [License](#10-license)
11. [Contact](#11-contact)

## 1. Introduction

This research explores the aesthetics of North Indian Khayal music by studying the flexibility artists employ in performing popular compositions. We analyze expressive timing and pitch variations of lyrical content within and across performances. Our work proposes computational representations capable of discriminating between different performances of the same song based on their expressive qualities and thereby creating grounds for generative models capturing expressiveness. The repository provides the necessary audio processing, annotation procedures, and analysis scripts to reproduce our observations and insights derived from a dataset of two songs in two ragas, each rendered by multiple prominent artists.

## 2. Paper and Presentation

* **Read full paper (ISMIR 2025) on ArXiv:** [here](https://arxiv.org/abs/2508.04430)
<!-- * **ArXiv: **  -->


## 3. Dataset

Our dataset comprises concert recordings of two popular Hindustani classical compositions: "Ja Ja Re" (in Raga Bhimpalasi) and "Yeri Aali" (in Raga Yaman). Each composition features multiple performances by several prominent artists.

| Raga        | Bandish    | Taal     | Swar (Notes)           | # Concerts | # Artists | # Repetitions (L1, L2, L3, L4) | Matra per min range |
| :---------- | :--------- | :------- | :--------------------- | :--------- | :-------- | :----------------------------- | :------------------ |
| Bhimpalasi  | Ja Ja Re   | Teentaal | S, R, g, m, P, D, n, S | 15         | 15        | 167, 39, 47, 47                | 138-200             |
| Yaman       | Yeri Aali  | Teentaal | S, R, G, M, P, D, N, S | 13         | 12        | 94, 32, 35, 23                 | 111-203             |

**Data Availability:**
Due to copyright and file size constraints, the raw audio files are not directly included in this repository. The YouTube video links for the audios is provided in the supplementary.
* **Raw Audio:** Please contact the authors (see [Contact](#11-contact) section) for access to the full dataset of concert recordings.
* **Processed Audio & Annotations:** Processed audio segments (vocal stems, segmented bandish lines) and all generated annotation files (syllable onsets, beat timings, pitch contours, PAA strings) will be made available upon request or through a dedicated download link (to be updated here).
* **Canonical Notation:** The machine-readable CSV files for the Bhatkhande notation are available in [`data/canonical_notation/`](./data/canonical_notation/).

## 4. Project Structure

```
.
├── data/
│   ├── bhimpalasi_ja_jare/             # bandish #1
│   ├── yaman_yeri_aali/                # bandish #2
│   ├── syllable_annotations/           # syllable level annotations (syllables as per canonical notation)
│   ├── salient_beat_annotations/       # Beat annotations (salient beat onsets)
│   └── canonical_notations/            # Machine-readable Bhatkhande notation (CSV files)
├── src/                                # All source code files - Python scripts and Jupyter notebooks
├── .gitignore                          # Specifies intentionally untracked files to ignore
├── requirements.txt                    # List of Python dependencies
├── LICENSE                             # License file
└── README.md                           # This file
```

## 5. Setup and Installation

### Prerequisites

* **Python 3.10**
* **Git**
* **Conda (Recommended)** or `pip`

### Environment Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Yash-Bhake/Performance-Expressiveness-in-HCM.git
    cd Performance-Expressiveness-in-HCM 
    ```

2.  **Create and activate a Conda environment (recommended):**
    ```bash
    conda env create -f environment.yml
    conda activate hindustani-expressiveness
    ```
    *If you prefer `pip`:*
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

### External Tools

Several components of our pipeline rely on external tools. Please ensure these are installed and configured correctly:

* **Praat:** Used for fundamental frequency (pitch) extraction and general audio manipulation.
    * Download: [https://www.fon.hum.uva.nl/praat/](https://www.fon.hum.uva.nl/praat/)
    * The `praat-parselmouth` Python library (included in `requirements.txt`) provides a convenient interface.
* **Kaldi Speech Recognition Toolkit:** Used for forced alignment (Hindi acoustic model).
    * Installation instructions: [https://kaldi-asr.org/doc/install.html](https://kaldi-asr.org/doc/install.html)
    * You will need to download and set up a Hindi acoustic model. Refer to Kaldi's official documentation for available models and setup. Our pipeline uses a Kaldi TDNN Hindi speech trained acoustic model [11].
* **OpenAI Whisper:** Used for speech-to-text conversion. The Python package `whisper` is included in `requirements.txt`.
* **Gaudiolab (Source Separation):** [https://www.gaudiolab.com/gaudio-studio](https://www.gaudiolab.com/gaudio-studio) Webtool was used for Vocal - accompaniments separation. API for this tool had not been released at the time of this research. If you do not have access, you may substitute it with an open-source alternative like Spleeter or Demucs, though results may vary.
    * Spleeter: [https://github.com/deezer/spleeter](https://github.com/deezer/spleeter)
    * Demucs: [https://github.com/facebookresearch/demucs](https://github.com/facebookresearch/demucs)
 
## 6. Usage and Reproducibility

### Generative Experiment

A preliminary experiment was conducted to generate temporal deviations and synthesize sine-tone based audio.
* Script: `src/generation/generate_audio_from_deviations.py`
    * This script uses the syllable pitch contours as well as the distributions of syllable deviations to generate new variations capturing the artists' style during that performance. A Genetic algorithm is used (presently being developed) to generate new pitch variations. 

* Generated audio examples are available in the [Supplementary Materials](#7-supplementary-materials).

## 7. Supplementary Materials

The `supplementary/` directory contains:
* Additional audio examples demonstrating expressive timing and pitch variations.
* Generated audio tracks from the generative experiment (canonical vs. generated).
* Further plots and data not included in the main paper due to space limitations.

Access the supplementary document referred in the paper [here](https://glamorous-nation-4cb.notion.site/Melodic-and-Metrical-Elements-of-Expression-in-Hindustani-Vocal-Music-ISMIR-2025-20619683367380caa928fe9dc55a2dd4?pvs=74).

## 8. Acknowledgments

We extend our sincere gratitude to Madhumitha S. for her foundational thesis work, and to Mr. Himanshu Sati and Mrs. Hemala Ranade for their invaluable musicological insights.

## 9. Citation

If you use this code or data in your research, please cite our paper:

```bibtex
@inproceedings{bhake2025melodic,
  author    = {Bhake, Yash and Anand, Ankit and Rao, Preeti},
  title     = {Melodic and Metrical Elements of Expressiveness in Hindustani Vocal Music},
  booktitle = {Proceedings of the 26th International Society for Music Information Retrieval Conference (ISMIR)},
  address   = {Daejeon, South Korea},
  year      = {2025}
}
```

## 10. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 11. Contact

For any questions, issues, or data access requests, please open an issue on this GitHub repository or contact the authors:
* Yash Bhake: `yash.bhake@iitb.ac.in, yashbhake1@gmail.com`
* Ankit Anand: `ankit0.anand0@gmail.com`
* Preeti Rao: `prao@ee.iitb.ac.in`
