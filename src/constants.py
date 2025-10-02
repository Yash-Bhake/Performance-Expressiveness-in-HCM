
BPM = 120
SR_NOV = 100
SR = 16000
HOP_LENGTH = int(SR/SR_NOV)  # 10ms hop at 16kHz


# List of artist IDs for different datasets
# These IDs correspond to different singers in the dataset
ARTISTS_JA_JARE = ["ABD", "AC", "AK", "DG", "IN", "KS", "MB", "NG", "PB", "PT", "PTJ", "RK", "RV", "SA", "SZ"] 
ARTISTS_YERI_AALI = ["ABD", "AC", "IN", "JV", "KA", "KA2", "KS", "MB", "RH", "SA", "SHS", "SS", "SZ"]

# Dictionary to convert Hindi syllables to English representation
# Used for mapping manual annotations to syllable IDs
# Note: Some syllables have numbers to distinguish between different instances of the same syllable
# (e.g., "जा1" and "जा2" represent two different occurrences of the syllable "जा")
# This is important for accurate mapping in cases where the same syllable
ENG_DICT = {
    "जा1": "Jaa1", "जा2": "Jaa2", "रे": "Re", "अ": "A", "प": "Pa", "ने": "Ne", "मं": "Man",
    "दि1": "di1", "र": "Ra", "वा": "Waa", "सु": "Su", "न1": "Na1", "पा": "Paa", "वे": "We",
    "गी": "Gii", "सा": "Saa", "स": "Sa", "न2": "Na2", "न3": "Na3", "दि2": "di2", "या": "Yaa",
    "न4": "Na4", "हो": "Ho", "दा": "daa", "रं": "Ran", "ग": "Ga", "तु": "tu", "म": "Ma",
    "को": "Ko", "चा": "Chaa", "ह": "Ha", "त": "ta", "हैं": "Hain", "क्या": "Kyaa",
    "छ": "Chha", "न5": "Na5", "दि3": "di3", "s": "-", "२": "2", "३": "3",
    "ए": "Ye", "री1": "Rii1", "आ": "Aa", "ली": "Lii", "पि1": "Pi1", "या1": "Yaa1",
    "बि": "Bi", "पि": "Pi", "न": "Na", "री": "Rii", "खी": "Khi", "क": "Ka",
    "ल": "La", "ना": "Naa", "मो": "Mo", "हे": "He", "घ": "Gha", "जि": "Ji",
    "दी": "dii", "ज": "Ja", "ब": "Ba", "से": "Se", "दे": "de", "श": "Sha",
    "व": "Wa", "की": "Kii", "नो": "No", "ति": "ti", "याँ": "Yaan", "ट": "Ta",
    "ता": "taa", "गि": "Gi"
}

# Praat pitch extraction parameters
# These parameters can be adjusted based on the characteristics of the audio data
# and the desired accuracy of pitch detection
# Refer to the Praat documentation for detailed explanations of each parameter
# https://www.fon.hum.uva.nl/praat/manual/Pitch_Analysis.html
CONTROLS = {
	"time_step": float(1/SR_NOV),  # time step between consecutive pitch measurements (in seconds)
	"pitch_floor": 75.0,  # minimum pitch value to detect (in Hz), useful for filtering out low-frequency noise
	"max_number_of_candidates": 15,  # max number of pitch candidates per frame to evaluate
	"very_accurate": False,  # increases accuracy at the cost of performance
	"silence_threshold": 0.03,  # energy threshold to distinguish silence from voiced parts
	"voicing_threshold": 0.35,  # threshold for deciding whether a frame is voiced
	"octave_cost": 0.1,  # cost for selecting a pitch candidate an octave apart from the previous one
	"octave_jump_cost": 0.5 ,  # penalty for sudden jumps between octaves
	"voiced_unvoiced_cost": 0.4,  # cost for transitioning between voiced and unvoiced frames
	"pitch_ceiling": 600  # maximum pitch to detect (in hz)
}
