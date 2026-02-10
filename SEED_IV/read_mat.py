import scipy.io as sio
import os
import numpy
import matplotlib as plt


print(" |    | PER-SUBJECT CLASS DISTRIBUTION ANALYSIS")
print(" ================================================")
print(" |  - |  NEUT  |  HAPP  |  SAD  |  FEAR")

session_label = [[1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
                 [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
                 [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]]

summation = [[0, 0, 0, 0] for _ in range(15)]

for session in range(3): # +1

    eeg_session_dir = os.path.join("eeg_feature_smooth", f'{session+1}')
    eye_session_dir = os.path.join("eye_feature_smooth", f'{session+1}')

    eeg_subjects = sorted([f for f in os.listdir(eeg_session_dir) if f.endswith(".mat")])
    eye_subjects = sorted([f for f in os.listdir(eye_session_dir) if f.endswith(".mat")])

    for subject in range(15): # +1
        eeg = sio.loadmat(os.path.join(eeg_session_dir, eeg_subjects[subject]))
        eog = sio.loadmat(os.path.join(eye_session_dir, eye_subjects[subject]))

        # print("|  - |   |  HAPPY |  SAD   |  NEUTRAL ")

        for trial in range(24): # +1

            if not(session==2 and trial==5 and subject==14):  
                eeg_dat = eeg[f'de_LDS{trial+1}']
                eog_dat = eog[f'eye_{trial+1}']

                summation[subject][session_label[session][trial]] += eeg_dat.shape[1]