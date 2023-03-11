from math import floor
import pympi
import cv2
import pandas as pd

directory = "C:/Users/matth/OneDrive/Documents/UQAI Internship/Code/MBCB1c2a.eaf"
eafob = pympi.Elan.Eaf(directory)

# Using the pympi Elan library to get the labeled annotations from the .eaf file.
listofstuff = list(eafob.get_gaps_and_overlaps('RH-IDgloss', 'LH-IDgloss'))

# We're interested in pauses/gaps in annotations, (i.e, the switches 
# from one sign to another).
pauses = []
for gaps in listofstuff:
    if gaps[2][0:2] == "P1" or gaps[2][0:2] == "P2":
        pauses.append(gaps)

print(pauses)


obj = pd.read_pickle("C:/Users/matth/OneDrive/Documents/UQAI Internship/Code/predictions.pkl")
print(obj)
print(len(obj))


cap = cv2.VideoCapture('C:/Users/matth/OneDrive/Documents/UQAI Internship/Code/video.mp4')

# Create a dictionary of frames and the corresponding timestamps
frame_no = 0
frames_and_times = dict()
frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for f in range(frame_total):
    frames_and_times[f] = -1

while(cap.isOpened()):
    frame_exists, curr_frame = cap.read()
    if frame_exists:
        frames_and_times[frame_no] = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    else:
        break
    frame_no += 1

cap.release()

print(frames_and_times)

# I only used part of the video, so we don't want to use all frames of
# the video. Hence, just use the frames that you ran the prediction on.
relevant_frames = dict(list(frames_and_times.items())[:len(obj)])

print(relevant_frames)
print("Frame Total: {}".format(frame_total))



max_time = relevant_frames[len(obj) - 1]
print(max_time)

# Remove irrelevant pause elements
for p in list(pauses):
    if p[0] > max_time and p[1] > max_time:
        pauses.remove(p)

print(pauses)

binary_labels = [0 for frame in range(len(relevant_frames))]

closest_frame_arr = []
for pause in pauses:
    min_value1 = 99999999
    min_value2 = 99999999
    idx_frame1 = -1
    idx_frame2 = -1
    for frame in range(len(relevant_frames)):
        if abs(pause[0] - relevant_frames[frame]) < min_value1:
            min_value1 = abs(pause[0] - relevant_frames[frame])
            idx_frame1 = frame
        if abs(pause[1] - relevant_frames[frame]) < min_value2:
            min_value2 = abs(pause[1] - relevant_frames[frame])
            idx_frame2 = frame
    closest_frame_arr.append((idx_frame1, idx_frame2))

print("closest: ", closest_frame_arr)

for closest_frames in closest_frame_arr:
    if closest_frames[0] == closest_frames[1]:
        binary_labels[closest_frames[0]] = 1
    else:
        i = closest_frames[0]
        while i <= closest_frames[1]:
            binary_labels[i] = 1
            i += 1

print(binary_labels)

i = 0
while i < len(binary_labels):
    if binary_labels[i] == obj[i] and binary_labels[i] == 1:
        print("successful at frame {}".format(i))
    i +=1

# Used to compute precision and recall evaluation metrics
truePositiveCount = 0
falsePositiveCount = 0
falseNegativeCount = 0

# Find true positives and false positives
i = 0
threshold = 4
while i < len(obj):
    if obj[i] == 1:
        predictedSum = 0
        predictedCount = 0
        while obj[i] == 1:
            predictedSum += i
            predictedCount += 1
            if (i + 1) >= len(obj) or obj[i + 1] == 0:
                break
            # Else, if next one is 1, continue
            i += 1
        if predictedCount > 0:
            predictedMean = predictedSum / predictedCount
            minRange = floor(predictedMean - threshold)
            maxRange = floor(predictedMean + threshold)
            successfulSegmentation = False
            j = minRange
            while j <= maxRange and j < len(binary_labels):
                if binary_labels[j] == 1:
                    successfulSegmentation = True
                    break
                j += 1
            if successfulSegmentation:
                truePositiveCount += 1
                print("TRUE POSITIVE SEGMENTATION AT MEAN: {}".format(predictedMean))
            else:
                falsePositiveCount += 1
                print("FALSE POSITIVE SEGMENTATION AT MEAN: {}".format(predictedMean))
    i += 1

# Find true negatives and false negatives
i = 0
threshold = 4
while i < len(binary_labels):
    if binary_labels[i] == 1:
        predictedSum = 0
        predictedCount = 0
        while binary_labels[i] == 1:
            predictedSum += i
            predictedCount += 1
            if (i + 1) >= len(binary_labels) or binary_labels[i + 1] == 0:
                break
            # Else, if next one is 1, continue
            i += 1
        if predictedCount > 0:
            predictedMean = predictedSum / predictedCount
            minRange = floor(predictedMean - threshold)
            maxRange = floor(predictedMean + threshold)
            successfulNonSegmentation = False
            j = minRange
            while j <= maxRange and j < len(obj):
                if obj[j] == 1:
                    successfulNonSegmentation = True
                    break
                j += 1
            if not successfulNonSegmentation:
                falseNegativeCount += 1
                print("FALSE NEGATIVE AT MEAN: {}".format(predictedMean))
    i += 1


# Print Evaluation Metrics
print("True Positives: {}".format(truePositiveCount))
print("False Positives: {}".format(falsePositiveCount))
print("False Negatives: {}".format(falseNegativeCount))

precision = truePositiveCount / (truePositiveCount + falsePositiveCount)
print("Precision: {}".format(precision))
recall = truePositiveCount / (truePositiveCount + falseNegativeCount)
print("Recall: {}".format(recall))

F1 = 2 * ((precision*recall) / (precision + recall))
print("F1 Score: {}".format(F1))