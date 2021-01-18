import csv

with open('resource/daily-facial-data.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    negativeEI = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
            continue
        # compute average negative facial emotion index
        negativeEI += float(row["Angry"]) + float(row["Contempt"]) + \
            float(row["Disgust"]) + float(row["Fear"]) + float(row["Sadness"])
        line_count += 1
    print(f'Negative Facial Emotion: ',
          "{:.2f}".format(negativeEI / line_count))
