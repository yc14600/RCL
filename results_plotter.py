import os
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

metrics = []
metric = []
metric_lines = []
metric_storage =[]
metric_container = []


classification_lines = []
classifications = []
classification = []
classification_storage = []
classification_container = []

names = []
number_of_tasks = 0
def get_classification_metrics(path):
    file = open(path)
    lines2 = file.read().splitlines()
    container = []
    for line in lines2:
        if "Top1_Acc_Exp/eval_phase/test_stream/Task000" in line:
            container.append(float(line.split("=")[1]))
    return container


for i in os.walk("runs"):
    folders = i[1]
    break
for name in folders:
    if os.path.exists("runs/"+name+"/results.txt"):
        names.append(name)
        classification_lines = get_classification_metrics("runs/"+name+"/log.txt")
        file1 = open("runs/"+name+"/results.txt")
        lines = file1.read().splitlines()
        for line in lines:
            metric_lines.append(float(line))

        number_of_tasks = int(math.sqrt(len(metric_lines)))
        for i in range(number_of_tasks):
            metric_storage.append(metric_lines[i*number_of_tasks:i*number_of_tasks+number_of_tasks])
            metric.append(sum(metric_lines[i*number_of_tasks:i*number_of_tasks+number_of_tasks])/number_of_tasks)
            classification_storage.append(classification_lines[i*number_of_tasks:i*number_of_tasks+number_of_tasks])
            classification.append(sum(classification_lines[i*number_of_tasks:i*number_of_tasks+number_of_tasks])/number_of_tasks)
        metrics.append(metric)
        classifications.append(classification)
        metric = []
        classification = []
        metric_container.append(metric_storage)
        classification_container.append(classification_storage)
        metric_storage = []
        classification_storage = []

    metric_lines = []

print(number_of_tasks)
#metrics is the average accuracy of all tasks at each run for all training methods
print(metrics[0])
print(classifications[0])
x = [i for i in range(1, number_of_tasks+1)]
print(x)
plt.figure()
data = []
columns = []
#for each training method


for i in range(len(metric_container)):
    intermediate = []
    for j in range(len(metric_container[i])):
        if len(intermediate) == 0:
            intermediate = metric_container[i][j]
        else:
            intermediate.extend(metric_container[i][j])
        intermediate.extend(classification_container[i][j])
    data.append(intermediate)

for i in range(number_of_tasks):
    for j in range(number_of_tasks):
        columns.append("Run " + str(i) + " Task" + str(j) + " loss")
    for j in range(0, number_of_tasks):
        columns.append("Run " + str(i) + " Task" + str(j) + " accuracy")


for i in range(number_of_tasks):
    for j in range(len(metric_container)):
        #first task accuracy for each training run for each training method
        y1 = [k[i] for k in metric_container[j]]
        plt.plot(x, y1, marker=".")
    plt.legend(names)
    plt.savefig("runs/metrics/task"+str(i)+"_reconstruction_loss.jpg")
    plt.figure()

#average accuracy for each training method

for i in range(len(metrics)):
    data[i].extend(metrics[i])

    #first task accuracy for each training run for each metric
    plt.plot(x, metrics[i], marker=".")
for i in range(number_of_tasks):
    columns.append("Run " + str(i) + " average loss")


plt.legend(names)
plt.savefig("runs/metrics/average_reconstruction_loss.jpg")
plt.figure()

for i in range(number_of_tasks):
    for j in range(len(classification_container)):
        #first task accuracy for each training run for each training method
        y1 = [k[i] for k in classification_container[j]]
        plt.plot(x, y1, marker=".")
    plt.legend(names)
    plt.savefig("runs/metrics/task"+str(i)+"_classification_accuracy.jpg")
    plt.figure()

#average accuracy for each training method

for i in range(len(classifications)):
    #first task accuracy for each training run for each metric
    data[i].extend(classifications[i])

    plt.plot(x, classifications[i], marker=".")
plt.legend(names)
plt.savefig("runs/metrics/average_accuracy.jpg")
plt.figure()

for i in range(number_of_tasks):
    columns.append("Run " + str(i) + " average accuracy")

print(len(columns))
print(len(data[0]))

df = pd.DataFrame(data=data, columns=columns)
df.index = names


new_df = df.corr()
new_df.to_csv('runs/metrics/correlations.csv')
df.to_csv('runs/metrics/dataframe.csv')