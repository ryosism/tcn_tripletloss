# In [Top5, Top5, Top5, ...]

start_border186 = [2280, 6270, 12960, 20880, 20880, 24480, 32400, 44550, 45810, 61620, 65070, 70440]
end_border186   = [5130, 12960, 20850, 24480, 24480, 29880, 44550, 44970, 61620, 63000, 70440, 71520]


def calcAccuracy(top5Indexes):
    correct = 0
    for order, top5 in enumerate(top5Indexes):
        for index in top5:
            if calcTop5(order, index):
                correct += 1
                break


def calcTop5(order, index):
    if start_border186[order] > index > end_border186[order]:
        return True
    else:
        return False
