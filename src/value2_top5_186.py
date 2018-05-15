# In [Top5, Top5, Top5, ...]

def count_value2_186(a):
    correct = 0

    if 2280<=int(a[0])<5130:    # 手順1
        correct += 1
    if 6270<=int(a[5])<12960: # 手順2
        correct += 1
    if 12960<=int(a[6])<20850:        # 手順3p
        correct += 1
    if 20880<=int(a[7])<24480:        # 手順4
        correct += 1
    if 20880<=int(a[8])<24480:        # 手順5
        correct += 1
    if 24480<=int(a[9])<29880:        # 手順６
        correct += 1
    if 32400<=int(a[10])<44550:        # 手順7
        correct += 1
    if 44550<=int(a[11])<44970:        # 手順8
        correct += 1
    if 45810<=int(a[12])<61620:        # 手順9
        correct += 1
    if 61620<=int(a[1])<63000 or 45150<=int(a[1])<45840:        # 手順10
        correct += 1
    if 65070<=int(a[2])<68220:        # 手順11
        correct += 1
    if 68220<=int(a[3])<70440:        # 手順12
        correct += 1
    if 70440<=int(a[4])<71520:        # 手順13
        correct += 1
    return correct

def calcAccuracy(top5Indexes):
    for order, top5 in enumerate(top5Indexes):
        for index in top5:
