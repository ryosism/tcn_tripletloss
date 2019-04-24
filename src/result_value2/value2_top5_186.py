# In [Top5, Top5, Top5, ...]

start_border186 = [2280, 6270, 12960, 20880, 20880, 24480, 32400, 44550, 45810, 61620, 65070, 68220, 70440]
end_border186   = [5130, 12960, 20850, 24480, 24480, 29880, 44550, 44970, 61620, 63000, 68220, 70440, 71520]


def calcAccuracy(top5Indexes, logger):
    print("top5Indexes", top5Indexes)
    correct = 0
    for order, top5 in enumerate(top5Indexes):
        logger.log(30, "[order = {}]".format(order))
        for index in top5:
            logger.log(30, "index = {}".format(index*30))
            logger.log(30, "{} < {} < {}".format(start_border186[order], index*30, end_border186[order]))
            if start_border186[order] < index*30 < end_border186[order]:
                correct += 1
                logger.log(30, "==================CORRECT=================")
                break

    return correct
