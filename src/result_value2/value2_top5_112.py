# In [Top5, Top5, Top5, ...]

start_border112 = [2280, 12450, 24480, 34000, 61620]
end_border112   = [5130, 20850, 29880, 44520, 63000]


def calcAccuracy(top5Indexes, logger):
    print("top5Indexes", top5Indexes)
    correct = 0
    for order, top5 in enumerate(top5Indexes):
        logger.log(30, "[order = {}]".format(order))
        for index in top5:
            logger.log(30, "index = {}".format(index*30))
            logger.log(30, "{} < {} < {}".format(start_border112[order], index*30, end_border112[order]))
            if start_border112[order] < index*30 < end_border112[order]:
                correct += 1
                logger.log(30, "==================CORRECT=================")
                break

    return correct
