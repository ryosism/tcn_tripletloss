# In [Top5, Top5, Top5, ...]

start_border186 = [0, 480, 2190, 4380, 4380, 4890, 5670, 6750, 7080, 7380, 8190, 8190, 10110]
end_border186   = [0, 2190, 4380, 4380, 4890, 5670, 6750, 7080, 7380, 8190, 8190, 10110, 10680]


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
