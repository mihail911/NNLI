import collections
import cPickle
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

# Hack for command line invocations                                                                                                     
if __name__ == '__main__':
    import os, sys
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(root_dir)

# Use our ouwn logger instead
try:
    from util.afs_safe_logger import Logger
except ImportError as e:
    import warnings
    warnings.warn("Could not find logger.  Using base logger instead.")
    from util.BaseLogger import BaseLogger as Logger
# Number of seconds in an hour
SEC_HOUR = 3600

class Stats(object):
    """
    General purpose object for recording and logging statistics/run of model run including
    accuracies and cost values. Will also be used to plot appropriate graphs.
    Note 'expName' must be full path for where to log experiment info.
    """
    def __init__(self, expName):
        self.logger = Logger(expName)
        self.startTime = time.time()
        self.acc = collections.defaultdict(list)
        self.cost = []
        self.totalNumEx = 0
        self.expName = expName


    def log(self, message):
        self.logger.Log(message)


    def reset(self):
        self.acc.clear()
        self.cost = []
        self.totalNumEx = 0


    def recordAcc(self, numEx, acc, dataset="train"):
        self.acc[dataset].append((numEx, acc))
        self.logger.Log("Current " + dataset + " accuracy after {0} examples:"
                                               " {1}".format(numEx, acc))
        if dataset == "train":
            ex = self.getExNum(dataList=dataset)
            acc = self.getAcc(dataList=dataset)
        elif dataset == "dev":
            ex = self.getExNum(dataList=dataset)
            acc = self.getAcc(dataList=dataset)
        # TODO: Support "test" computation as well

        self.plotAndSaveFig(self.expName+"_"+dataset+"Acc.png", dataset +
                            "Accuracy vs. Num Examples", "Num Examples", dataset +
                            " Accuracy", ex, acc)


    def recordCost(self, numEx, cost):
        self.cost.append((numEx, cost))
        self.logger.Log("Current cost: {0}".format(cost))
        numEx = self.getExNum(dataList="cost")
        cost = self.getCost()
        self.plotAndSaveFig(self.expName+"_cost.png", "Cost vs. Num Examples", "Num Examples",
                     "Cost", numEx, cost)



    def plotAndSaveFig(self, fileName, title, xLabel, yLabel, xCoord, yCoord): 
        plt.plot(xCoord, yCoord)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.title(title)
        plt.savefig(fileName)
        plt.clf()


    def plotAndSaveFigs(self, fileName, title, xLabel, yLabel, coordList):
        for x, y in coordList:
            plt.plot(x, y)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.title(title)
        plt.savefig(fileName)
        plt.clf()


    def getCost(self):
        return [stat[1] for stat in self.cost]


    def getExNum(self, dataList="train"):
        """
        Get total examples in data list specified returned as a list.
        Options include "train", "dev", "cost".
        Eventually will support "test" as well.
        :param dataList:
        :return:
        """
        if dataList == "cost":
            return [stat[0] for stat in self.cost]
        else:
            return [stat[0] for stat in self.acc[dataList]]


    def getAcc(self, dataList="train"):
        return [stat[1] for stat in self.acc[dataList]]


    def recordFinalStats(self, numEx, trainAcc, devAcc):
        # TODO: Eventually support test accuracy computation as well
        self.totalNumEx = numEx
        self.acc["train"].append((numEx, trainAcc))
        self.acc["dev"].append((numEx, devAcc))
        self.logger.Log("Final training accuracy after {0} examples: {1}".format(numEx, trainAcc))
        self.logger.Log("Final validation accuracy after {0} examples: {1}".format(numEx, devAcc))

        # Pickle accuracy and cost
        with open(self.expName+".pickle", "w") as f:
            cPickle.dump(self.acc, f)
            cPickle.dump(self.cost, f)

        # Plot accuracies and loss function
        numEx = self.getExNum(dataList="cost")
        cost = self.getCost()

        trainEx = self.getExNum(dataList="train")
        trainAcc = self.getAcc(dataList="train")

        devEx = self.getExNum(dataList="dev")
        devAcc = self.getAcc(dataList="dev")

        self.plotAndSaveFig(self.expName+"_cost.png", "Cost vs. Num Epochs", "Num Epochs",
                     "Cost", numEx, cost)

        self.plotAndSaveFigs(self.expName+"_trainDevAcc.png", "Train / Dev F1 vs. Num Epochs", "Num Epochs",
                     "Accuracy", [(trainEx, trainAcc), (devEx, devAcc)])
        '''
        self.plotAndSaveFig(self.expName+"_devAcc.png", "Dev F1 vs. Num Epochs", "Num Epochs",
                     "Accuracy", devEx, devAcc)
        '''
        self.logger.Log("Training complete! "
                        "Total training time: {0} ".format((time.time() -self.startTime)/SEC_HOUR))


        self.reset()
