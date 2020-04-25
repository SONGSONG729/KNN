import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    '''
    使用k-近邻算法将每组数据划分到某个类中
    :param inX:
    :param dataSet:
    :param labels:
    :param k:
    :return:
    '''
    dataSetSize = dataSet.shape[0]
    # 距离计算
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5

    sortedDistIndicies = distances.argsort()
    classCount = {}
    # 选择距离最小的k个点
    for i in range(k):
        voteIlable = labels[sortedDistIndicies[i]]
        classCount[voteIlable] = classCount.get(voteIlable, 0) + 1
    # 排序  修改iteritems——>items
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    '''
    1. 准备数据：从文本中解析数据
    :param filename:
    :return:
    '''
    fr = open(filename)
    arrayOLines = fr.readlines()
    # 得到文件行数
    numberOfLines = len(arrayOLines)
    # 创建返回的Numpy矩阵
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    # 解析文件数据到列表
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0: 3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    '''
    3. 准备数据：归一化数值
    :param dataSet:
    :return:
    '''
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    # 特征值相除
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def datingClassTest():
    '''
    4. 测试算法：作为完整程序验证分类器
    :return:
    '''
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs: m, :], datingLabels[numTestVecs:m], 4)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
        print("the total error rate is: %f" % (errorCount/float(numTestVecs)))

def classifyPerson():
    """
    5. 使用算法：构建完整可用系统
    :return:
    """
    resultList = ['not at all', 'in small doses', 'in large doses']
    # row_input——>input
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMailes = float(input("frequent flier miles earned per year?"))
    iceCreams = float(input("liters of ice cream consumed per years?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMailes, percentTats, iceCreams])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 4)
    print("You will probably like this person:", resultList[classifierResult - 1])

def main():
    '''group, labels = createDataSet()
    print("group:\n", group)
    print("labels:\n", labels)
    print(classify0([0, 0], group, labels, 3))
    '''


    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    # print(datingDataMat)

    # 2. 分析数据：使用Matplotlib创建散点图
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],
    #            15.0 * np.array(datingLabels), 15.0 * np.array(datingLabels))
    # plt.show()

    # normMat, ranges, minVals = autoNorm(datingDataMat)
    # print("normMat:\n", normMat)
    # print("ranges:\n", ranges)
    # print("minVals:\n", minVals)

    # datingClassTest()
    classifyPerson()
    '''
    percentage of time spent playing video games?10
    frequent flier miles earned per year?10000
    liters of ice cream consumed per years?0.5
    You will probably like this person: in small doses
    '''

if __name__ == '__main__':
    main()