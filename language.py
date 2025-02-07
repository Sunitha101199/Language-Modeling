"""
Language Modeling Project
Name: Sunitha
Roll No: 2021501001
"""

from matplotlib.pyplot import text
import language_tests as test

project = "Language" # don't edit this

### WEEK 1 ###

'''
loadBook(filename)
#1 [Check6-1]
Parameters: str
Returns: 2D list of strs
'''
def loadBook(filename):
    f = open(filename,"r")
    corpus = []
    for i in f.readlines():
        if i!="\n":
            line = i.split("\n")[0]
            lines = line.split(" ")
        else:
            continue
        corpus.append(lines)
    return corpus


'''
getCorpusLength(corpus)
#2 [Check6-1]
Parameters: 2D list of strs
Returns: int
'''
def getCorpusLength(corpus):
    length = 0
    for i in corpus:
        length+=len(i)
    return length


'''
buildVocabulary(corpus)
#3 [Check6-1]
Parameters: 2D list of strs
Returns: list of strs
'''
def buildVocabulary(corpus):
    unique = []
    for i in corpus:
        for j in i:
            if j not in unique:
                unique.append(j)
    return unique


'''
countUnigrams(corpus)
#4 [Check6-1]
Parameters: 2D list of strs
Returns: dict mapping strs to ints
'''
def countUnigrams(corpus):
    dictionary = {}
    for i in corpus:
        for j in i:
            if j not in dictionary:
                dictionary[j]=0
            dictionary[j]+=1
    return dictionary


'''
getStartWords(corpus)
#5 [Check6-1]
Parameters: 2D list of strs
Returns: list of strs
'''
def getStartWords(corpus):
    startwords = []
    for i in corpus:
        if i[0] not in startwords:
            startwords.append(i[0])
    return startwords


'''
countStartWords(corpus)
#5 [Check6-1]
Parameters: 2D list of strs
Returns: dict mapping strs to ints
'''
def countStartWords(corpus):
    dictionary = {}
    for i in corpus:
        if i[0] not in dictionary:
            dictionary[i[0]]=0
        dictionary[i[0]]+=1
    return dictionary


'''
countBigrams(corpus)
#6 [Check6-1]
Parameters: 2D list of strs
Returns: dict mapping strs to (dicts mapping strs to ints)
'''
def countBigrams(corpus):
    dictionary = {}
    for sentence in corpus:
        for i in range(0,len(sentence)-1):
            if sentence[i] not in dictionary:
                dictionary[sentence[i]] = {}
            if sentence[i+1] not in dictionary[sentence[i]]:
                dictionary[sentence[i]][sentence[i+1]] = 0
            dictionary[sentence[i]][sentence[i+1]]+=1
    return dictionary


### WEEK 2 ###

'''
buildUniformProbs(unigrams)
#1 [Check6-2]
Parameters: list of strs
Returns: list of floats
'''
def buildUniformProbs(unigrams):
    probabilities = []
    i=0
    while i<len(unigrams):
        probabilities.append(1/len(unigrams))
        i+=1
    return probabilities


'''
buildUnigramProbs(unigrams, unigramCounts, totalCount)
#2 [Check6-2]
Parameters: list of strs ; dict mapping strs to ints ; int
Returns: list of floats
'''
def buildUnigramProbs(unigrams, unigramCounts, totalCount):
    unigramProbs = []
    for i in unigrams:
        if i in unigramCounts:
           unigramProbs.append(unigramCounts[i]/totalCount)
        else:
            unigramProbs.append(0)
    return unigramProbs


'''
buildBigramProbs(unigramCounts, bigramCounts)
#3 [Check6-2]
Parameters: dict mapping strs to ints ; dict mapping strs to (dicts mapping strs to ints)
Returns: dict mapping strs to (dicts mapping strs to (lists of values))
'''
def buildBigramProbs(unigramCounts, bigramCounts):
    bigramProbs = {}
    for prevWord in bigramCounts:
        words, probs = [], []
        tempDict = {}
        for i in bigramCounts[prevWord]:
            words.append(i)
            probs.append(bigramCounts[prevWord][i]/unigramCounts[prevWord])
        tempDict["words"] = words
        tempDict["probs"] = probs
        bigramProbs[prevWord] = tempDict
    return bigramProbs


'''
getTopWords(count, words, probs, ignoreList)
#4 [Check6-2]
Parameters: int ; list of strs ; list of floats ; list of strs
Returns: dict mapping strs to floats
'''
def getTopWords(count, words, probs, ignoreList):
    dictionary = {}
    topWords = {}
    for i in range(len(words)):
        dictionary[words[i]]=probs[i]
    for key,value in sorted(dictionary.items(), key=lambda item: item[1], reverse = True):
        if key not in ignoreList:
            topWords[key] = value
        if len(topWords)==count:
            break
    return topWords


'''
generateTextFromUnigrams(count, words, probs)
#5 [Check6-2]
Parameters: int ; list of strs ; list of floats
Returns: str
'''
from random import choices
def generateTextFromUnigrams(count, words, probs):
    textFromUnigrams = ""
    j = 0
    while j!=count:
        for i in choices(words, weights=probs):
            textFromUnigrams=textFromUnigrams+i+" "
            j+=1
    return textFromUnigrams


'''
generateTextFromBigrams(count, startWords, startWordProbs, bigramProbs)
#6 [Check6-2]
Parameters: int ; list of strs ; list of floats ; dict mapping strs to (dicts mapping strs to (lists of values))
Returns: str
'''
def generateTextFromBigrams(count, startWords, startWordProbs, bigramProbs):
    textFromBigrams = ""
    i=0
    prevWords = ""
    while i!=count:
        if textFromBigrams=="" or prevWords==".": 
            for j in choices(startWords, weights=startWordProbs):
                prevWords = j
                textFromBigrams=textFromBigrams+j+" "
                i+=1
        else:
            for j in choices(bigramProbs[prevWords]["words"], weights=bigramProbs[prevWords]["probs"]):
                prevWords=j
                textFromBigrams=textFromBigrams+j+" "
                i+=1
    return textFromBigrams


### WEEK 3 ###

ignore = [ ",", ".", "?", "'", '"', "-", "!", ":", ";", "by", "around", "over",
           "a", "on", "be", "in", "the", "is", "on", "and", "to", "of", "it",
           "as", "an", "but", "at", "if", "so", "was", "were", "for", "this",
           "that", "onto", "from", "not", "into" ]

'''
graphTop50Words(corpus)
#3 [Hw6]
Parameters: 2D list of strs
Returns: None
'''
def graphTop50Words(corpus):
    top50 = getTopWords(50,buildVocabulary(corpus), buildUnigramProbs(buildVocabulary(corpus),countUnigrams(corpus),50), ignore)
    barPlot(top50, "Top 50 Unigrams based on their Probabilities")
    return


'''
graphTopStartWords(corpus)
#4 [Hw6]
Parameters: 2D list of strs
Returns: None
'''
def graphTopStartWords(corpus):
    topStartWords = getTopWords(50, getStartWords(corpus), buildUnigramProbs(getStartWords(corpus), countStartWords(corpus), 50), ignore)
    barPlot(topStartWords, "Top 50 Most frequent Start Words")
    return


'''
graphTopNextWords(corpus, word)
#5 [Hw6]
Parameters: 2D list of strs ; str
Returns: None
'''
def graphTopNextWords(corpus, word):
    dictionary = buildBigramProbs(countUnigrams(corpus), countBigrams(corpus))
    top10NextWords = getTopWords(10, dictionary[word]["words"], dictionary[word]["probs"] , ignore)
    title = "Top 10 words occurring after \""+word+"\" word"
    barPlot(top10NextWords, title)
    return


'''
setupChartData(corpus1, corpus2, topWordCount)
#6 [Hw6]
Parameters: 2D list of strs ; 2D list of strs ; int
Returns: dict mapping strs to (lists of values)
'''
def setupChartData(corpus1, corpus2, topWordCount):
    uniCorpus1 = buildVocabulary(corpus1)
    uniCorpus2 = buildVocabulary(corpus2)
    topWords = []
    uniCor1Probs = buildUnigramProbs(uniCorpus1, countUnigrams(corpus1),getCorpusLength(corpus1))
    uniCor2Probs = buildUnigramProbs(uniCorpus2, countUnigrams(corpus2),getCorpusLength(corpus2))
    dictCorpus1 = getTopWords(topWordCount, uniCorpus1, uniCor1Probs, ignore)
    dictCorpus2 = getTopWords(topWordCount, uniCorpus2, uniCor2Probs, ignore)
    topWords1 = list(dictCorpus1.keys())
    topWords2 = list(dictCorpus2.keys())
    for i in topWords1+topWords2:
        if i not in topWords:
            topWords.append(i)
    topWords1Probs = buildUnigramProbs(topWords, countUnigrams(corpus1),getCorpusLength(corpus1))
    topWords2Probs = buildUnigramProbs(topWords, countUnigrams(corpus2),getCorpusLength(corpus2))
    dictionary = {"topWords": topWords, "corpus1Probs": topWords1Probs, 'corpus2Probs': topWords2Probs}      
    return dictionary


'''
graphTopWordsSideBySide(corpus1, name1, corpus2, name2, numWords, title)
#6 [Hw6]
Parameters: 2D list of strs ; str ; 2D list of strs ; str ; int ; str
Returns: None
'''
def graphTopWordsSideBySide(corpus1, name1, corpus2, name2, numWords, title):
    topWords = setupChartData(corpus1,corpus2,numWords)
    sideBySideBarPlots(topWords["topWords"], topWords["corpus1Probs"], topWords["corpus2Probs"], name1, name2, title)
    return


'''
graphTopWordsInScatterplot(corpus1, corpus2, numWords, title)
#6 [Hw6]
Parameters: 2D list of strs ; 2D list of strs ; int ; str
Returns: None
'''
def graphTopWordsInScatterplot(corpus1, corpus2, numWords, title):
    topWords = setupChartData(corpus1, corpus2, numWords)
    scatterPlot(list(topWords["corpus1Probs"]), list(topWords["corpus2Probs"]), list(topWords["topWords"]), title)
    return


### WEEK 3 PROVIDED CODE ###

"""
Expects a dictionary of words as keys with probabilities as values, and a title
Plots the words on the x axis, probabilities as the y axis and puts a title on top.
"""
def barPlot(dict, title):
    import matplotlib.pyplot as plt

    names = []
    values = []
    for k in dict:
        names.append(k)
        values.append(dict[k])

    plt.bar(names, values)

    plt.xticks(rotation='vertical')
    plt.title(title)

    plt.show()

"""
Expects 3 lists - one of x values, and two of values such that the index of a name
corresponds to a value at the same index in both lists. Category1 and Category2
are the labels for the different colors in the graph. For example, you may use
it to graph two categories of probabilities side by side to look at the differences.
"""
def sideBySideBarPlots(xValues, values1, values2, category1, category2, title):
    import matplotlib.pyplot as plt

    w = 0.35  # the width of the bars

    plt.bar(xValues, values1, width=-w, align='edge', label=category1)
    plt.bar(xValues, values2, width= w, align='edge', label=category2)

    plt.xticks(rotation="vertical")
    plt.legend()
    plt.title(title)

    plt.show()

"""
Expects two lists of probabilities and a list of labels (words) all the same length
and plots the probabilities of x and y, labels each point, and puts a title on top.
Note that this limits the graph to go from 0x0 to 0.02 x 0.02.
"""
def scatterPlot(xs, ys, labels, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    plt.scatter(xs, ys)

    # make labels for the points
    for i in range(len(labels)):
        plt.annotate(labels[i], # this is the text
                    (xs[i], ys[i]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0, 10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    plt.title(title)
    plt.xlim(0, 0.02)
    plt.ylim(0, 0.02)

    # a bit of advanced code to draw a y=x line
    ax.plot([0, 1], [0, 1], color='black', transform=ax.transAxes)

    plt.show()


### RUN CODE ###

# This code runs the test cases to check your work
if __name__ == "__main__":
    print("\n" + "#"*15 + " WEEK 1 TESTS " +  "#" * 16 + "\n")
    test.week1Tests()
    print("\n" + "#"*15 + " WEEK 1 OUTPUT " + "#" * 15 + "\n")
    test.runWeek1()

    ## Uncomment these for Week 2 ##
    print("\n" + "#"*15 + " WEEK 2 TESTS " +  "#" * 16 + "\n")
    test.week2Tests()
    print("\n" + "#"*15 + " WEEK 2 OUTPUT " + "#" * 15 + "\n")
    test.runWeek2()

    ## Uncomment these for Week 3 ##

    print("\n" + "#"*15 + " WEEK 3 OUTPUT " + "#" * 15 + "\n")
    test.runWeek3()
