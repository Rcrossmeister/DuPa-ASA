
import re
import json
import collections
from nltk.tokenize import sent_tokenize, word_tokenize




class Opinion(object):
    def __init__(self, sentence, opinion, keyword):
        self.opinion = opinion
        self.sentence = sentence
        self.keyword = keyword
        self.cluster = None

    def updateCluster(self, cluster):
        self.cluster = cluster


class OpinionCluster(object):

    def __init__(self):
        self._opinions = []

    def addOpinion(self, opinion):
        self._opinions.append(opinion)
        opinion.updateCluster(self)

    def getOpinions(self):
        return self._opinions

    def getSummary(self, freqStrLen):
        opinionStrs = []
        for op in self._opinions:
            opinion = op.opinion
            opinionStrs.append(opinion)

        # 统计字频率
        word_counter = collections.Counter(list("".join(opinionStrs))).most_common()

        freqStr = ""
        for item in word_counter:
            if item[1] >= freqStrLen:
                freqStr += item[0]

        maxSim = -1
        maxOpinion = ""
        for opinion in opinionStrs:
            sim = similarity(freqStr, opinion)
            if sim > maxSim:
                maxSim = sim
                maxOpinion = opinion

        return maxOpinion


class OpinionExtraction(object):

    def __init__(self, sentences = [], sentenceFile = "", keywordFile = ""):
        self.json_config = self.loadConfig()  # 加载配置文件

        if sentenceFile:
            self.sentences = self.filterSentence(readFile(sentenceFile)[:self.json_config["dataLen"]])
        else:
            self.sentences = self.filterSentence(sentences[:self.json_config["dataLen"]])

        self.keyword = readFile(keywordFile)


    def loadConfig(self):
        f = open("./config.json", "r", encoding='utf-8')
        config = json.load(f)
        return config


    def filterSentence(self, sentences):
        newSentences = []
        for sent in sentences:
            # 长度太短
            if len(sent) < 4:
                continue

            addFlag = True
            sentLower = sent.lower()

            # 关键字过滤
            for exceptWord in self.json_config["exceptWordList"]:
                if exceptWord in sentLower:
                    addFlag = False
                    break
            if not addFlag:
                continue

            # 不过滤的关键字
            for includeWord in self.json_config["includeWordList"]:
                if includeWord in sentLower:
                    newSentences.append(sent)
                    addFlag = False
                    break
            if not addFlag:
                continue

            # 重复过滤
            if sent in newSentences:
                continue

            if addFlag:
                newSentences.append(sent)

        return newSentences


    def extractor(self):
        de = DependencyExtraction()
        opinionList = OpinionCluster()
        for sent in self.sentences:
            keyword = ""
            if not self.keyword:
                keyword = ""
            else:
                checkSent = []
                for word in self.keyword:
                    if sent not in checkSent and word in sent:
                        keyword = word
                        checkSent.append(sent)
                        break

            opinion = " ".join(de.parseSentWithKey(sent, keyword))
            if self.filterOpinion(opinion):
                opinionList.addOpinion(Opinion(sent, opinion, keyword))

        # 这步跳过前面的依存分析，加快调试
        # opinionList = self.getFirstCluster()

        '''
            这里设置两个阈值，先用小阈值把一个大数据切成小块，由于是小阈值，所以本身是一类的基本也能分到一类里面。
            由于分成了许多小块，再对每个小块做聚类，聚类速度大大提升，[0.2, 0.6]比[0.6]速度高30倍左右。
            但是[0.2, 0.6]和[0.6]最后的结果不是一样的，会把一些相同的观点拆开。
        '''
        thresholds = self.json_config["thresholds"]  # 阈值
        clusters = [opinionList]
        for threshold in thresholds:
            newClusters = []
            for cluster in clusters:
                newClusters += self.clusterOpinion(cluster, threshold)
            clusters = newClusters

        resMaxLen = {}
        for oc in clusters:
            if len(oc.getOpinions()) >= self.json_config["minClusterLen"]:
                summaryStr = oc.getSummary(self.json_config["freqStrLen"])
                resMaxLen[summaryStr] = oc.getOpinions()

        return self.sortRes(resMaxLen)


    def sortRes(self, res):
        return sorted(res.items(), key=lambda item :len(item[1]), reverse=True)


    def getFirstCluster(self):
        opinions = []
        with open("./data/opinion.txt", "r", encoding="utf-8") as f:
            for line in f:
                lineSplit = line.strip().split(",")
                opinions.append(lineSplit)


        opinions = opinions[:self.json_config["dataLen"]]

        firstCluster = OpinionCluster()
        for op in opinions:
            op = op + [""]
            firstCluster.addOpinion(Opinion(*op))
        return firstCluster


    def filterOpinion(self, opinion):
        check = True
        if len(opinion) <= self.json_config["minOpinionLen"]:
            check = False
        elif opinion.isdigit():
            check = False
        return check

    # 复杂度是O(n2)，速度比较慢。
    def clusterOpinion(self, cluster, threshold):
        opinions = cluster.getOpinions()
        num = len(opinions)
        clusters = []
        checked1 = []
        for i in range(num):
            oc = OpinionCluster()
            opinion1 = opinions[i]
            if opinion1 in checked1:
                continue
            if opinion1 not in oc.getOpinions():
                oc.addOpinion(opinion1)
            checked1.append(opinion1)
            for j in range(i + 1, num):
                opinion2 = opinions[j]
                if opinion2 in checked1:
                    continue
                sim = similarity(opinion1.opinion, opinion2.opinion)
                if sim > threshold:
                    if opinion2 not in oc.getOpinions():
                        oc.addOpinion(opinion2)
                    checked1.append(opinion2)
            clusters.append(oc)
        return clusters


import re
import difflib

# 判断中文
def findChineseWord(word):
    chinese_pattern = '[\u4e00-\u9fa5]+'
    says = re.findall(chinese_pattern, word)

    if not says:
        return False

    return True

# 读取文件
def readFile(path):
    content = []
    if not path:
        return content
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            content.append(line.strip())
    return content

def similarity(opinion1, opinion2):
    return difflib.SequenceMatcher(a = opinion1, b = opinion2).quick_ratio()

import spacy
import operator

class DependencyExtraction(object):

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.jump_relation = set(['amod', 'advcl', 'nsubj'])
        self.reverse_relation = set(['xcomp', 'dobj', 'prep'])
        self.main_relation = set(['ROOT'])
        self.remove_relate = set(['punct'])
        self.include = set()
        self.group = {}

    # 句子的观点提取，单root，从root出发，1.找前面最近的修饰词。2.找后面距离为1的reverse_relation
    def parseSentence(self, sentence):
        reverse_target = {}
        # parse_result = str(self.hanlp.parseDependency(sentence)).strip().split('\n')
        doc =self.nlp(sentence)
        parse_result = []
        for i in range(len(doc)):
            row = []
            for j in range(10):
                row.append(j)
            parse_result.append(row)
        for i ,token in enumerate(doc):
            parse_result[i].append(i)
            parse_result[i].append(token.text)
            parse_result[i].append(token.text)
            parse_result[i].append(token.pos_)
            parse_result[i].append(token.pos_)
            parse_result[i].append(0)
            parse_result[i].append(token.root.i)
            parse_result[i].append(token.dep_)
            parse_result[i].append(0)
            parse_result[i].append(0)
        for p in parse_result:
            print(p)
        for i in range(len(parse_result)):
            parse_result[i] = parse_result[i].split('\t')
            self_index = int(parse_result[i][0])
            target_index = int(parse_result[i][6])
            relation = parse_result[i][7]
            if relation in self.remove_relate:
                continue
            if target_index > self_index:
                reverse_target[target_index] = self_index
        result = {}
        checked = set()
        related_words = set()
        for item in parse_result:
            relation = item[7]
            target = int(item[6])
            index = int(item[0])
            if index in checked:
                continue
            while relation in self.jump_relation:
                checked.add(index)
                next_item = parse_result[target - 1]
                relation = next_item[7]
                target = int(next_item[6])
                index = int(next_item[0])

            if relation in self.reverse_relation and target in result and target not in related_words:
                result[index] = parse_result[index - 1][1]
                if index in reverse_target:
                    reverse_target_index = reverse_target[index]
                    if abs(index - reverse_target[index]) <= 1:
                        result[reverse_target_index] = parse_result[reverse_target_index - 1][1]
                        related_words.add(reverse_target_index)

            if relation in self.main_relation:
                result[index] = parse_result[index - 1][1]
                if index in reverse_target:
                    reverse_target_index = reverse_target[index]
                    if abs(index - reverse_target_index) <= 1:
                        result[reverse_target_index] = parse_result[reverse_target_index - 1][1]
                        related_words.add(reverse_target_index)
            checked.add(index)

        for item in parse_result:
            word = item[1]
            if word in self.include:
                result[int(item[0])] = word

        sorted_keys = sorted(result.items(), key=operator.itemgetter(0))
        selected_words = [w[1] for w in sorted_keys]
        return selected_words

    ''' 
    关键词观点提取，根据关键词key，找到关键处的rootpath，寻找这个root中的观点，观点提取方式和parseSentence的基本一样。
    支持提取多个root的观点。
    '''
    def parseSentWithKey(self, sentence, key=None):
        if key:
            keyIndex = 0
            if key not in sentence:
                return []
        rootList = []
        doc = self.nlp(sentence)
        parse_result = []
        for i in range(len(doc)):
            row = []
            for j in range(10):
                row.append(j)
            parse_result.append(row)
        for i, token in enumerate(doc):
            parse_result[i][0] = i
            parse_result[i][1] = token.text
            parse_result[i][2] = token.text
            parse_result[i][3] = token.pos_
            parse_result[i][4] = token.pos_
            parse_result[i][5] = 0
            parse_result[i][6] = token.head.i
            parse_result[i][7] = token.dep_
            parse_result[i][8] = 0
            parse_result[i][9] = 0

        for i in range(len(parse_result)):
            self_index = int(parse_result[i][0])
            target_index = int(parse_result[i][6])
            relation = parse_result[i][7]
            if relation in self.main_relation:
                if self_index not in rootList:
                    rootList.append(self_index)
            elif relation == "cc" and target_index in rootList:
                if self_index not in rootList:
                    rootList.append(self_index)

            if len(parse_result[target_index]) == 10:
                parse_result[target_index].append([])

            if target_index != -1 and not (relation == "cc" and target_index in rootList):
                parse_result[target_index][10].append(self_index)

        if key:
            rootIndex = 0
            if len(rootList) > 1:
                target = keyIndex
                while True:
                    if target in rootList:
                        rootIndex = rootList.index(target)
                        break
                    next_item = parse_result[target]
                    target = int(next_item[6])
            loopRoot = [rootList[rootIndex]]
        else:
            loopRoot = rootList

        result = {}
        related_words = set()
        for root in loopRoot:
            if key:
                self.addToResult(parse_result, keyIndex, result, related_words)
            self.addToResult(parse_result, root, result, related_words)

        for item in parse_result:
            relation = item[7]
            target = int(item[6])
            index = int(item[0])
            if relation in self.reverse_relation and target in result and target not in related_words:
                self.addToResult(parse_result, index, result, related_words)

        for item in parse_result:
            word = item[1]
            if word == key:
                result[int(item[0])] = word

        sorted_keys = sorted(result.items(), key=operator.itemgetter(0))
        selected_words = [w[1] for w in sorted_keys]
        return selected_words


    def addToResult(self, parse_result, index, result, related_words):
        result[index] = parse_result[index][1]
        if len(parse_result[index]) == 10:
            return
        reverse_target_index = 0
        for i in parse_result[index][10]:
            if i < index and i > reverse_target_index:
                reverse_target_index = i
        if abs(index - reverse_target_index) <= 1:
            result[reverse_target_index] = parse_result[reverse_target_index][1]
            related_words.add(reverse_target_index)

if __name__=="__main__":
    f = open('./text3.txt', encoding='utf-8')
    data = f.readlines()
    f.close()
    text = ""
    for line in data:
        text += line
    sentences = sent_tokenize(text)
    oe = OpinionExtraction(sentences=sentences)
    sortedKeys = oe.extractor()
    result =[]
    for sortK in sortedKeys:
        str = []
        for op in sortK[1]:
            str.append(op.opinion)
        result.append(sortK[0])
    print(result)