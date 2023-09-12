from collections import defaultdict
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import math
import re, os
import jieba.posseg as pseg
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
import json
import collections
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import difflib
import operator
from tqdm import tqdm


class ExtractEvent:
    def __init__(self):
        self.text = None
        self.map_dict = self.load_mapdict()
        self.minlen = 2
        self.maxlen = 300
        self.keywords_num = 20
        self.limit_score = 10
        self.IP = "(([NERMQ]*P*[ABDP]*)*([ABDV]{1,})*([NERMQ]*)*([VDAB]$)?([NERMQ]*)*([VDAB]$)?)*"
        self.IP = "([NER]*([PMBQADP]*[NER]*)*([VPDA]{1,}[NEBRVMQDA]*)*)"
        self.MQ = '[DP]*M{1,}[Q]*([VN]$)?'
        self.VNP = 'V*N{1,}'
        self.NP = '[NER]{1,}'
        self.REN = 'R{2,}'
        self.VP = 'P?(V|A$|D$){1,}'
        self.PP = 'P?[NERMQ]{1,}'
        self.SPO_n = "n{1,}"
        self.SPO_v = "v{1,}"
        self.stop_tags = {'PART', 'INTJ', 'ADP', 'CCONJ', 'NUM', 'X', 'DET'}
        self.combine_words = {"first", "then", "before", "after"}

    """构建映射字典"""

    def load_mapdict(self):
        tag_dict = {
            'B': 'DET'.split(),  # 时间词
            'A': 'ADJ ADV'.split(),  # 时间词
            'D': "ADV".split(),  # 限定词
            'N': "NOUN X PRON ADP".split(),  # 名词
            "E": "PROPN ADJ".split(),  # 实体词
            "R": "PROPN".split(),  # 人物
            'G': "X".split(),  # 语素
            'V': "VERB".split(),  # 动词
            'P': "ADP".split(),  # 介词
            "M": "NUM".split(),  # 数词
            "Q": "NUM".split(),  # 量词
            "v": "V".split(),  # 动词短语
            "n": "N".split(),  # 名词介宾短语
        }
        map_dict = {}
        for flag, tags in tag_dict.items():
            for tag in tags:
                map_dict[tag] = flag
        return map_dict

    """根据定义的标签,对词性进行标签化"""

    def transfer_tags(self, postags):
        tags = [self.map_dict.get(tag, 'W') for tag in postags]
        return ''.join(tags)

    """抽取出指定长度的ngram"""

    def extract_ngram(self, pos_seq, regex):
        ss = self.transfer_tags(pos_seq)

        def gen():
            for s in range(len(ss)):
                for n in range(self.minlen, 1 + min(self.maxlen, len(ss) - s)):
                    e = s + n
                    substr = ss[s:e]
                    if re.match(regex + "$", substr):
                        yield (s, e)

        return list(gen())

    '''抽取ngram'''

    def extract_sentgram(self, pos_seq, regex):
        ss = self.transfer_tags(pos_seq)

        def gen():
            for m in re.finditer(regex, ss):
                yield (m.start(), m.end())

        return list(gen())

    """指示代词替换，消解处理"""

    def cite_resolution(self, words, postags, persons):
        if not persons and 'r' not in set(postags):
            return words, postags
        elif persons and 'r' in set(postags):
            cite_index = postags.index('r')
            if words[cite_index] in {"it", "he", "she", "i"}:
                words[cite_index] = persons[-1]
                postags[cite_index] = 'PROPN'
        elif 'r' in set(postags):
            cite_index = postags.index('r')
            if words[cite_index] in {"why", "how"}:
                postags[cite_index] = 'X'
        return words, postags

    """抽取量词性短语"""

    def extract_mqs(self, wds, postags):
        phrase_tokspans = self.extract_sentgram(postags, self.MQ)
        if not phrase_tokspans:
            return []
        phrases = [' '.join(wds[i[0]:i[1]]) for i in phrase_tokspans]
        return phrases

    '''抽取动词性短语'''

    def get_ips(self, wds, postags):
        ips = []
        phrase_tokspans = self.extract_sentgram(postags, self.IP)
        if not phrase_tokspans:
            return []
        phrases = [' '.join(wds[i[0]:i[1]]) for i in phrase_tokspans]
        phrase_postags = [''.join(postags[i[0]:i[1]]) for i in phrase_tokspans]
        for phrase, phrase_postag_ in zip(phrases, phrase_postags):
            if not phrase:
                continue
            phrase_postags = ''.join(phrase_postag_).replace('NUM', '').replace('ADJ', '')
            if phrase_postags.startswith('NOUN'):
                has_subj = 1
            else:
                has_subj = 0
            ips.append((has_subj, phrase))
        return ips

    """分短句处理"""

    def split_short_sents(self, text):
        return [i for i in re.split(r'[,，]', text) if len(i) > 2]

    """分段落"""

    def split_paras(self, text):
        return [i for i in re.split(r'[\n\r]', text) if len(i) > 4]

    """分长句处理"""

    def split_long_sents(self, text):
        return [i for i in re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text) if len(i) > 4]

    """移出噪声数据"""

    def remove_punc(self, text):
        text = str(text).replace('\u3000', '').replace("'", '').replace('“', '').replace('”', '').replace('▲',
                                                                                                          '').replace(
            '” ', "”")
        tmps = re.findall('[\(|（][^\(（\)）]*[\)|）]', text)
        for tmp in tmps:
            text = text.replace(tmp, '')
        return text

    """保持专有名词"""

    def zhuanming(self, text):
        books = re.findall('([<《])([^《》]+)([》>])', text)
        return books

    """对人物类词语进行修正"""

    def modify_nr(self, wds, postags):
        phrase_tokspans = self.extract_sentgram(postags, self.REN)
        wds_seq = ' '.join(wds)
        pos_seq = ' '.join(postags)
        if not phrase_tokspans:
            return wds, postags
        else:
            wd_phrases = [' '.join(wds[i[0]:i[1]]) for i in phrase_tokspans]
            postag_phrases = [' '.join(postags[i[0]:i[1]]) for i in phrase_tokspans]
            for wd_phrase in wd_phrases: #整合人物类词语
                tmp = wd_phrase.replace(' ', '')
                wds_seq = wds_seq.replace(wd_phrase, tmp)
            for postag_phrase in postag_phrases:
                pos_seq = pos_seq.replace(postag_phrase, 'PROPN')
        words = [i for i in wds_seq.split(' ') if i]
        postags = [i for i in pos_seq.split(' ') if i]
        return words, postags

    """对复制进行修正"""

    def modify_duplicate(self, wds, postags, regex, tag):
        phrase_tokspans = self.extract_sentgram(postags, regex)
        wds_seq = ' '.join(wds)
        pos_seq = ' '.join(postags)
        if not phrase_tokspans:
            return wds, postags
        else:
            wd_phrases = [' '.join(wds[i[0]:i[1]]) for i in phrase_tokspans]
            postag_phrases = [' '.join(postags[i[0]:i[1]]) for i in phrase_tokspans]
            for wd_phrase in wd_phrases:
                tmp = wd_phrase.replace(' ', '')
                wds_seq = wds_seq.replace(wd_phrase, tmp)
            for postag_phrase in postag_phrases:
                pos_seq = pos_seq.replace(postag_phrase, tag)
        words = [i for i in wds_seq.split(' ') if i]
        postags = [i for i in pos_seq.split(' ') if i]
        return words, postags

    '''对句子进行分词处理'''

    def cut_wds(self, sent):
        # wds = list(pseg.cut(sent))
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(sent)
        wds = [(token.text, token.pos_) for token in doc]
        postags = [w[1] for w in wds]
        words = [w[0] for w in wds]
        return self.modify_nr(words, postags)

    """移除噪声词语"""

    def clean_wds(self, words, postags):
        wds = []
        poss = []
        for wd, postag in zip(words, postags):
            if postag.lower() in self.stop_tags:
                continue
            wds.append(wd)
            poss.append(postag)
        return wds, poss

    """检测是否成立, 肯定需要包括名词"""

    def check_flag(self, postags):
        if not {"VERB", 'ADJ', 'NOUN'}.intersection(postags):
            return 0
        return 1

    """识别出人名实体"""

    def detect_person(self, words, postags):
        persons = []
        for wd, postag in zip(words, postags):
            if postag == 'PROPN':
                persons.append(wd)
        return persons

    """识别出名词性短语"""

    def get_nps(self, wds, postags):
        phrase_tokspans = self.extract_sentgram(postags, self.NP)
        if not phrase_tokspans:
            return [], []
        phrases_np = [' '.join(wds[i[0]:i[1]]) for i in phrase_tokspans]
        return phrase_tokspans, phrases_np

    """识别出介宾短语"""

    def get_pps(self, wds, postags):
        phrase_tokspans = self.extract_sentgram(postags, self.PP)
        if not phrase_tokspans:
            return [], []
        phrases_pp = [' '.join(wds[i[0]:i[1]]) for i in phrase_tokspans]
        return phrase_tokspans, phrases_pp

    """识别出动词短语"""

    def get_vps(self, wds, postags):
        phrase_tokspans = self.extract_sentgram(postags, self.VP)
        if not phrase_tokspans:
            return [], []
        phrases_vp = [' '.join(wds[i[0]:i[1]]) for i in phrase_tokspans]
        return phrase_tokspans, phrases_vp

    """抽取名动词性短语"""

    def get_vnps(self, s):
        wds, postags = self.cut_wds(s)
        if not postags:
            return [], []
        if not (postags[-1].endswith("NOUN")):
            return [], []
        phrase_tokspans = self.extract_sentgram(postags, self.VNP)
        if not phrase_tokspans:
            return [], []
        phrases_vnp = [' '.join(wds[i[0]:i[1]]) for i in phrase_tokspans]
        phrase_tokspans2 = self.extract_sentgram(postags, self.NP)
        if not phrase_tokspans2:
            return [], []
        phrases_np = [' '.join(wds[i[0]:i[1]]) for i in phrase_tokspans2]
        return phrases_vnp, phrases_np

    def text_clean(self):
        # 清洗文本
        stopwords = self.stopwordslist()
        self.text = self.text.lower()
        self.text = re.sub(r'，', '。', self.text)
        self.text = re.sub(r'[[0-9]*]', ' ', self.text)  # 去除类似[1]，[2]
        self.text = re.sub(r'\s+', ' ', self.text)  # 用单个空格替换了所有额外的空格
        sentences = re.split('。', self.text)  # 分句
        sentences_tmp = sentences
        sentences = []
        string = ''
        for sentence in sentences_tmp:
            words = word_tokenize(sentence)
            for word in words:
                if word not in stopwords:
                    string = string + word
            sentences.append(string)
            string = ''
        if "" in sentences or " " in sentences or None in sentences:
            sentences = [i for i in sentences if i not in ["", " ", None]]
        return sentences

    """提取短语"""

    def phrase_ip(self, content):
        spos = []
        events = []
        content = self.remove_punc(content)
        paras = self.split_paras(content)
        for para in paras:
            long_sents = self.split_long_sents(para)
            for long_sent in long_sents:
                persons = []
                short_sents = self.split_short_sents(long_sent)
                for sent in short_sents:
                    words, postags = self.cut_wds(sent)
                    person = self.detect_person(words, postags)
                    words, postags = self.cite_resolution(words, postags, persons) #指示代词替换
                    words, postags = self.clean_wds(words, postags)
                    # print(words,postags)
                    ips = self.get_ips(words, postags) #抽取动词性短语
                    persons += person
                    for ip in ips:
                        events.append(ip[1])
                        wds_tmp = []
                        postags_tmp = []
                        words, postags = self.cut_wds(ip[1])
                        verb_tokspans, verbs = self.get_vps(words, postags) #识别出动词
                        pp_tokspans, pps = self.get_pps(words, postags) #识别出介宾词
                        tmp_dict = {str(verb[0]) + str(verb[1]): ['V', verbs[idx]] for idx, verb in
                                    enumerate(verb_tokspans)}
                        pp_dict = {str(pp[0]) + str(pp[1]): ['N', pps[idx]] for idx, pp in enumerate(pp_tokspans)}
                        tmp_dict.update(pp_dict) #整合到一个dict中
                        sort_keys = sorted([int(i) for i in tmp_dict.keys()])
                        for i in sort_keys:
                            if i < 10:
                                i = '0' + str(i)
                            else:
                                i = str(i)
                            wds_tmp.append(tmp_dict[str(i)][-1])
                            postags_tmp.append(tmp_dict[str(i)][0])
                        # wds_tmp, postags_tmp = self.modify_duplicate(wds_tmp, postags_tmp, self.SPO_v, 'V')
                        # wds_tmp, postags_tmp = self.modify_duplicate(wds_tmp, postags_tmp, self.SPO_n, 'N')
                        if len(postags_tmp) < 2:
                            continue
                        seg_index = []
                        i = 0
                        for wd, postag in zip(wds_tmp, postags_tmp):
                            if postag == 'V':
                                seg_index.append(i)
                            i += 1
                        spo = []
                        for indx, seg_indx in enumerate(seg_index):
                            if indx == 0:
                                pre_indx = 0
                            else:
                                pre_indx = seg_index[indx - 1]
                            if pre_indx < 0:
                                pre_indx = 0
                            if seg_indx == 0:
                                spo.append((' ', wds_tmp[seg_indx], ' '.join(wds_tmp[seg_indx + 1:])))
                            elif seg_indx > 0 and indx < 1:
                                spo.append(
                                    (' '.join(wds_tmp[:seg_indx]), wds_tmp[seg_indx], ' '.join(wds_tmp[seg_indx + 1:])))
                            else:
                                spo.append((' '.join(wds_tmp[pre_indx + 1:seg_indx]), wds_tmp[seg_indx],
                                            ' '.join(wds_tmp[seg_indx + 1:])))
                        sentences=[]
                        sentence=''
                        for spo_ in spo:
                            for word in spo_:
                                sentence=sentence+word+' '
                            sentence=sentence.rstrip()
                            sentences.append(sentence)
                        spos += sentences

        return events, spos

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

class Extract_Summary_English:
    def __init__(self, text):
        self.text = text

    def text_clean(self):
        sentences = sent_tokenize(self.text)
        sentences_=sentences
        sentences=[]
        for sentence in sentences_:
            sentence=sentence.split(",")
            for sent in sentence:
                sentences.append(sent)
        stop_words = set(stopwords.words('english'))
        sentences_=sentences
        sentences=[]
        for sentence in sentences_:
            sentence_words = word_tokenize(sentence.lower())
            sentence_words = [word for word in sentence_words if word.isalnum() and word not in stop_words]
            sentences.append(" ".join(sentence_words))
        return sentences

    def sentences_tf_idf(self, sentences):
        # 计算每个句子的tf-idf值
        vectorizer = TfidfVectorizer(use_idf=True)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        # 获取单词列表
        feature_names = vectorizer.get_feature_names_out()
        # 获取每个句子的tf-idf值之和
        sums = tfidf_matrix.sum(axis=1)
        # 计算每个句子的重要性得分
        scores = []
        for i in range(len(sentences)):
            score = sums[i].item()
            scores.append(score)
        sentences_tf_idf = {}
        for sentence, score in zip(sentences, scores):
            sentences_tf_idf[sentence] = score
        max_score = max(sentences_tf_idf.values())
        min_score = min(sentences_tf_idf.values())
        for i in sentences_tf_idf.keys():  # 权重归一化
            if (max_score - min_score != 0):
                sentences_tf_idf[i] = (sentences_tf_idf[i] - min_score) / (max_score - min_score)
            else:
                sentences_tf_idf[i] = sentences_tf_idf[i]
        return sentences_tf_idf

    def cos_sim(self, a, b):
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm != 0 and b_norm != 0:
            cos = np.dot(a, b) / (a_norm * b_norm)
        else:
            cos = 0
        return cos

    def sentences_cos_sim(self, sentences):
        word_list = []
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            for word in words:
                if word not in word_list:
                    word_list.append(word)
        word_frequency_vector = [[0 for i in range(len(word_list))] for j in range(len(sentences))]
        cnt = 0
        cnt1 = 0
        for word in word_list:
            for sentence in sentences:
                for s_word in nltk.word_tokenize(sentence):
                    if s_word == word:
                        if cnt1 < len(sentences):
                            word_frequency_vector[cnt1][cnt] = word_frequency_vector[cnt1][cnt] + 1
                cnt1 = cnt1 + 1
            cnt = cnt + 1
            cnt1 = 0
        theme = self.Find_Theme(sentences)
        cnt = 0
        num = 0
        for sentence in sentences:
            if sentence == theme:
                num = cnt
            cnt = cnt + 1
        sentence_cos_sim = {}
        tenor_sentence_vector = word_frequency_vector[num]
        cnt = 0
        for sentence in sentences:
            sentence_cos_sim[sentence] = self.cos_sim(tenor_sentence_vector, word_frequency_vector[cnt])
            cnt = cnt + 1
        list1 = sentence_cos_sim.values()
        max_score = max(list1)
        min_score = min(list1)

        for i in sentence_cos_sim.keys():
            if max_score - min_score != 0:
                sentence_cos_sim[i] = (sentence_cos_sim[i] - min_score) / (max_score - min_score)
            else:
                sentence_cos_sim[i] = sentence_cos_sim[i]

        return sentence_cos_sim

        # 计算每个句子的位置权重
    def sentences_pos(self, sentences):
        n = len(sentences)
        sentence_pos = {}
        i = 1
        for sentence in sentences:
            sentence_pos[sentence] = 1-math.pow(i,0.5)
            i = i + 1

        max_score = max(sentence_pos.values())
        min_score = min(sentence_pos.values())

        for i in sentence_pos.keys():
            if max_score - min_score != 0:
                sentence_pos[i] = (sentence_pos[i] - min_score) / (max_score - min_score)
            else:
                 sentence_pos[i] = sentence_pos[i]

        return sentence_pos

    # 计算每个句子的长度权重
    def sentences_len(self, sentences):
        total_length = 0
        n = len(sentences)
        for sentence in sentences:
            total_length = total_length + len(sentence)
        miu = total_length / n

        sigma2 = 0
        for sentence in sentences:
            sigma2 = sigma2 + math.pow((len(sentence) - miu), 2)
        sigma2 = sigma2 / n

        sentence_len = {}
        for sentence in sentences:
            if sigma2 != 0:
                sentence_len[sentence] = (1 / (math.pow(2 * math.pi * sigma2, 0.5))) * math.pow(math.e, -(
                        math.pow((len(sentence) - miu), 2) / (2 * sigma2)))
            else:
                sentence_len[sentence] = 1

        max_score = max(sentence_len.values())
        min_score = min(sentence_len.values())

        for i in sentence_len.keys():
            if max_score - min_score != 0:
                sentence_len[i] = (sentence_len[i] - min_score) / (max_score - min_score)
            else:
                sentence_len[i] = sentence_len[i]

        return sentence_len

        # 计算事件抽取权重
    def sentence_EE(self, sentences):
        handler = ExtractEvent()
        sentence_EE = {}
        for sentence in sentences:
            sentence_EE[sentence]=0.2
        x,events=handler.phrase_ip(sentences)
        for sentence in sentences:
            sentence_="".join(sentence.split())
            for event in events:
                event="".join(event.split())
                if event in sentence_:
                    sentence_EE[sentence] = 0.8
        return sentence_EE

        # 根据观点抽取获得文本主旨句
    def Find_Theme(self, sentences):
        result = []
        backup_sentences = sentences
        old_sentences = sentences
        while set(old_sentences) != set(result):
            result = []
            oe = OpinionExtraction(sentences=sentences)
            sortedKeys = oe.extractor()
            for sortK in sortedKeys:
                str = []
                for op in sortK[1]:
                    str.append(op.opinion)
                result.append(sortK[0])
            old_sentences = sentences
            sentences = result
        if len(sentences)==0:
            theme=backup_sentences[0]
        else:
            theme = sentences[0]
        for sentence in backup_sentences:
            if theme in sentence:
                return sentence

    #计算句子的综合得分
    def sentences_score(self, sentence_tf_idf, sentence_cos_sim, sentence_pos, sentence_len, sentence_EE):
        sentence_score = {}
        for i in sentence_pos.keys():
            sentence_score[i] = sentence_tf_idf[i] + 2 * sentence_cos_sim[i] + 0.5 * sentence_pos[i] + sentence_len[i] + sentence_EE[i]
        return sentence_score

        # 字典排序
    def dic_order_value_and_get_key(self, dicts, count):
        # by hellojesson
         # 字典根据value排序，并且获取value排名前几的key
        final_result = []
        # 先对字典排序
        sorted_dic = sorted([(k, v) for k, v in dicts.items()], reverse=True)
        tmp_set = set()  # 定义集合 会去重元素
        for item in sorted_dic:
            tmp_set.add(item[1])
        for list_item in sorted(tmp_set, reverse=True)[:count]:
            for dic_item in sorted_dic:
                if dic_item[1] == list_item:
                    final_result.append(dic_item[0])
        return final_result

# 根据各个指标得到抽取式摘要结果
    def ExtractSummary(self) -> object:
        sentences = self.text_clean()
        sentence_tf_idf = self.sentences_tf_idf(sentences)
        sentence_cos_sim = self.sentences_cos_sim(sentences)
        sentence_pos = self.sentences_pos(sentences)
        sentence_len = self.sentences_len(sentences)
        sentence_EE = self.sentence_EE(sentences)
        sentence_score = self.sentences_score(sentence_tf_idf, sentence_cos_sim, sentence_pos, sentence_len,sentence_EE)
        final_result = self.dic_order_value_and_get_key(sentence_score, 8)
        finals = []
        for sentence in sentences:
            for final in final_result[:6]:
                if sentence == final:
                    finals.append(sentence)
        return finals



if __name__ == "__main__":
    f = open('/home/hzj/NLP1/SentimentAnalysis/Data/train.txt', encoding='utf-8')
    data = f.readlines()
    f.close()

    f = open('/home/hzj/NLP1/SentimentAnalysis/Data/test1.txt', 'w', encoding='utf-8')
    # 使用tqdm显示进度条
    for line in tqdm(data, desc='Processing'):
        sentence, label = line.split('\t')
        ese = Extract_Summary_English(sentence)
        summaries = ese.ExtractSummary()
        # 将摘要合并为一整段完整的摘要
        full_summary = ' '.join(summaries)

        f.write(f"{full_summary}\t{label}")

    f.close()

    print("提取的摘要已保存到test1.txt文件中。")