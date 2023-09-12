# coding=utf-8
import re, os
import jieba.posseg as pseg
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize


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


import time

if __name__ == "__main__":
    handler = ExtractEvent()
    start = time.time()
    with open("./text3.txt", 'r', encoding='utf-8') as tt:
        content1 = tt.readlines()
    events, spos = handler.phrase_ip(content1)
    print("result")
    print(spos)

