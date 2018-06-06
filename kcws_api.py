# -*- coding: utf-8 -*-
# @Author: Alley
# @Date:   2018-05-10

# Class usage sequence:
# CwsTrain --> PosTrain --> CwsPosUse

import os
import sys
import ctypes
import json

sys.path.append(r"kcws/train")
sys.path.append(r"kcws/cc")
sys.path.append(r"tools")
sys.path.append(r"bazel-bin/kcws/train")
sys.path.append(r"bazel-bin/kcws/cc/")

import process_anno_file as paf
import replace_unk as ru
import generate_training as gt
import filter_sentence as fs
import train_cws_hy as tc
import py_kcws_pos
import prepare_pos as pp
import stats_pos as sp
import generate_pos_train as gps
import dump_vocab as dv
import train_pos_hy as tp
import freeze_graph as fp


# Docoder parameters.json file
class JsonDecoder:
    pars = {}
    @staticmethod
    def load(jsonfilepath):
        '''
            描述:
                解析jsonfilepath文件
            参数:
                jsonfilepath: [IN] json文件路径
        '''
        jfile = open(jsonfilepath,"rb")
        JsonDecoder.pars = json.load(jfile)

    @staticmethod
    def getPars(category):
        '''
            描述:
                获取训练参数
            参数:
                category: 获取的参数类别,只能为下面五种
                    "cws_word2vec"
                    "cws_train"
                    "pos_word2vec"
                    "pos_train"
                    "cws_pos_use"
            返回值: 参数的字典
        '''
        return JsonDecoder.pars[category]


# Use Chinese Word Segment and Part-of-speech tagging model --cwsposuse--
class CwsPosUse:
    '''
        CwsPosUse 为经过CwsPosTrain训练好模型后,可通过CwsPosUse使用模型进行分词和词性标注
    '''
    def __init__(self, maxSentenceLen=80, maxWordNum=50, usePos=False, debug=False):
        '''
            描述:
                初始化环境,即设置模型文件和各参数,
            参数:
                  maxSentenceLen: 最大句子长度值,默认值为80
                  maxWordNum: 最大单词长度值,默认值为50
                  usePos: 是否使用词性标注, 默认为True,若为false则只进行分词
                  debug: 若为True, 分词结果每次都会打印到屏幕
        '''
        JsonDecoder.load("parameters.json")
        pars = JsonDecoder.getPars("cws_pos_use")

        self.kp = py_kcws_pos.kcwsPosProcess()
        self.debug = debug

        self.kp.kcwsSetEnvfilePars(pars["fcwsModel"], pars["fcwsVocab"], pars["fposModel"], 
                pars["fposVocab"], maxSentenceLen, maxWordNum, pars["fuserDict"], usePos)

    def preocessSentence(self, srcstr):
        '''
            描述:
                使用模型进行分词[和词性标注]
            参数:
                srcstr: [IN] 原始字符串
            返回值: 
                [OUT] 分词后[和词性标注后]的字符串
        '''
        outstr = self.kp.kcwsPosProcessSentence(srcstr)
        if self.debug:
            print '[IN]:  \033[92m' + srcstr + '\033[0m'
            print '[OUT]: \033[92m' + outstr + '\033[0m'
        return outstr


# Chinese Word Segment --cws--
class CwsTrain:
    '''
        对语料进行预处里和分词训练
    '''

    def __init__(self, corpusdir):
        '''
            描述:
                设置语料库目录,并加载word2vec_动态库,
                创建temp文件夹,设置中间文件存放位置用来存放一些中间文件
            参数:
                corpusdir: [IN] 语料库目录路径
        '''
        self.corpusdir = corpusdir
        self.w2v = ctypes.cdll.LoadLibrary("bazel-bin/third_party/word2vec/libword2vec_hy.so")
        self.tmpdir = "cws_train_tmp/"
        os.system("mkdir -p " + self.tmpdir)
        self.fcharsw2v = self.tmpdir + "chars_for_w2v.txt"
        self.fcwsTrain = self.tmpdir + "train.txt"
        self.fcharvec = self.tmpdir + "chars_vec.txt"
        self.fcwsTest = self.tmpdir + "test.txt"
        self.cwslogdir="cws_logs"
        JsonDecoder.load("parameters.json")

    def prepWord2vec(self):
        '''
            描述:对语料库进行一定的处理,以便适合word2vec进行train
        '''
        fpreCharsw2v = self.tmpdir + "pre_chars_for_w2v.txt"
        paf.processAnnoFile(self.corpusdir, fpreCharsw2v)

        minCount = 3
        fpreVocab = self.tmpdir + "pre_vocab.txt"
        self.w2v.word2vec_get_vocab.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        self.w2v.word2vec_get_vocab(fpreCharsw2v, fpreVocab, minCount)
        ru.replaceUNK(fpreVocab, fpreCharsw2v, self.fcharsw2v)

    def word2vecTrain(self, size=100, mincount=5):
        '''
        描述:
            通过word2vec训练字频表生成字特征向量
        参数:
            size: 特征向量的维度,默认100
            mincount: 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉,默认值5
        '''
        pars = JsonDecoder.getPars("cws_word2vec")
                
        self.w2v.word2vec_train.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, 
                ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.w2v.word2vec_train(self.fcharsw2v, self.fcharvec, size, pars["sample"], 
                pars["negative"], pars["hs"], pars["binary"], pars["iter"], pars["window"],
                pars["cbow"], mincount)

    def prepCws(self):
        '''
            描述:对word2vec生成的vec处理成cwstrain需要的文件
        '''
        allfile = self.tmpdir + "cws_all.file"
        gt.generateTraining(self.fcharvec, self.corpusdir, allfile)
        fs.filter_sentence(allfile, self.fcwsTrain, self.fcwsTest)

    def cwsTrain(self, useIdcnn=True, maxSentenceLen=80, embeddingSize=50):
        '''
        描述:
            进行分词模型训练
        参数:
            useIdcnn: 使用Idcnn算法还是Bi-LTSM算法,默认为True用Idcnn
            kwargs:
                maxSentenceLen: 最大句子长度,默认值80
                embeddingSize: 特征向量维度,默认值50
        '''
        pars = JsonDecoder.getPars("cws_train")
        tc.cws_train(self.fcwsTrain, self.fcwsTest, self.fcharvec, self.cwslogdir,
                pars["numHidden"], pars["batchSize"], pars["trainSteps"], pars["trackHistory"],
                maxSentenceLen, embeddingSize, pars["numTags"], pars["learningRate"], useIdcnn)

    def freeGraph(self, cwsVocabfile=None, outputGraphfile=None):
        '''
        描述:
            导出vocab和model
        参数:
            cwsVocabfile: [OUT] 导出的vocab文件
            outputGraphfile: [OUT] 导出的graph文件
        '''

        cwsVocab = cwsVocabfile if cwsVocabfile else "kcws/models/cws_vocab.txt"
        dv.dumpVocab(self.fcharvec, cwsVocab)

        outputGraph = outputGraphfile if outputGraphfile else "kcws/models/cws_model.pbtxt"
        inputGraph = self.cwslogdir + "/graph.pbtxt"
        inputCheckPoint = self.cwslogdir + "/model.ckpt"
        outputNodeNames = "transitions,Reshape_7"
        fp.freeze_graph(inputGraph, inputCheckPoint, outputNodeNames, outputGraph)

# Part-of-speech tagging --pos--
class PosTrain:
    '''
        对语料进行预处理和词性标注训练
    '''
    def __init__(self, corpusdir):
        '''
            描述:
                设置语料库目录,并加载word2vec_动态库,
                创建temp文件夹用来存放一些中间文件
            参数:
                corpusdir: [IN] 语料库目录路径
        '''
        self.corpusdir = corpusdir
        self.w2v = ctypes.cdll.LoadLibrary("bazel-bin/third_party/word2vec/libword2vec_hy.so")
        self.tmpdir = "pos_train_tmp/"
        os.system("mkdir -p " + self.tmpdir)
        
        self.fposlinesUnk = self.tmpdir + "pos_lines_with_unk.txt"
        self.fwordvec = self.tmpdir + "word_vec.txt"
        self.fposTrain = self.tmpdir + "train.txt"
        self.fposTest = self.tmpdir + "test.txt"
        # fcharvec使用cwsword2vec训练结果文件
        self.fcharvec = "cws_train_tmp/chars_vec.txt"
        self.poslogdir = "pos_logs"
        JsonDecoder.load("parameters.json")

    def prepWord2vec(self):
        '''
            描述:
                处理语料库文件以便适合word2vec进行train
        '''
        fposLines = self.tmpdir + "pos_lines.txt"
        pp.prepare_pos(self.corpusdir, fposLines)

        minCount = 5
        fvocab = self.tmpdir + "pre_word_vec.txt"
        self.w2v.word2vec_get_vocab.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        self.w2v.word2vec_get_vocab(fposLines, fvocab, minCount)

        ru.replaceUNK(fvocab, fposLines, self.fposlinesUnk)

    def word2vecTrain(self, size=100, mincount=5):
        '''
        描述:
            通过word2vec训练词频表生成词特征向量
        参数:
            size: 词特征向量的维度,默认值100
            mincount: 可以对词典做截断. 词频少于min_count次数的单词会被丢弃掉,默认值5
        '''
        pars = JsonDecoder.getPars("pos_word2vec")

        self.w2v.word2vec_train.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, 
                ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.w2v.word2vec_train(self.fposlinesUnk, self.fwordvec, size, pars["sample"], 
                pars["negative"], pars["hs"], pars["binary"], pars["iter"], pars["window"],
                pars["cbow"], mincount)

    def prepPos(self):
        '''
            描述:
                通过词向量,字向量生成训练所需文本
        '''
        linPos = self.tmpdir + "lines_withpos.txt"
        fposVocab = self.tmpdir + "pos_vocab.txt"
        sp.stats_pos(self.corpusdir, fposVocab, linPos)
        allfile = self.tmpdir + "pos_all.txt"

        gps.generatepostrain(self.fwordvec, self.fcharvec, fposVocab, self.corpusdir, allfile)
        lines = len(open(allfile,'rU').readlines())
        print "allfile lines: ", lines
        os.system("sort -u "+ allfile + " > pos_train_tmp/pos_train.u")
        os.system("shuf pos_train_tmp/pos_train.u > " + allfile)
        os.system("head -n "+ str(int(lines*0.75)) +" " + allfile +" > " + self.fposTrain)
        os.system("tail -n "+ str(int(lines*0.25)) +" " + allfile +" > " + self.fposTest)
        os.system("cp " + fposVocab + " kcws/models/")
        
    def posTrain(self, maxSentenceLen=50, embeddingWordSize=150, embeddingCharSize=50):
        '''
            描述:
                进行词性标注训练
            参数:
                maxSentenceLen: 最大句子长度,默认值50
                embeddingWordSize: 词特征向量维度,默认值150
                embeddingCharSize: 字特征向量维度,默认值50
        '''
        pars = JsonDecoder.getPars("pos_train")

        tp.pos_train(self.fposTrain, self.fposTest, self.fwordvec, self.fcharvec, self.poslogdir, 
                maxSentenceLen, embeddingWordSize, embeddingCharSize, pars["numTags"], 
                pars["charWindowSize"], pars["maxCharsPerWord"], pars["numHidden"],
                pars["batchSize"], pars["trainSteps"], pars["learningRate"])

    def freeGraph(self, outputGraphfile=None):
        '''
        描述:
            导出posTrain训练好的model
        参数:
            outputGraphfile: [OUT] 导出的graph文件
        '''
        inputGraph = self.poslogdir + "/graph.pbtxt"
        inputCheckPoint = self.poslogdir + "/model.ckpt"
        outputNodeNames = "transitions,Reshape_9"
        outputGraph = outputGraphfile if outputGraphfile else "kcws/models/pos_model.pbtxt"
        fp.freeze_graph(inputGraph, inputCheckPoint, outputNodeNames, outputGraph)
