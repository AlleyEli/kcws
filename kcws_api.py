# -*- coding: utf-8 -*-
# @Author: Alley
# @Date:   2018-05-10

# Class usage sequence:
# CwsTrain --> PosTrain --> CwsPosUse

import os
import sys
import ctypes

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


# class Log:
#     HEAD = '\033[92m'
#     TAIL = '\033[0m'

#     @staticmethod
#     def p(var):
#         '''
#             print coloured log
#         '''
#         print Log.HEAD, var, Log.TAIL


class CwsPosUse:
    '''
        CwsPosUse 为经过CwsPosTrain训练好模型后,可通过CwsPosUse使用模型进行分词和词性标注
    '''

    def setEnv(self, cwsModelfile=None, cwsVocabfile=None, posModelfile=None, 
            posVocabfile=None, **kwargs):
        '''
            描述:
                设置模型文件和参数,
            参数:
                cwsModelfile: [IN] 分词模型训练后导出后的文件
                cwsVocabfile: [IN] 分词语料生成特征向量导出后的文件
                posModelfile: [IN] 词性标注模型训练后导出后的文件
                posVocabfile: [IN] 词性标注语料生成特征向量导出后的文件
                kwargs:
                  maxSentenceLen: 最大句子长度值,默认值为80
                  maxWordNum: 最大单词长度值,默认值为50
                  userDictfile: [IN] 用户词典文件,可以不设置
        '''
        cwsModel = cwsModelfile if cwsModelfile else "kcws/models/cws_model.pbtxt"
        cwsVocab = cwsVocabfile if cwsVocabfile else "kcws/models/cws_vocab.txt"
        posModel = posModelfile if posModelfile else "kcws/models/pos_model.pbtxt"
        posVocab = posVocabfile if posVocabfile else "kcws/models/pos_vocab.txt"
        wordVocab = "kcws/models/word_vocab.txt"

        _maxSentenceLen = kwargs["maxSentenceLen"] if "maxSentenceLen" in kwargs else 80
        _maxWordNum = kwargs["maxWordNum"] if "maxWordNum" in kwargs else 50
        _userDict = kwargs["userDictfile"] if "userDictfile" in kwargs else ""
        self.kp = py_kcws_pos.kcwsPosProcess()
        self.kp.kcwsSetEnvfilePars(cwsModel, cwsVocab, posModel, wordVocab, posVocab,
                _maxSentenceLen, _maxWordNum, _userDict)

    def preocessSentence(self, srcstr, deststr, usePos=True):
        '''
            描述:
                使用模型进行分词和词性标注
            参数:
                srcstr: [IN] 原始字符串
                deststr: [OUT] 分词后和词性标注后的字符串
                usePos: 是否使用词性标注,若为false则只进行分词
        '''
        self.kp.kcwsPosProcessSentence(srcstr, deststr, usePos)


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
        os.system("mkdir " + tmpdir)
        self.fcharsw2v = self.tmpdir + "chars_for_w2v.txt"
        self.fcwsTrain = self.tmpdir + "train.txt"
        self.fcharvec = self.tmpdir + "chars_vec.txt"
        self.fcwsTest = self.tmpdir + "test.txt"
        self.cwslogdir="cws_logs"

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
    
    def word2vecTrain(self, charvecfile=None, **kwargs):
        '''
        描述:
            通过word2vec训练字频表生成字特征向量
        参数:
            charsW2Vfile: [OUT] 生成的字特征向量表文件
            kwargs
                size: 特征向量的维度,默认100
                sample: 高频词汇的随机降采样的配置阈值,默认1e-3
                negative: 是否使用Negative Sampling, 0不使用, >0为使用
                hs: 是否使用Hierarchical Softmax算法,默认为0不使用, 1为使用,与negative互斥
                binary: 将生成的向量保存在二进制代码中,默认为0不保存
                iter: 迭代次数,默认为5
                window: 表示当前词与预测词在一个句子中的最大距离是多少,默认值5
                cbow: 是否使用cbow模型,0表示使用skip-gram模型,1表示使用cbow模型,默认1
                mincount: 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉,默认值5
        '''
        _size = kwargs["size"] if "size" in kwargs else 100
        _sample = kwargs["sample"] if "sample" in kwargs else 1e-3
        _negative = kwargs["negative"] if "negative" in kwargs else 5
        _hs = kwargs["hs"] if "hs" in kwargs else 0
        _binary = kwargs["binary"] if "binary" in kwargs else 0
        _iter = kwargs["iter"] if "iter" in kwargs else 5
        _window = kwargs["window"] if "window" in kwargs else 5
        _cbow = kwargs["cbow"] if "cbow" in kwargs else 1
        _mincount = kwargs["mincount"] if "mincount" in kwargs else 5

        if charvecfile:
            self.fcharvec = charvecfile
                
        self.w2v.word2vec_train.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, 
                ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.w2v.word2vec_train(self.fcharsw2v, self.fcharvec, _size, _sample, _negative, 
                _hs, _binary, _iter, _window, _cbow, _mincount)

    def prepCws(self):
        '''
            描述:对word2vec生成的vec处理成cwstrain需要的文件
        '''
        allfile = self.tmpdir + "cws_all.file"
        gt.generateTraining(self.fcharvec, self.corpusdir, allfile)
        fs.filter_sentence(allfile, self.fcwsTrain, self.fcwsTest)

    def cwsTrain(self, useIdcnn=True, **kwargs):
        '''
        描述:
            进行分词模型训练
        参数:
            useIdcnn: 使用Idcnn算法还是Bi-LTSM算法,默认为True用Idcnn
            kwargs:
                maxSentenceLen: 最大句子长度,默认值80
                embeddingSize: 特征向量维度,默认值50
                numTags: 标签数量,默认值4
                numHidden: 隐含层单元数量,默认值100
                batchSize: 每次送给神经网络的样本数量,默认值100
                trainSteps:训练次数,默认值150000
                learningRate: 学习率,默认值0.001
                trackHistory: 最大历史精度跟踪次数,默认值6
        '''

        _maxSentenceLen = kwargs["maxSentenceLen"] if "maxSentenceLen" in kwargs else 80
        _embeddingSize = kwargs["embeddingSize"] if "embeddingSize" in kwargs else 50
        _numTags = kwargs["numTags"] if "numTags" in kwargs else 4
        _numHidden = kwargs["numHidden"] if "numHidden" in kwargs else 100
        _batchSize = kwargs["batchSize"] if "batchSize" in kwargs else 100
        _trainSteps = trainSteps["trainSteps"] if "trainSteps" in kwargs else 150000
        _learningRate = kwargs["learningRate"] if "learningRate" in kwargs else 0.001
        _trackHistory = kwargs["trackHistory"] if "trackHistory" in kwargs else 6

        tc.cws_train(self.fcwsTrain, self.fcwsTest, self.fcharvec, self.cwslogdir, _numHidden, 
                _batchSize, _trainSteps, _trackHistory, _maxSentenceLen, _embeddingSize, 
                _numTags, _learningRate, useIdcnn)

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
        os.system("mkdir " + self.tmpdir)
        
        self.fposlinesUnk = self.tmpdir + "pos_lines_with_unk.txt"
        self.fwordvec = self.tmpdir + "word_vec.txt"
        self.posTrain = self.tmpdir + "train.txt"
        self.posTest = self.tmpdir + "test.txt"
        # fcharvec使用cwsword2vec训练结果文件
        self.fcharvec = "cws_train_tmp/chars_vec.txt"
        self.poslogdir = "pos_logs"

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

    def word2vecTrain(self, worldVecfile=None, **kwargs):
        '''
        描述:
            通过word2vec训练词频表生成词特征向量
        参数:
            charsW2Vfile: [OUT] 生成的词特征向量表文件
            kwargs
                size: 词特征向量的维度,默认值100
                sample: 高频词汇的随机降采样的配置阈值,默认值1e-3
                negative: 是否使用Negative Sampling, 0不使用, >0为使用
                hs: 是否使用Hierarchical Softmax算法,默认为0不使用, 1为使用,与negative互斥
                binary: 将生成的向量保存在二进制代码中,默认为0不保存
                iter: 迭代次数,默认值5
                window: 表示当前词与预测词在一个句子中的最大距离是多少,默认值5
                cbow: 是否使用cbow模型,0表示使用skip-gram模型,1表示使用cbow模型,默认值1
                mincount: 可以对词典做截断. 词频少于min_count次数的单词会被丢弃掉,默认值5
        '''
        _size = kwargs["size"] if "size" in kwargs else 100
        _sample = kwargs["sample"] if "sample" in kwargs else 1e-3
        _negative = kwargs["negative"] if "negative" in kwargs else 5
        _hs = kwargs["hs"] if "hs" in kwargs else 0
        _binary = kwargs["binary"] if "binary" in kwargs else 0
        _iter = kwargs["iter"] if "iter" in kwargs else 5
        _window = kwargs["window"] if "window" in kwargs else 5
        _cbow = kwargs["cbow"] if "cbow" in kwargs else 1
        _mincount = kwargs["mincount"] if "mincount" in kwargs else 5

        if worldVecfile:
            self.fwordvec = worldVecfile

        self.w2v.word2vec_train.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, 
                ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.w2v.word2vec_train(self.fposlinesUnk, self.fwordvec, _size, _sample, _negative, 
                _hs, _binary, _iter, _window, _cbow, _mincount)

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
        os.system("head -n "+ str(int(lines*0.75)) +" " + allfile +" > " + self.posTrain)
        os.system("tail -n "+ str(int(lines*0.25)) +" " + allfile +" > " + self.posTest)
        
    def posTrain(self, **kwargs):
        '''
            描述:
                进行词性标注训练
            参数:
                kwargs:
                  maxSentenceLen: 最大句子长度,默认值50
                  embeddingWordSize: 词特征向量维度,默认值150
                  embeddingCharSize: 字特征向量维度,默认值50
                  numTags: 词性标注标签数量,默认值74
                  numHidden: 隐含层单元数量,默认值100
                  batchSize: 每次送给神经网络的样本数量,默认值64
                  trainSteps:训练次数,默认值50000
                  learningRate: 学习率,默认值0.001
                  charWindowSize: 字符卷积的窗口大小,默认值2
                  maxCharsPerWord: 单词最大字符数量,默认值5
        '''

        _maxSentenceLen = kwargs["maxSentenceLen"] if "maxSentenceLen" in kwargs else 50
        _embeddingWordSize = kwargs["embeddingWordSize"] if "embeddingWordSize" in kwargs else 150
        _embeddingCharSize = kwargs["embeddingCharSize"] if "embeddingCharSize" in kwargs else 50
        _numTags = kwargs["numTags"] if "numTags" in kwargs else 74
        _numHidden = kwargs["numHidden"] if "numHidden" in kwargs else 100
        _batchSize = kwargs["batchSize"] if "batchSize" in kwargs else 64
        _trainSteps = trainSteps["trainSteps"] if "trainSteps" in kwargs else 50000
        _learningRate = kwargs["learningRate"] if "learningRate" in kwargs else 0.001
        _charWindowSize = kwargs["charWindowSize"] if "charWindowSize" in kwargs else 2
        _maxCharsPerWord = kwargs["maxCharsPerWord"] if "maxCharsPerWord" in kwargs else 5

        tp.pos_train(self.posTrain, self.posTest, self.fwordvec, self.fcharvec, self.poslogdir, 
                _maxSentenceLen, _embeddingWordSize, _embeddingCharSize, _numTags, 
                _charWindowSize, _maxCharsPerWord, _numHidden, _batchSize, _trainSteps, _learningRate)

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