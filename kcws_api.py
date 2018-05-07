# -*- coding: utf-8 -*-
# @Author: Alley
# @Date:   2016-04-25

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


class Ikcws:
    def __init__(self, corpusdir):
        self.corpusdir = corpusdir
        self.w2v = ctypes.cdll.LoadLibrary("bazel-bin/third_party/word2vec/libword2vec_hy.so")
        os.system("mkdir cws_train_tmp")
        os.system("mkdir pos_train_tmp")

    def processAnnoFile(self, preCharsW2Vfile):
        paf.processAnnoFile(self.corpusdir, preCharsW2Vfile)
        
    def processAnnoSentence(self, srcstr):
        return paf.processAnnoSentence(srcstr)

    def getVocab(self, preCharsW2Vfile, vocabfile, minCount):
        self.w2v.word2vec_get_vocab.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        self.w2v.word2vec_get_vocab(preCharsW2Vfile, vocabfile, minCount)

    def replaceUNK(self, preCharsW2Vfile, vocabfile, charsW2Vfile):
        ru.replaceUNK(vocabfile, preCharsW2Vfile, charsW2Vfile)

    def word2vecTrain(self, charsW2Vfile, charvecfile, size=100, sample=1e-3, negative=5,
            hs=0, binary=0, iter=0, window=5, cbow=1, mincount=5):
        self.w2v.word2vec_train.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, 
                ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.w2v.word2vec_train(charsW2Vfile, charvecfile, size, sample, negative, 
                hs, binary, iter, window, cbow, mincount)

    def generateTrainFile(self, charvecfile, allfile):
        gt.generateTraining(charvecfile, self.corpusdir, allfile)

    def filterCwsTrainfile(self, allfile, trainfile, testfile):
        fs.filter_sentence(allfile, trainfile, testfile)

    def cwsTrain(self, trainfile, testfile, charvecfile, logdir="logs", maxSentenceLen=80,
            embeddingSize=50, numTags=4, numHidden=100, batchSize=100, trainSteps=150000,
            learningRate=0.001, useIdcnn=True, trackHistory=6):
        tc.cws_train(trainfile, testfile, charvecfile, logdir, numHidden, batchSize, trainSteps,
                trackHistory, maxSentenceLen, embeddingSize, numTags,learningRate, useIdcnn)

    def dumpVocab(self, charvecfile, basicVocabfile):
        dv.dumpVocab(charvecfile, basicVocabfile)


   # Pos fun
    def preparePos(self, pos_linesfile):
        pp.prepare_pos(self.corpusdir, pos_linesfile)

    def statsPos(self, pos_vobfile, linesPosfile):
        sp.stats_pos(self.corpusdir, pos_vobfile, linesPosfile)

    def generatePosTrainFile(self, word_vobfile, char_vobfile, pos_vobfile, pos_trainfile):
        gps.generatepostrain(word_vobfile, char_vobfile, pos_vobfile, self.corpusdir,  pos_trainfile)
        
    def filterPosTrainfile(self, pos_trainfile, trainfile, testfile):
        lines = len(open(pos_trainfile,'rU').readlines())
        print "pos_trainfile lines: ", lines
        os.system("sort -u "+ pos_trainfile + " > pos_train.u")
        os.system("shuf pos_train.u > " + pos_trainfile)
        os.system("head -n "+ str(int(lines*0.75)) +" " + pos_trainfile +" > " + trainfile)
        os.system("tail -n "+ str(int(lines*0.25)) +" " + pos_trainfile +" > " + testfile)
        os.system("rm pos_train.u")

    def posTrain(self, trainfile, testfile, word_vecfile, charvecfile, log_dir="poslogs",
            maxSentenceLen=50, embeddingWordSize=150, embeddingCharSize=50,
            numTags=74, charWindowSize=2, maxCharsPerWord=5, numHidden=100,
            batchSize=64, trainSteps=50000, learningRate=0.001):
        tp.pos_train(trainfile, testfile, word_vecfile, charvecfile, log_dir, maxSentenceLen, 
                embeddingWordSize,embeddingCharSize, numTags,charWindowSize, maxCharsPerWord,
                numHidden, batchSize, trainSteps, learningRate)

    def posFreeGraph(self, inputGraph, inputCheckPoint, outputNodeNames, outputGraph):
        fp.freeze_graph(inputGraph, inputCheckPoint, outputNodeNames, outputGraph)

    def kcwsUseSetEnv(self, modelfile="kcws/models/seg_model.pbtxt", vocabfile="kcws/models/basic_vocab.txt", 
            posModelfile="kcws/models/pos_model.pbtxt", wordVocabfile="kcws/models/word_vocab.txt", 
            posVocabfile="kcws/models/pos_vocab.txt", maxSentenceLen=80,maxWordNum=50,userDictfile=""):
        self.kp = py_kcws_pos.kcwsPosProcess()
        self.kp.kcwsSetEnvfilePars(modelfile, vocabfile, posModelfile, wordVocabfile, posVocabfile,
                maxSentenceLen, maxWordNum, userDictfile)

    def kcwsUseProcessSentence(self, srcstr, deststr, usePos=True):
        self.kp.kcwsPosProcessSentence(srcstr, deststr, usePos)