# -*- coding: utf-8 -*-
# @Author: Alley
# @Date:   2016-05-02


import kcws_api
# cws
def cwstrain():
        kcws = kcws_api.Ikcws("../../2014")
        kcws.processAnnoFile("test2/preCharsW2Vfile.txt")
        kcws.getVocab("test2/preCharsW2Vfile.txt","test2/pre_vocab.txt", 3)
        kcws.replaceUNK("test2/preCharsW2Vfile.txt", "test2/pre_vocab.txt", "test2/chars_for_w2v.txt")
        kcws.word2vecTrain("test2/chars_for_w2v.txt", "test2/charvec.txt", size=50, sample=1e-4, negative=5, hs=1, binary=0, iter=5)
        kcws.generateTrainFile("test2/charvec.txt","test2/all.txt")
        kcws.filterCwsTrainfile("test2/all.txt", "test2/train.txt", "test2/test.txt")
        kcws.cwsTrain("test2/train.txt", "test2/test.txt", "test2/charvec.txt", maxSentenceLen=80, learningRate=0.001)
        kcws.dumpVocab("test2/charvec.txt","kcws/models/basic_vocab.txt")


def postrain():
        kcws = kcws_api.Ikcws("../../2014")
        kcws.preparePos("test2/pos_lines.txt")
        kcws.getVocab("test2/pos_lines.txt", "test2/pre_word_vec.txt", 3)
        kcws.replaceUNK("test2/pre_word_vec.txt", "test2/pos_lines.txt", "test2/pos_lines_with_unk.txt")
        kcws.word2vecTrain("test2/pos_lines_with_unk.txt", "test2/word_vec.txt", size=150, sample=1e-4,
                        negative=5, binary=0, cbow=0, iter=3, mincount=5, hs=1)
        kcws.statsPos("test2/pos_vobcab.txt", "test2/lines_withpos.txt")
        kcws.generatePosTrainFile("test2/word_vec.txt", "test2/charvec.txt", "test2/pos_vobcab.txt", "test2/pos_train.txt")
        kcws.filterPosTrainfile("test2/pos_train.txt", "test2/ptrain.txt", "test2/ptest.txt")
        kcws.posTrain("test2/ptrain.txt", "test2/ptest.txt","test2/word_vec.txt", "test2/charvec.txt")
        kcws.posFreeGraph("pos_logs/graph.pbtxt", "pos_logs/model.ckpt", "transitions,Reshape_9", "kcws/models/pos_model.pbtxt")

def usekcws():
        import kcws_api
        kcws = kcws_api.Ikcws("../../2014")
        srcstr= "她还专门“培养”了一名学生，每天记下各科的作业，并且直接登录校讯通发给家长。"
        kcws.kcwsUseSetEnv()
        outstr=""
        kcws.kcwsUseProcessSentence(srcstr, outstr)

usekcws()



