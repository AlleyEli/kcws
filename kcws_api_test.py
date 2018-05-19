# -*- coding: utf-8 -*-
# @Author: Alley
# @Date:   2016-05-10

import kcws_api
# 语料库目录在../2014

def cwstrain():
        cws = kcws_api.CwsTrain("../2014")
        cws.prepWord2vec()
        cws.word2vecTrain(size=50, sample=1e-4, hs=1)
        cws.prepCws()
        cws.cwsTrain(embeddingSize=50)
        cws.freeGraph()

def postrain():
        pos = kcws_api.PosTrain("../2014")
        pos.prepWord2vec()
        pos.word2vecTrain(size=100, sample=1e-4, hs=1, cbow=0)
        pos.prepPos()
        pos.posTrain(embeddingWordSize=100)
        pos.freeGraph()

def usekcws():
        cp = kcws_api.CwsPosUse()
        # cp.setEnv(usePos=False)
        cp.setEnv(usePos=True)
        # "梁伟新/nr 出任/v 漳州市/ns 副市长"
        srcstr="梁伟新出任漳州市副市长"
        outstr = cp.preocessSentence(srcstr)
        print "stcstr : " + srcstr
        print "outstr : " + outstr


# cwstrain()
# postrain()
usekcws()



