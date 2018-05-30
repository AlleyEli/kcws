# -*- coding: utf-8 -*-
# @Author: Alley
# @Date:   2016-05-10

import kcws_api
# 语料库目录在../2014

def cwstrain():
        cws = kcws_api.CwsTrain("../2014")
        cws.prepWord2vec()
        cws.word2vecTrain(size=50)
        cws.prepCws()
        cws.cwsTrain(embeddingSize=50)
        cws.freeGraph()

def postrain():
        pos = kcws_api.PosTrain("../2014")
        pos.prepWord2vec()
        pos.word2vecTrain(size=100)
        pos.prepPos()
        pos.posTrain(embeddingWordSize=100)
        pos.freeGraph()

def usekcws():
        cp = kcws_api.CwsPosUse(debug=True)

        srcstr ="梁伟新出任漳州市副市长"
        print srcstr
        outstr = cp.preocessSentence(srcstr)

# cwstrain()
# postrain()
usekcws()



