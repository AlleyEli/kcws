# -*- coding: utf-8 -*-
# @Author: Alley
# @Date:   2016-05-02


import kcws_api3
# cws
def cwstrain():
        cws = kcws_api3.CwsTrain("test/0103")
        cws.prepWord2vec()
        cws.word2vecTrain(size=50, sample=1e-4)
        cws.prepCws()
        # cws.cwsTrain()
        # cws.freeGraph()

def postrain():
        pos = kcws_api3.PosTrain("test/0103")
        pos.prepWord2vec()
        pos.word2vecTrain()
        pos.prepPos()
        # pos.posTrain()
        # pos.freeGraph()
        

def usekcws():
        cp = kcws_api3.CwsPosUse()
        cp.setEnv()
        outstr = ""
        cp.preocessSentence("我是中国人", outstr)
       

#cwstrain()
#postrain()
usekcws()



