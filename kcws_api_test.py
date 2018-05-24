# -*- coding: utf-8 -*-
# @Author: Alley
# @Date:   2016-05-10

import kcws_api
# 语料库目录在../2014

def cwstrain():
        cws = kcws_api.CwsTrain("test/0103")
        cws.prepWord2vec()
        cws.word2vecTrain(size=50)
        cws.prepCws()
        cws.cwsTrain(embeddingSize=50)
        #cws.freeGraph()

def postrain():
        pos = kcws_api.PosTrain("test/0103")
        pos.prepWord2vec()
        pos.word2vecTrain(size=100)
        pos.prepPos()
        pos.posTrain(embeddingWordSize=100)
        #pos.freeGraph()

def usekcws():
        cp = kcws_api.CwsPosUse()
        # cp.setEnv(usePos=False)
        cp.setEnv(usePos=False)
        # "梁伟新/nr 出任/v 漳州市/ns 副市长"
        srcstr ="用毒毒毒蛇毒蛇会不会被毒毒死？"
        outstr = cp.preocessSentence(srcstr)
        print "stcstr : " + srcstr
        print "outstr : " + outstr
        srcstr="她还专门“培养”了一名学生，每天记下各科的作业，并且直接登录校讯通发给家长。"
        outstr = cp.preocessSentence(srcstr)
        print "stcstr : " + srcstr
        print "outstr : " + outstr

        srcstr="来到杨过曾经生活的地方,小龙女动情地说:'我也想过过过儿过过的生活'"
        outstr = cp.preocessSentence(srcstr)
        print "stcstr : " + srcstr
        print "outstr : " + outstr


#cwstrain()
postrain()
#usekcws()



