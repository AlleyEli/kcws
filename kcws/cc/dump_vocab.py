# -*- coding: utf-8 -*-
# @Author: Koth
# @Date:   2016-11-20 15:04:18
# @Last Modified by:   Koth
# @Last Modified time: 2016-11-20 15:07:51
import sys
import os
sys.path.append(r"bazel-bin/utils")
import w2v


def dumpVocab(charVecfile, basicVocabfile):
  vob = w2v.Word2vecVocab()
  vob.Load(charVecfile)
  vob.DumpBasicVocab(basicVocabfile)