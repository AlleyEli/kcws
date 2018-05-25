

# 构建及API使用

*-alley-*


* 构建  
  1. 环境配置: **python 2.7.15 + bazel0.5.4 + tensorflow1.4**  
  2. 切换到本项目代码目录，运行  
    > **./configure**  
    > **./compiletool**  
* 参数配置: 修改parameters.json文件即可
* 使用  
  参考: **kcws_api_test.py** 和 **kcws_api.py** 直接训练和使用  
  * 注意类使用顺序: CwsTrain --> PosTrain --> CwsPosUse  
  * 测试时先可以用: python kcws_api_test.py执行代码(已屏蔽代码训练,只是执行使用模型)  
    看看测试结果:   
  ```
  2018-05-19 09:42:11.919420: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
  2018-05-19 09:42:11.919452: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
  WARNING: Logging before InitGoogleLogging() is written to STDERR
  I0519 09:42:11.927290 10086 tfmodel.cc:64] Reading file to proto: kcws/models/cws_model.pbtxt
  I0519 09:42:11.930253 10086 tfmodel.cc:69] Creating session.
  I0519 09:42:11.960114 10086 tfmodel.cc:77] Tensorflow graph loaded from: kcws/models/cws_model.pbtxt
  2018-05-19 09:42:11.961064: I kcws/cc/tf_seg_model.cc:239] Reading from layer transitions
  2018-05-19 09:42:11.961097: I kcws/cc/tf_seg_model.cc:245] got num tag:4
  2018-05-19 09:42:11.980288: I kcws/cc/tf_seg_model.cc:258] Total word :5169
  I0519 09:42:11.980355 10086 tfmodel.cc:64] Reading file to proto: kcws/models/pos_model.pbtxt
  I0519 09:42:12.002810 10086 tfmodel.cc:69] Creating session.
  I0519 09:42:12.009234 10086 tfmodel.cc:77] Tensorflow graph loaded from: kcws/models/pos_model.pbtxt
  2018-05-19 09:42:12.026460: I kcws/cc/pos_tagger.cc:218] Reading from layer transitions
  2018-05-19 09:42:12.026487: I kcws/cc/pos_tagger.cc:224] got num tag:74
  stcstr : 梁伟新出任漳州市副市长
  outstr : 梁伟新/nr 出任/nz 漳州市/nz 副市长/nn
  ```

  * 具体使用参考kcws_api_test.py修改即可.用法基本一样

