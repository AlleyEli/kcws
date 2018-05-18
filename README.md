

# 构建及API使用


* 构建
  1. 环境配置: **python 2.7.15 + bazel0.5.4 + tensorflow1.4**
  2. 切换到本项目代码目录，运行
    > **./configure  **
    > **./compiletool**
* 使用
  参考: **kcws_api_test.py** 和 **kcws_api.py** 直接训练和使用
  * 注意类使用顺序: cwsTrain --> posTrain --> CwsPosUse
