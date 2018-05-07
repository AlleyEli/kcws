licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "word2vec",
    srcs = [
        "word2vec.c",
    ],
    linkopts = [
        "-pthread",
    ],
)

cc_library(
    name = "word2vec_hy",
    srcs = [
        "word2vec_hy.c",
    ],
    hdrs = ["word2vec_hy.h"],
    linkopts = [
        "-pthread",
    ],
)

cc_binary(
    name = "distance",
    srcs = [
        "distance.cc",
    ],
    deps = [
        "//utils:basic_string_util",
    ],
)
