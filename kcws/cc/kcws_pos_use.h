/*
 *
 * Filename:  kcws_pos_use.cc
 * Author:  Alley
 * Create Time: 2018-04-28
 * Description: use model api
 *
 */
#ifndef __SEG_BACKEND_API_HY__
#define __SEG_BACKEND_API_HY__


#include <string>
#include <thread>
#include <memory>

#include "base/base.h"
#include "utils/basic_string_util.h"
#include "kcws/cc/tf_seg_model.h"
#include "kcws/cc/pos_tagger.h"

class kcwsPosProcess{
public:
    kcwsPosProcess(){}
    void kcws_set_envfile_pars(const char * model_file, const char * vocab_file, const char * pos_model_file,
            const char * word_vocab_file, const char * pos_vocab_file, const int max_sentence_len,
            const int max_word_num, const char * user_dict_file, bool use_pos);
    char* kcws_pos_process(const char* srcsentence);

private:
    kcws::TfSegModel model;
    bool usePos;
};

#endif