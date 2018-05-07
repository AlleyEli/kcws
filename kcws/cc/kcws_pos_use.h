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
                            const char * word_vocab_file, const char * pos_vocab_file, const int max_sentencelen,
                            const int max_wordnum, const char * user_dict_file);
    void kcws_pos_process(const char* srcsentence, char * outsentence, int use_pos);

private:
    std::string model_path;
    std::string vocab_path;
    std::string pos_model_path;
    std::string word_vocab_path;
    std::string pos_vocab_path;
    std::string user_dict_path;

    int max_sentence_len;
    int max_word_num;
};

#endif