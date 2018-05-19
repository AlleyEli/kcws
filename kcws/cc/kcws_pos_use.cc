/*
 *
 * Filename:  kcws_pos_use.cc
 * Author:  Alley
 * Create Time: 2018-04-28
 * Description: use model api
 *
 */
#include "kcws_pos_use.h"


void kcwsPosProcess::kcws_set_envfile_pars(const char * cws_model_file,
                           const char * cws_vocab_file,
                           const char * pos_model_file,
                           const char * pos_vocab_file,
                           const int max_sentence_len,
                           const int max_word_num,
                           const char * user_dict_file,
                           const bool use_pos){
    usePos = use_pos;
    CHECK(model.LoadModel(cws_model_file,
                          cws_vocab_file,
                          max_sentence_len,
                          user_dict_file)) << "Load cws model error";
    if (usePos) {
        kcws::PosTagger* tagger = new kcws::PosTagger;
        CHECK(tagger->LoadModel(pos_model_file,
                                pos_vocab_file,
                                cws_vocab_file,
                                pos_vocab_file,
                                max_word_num)) << "load pos model error";
        model.SetPosTagger(tagger);
    }
}


char* kcwsPosProcess::kcws_pos_process(const char* srcsentence) {
    std::string sentence = srcsentence;
    std::string resultsentence = "";
    std::vector<std::string> result;
    std::vector<std::string> tags;

    if (usePos) {
        CHECK(model.Segment(sentence, &result, &tags)) << "segment error 1";
        if (result.size() == tags.size()) {
            int nl = result.size();
            for (int i = 0; i < nl; i++) {
                resultsentence += result[i] + "/" + tags[i] + " ";
            }
        } else {
            for (std::string str : result) {
                resultsentence += str + ' ';
            }
        }
    } else {
        CHECK(model.Segment(sentence, &result)) << "segment error 2";
        for (std::string str : result) {
            resultsentence += str + ' ';
        }
    }
    return (char *)resultsentence.data();
}
