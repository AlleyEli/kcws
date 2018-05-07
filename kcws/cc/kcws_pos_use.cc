#include "kcws_pos_use.h"


void kcwsPosProcess::kcws_set_envfile_pars(const char * model_file,
                           const char * vocab_file,
                           const char * pos_model_file,
                           const char * word_vocab_file,
                           const char * pos_vocab_file,
                           const int max_sentencelen,
                           const int max_wordnum,
                           const char * user_dict_file){
    model_path = model_file;
    vocab_path = vocab_file;
    pos_model_path = pos_model_file;
    word_vocab_path = word_vocab_file;
    pos_vocab_path = pos_vocab_file;
    max_sentence_len = max_sentencelen;
    max_word_num = max_wordnum;
    user_dict_path = user_dict_file;
}


void kcwsPosProcess::kcws_pos_process(const char* srcsentence, char * outsentence, int use_pos) {
    std::string sentence = srcsentence;
    std::string resultsentence = "";
    kcws::TfSegModel model;
    std::vector<std::string> result;
    std::vector<std::string> tags;
    CHECK(model.LoadModel(model_path, 
                          vocab_path,
                          max_sentence_len, 
                          user_dict_path)) << "Load model error";
    if (!pos_model_path.empty()) {
      kcws::PosTagger* tagger = new kcws::PosTagger;
      CHECK(tagger->LoadModel(pos_model_path,
                              word_vocab_path,
                              vocab_path,
                              pos_vocab_path,
                              max_word_num)) << "load pos model error";
      model.SetPosTagger(tagger);
    }
    if (model.Segment(sentence, &result, &tags)) {
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
    }

    outsentence = (char *)resultsentence.data();
} 

/*
int cws_process(){

}*/
