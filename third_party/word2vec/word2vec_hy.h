#ifndef __WORD2VEC_HY__
#define __WORD2VEC_HY__


/**
 * Description: get base vocabulary
 */
void word2vec_get_vocab(char* srcfile, char* destfile, int min_count=5);


/**
* Description: train vocab get wordvec
*/
void word2vec_train(char* srcfile, char* destfile, int _size=100, float _sample=1e-3,
                         int _negative=5, int _hs=0, int _binary=0, int _iter=0,
                         int _window = 5, int _cbow = 1, int _mincount=5);

#endif