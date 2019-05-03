// nnet4/nnet-example.cc

// Copyright 2012-2013  Johns Hopkins University (author: Daniel Povey)
//                2014  Vimal Manohar
//				  2019. xutao
// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.



#include "nnet4/nnet-example.h"



namespace kaldi{
namespace nnet4{





NnetExample::NnetExample(std::string key, Matrix<BaseFloat>& feature,
						 Posterior& post, Vector<BaseFloat>& weight){
	key_ = key;
	mat_ = feature;
	tgt_ = post;
	weight_ = weight;

}

void ExamplesRepository::AcceptExamples(NnetExample *example) {
  empty_semaphore_.Wait();
  examples_.push_back(example);
  full_semaphore_.Signal();
}

void ExamplesRepository::ExamplesAcceptDone() {
  done_ = true;
}

bool ExamplesRepository::ExampleEnd(){
  
  if(done_){
    if(examples_.empty()){
      return true;
    }else{
      return false;
    }
  }else{
    return false;
  }
}

bool ExamplesRepository::ProvideExamples(NnetExample *example) {
  full_semaphore_.Wait();
  if (done_) {
    if(examples_.empty()){
      full_semaphore_.Signal();
      return false;
    }else{
      example = examples_.front();
  	  examples_.erase(examples_.begin());
      return true;
    }
  } else {
  	example = examples_.front();
  	examples_.erase(examples_.begin());
    empty_semaphore_.Signal();
    return true;
  }
}

}// end of namespace nnet4

}// end of namespace kaldi