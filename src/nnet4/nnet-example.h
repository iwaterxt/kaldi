// nnet4/nnet-example.h

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)
//           2014  Vimal Manohar
//			 2019 xutao

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

#ifndef KALDI_NNET4_NNET_EXAMPLE_H_
#define KALDI_NNET4_NNET_EXAMPLE_H_

#include <queue>
#include "nnet4/nnet-nnet.h"
#include "util/table-types.h"
#include "util/kaldi-semaphore.h"




namespace kaldi{
namespace nnet4{

struct NnetExample
{
	Posterior tgt_;
	Matrix<BaseFloat> mat_;
	Vector<BaseFloat> weight_;
	std::string key_; 

	NnetExample(){}

	NnetExample(std::string key, Matrix<BaseFloat>& feature, Posterior& post, Vector<BaseFloat>& weight);
	
};

class ExamplesRepository{

public:

	void AcceptExamples(NnetExample* example);

	void ExamplesDone();

	bool ProvideExamples(NnetExample* example);

	ExamplesRepository():empty_semaphore_(1),done_(false) {}

private:
	Semaphore full_semaphore_;
	Semaphore empty_semaphore_;

	std::vector<NnetExample*> examples_;
	bool done_;

	KALDI_DISALLOW_COPY_AND_ASSIGN(ExamplesRepository);
};

}//end of namespace nnet4
}//end of namespace kaldi


#endif