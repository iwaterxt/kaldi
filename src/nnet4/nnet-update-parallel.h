//nnet4/nnet-update-parallel.h
//copyright 2019 xutao
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
#ifndef KALDI_NNET4_NNET_UPDATE_PARALLEL_H_
#define KALDI_NNET4_NNET_UPDATE_PARALLEL_H_


#include "util/table-types.h"
#include "util/kaldi-semaphore.h"
#include "util/kaldi-thread.h"
#include "util/common-utils.h"
#include "itf/options-itf.h"
#include "nnet4/nnet-nnet.h"
#include "nnet4/nnet-trnopts.h"
#include "nnet4/nnet-example.h"
#include "nnet/nnet-randomizer.h"


namespace kaldi{
namespace nnet4{


struct NnetParallelTrainOptions{

	bool binary;
	bool crossvalidate;
	bool randomize;
	std::string feature_transform;
	std::string objective_function;
	std::string frame_weights;
	std::string utt_weights;
	std::string use_gpu;
	int32 max_frames;
	int32 length_tolerance;
	int32 num_threads;

	NnetParallelTrainOptions():
		binary(true),
		crossvalidate(false),
		randomize(true),
		feature_transform(""),
		objective_function("xent"),
		frame_weights(""),
		utt_weights(""),
		use_gpu("yes"),
		max_frames(360000),
		length_tolerance(5),
		num_threads(1)
	{}

	void Register(OptionsItf *opts){

	    bool binary = true;
	    opts->Register("binary", &binary, "Write output in binary mode");

	    bool crossvalidate = false;
	    opts->Register("cross-validate", &crossvalidate,
	        "Perform cross-validation (don't back-propagate)");

	    bool randomize = true;
	    opts->Register("randomize", &randomize,
	        "Perform the frame-level shuffling within the Cache::");

	    std::string feature_transform;
	    opts->Register("feature-transform", &feature_transform,
	        "Feature transform in Nnet format");

	    std::string objective_function = "xent";
	    opts->Register("objective-function", &objective_function,
	        "Objective function : xent|mse|multitask");

	    int32 max_frames = 360000;
	    opts->Register("max-frames", &max_frames,
	        "Maximum number of frames an utterance can have (skipped if longer)");

	    int32 length_tolerance = 5;
	    opts->Register("length-tolerance", &length_tolerance,
	        "Allowed length mismatch of features/targets/weights "
	        "(in frames, we truncate to the shortest)");

	    int32 num_threads = 1;
	    opts->Register("num-threads", &num_threads,
	    			"number of threads to training neural network");

	    std::string frame_weights;
	    opts->Register("frame-weights", &frame_weights,
	        "Per-frame weights, used to re-scale gradients.");

	    std::string utt_weights;
	    opts->Register("utt-weights", &utt_weights,
	        "Per-utterance weights, used to re-scale frame-weights.");

	    std::string use_gpu="yes";
	    opts->Register("use-gpu", &use_gpu,
	        "yes|no|optional, only has effect if compiled with CUDA");
	}

};


double DNNDoBackpropParallel(const Nnet& nnet,
						  SequentialBaseFloatMatrixReader& feature_reader,
						  RandomAccessPosteriorReader& targets_reader,
						  RandomAccessBaseFloatVectorReader& weights_reader,
						  RandomAccessBaseFloatReader& utt_weights_reader,
						  NnetTrainOptions& trn_opts,
						  NnetDataRandomizerOptions rnd_opts,
						  NnetParallelTrainOptions parallel_opts,
						  std::string& target_model_filename);




}//namespace nnet4
}//namespace kaldi





#endif // KALDI_NNET4_NNET_UPDATE_PARALLEL_H_
