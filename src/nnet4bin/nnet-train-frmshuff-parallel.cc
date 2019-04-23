// nnet4bin/nnet-train-frmshuff.cc

// Copyright 2013-2016  Brno University of Technology (Author: Karel Vesely)

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

#include "nnet4/nnet-trnopts.h"
#include "nnet4/nnet-nnet.h"
#include "nnet4/nnet-loss.h"
#include "nnet4/nnet-randomizer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "nnet4/nnet-update-parallel.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet4;
  typedef kaldi::int32 int32;

  try {
    const char *usage =
      "Perform one iteration (epoch) of Neural Network training with\n"
      "mini-batch Stochastic Gradient Descent. The training targets\n"
      "are usually pdf-posteriors, prepared by ali-to-post.\n"
      "Usage:  nnet-train-frmshuff-parallel [options] <feature-rspecifier> <targets-rspecifier> <model-in> [<model-out>]\n"
      "e.g.: nnet-train-frmshuff-parallel scp:feats.scp ark:posterior.ark nnet.init nnet.iter1\n";

    ParseOptions po(usage);

    NnetTrainOptions trn_opts;
    trn_opts.Register(&po);
    NnetDataRandomizerOptions rnd_opts;
    rnd_opts.Register(&po);
    LossOptions loss_opts;
    loss_opts.Register(&po);
    NnetParallelTrainOptions parallel_opts;
    parallel_opts.Register(&po);

    po.Read(argc, argv);
    if (po.NumArgs() != 3 + (parallel_opts.crossvalidate ? 0 : 1)) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1),
      targets_rspecifier = po.GetArg(2),
      model_filename = po.GetArg(3);

    std::string target_model_filename;
    if (!parallel_opts.crossvalidate) {
      target_model_filename = po.GetArg(4);
    }

    using namespace kaldi;
    using namespace kaldi::nnet4;
    typedef kaldi::int32 int32;


    Nnet nnet;
    nnet.Read(model_filename);
    nnet.SetTrainOptions(trn_opts);

    if (parallel_opts.crossvalidate) {
      nnet.SetDropoutRate(0.0);
    }

    kaldi::int64 total_frames = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessPosteriorReader targets_reader(targets_rspecifier);
    RandomAccessBaseFloatVectorReader weights_reader;
    if (parallel_opts.frame_weights != "") {
      weights_reader.Open(frame_weights);
    }
    RandomAccessBaseFloatReader utt_weights_reader;
    if (parallel_opts.utt_weights != "") {
      utt_weights_reader.Open(utt_weights);
    }

    DNNDoBackpropParallel(nnet,
    				   feature_reader,
    				   targets_reader,
    				   weights_reader,
    				   utt_weights_reader,
    				   trn_opts,
               loss_opts,
    				   rnd_opts,
    				   parallel_opts,
    				   target_model_filename);

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
