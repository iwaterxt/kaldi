//nnet4/nnet-update-parallel.cc
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

#include <numeric>
#include "nnet4/nnet-update-parallel.h"
#include "nnet4/nnet-nnet.h"
#include "util/kaldi-thread.h"
#include "nnet4/nnet-loss.h"


namespace kaldi{
namespace nnet4{

class DNNDoBackpropParallelClass: public MultiThreadable{

public:
	DNNDoBackpropParallelClass(const Nnet &nnet,
							   ExamplesRepository* repository,
							   NnetTrainOptions& trn_opts,
							   NnetDataRandomizerOptions& rnd_opts,
							   NnetParallelTrainOptions& parallel_opts,
							   std::string target_model_filename):
			nnet_(nnet), repository_(repository), trn_opts_(trn_opts),
			rnd_opts_(rnd_opts), parallel_opts_(parallel_opts),target_model_filename_(target_model_filename){}

	void operator () (){
		using namespace kaldi;
		using namespace kaldi::nnet4;
		typedef kaldi::int32 int32;

		#if HAVE_CUDA == 1
			CuDevice::Instantiate().AllowMultithreading();
			CuDevice::Instantiate().SelectGpuId(parallel_opts_.use_gpu);
		#endif
			Nnet nnet_transf;
			if(parallel_opts_.feature_transform !=""){
				nnet_transf.Read(parallel_opts_.feature_transform);
			}

			CuMatrix<BaseFloat> feats_transf, nnet_out, obj_diff;
	    RandomizerMask randomizer_mask(rnd_opts_);
	    MatrixRandomizer feature_randomizer(rnd_opts_);
	    PosteriorRandomizer targets_randomizer(rnd_opts_);
	    VectorRandomizer weights_randomizer(rnd_opts_);
	    Xent xent;
	    Mse mse;
	    MultiTaskLoss multitask;
	    if (0 == parallel_opts_.objective_function.compare(0, 9, "multitask")) {
	      // objective_function contains something like :
	      // 'multitask,xent,2456,1.0,mse,440,0.001'
	      //
	      // the meaning is following:
	      // 'multitask,<type1>,<dim1>,<weight1>,...,<typeN>,<dimN>,<weightN>'
	      multitask.InitFromString(parallel_opts_.objective_function);
	    }
		int32 num_done = 0;
		int32 total_frames = 0;
		while(1){

			NnetExample* example;
			while(repository_->ProvideExamples(example)){
		        if (feature_randomizer.IsFull()) {
		          // break the loop without calling Next(),
		          // we keep the 'utt' for next round,
		          break;
		        }
				Matrix<BaseFloat> mat = example->mat_;
				Posterior targets = example->tgt_;
				Vector<BaseFloat> weights = example->weight_;

				nnet_transf_.Feedforward(CuMatrix<BaseFloat>(mat), &feats_transf);
		        // remove frames with '0' weight from training,

		        // pass data to randomizers,
		        KALDI_ASSERT(feats_transf.NumRows() == targets.size());
		        feature_randomizer.AddData(feats_transf);
		        targets_randomizer.AddData(targets);
		        weights_randomizer.AddData(weights);
		        num_done++;
			}
		  // randomize,
		  if (!parallel_opts_.crossvalidate && parallel_opts_.randomize) {
		      const std::vector<int32>& mask =
		      randomizer_mask.Generate(feature_randomizer.NumFrames());
		      feature_randomizer.Randomize(mask);
		      targets_randomizer.Randomize(mask);
		      weights_randomizer.Randomize(mask);
		  }
		  for(; !feature_randomizer.Done(); feature_randomizer.Next(),
		    								  targets_randomizer.Next(),
		    								  weights_randomizer.Next()){

		        // get block of feature/target pairs,
		        const CuMatrixBase<BaseFloat>& nnet_in = feature_randomizer.Value();
		        const Posterior& nnet_tgt = targets_randomizer.Value();
		        const Vector<BaseFloat>& frm_weights = weights_randomizer.Value();
		        // forward pass,
		        nnet_.Propagate(nnet_in, &nnet_out);

		        // evaluate objective function we've chosen,
		        if (parallel_opts_.objective_function == "xent") {
		          // gradients re-scaled by weights in Eval,
		          xent.Eval(frm_weights, nnet_out, nnet_tgt, &obj_diff);
		        } else if (parallel_opts_.objective_function == "mse") {
		          // gradients re-scaled by weights in Eval,
		          mse.Eval(frm_weights, nnet_out, nnet_tgt, &obj_diff);
		        } else if (0 == parallel_opts_.objective_function.compare(0, 9, "multitask")) {
		          // gradients re-scaled by weights in Eval,
		          multitask.Eval(frm_weights, nnet_out, nnet_tgt, &obj_diff);
		        } else {
		          KALDI_ERR << "Unknown objective function code : " << parallel_opts_.objective_function;
		        }

		        if (!parallel_opts_.crossvalidate) {
		          // back-propagate, and do the update,
		          nnet_.Backpropagate(obj_diff, NULL);
		        }

		        // 1st mini-batch : show what happens in network,
		        if (total_frames == 0) {
		          KALDI_LOG << "### After " << total_frames << " frames,";
		          KALDI_LOG << nnet_.InfoPropagate();
		          if (!parallel_opts_.crossvalidate) {
		            KALDI_LOG << nnet_.InfoBackPropagate();
		            KALDI_LOG << nnet_.InfoGradient();
		          }
		        }
		        total_frames += nnet_in.NumRows();
		  }

		  if(repository_->ExamplesDone()){
		    break;
		  }
		}


	}

~DNNDoBackpropParallelClass(){}

private:
	Nnet nnet_;
	ExamplesRepository* repository_;
	NnetTrainOptions trn_opts_;
	NnetDataRandomizerOptions rnd_opts_;
	NnetParallelTrainOptions parallel_opts_;
	std::string target_model_filename_;

};

double DNNDoBackpropParallel(const Nnet& nnet,
						  SequentialBaseFloatMatrixReader& feature_reader,
						  RandomAccessPosteriorReader& targets_reader,
						  RandomAccessBaseFloatVectorReader& weights_reader,
						  RandomAccessBaseFloatReader& utt_weights_reader,
						  NnetTrainOptions& trn_opts,
						  NnetDataRandomizerOptions& rnd_opts,
						  NnetParallelTrainOptions& parallel_opts,
						  std::string& target_model_filename){

	ExamplesRepository repository;
	int32 num_no_tgt_mat = 0;
	int32 num_other_error = 0;
	DNNDoBackpropParallelClass c(nnet,
								 repository,
								 trn_opts,
								 rnd_opts,
								 parallel_opts,
								 target_model_filename);

		MultiThreader<DNNDoBackpropParallel> m(parallel_opts.num_threads, c);
		for(; !feature_reader.Done(); feature_reader.Netx()){
			std::string utt = feature_reader.Key();
	        KALDI_VLOG(3) << "Reading " << utt;
	        // check that we have targets,
	        if (!targets_reader.HasKey(utt)) {
	          KALDI_WARN << utt << ", missing targets";
	          num_no_tgt_mat++;
	          continue;
	        }
	        // check we have per-frame weights,
	        if (frame_weights != "" && !weights_reader.HasKey(utt)) {
	          KALDI_WARN << utt << ", missing per-frame weights";
	          num_other_error++;
	          continue;
	        }
	        // check we have per-utterance weights,
	        if (utt_weights != "" && !utt_weights_reader.HasKey(utt)) {
	          KALDI_WARN << utt << ", missing per-utterance weight";
	          num_other_error++;
	          continue;
	        }
	        Vector<BaseFloat> weights;
	        if (frame_weights != "") {
	          weights = weights_reader.Value(utt);
	        } else {  // all per-frame weights are 1.0,
	          weights.Resize(mat.NumRows());
	          weights.Set(1.0);
	        }
	        // multiply with per-utterance weight,
	        if (utt_weights != "") {
	          BaseFloat w = utt_weights_reader.Value(utt);
	          KALDI_ASSERT(w >= 0.0);
	          if (w == 0.0) continue;  // remove sentence from training,
	          weights.Scale(w);
	        }
			NnetExample example(utt, feature_reader.Value(), targets_reader.Value(), weights)

			repository.AcceptExamples(&example);
		}


}

}// end of namespace kaldi
}// end of namespace nnet4
