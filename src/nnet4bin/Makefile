
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

BINFILES = nnet-train-frmshuff \
        nnet-train-perutt \
        nnet-train-mmi-sequential \
        nnet-train-mpe-sequential \
        nnet-train-multistream nnet-train-multistream-perutt \
        rbm-train-cd1-frmshuff rbm-convert-to-nnet \
        nnet-forward nnet-copy nnet-info nnet-concat \
        transf-to-nnet cmvn-to-nnet nnet-initialize \
	feat-to-post paste-post train-transitions \
	cuda-gpu-available nnet-set-learnrate nnet-train-frmshuff-parallel

OBJFILES =



TESTFILES =

ADDLIBS = ../nnet4/kaldi-nnet4.a ../cudamatrix/kaldi-cudamatrix.a \
          ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a \
          ../util/kaldi-util.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a 

include ../makefiles/default_rules.mk
