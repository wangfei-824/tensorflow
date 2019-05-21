#define EIGEN_USE_THREADS

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <iostream>
#include <sys/time.h>
#include <stdlib.h>
#include <vector>
#include <math.h>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"
#include "tensorflow/core/util/sparse/dim_comparator.h"


using namespace tensorflow; 


REGISTER_OP("SortReindex")
.Input("indices_all: T")
.Input("indices_selected: int64")
.Input("layer_num: int32")
.Output("permutation: int64")
.Attr("T: type")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
  ::tensorflow::shape_inference::ShapeHandle indices;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &indices));

  c->set_output(0, c->Vector(c->Dim(indices, 0)));

  return Status::OK();
}).Doc(R"doc(
Indices should be [N, X, Y, Z].
)doc");



template <typename T>
class SortReindexOp : public OpKernel {

public:
  explicit SortReindexOp(OpKernelConstruction* context) : OpKernel(context) {}


  bool cmp(const typename TTypes<T>::ConstMatrix& ix_,
  					const typename TTypes<int64>::ConstVec& sel_,
  					const int64 i, const int64 j) const {
	if (ix_(sel_(i), 0) < ix_(sel_(j), 0)) return true;
    if (ix_(sel_(i), 0) > ix_(sel_(j), 0)) return false;

    for (int64 d = 1; d < 4; ++d) {
      if ((ix_(sel_(i), d) >> move_) < (ix_(sel_(j), d) >> move_)) return true;
      if ((ix_(sel_(i), d) >> move_) > (ix_(sel_(j), d) >> move_)) return false;
    }

    return false;
  }

  int64 next(const typename TTypes<T>::ConstMatrix& ix,
  				const typename TTypes<int64>::ConstVec& sel,
  				const int64 cur, const int64 i, const int64 j) const {
	for (int64 next=cur+1; next<sel.size(); ++next) {
		if (((ix(sel(next), 1)>>(move_-1))&0x1) == i)
			if (((ix(sel(next), 2)>>(move_-1))&0x1) == j)
				return next;
	}
	return sel.size();
  }

  void sort(const typename TTypes<T>::ConstMatrix& ix,
  					const typename TTypes<int64>::ConstVec& sel,
  					std::vector<int64>& permutation) const {
	int64 ids[4] = {-1};
	for (int i=0; i<4; ++i)
		ids[i] = next(ix, sel, ids[i], i/2, i%2);

	int64 loc = 0;
	int m, n;
	while(ids[0]!=sel.size() || ids[1]!=sel.size() ||
			ids[2]!=sel.size() || ids[3]!=sel.size()) {
		for (m=0; m<4; ++m) 
			if (ids[m]!=sel.size()) 
				break;
		for (n=m+1; n<4; ++n)
			if (ids[n]!=sel.size()) 
				if (cmp(ix, sel, ids[n], ids[m]))
					m = n;
		permutation[loc++] = ids[m];
		ids[m] = next(ix, sel, ids[m], m/2, m%2);
	}
  }


  void Compute(OpKernelContext* context) override {
    const Tensor& indices_all = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(indices_all.shape()),
      errors::InvalidArgument(
        "Input indices should be a matrix but received shape ",
        indices_all.shape().DebugString()));

    const Tensor& indices_selected = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(indices_selected.shape()),
      errors::InvalidArgument(
        "Input indices should be a vector but received shape ",
        indices_selected.shape().DebugString()));

    const int layer_num = context->input(2).scalar<int>()();
    move_ = layer_num + 1;
    and_ = 0x1 << layer_num;

    const int64 nnz = indices_selected.shape().dim_size(0);
    std::vector<int64> reorder(nnz);
    // std::iota(reorder.begin(), reorder.end(), 0);

    // MyDimComparator sorter(indices_all.matrix<T>(), 
    // 						indices_selected.vec<int64>(),
    // 						layer_num);
    // std::sort(reorder.begin(), reorder.end(), sorter); 

	sort(indices_all.matrix<T>(), 
    	indices_selected.vec<int64>(),
    	reorder);

    Tensor* permutation = nullptr;
    OP_REQUIRES_OK(context,
            context->allocate_output(0, TensorShape({nnz}), &permutation));

    auto permutation_vec = permutation->vec<int64>();
    std::copy_n(reorder.begin(), nnz, &permutation_vec(0));

  }


 private:
 	int move_;
  	int64 and_;
};


#define REGISTER_KERNELS(type)                                     \
REGISTER_KERNEL_BUILDER(                                            \
  Name("SortReindex").Device(DEVICE_CPU).TypeConstraint<type>("T"),  \
  SortReindexOp<type>)

REGISTER_KERNELS(int64);
REGISTER_KERNELS(int32);
REGISTER_KERNELS(int16);
REGISTER_KERNELS(int8);
#undef REGISTER_KERNELS

