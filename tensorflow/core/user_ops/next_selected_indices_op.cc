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


REGISTER_OP("NextSelectedIndices")
.Input("indices_all: T")
.Input("indices_selected: int64")
.Input("layer_num: int32")
.Input("permutation: int64")
.Output("next_selected_indices: int64")
.Attr("T: type")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
  ::tensorflow::shape_inference::ShapeHandle indices;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &indices));

  c->set_output(0, c->Vector(::tensorflow::shape_inference::InferenceContext::kUnknownDim));

  return Status::OK();
}).Doc(R"doc(
Indices should be [N, X, Y, Z].
)doc");



template <typename T>
class NextSelectedIndicesOp : public OpKernel {

public:
  explicit NextSelectedIndicesOp(OpKernelConstruction* context) : OpKernel(context) {}


  bool equal(const typename TTypes<T>::ConstMatrix& ix_,
  					const typename TTypes<int64>::ConstVec& sel_,
                                        const int64 i, const int64 j) const {
      if (ix_(sel_(i), 0) != ix_(sel_(j), 0)) return false;

    for (int64 d = 1; d < 4; ++d) {
      if ((ix_(sel_(i), d) >> move_) != (ix_(sel_(j), d) >> move_)) return false;
    }

    return true;
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& indices_all = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(indices_all.shape()),
      errors::InvalidArgument(
        "Input indices should be a matrix but received shape ",
        indices_all.shape().DebugString()));
    auto indices_all_mat = indices_all.matrix<T>();

    const Tensor& indices_selected = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(indices_selected.shape()),
      errors::InvalidArgument(
        "Input indices should be a vector but received shape ",
        indices_selected.shape().DebugString()));
    auto indices_selected_vec = indices_selected.vec<int64>();

    const int layer_num = context->input(2).scalar<int>()();
    move_ = layer_num + 1;

    const Tensor& permutation = context->input(3);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(permutation.shape()),
      errors::InvalidArgument(
        "Permutation should be a vector but received shape ",
        permutation.shape().DebugString()));
    auto permutation_vec = permutation.vec<int64>();

    const int64 all = indices_selected.shape().dim_size(0);
    std::vector<int64> next_indices(all);
    next_indices[0] = indices_selected_vec(permutation_vec(0));

    int64 count = 1;
    for (size_t i=1; i<permutation_vec.size(); ++i) {
        if (! equal(indices_all_mat, indices_selected_vec, permutation_vec(i), permutation_vec(i-1))) {
            next_indices[count++] = indices_selected_vec(permutation_vec(i));
        }
    }

    Tensor* next_selected_indices = nullptr;
    OP_REQUIRES_OK(context,
            context->allocate_output(0, TensorShape({count}), &next_selected_indices));

    auto next_selected_indices_vec = next_selected_indices->vec<int64>();
    std::copy_n(next_indices.begin(), count, &next_selected_indices_vec(0));
  }

 private:
 	int move_;
};


#define REGISTER_KERNELS(type)                                     \
REGISTER_KERNEL_BUILDER(                                            \
  Name("NextSelectedIndices").Device(DEVICE_CPU).TypeConstraint<type>("T"),  \
  NextSelectedIndicesOp<type>)

REGISTER_KERNELS(int64);
REGISTER_KERNELS(int32);
REGISTER_KERNELS(int16);
REGISTER_KERNELS(int8);
#undef REGISTER_KERNELS

