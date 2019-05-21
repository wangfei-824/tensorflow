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


REGISTER_OP("ScatterOrder")
.Input("indices_all: T")
.Input("indices_selected: int64")
.Input("layer_num: int32")
.Input("permutation: int64")
.Output("scatter_order: int64")
.Output("count: int64")
//.Output("shape: int64")
.Attr("T: type")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
  ::tensorflow::shape_inference::ShapeHandle indices;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &indices));

  c->set_output(0, c->Matrix(c->Dim(indices, 0), 1));
  c->set_output(1, c->Vector(1));
//  c->set_output(0, c->Matrix(c->Dim(indices, 0), 4));
//  c->set_output(1, c->Vector(4));

  return Status::OK();
}).Doc(R"doc(
Indices should be [N, X, Y, Z].
)doc");



template <typename T>
class ScatterOrderOp : public OpKernel {

public:
  explicit ScatterOrderOp(OpKernelConstruction* context) : OpKernel(context) {}

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
      and_ = 0x1 << layer_num;

      const Tensor& permutation = context->input(3);
      OP_REQUIRES(context, TensorShapeUtils::IsVector(permutation.shape()),
        errors::InvalidArgument(
          "Permutation should be a vector but received shape ",
          permutation.shape().DebugString()));
      auto permutation_vec = permutation.vec<int64>();

      const int64 nnz = indices_selected.shape().dim_size(0);
      std::vector<int64> scatter_order(nnz);
      scatter_order[permutation_vec(0)] = ((indices_all_mat(indices_selected_vec(permutation_vec(0)), 1)>>(move_-1))&0x1) * 4 +
                            ((indices_all_mat(indices_selected_vec(permutation_vec(0)), 2)>>(move_-1))&0x1) * 2 +
                            ((indices_all_mat(indices_selected_vec(permutation_vec(0)), 3)>>(move_-1))&0x1);

      int64 count = 0;
      for (size_t i=1; i<nnz; ++i) {
          if (! equal(indices_all_mat, indices_selected_vec, permutation_vec(i), permutation_vec(i-1))) {
              count += 8;
          }

          scatter_order[permutation_vec(i)] = ((indices_all_mat(indices_selected_vec(permutation_vec(i)), 1)>>(move_-1))&0x1) * 4 +
                                                ((indices_all_mat(indices_selected_vec(permutation_vec(i)), 2)>>(move_-1))&0x1) * 2 +
                                                ((indices_all_mat(indices_selected_vec(permutation_vec(i)), 3)>>(move_-1))&0x1) + count;
      }
      count +=8;

    Tensor* scatter_order_tensor = nullptr;
    OP_REQUIRES_OK(context,
            context->allocate_output(0, TensorShape({nnz, 1}), &scatter_order_tensor));
    auto scatter_order_flat = scatter_order_tensor->flat<int64>();
    std::copy_n(scatter_order.begin(), nnz, &scatter_order_flat(0));

    Tensor* count_tensor = nullptr;
    OP_REQUIRES_OK(context,
            context->allocate_output(1, TensorShape({1}), &count_tensor));
    auto count_scalar = count_tensor->flat<int64>();
    count_scalar(0) = count;


//      int64 count = 0;
//      std::vector<int64> scatter_order(nnz*4);
//      scatter_order[permutation_vec(0)*4] = count;
//      scatter_order[permutation_vec(0)*4+1] = (indices_all_mat(indices_selected_vec(permutation_vec(0)), 1)>>(move_-1))&0x1;
//      scatter_order[permutation_vec(0)*4+2] = (indices_all_mat(indices_selected_vec(permutation_vec(0)), 2)>>(move_-1))&0x1;
//      scatter_order[permutation_vec(0)*4+3] = (indices_all_mat(indices_selected_vec(permutation_vec(0)), 3)>>(move_-1))&0x1;

//      for (size_t i=1; i<nnz; ++i) {
//          if (! equal(indices_all_mat, indices_selected_vec, permutation_vec(i), permutation_vec(i-1))) {
//              count ++;
//          }

//          scatter_order[permutation_vec(i) * 4] = count;

//          scatter_order[permutation_vec(i) * 4 + 1] = (indices_all_mat(indices_selected_vec(permutation_vec(i)), 1)>>(move_-1))&0x1;
//          scatter_order[permutation_vec(i) * 4 + 2] = (indices_all_mat(indices_selected_vec(permutation_vec(i)), 2)>>(move_-1))&0x1;
//          scatter_order[permutation_vec(i) * 4 + 3] = (indices_all_mat(indices_selected_vec(permutation_vec(i)), 3)>>(move_-1))&0x1;
//      }
//      count ++;

//    Tensor* scatter_order_tensor = nullptr;
//    OP_REQUIRES_OK(context,
//            context->allocate_output(0, TensorShape({nnz, 4}), &scatter_order_tensor));
//    auto scatter_order_flat = scatter_order_tensor->flat<int64>();
//    std::copy_n(scatter_order.begin(), nnz*4, &scatter_order_flat(0));

//    Tensor* count_tensor = nullptr;
//    OP_REQUIRES_OK(context,
//            context->allocate_output(1, TensorShape({4}), &count_tensor));
//    auto count_scalar = count_tensor->flat<int64>();
//    count_scalar(0) = count;
//    count_scalar(1) = 2;
//    count_scalar(2) = 2;
//    count_scalar(3) = 2;
  }


 private:
 	int move_;
  	int64 and_;
};


#define REGISTER_KERNELS(type)                                     \
REGISTER_KERNEL_BUILDER(                                            \
  Name("ScatterOrder").Device(DEVICE_CPU).TypeConstraint<type>("T"),  \
  ScatterOrderOp<type>)

REGISTER_KERNELS(int64);
REGISTER_KERNELS(int32);
REGISTER_KERNELS(int16);
REGISTER_KERNELS(int8);
#undef REGISTER_KERNELS

