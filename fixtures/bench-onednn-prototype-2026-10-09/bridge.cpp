#include <oneapi/dnnl/dnnl.hpp>
#include <vector>
#include <memory>
#include <mutex>
#include <iostream>
using namespace dnnl;
struct Weight {const float* identity; int k,n; memory data;};
struct Plan {const float* identity; int m,k,n; inner_product_forward op; memory src,weight,dst;};
static engine eng(engine::kind::cpu,0);
static stream str(eng);
static std::mutex mutex;
static std::vector<Weight>weights;
static std::vector<Plan>plans;
extern "C" int diagnostic_run(const float*x,int m,int k,const float*w,int n,float*y){try{
 std::lock_guard<std::mutex>guard(mutex);
 Plan*p=nullptr;
 for(auto&v:plans)if(v.identity==w&&v.m==m&&v.k==k&&v.n==n){p=&v;break;}
 if(!p){
 auto srcmd=memory::desc({m,k},memory::data_type::f32,memory::format_tag::ab);
 auto dstmd=memory::desc({m,n},memory::data_type::f32,memory::format_tag::ab);
 auto wmd=memory::desc({n,k},memory::data_type::f32,memory::format_tag::any);
 primitive_attr attr;attr.set_fpmath_mode(fpmath_mode::strict);post_ops ops;ops.append_sum(1.f);attr.set_post_ops(ops);
 auto pd=inner_product_forward::primitive_desc(eng,prop_kind::forward_inference,srcmd,wmd,dstmd,attr);
 memory packed;
 for(auto&v:weights)if(v.identity==w&&v.k==k&&v.n==n&&v.data.get_desc()==pd.weights_desc()){packed=v.data;break;}
 if(!packed){
 packed=memory(pd.weights_desc(),eng);
 auto raw=memory(memory::desc({n,k},memory::data_type::f32,memory::format_tag::ab),eng,const_cast<float*>(w));
 reorder(raw,packed).execute(str,raw,packed);str.wait();weights.push_back({w,k,n,packed});
 }
 plans.push_back({w,m,k,n,inner_product_forward(pd),memory(srcmd,eng,DNNL_MEMORY_NONE),packed,memory(dstmd,eng,DNNL_MEMORY_NONE)});p=&plans.back();
 }
 p->src.set_data_handle(const_cast<float*>(x));p->dst.set_data_handle(y);
 p->op.execute(str,{{DNNL_ARG_SRC,p->src},{DNNL_ARG_WEIGHTS,p->weight},{DNNL_ARG_DST,p->dst}});str.wait();return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
