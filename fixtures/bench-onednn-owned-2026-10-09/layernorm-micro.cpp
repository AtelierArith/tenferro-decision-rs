#include <oneapi/dnnl/dnnl.hpp>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iostream>
using namespace dnnl;
int main(){engine eng(engine::kind::cpu,0);stream str(eng);for(int rows:{8,16,64}){
 int d=1024;std::vector<float>x(rows*d),y(rows*d),w(d,1),b(d,.1f);
 for(size_t i=0;i<x.size();i++)x[i]=float(i%97)*.08f-3.84f;
 auto md=memory::desc({rows,d},memory::data_type::f32,memory::format_tag::ab);
 primitive_attr attr;attr.set_fpmath_mode(fpmath_mode::strict);attr.set_scratchpad_mode(scratchpad_mode::user);
 auto pd=layer_normalization_forward::primitive_desc(eng,prop_kind::forward_inference,md,md,1e-5f,normalization_flags::use_scale|normalization_flags::use_shift,attr);
 auto op=layer_normalization_forward(pd);auto src=memory(md,eng,x.data()),dst=memory(md,eng,y.data()),weight=memory(memory::desc({d},memory::data_type::f32,memory::format_tag::a),eng,w.data()),bias=memory(weight.get_desc(),eng,b.data()),scratch=memory(pd.scratchpad_desc(),eng);
 std::vector<double>times;for(int i=0;i<20;i++){auto start=std::chrono::steady_clock::now();op.execute(str,{{DNNL_ARG_SRC,src},{DNNL_ARG_DST,dst},{DNNL_ARG_SCALE,weight},{DNNL_ARG_SHIFT,bias},{DNNL_ARG_SCRATCHPAD,scratch}});str.wait();if(i>=5)times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());}
 std::sort(times.begin(),times.end());std::cout<<"rows="<<rows<<" norm_ms="<<times[7]<<" impl="<<pd.impl_info_str()<<std::endl;
}}
