#include <oneapi/dnnl/dnnl.hpp>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
using namespace dnnl;
int main(){engine eng(engine::kind::cpu,0);stream str(eng);
for(int rows:{8,16,64})for(bool fused:{false,true})try{
 int hidden=2624;std::vector<float>u(rows*hidden*2),out(rows*hidden);
 for(size_t i=0;i<u.size();i++)u[i]=float(i%97)*.08f-3.84f;
 auto srcmd=memory::desc({rows,hidden},memory::data_type::f32,memory::format_tag::ab);
 auto dstmd=memory::desc({rows,hidden},memory::data_type::f32,memory::format_tag::ab);
 primitive_attr attr;attr.set_fpmath_mode(fpmath_mode::strict);attr.set_scratchpad_mode(scratchpad_mode::user);
 if(fused){post_ops ops;ops.append_binary(algorithm::binary_mul,srcmd);attr.set_post_ops(ops);}
 auto pd=eltwise_forward::primitive_desc(eng,prop_kind::forward_inference,algorithm::eltwise_gelu_erf,srcmd,dstmd,attr);
 auto src=memory(srcmd,eng,u.data()),gate=memory(srcmd,eng,u.data()+hidden),dst=memory(dstmd,eng,out.data()),scratch=memory(pd.scratchpad_desc(),eng);
 auto op=eltwise_forward(pd);std::vector<double>times;
 for(int i=0;i<20;i++){auto start=std::chrono::steady_clock::now();op.execute(str,{{DNNL_ARG_SRC,src},{DNNL_ARG_DST,dst},{DNNL_ARG_SCRATCHPAD,scratch},{DNNL_ARG_ATTR_MULTIPLE_POST_OP(0)|DNNL_ARG_SRC_1,gate}});str.wait();if(i>=5)times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());}
 std::sort(times.begin(),times.end());double error=0;
 for(int r=0;r<rows;r++)for(int h=0;h<hidden;h++){double v=u[r*hidden+h];double expected=.5*v*(1+std::erf(v/std::sqrt(2.)));if(fused)expected*=u[r*hidden+hidden+h];error=std::max(error,std::abs(expected-out[r*hidden+h]));}
 std::cout<<"rows="<<rows<<" fused="<<fused<<" ms="<<times[7]<<" error="<<error<<" impl="<<pd.impl_info_str()<<std::endl;
 }catch(const std::exception&e){std::cout<<"rows="<<rows<<" fused="<<fused<<" error="<<e.what()<<std::endl;}}
