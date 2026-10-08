#include <oneapi/dnnl/dnnl.hpp>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
using namespace dnnl;
using Clock=std::chrono::steady_clock;
int main(){
 engine eng(engine::kind::cpu,0);stream str(eng);
 for(auto shape: {std::pair<int,int>{1024,1024},{1024,3072},{1024,5248},{2624,1024}}){
 int k=shape.first,n=shape.second;
 for(int m:{1,3,7,8,9,16,64,65}){
 std::vector<float>x(m*k),w(n*k),y(m*n);
 for(size_t i=0;i<x.size();i++)x[i]=(i%97)*.001f-.048f;
 for(size_t i=0;i<w.size();i++)w[i]=(i%89)*.002f-.088f;
 auto srcmd=memory::desc({m,k},memory::data_type::f32,memory::format_tag::ab);
 auto rawmd=memory::desc({n,k},memory::data_type::f32,memory::format_tag::ab);
 auto wmd=memory::desc({n,k},memory::data_type::f32,memory::format_tag::any);
 auto dstmd=memory::desc({m,n},memory::data_type::f32,memory::format_tag::ab);
 primitive_attr attr;attr.set_fpmath_mode(fpmath_mode::strict);
 auto pd=inner_product_forward::primitive_desc(eng,prop_kind::forward_inference,srcmd,wmd,dstmd,attr);
 auto src=memory(srcmd,eng,x.data()),raw=memory(rawmd,eng,w.data()),dst=memory(dstmd,eng,y.data()),packed=memory(pd.weights_desc(),eng);
 auto start=Clock::now();reorder(raw,packed).execute(str,raw,packed);str.wait();
 double pack_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
 auto op=inner_product_forward(pd);
 std::vector<double>times;
 for(int i=0;i<20;i++){start=Clock::now();op.execute(str,{{DNNL_ARG_SRC,src},{DNNL_ARG_WEIGHTS,packed},{DNNL_ARG_DST,dst}});str.wait();if(i>=5)times.push_back(std::chrono::duration<double,std::milli>(Clock::now()-start).count());}
 std::sort(times.begin(),times.end());float err=0;
 for(int row=0;row<m;row++)for(int o=0;o<n;o++){double v=0;for(int i=0;i<k;i++)v+=double(x[row*k+i])*w[o*k+i];err=std::max(err,float(std::abs(v-y[row*n+o])));}
 if(err>=1e-4)return 2;
 std::cout<<"in="<<k<<" out="<<n<<" rows="<<m<<" onednn_ms="<<times[7]<<" packed_bytes="<<pd.weights_desc().get_size()<<" pack_ms="<<pack_ms<<" error="<<err<<" impl="<<pd.impl_info_str()<<std::endl;
 }
 }
}
