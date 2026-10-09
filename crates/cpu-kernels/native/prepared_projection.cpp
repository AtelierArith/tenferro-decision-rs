// Standard oneDNN primitives only; no model arithmetic or bespoke kernels.
#include <oneapi/dnnl/dnnl.hpp>
#include <algorithm>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <memory>
#include <vector>
using namespace dnnl;
namespace {
struct Plan {
    int rows;
    bool accumulate;
    inner_product_forward op;
    memory src, weights, dst, scratch;
    size_t scratch_bytes;
};
struct GatedPlan {
    int rows;
    eltwise_forward gelu;
    binary multiply;
    memory value, gate, gelu_scratch, multiply_scratch;
    size_t gate_bytes, required_bytes;
};
struct Projection {
    engine eng{engine::kind::cpu, 0};
    stream str{eng};
    int input, output;
    size_t retained_bytes = 0;
    std::vector<memory> weights;
    std::vector<Plan> plans;
    std::vector<GatedPlan> gated_plans;
    Projection(const float* data, int k, int n) : input(k), output(n) {
        auto pd = descriptor(8, false);
        auto raw = memory(memory::desc({n,k}, memory::data_type::f32,
            memory::format_tag::ab), eng, const_cast<float*>(data));
        auto packed = memory(pd.weights_desc(), eng);
        reorder(raw, packed).execute(str, raw, packed);
        str.wait();
        weights.push_back(packed);
        retained_bytes += packed.get_desc().get_size();
    }
    inner_product_forward::primitive_desc descriptor(int rows, bool accumulate) {
        primitive_attr attr;
        attr.set_fpmath_mode(fpmath_mode::strict);
        // User-owned scratch permits execution on another thread regardless
        // of the system library's concurrent-execution build option.
        attr.set_scratchpad_mode(scratchpad_mode::user);
        if (accumulate) {
            post_ops ops;
            ops.append_sum(1.f);
            attr.set_post_ops(ops);
        }
        return inner_product_forward::primitive_desc(eng, prop_kind::forward_inference,
            memory::desc({rows,input}, memory::data_type::f32, memory::format_tag::ab),
            memory::desc({output,input}, memory::data_type::f32, memory::format_tag::any),
            memory::desc({rows,output}, memory::data_type::f32, memory::format_tag::ab), attr);
    }
    Plan& prepare(int rows, bool accumulate) {
        Plan* plan = nullptr;
        for (auto& candidate : plans) {
            if (candidate.rows == rows && candidate.accumulate == accumulate) {
                plan = &candidate;
                break;
            }
        }
        if (!plan) {
            auto pd = descriptor(rows, accumulate);
            memory packed;
            for (auto& candidate : weights) {
                if (candidate.get_desc() == pd.weights_desc()) {
                    packed = candidate;
                    break;
                }
            }
            if (!packed) {
                packed = memory(pd.weights_desc(), eng);
                reorder(weights.front(), packed).execute(str, weights.front(), packed);
                str.wait();
                weights.push_back(packed);
                retained_bytes += packed.get_desc().get_size();
            }
            // Bound dynamic-length plan metadata; packed formats remain reusable.
            if (plans.size() == 32) {
                plans.erase(plans.begin());
            }
            plans.push_back({rows, accumulate, inner_product_forward(pd),
                memory(pd.src_desc(), eng, DNNL_MEMORY_NONE), packed,
                memory(pd.dst_desc(), eng, DNNL_MEMORY_NONE),
                memory(pd.scratchpad_desc(), eng, DNNL_MEMORY_NONE), pd.scratchpad_desc().get_size()});
            plan = &plans.back();
        }
        return *plan;
    }
    GatedPlan& prepare_geglu(int rows) {
        if (output % 2) throw std::invalid_argument("GeGLU requires even output width");
        for (auto& plan : gated_plans) if (plan.rows == rows) return plan;
        int hidden = output / 2;
        auto md = memory::desc({rows,hidden}, memory::data_type::f32, memory::format_tag::ab);
        primitive_attr attr;
        attr.set_fpmath_mode(fpmath_mode::strict);
        attr.set_scratchpad_mode(scratchpad_mode::user);
        auto gelu_pd = eltwise_forward::primitive_desc(eng,prop_kind::forward_inference,
            algorithm::eltwise_gelu_erf,md,md,attr);
        auto multiply_pd = binary::primitive_desc(eng,algorithm::binary_mul,md,md,md,attr);
        size_t gate_bytes = (size_t(rows) * hidden * sizeof(float) + 63) / 64 * 64;
        size_t scratch_bytes = std::max(gelu_pd.scratchpad_desc().get_size(),multiply_pd.scratchpad_desc().get_size());
        if (gated_plans.size() == 32) gated_plans.erase(gated_plans.begin());
        gated_plans.push_back({rows,eltwise_forward(gelu_pd),binary(multiply_pd),
            memory(md,eng,DNNL_MEMORY_NONE),memory(md,eng,DNNL_MEMORY_NONE),
            memory(gelu_pd.scratchpad_desc(),eng,DNNL_MEMORY_NONE),
            memory(multiply_pd.scratchpad_desc(),eng,DNNL_MEMORY_NONE),gate_bytes,gate_bytes+scratch_bytes});
        return gated_plans.back();
    }
    void geglu(const float* projected, int rows, float* result, void* workspace, size_t capacity) {
        auto& plan = prepare_geglu(rows);
        if (capacity < plan.required_bytes || !workspace) throw std::invalid_argument("insufficient GeGLU workspace");
        int hidden = output / 2;
        auto* gate = static_cast<float*>(workspace);
        // Layout movement only. Arithmetic stays in standard oneDNN primitives.
        for (int row=0;row<rows;++row) {
            std::copy_n(projected+size_t(row)*output,hidden,result+size_t(row)*hidden);
            std::copy_n(projected+size_t(row)*output+hidden,hidden,gate+size_t(row)*hidden);
        }
        auto* scratch = static_cast<unsigned char*>(workspace)+plan.gate_bytes;
        plan.value.set_data_handle(result);
        plan.gate.set_data_handle(gate);
        plan.gelu_scratch.set_data_handle(scratch);
        plan.multiply_scratch.set_data_handle(scratch);
        auto clear = [&] {
            plan.value.set_data_handle(DNNL_MEMORY_NONE);
            plan.gate.set_data_handle(DNNL_MEMORY_NONE);
            plan.gelu_scratch.set_data_handle(DNNL_MEMORY_NONE);
            plan.multiply_scratch.set_data_handle(DNNL_MEMORY_NONE);
        };
        try {
            plan.gelu.execute(str,{{DNNL_ARG_SRC,plan.value},{DNNL_ARG_DST,plan.value},
                {DNNL_ARG_SCRATCHPAD,plan.gelu_scratch}});
            str.wait();
            plan.multiply.execute(str,{{DNNL_ARG_SRC_0,plan.value},{DNNL_ARG_SRC_1,plan.gate},
                {DNNL_ARG_DST,plan.value},{DNNL_ARG_SCRATCHPAD,plan.multiply_scratch}});
            str.wait();
        } catch (...) {clear();throw;}
        clear();
    }
    void run(const float* x, int rows, float* y, bool accumulate, void* scratch, size_t capacity) {
        auto* plan = &prepare(rows, accumulate);
        if (capacity < plan->scratch_bytes || (plan->scratch_bytes && !scratch))
            throw std::invalid_argument("insufficient scratchpad capacity");
        plan->scratch.set_data_handle(scratch);
        plan->src.set_data_handle(const_cast<float*>(x));
        plan->dst.set_data_handle(y);
        try {
            plan->op.execute(str, {{DNNL_ARG_SRC,plan->src},
                {DNNL_ARG_WEIGHTS,plan->weights}, {DNNL_ARG_DST,plan->dst},
                {DNNL_ARG_SCRATCHPAD,plan->scratch}});
            str.wait();
        } catch (...) {
            plan->src.set_data_handle(DNNL_MEMORY_NONE);
            plan->dst.set_data_handle(DNNL_MEMORY_NONE);
            plan->scratch.set_data_handle(DNNL_MEMORY_NONE);
            throw;
        }
        // Do not retain a caller's potentially short-lived buffers.
        plan->src.set_data_handle(DNNL_MEMORY_NONE);
        plan->dst.set_data_handle(DNNL_MEMORY_NONE);
        plan->scratch.set_data_handle(DNNL_MEMORY_NONE);
    }
};
struct Normalization {
    engine eng{engine::kind::cpu,0};
    stream str{eng};
    layer_normalization_forward op;
    memory source, result, scale, shift, scratch;
    size_t required_bytes;
    bool with_bias;
    Normalization(int rows,int width,float epsilon,bool bias):with_bias(bias) {
        auto md=memory::desc({rows,width},memory::data_type::f32,memory::format_tag::ab);
        primitive_attr attr;attr.set_fpmath_mode(fpmath_mode::strict);attr.set_scratchpad_mode(scratchpad_mode::user);
        auto flags=normalization_flags::use_scale;
        if(bias)flags=flags|normalization_flags::use_shift;
        auto pd=layer_normalization_forward::primitive_desc(eng,prop_kind::forward_inference,md,md,epsilon,flags,attr);
        op=layer_normalization_forward(pd);
        source=memory(md,eng,DNNL_MEMORY_NONE);result=memory(md,eng,DNNL_MEMORY_NONE);
        auto weight_md=memory::desc({width},memory::data_type::f32,memory::format_tag::a);
        scale=memory(weight_md,eng,DNNL_MEMORY_NONE);shift=memory(weight_md,eng,DNNL_MEMORY_NONE);
        scratch=memory(pd.scratchpad_desc(),eng,DNNL_MEMORY_NONE);
        required_bytes=pd.scratchpad_desc().get_size();
    }
    void run(const float* x,const float* weight,const float* bias,float* y,void* workspace,size_t capacity) {
        if(capacity<required_bytes || (required_bytes&&!workspace))throw std::invalid_argument("insufficient normalization workspace");
        source.set_data_handle(const_cast<float*>(x));result.set_data_handle(y);
        scale.set_data_handle(const_cast<float*>(weight));
        if(with_bias)shift.set_data_handle(const_cast<float*>(bias));
        scratch.set_data_handle(workspace);
        auto clear=[&]{source.set_data_handle(DNNL_MEMORY_NONE);result.set_data_handle(DNNL_MEMORY_NONE);scale.set_data_handle(DNNL_MEMORY_NONE);shift.set_data_handle(DNNL_MEMORY_NONE);scratch.set_data_handle(DNNL_MEMORY_NONE);};
        try {
            std::unordered_map<int,memory>args={{DNNL_ARG_SRC,source},{DNNL_ARG_DST,result},{DNNL_ARG_SCALE,scale},{DNNL_ARG_SCRATCHPAD,scratch}};
            if(with_bias)args.emplace(DNNL_ARG_SHIFT,shift);
            op.execute(str,args);str.wait();
        }catch(...){clear();throw;}
        clear();
    }
};
void error_message(char* dest, size_t capacity, const char* message) noexcept {
    if (!dest || !capacity) return;
    size_t count = std::min(capacity - 1, std::strlen(message));
    std::memcpy(dest, message, count);
    dest[count] = '\0';
}
}
extern "C" {
void* decision_norm_create(int rows,int width,float epsilon,bool bias,size_t* scratch_bytes,char* error,size_t capacity) noexcept {
    try {auto owned=std::make_unique<Normalization>(rows,width,epsilon,bias);*scratch_bytes=owned->required_bytes;return owned.release();}
    catch(const std::exception&e){error_message(error,capacity,e.what());}
    catch(...){error_message(error,capacity,"unknown oneDNN exception");}
    return nullptr;
}
int decision_norm_run(void* handle,const float* x,const float* weight,const float* bias,float* y,void* workspace,size_t workspace_capacity,char* error,size_t capacity) noexcept {
    try{static_cast<Normalization*>(handle)->run(x,weight,bias,y,workspace,workspace_capacity);return 0;}
    catch(const std::exception&e){error_message(error,capacity,e.what());}
    catch(...){error_message(error,capacity,"unknown oneDNN exception");}
    return 1;
}
void decision_norm_destroy(void* handle) noexcept {delete static_cast<Normalization*>(handle);}
void* decision_projection_create(const float* weights, int input, int output,
    char* error, size_t capacity) noexcept {
    try { return new Projection(weights,input,output); }
    catch (const std::exception& e) { error_message(error,capacity,e.what()); }
    catch (...) { error_message(error,capacity,"unknown oneDNN exception"); }
    return nullptr;
}
int decision_projection_run(void* handle, const float* x, int rows, float* y,
    bool accumulate, void* scratch, size_t scratch_capacity, char* error, size_t capacity) noexcept {
    try { static_cast<Projection*>(handle)->run(x,rows,y,accumulate,scratch,scratch_capacity); return 0; }
    catch (const std::exception& e) { error_message(error,capacity,e.what()); }
    catch (...) { error_message(error,capacity,"unknown oneDNN exception"); }
    return 1;
}
int decision_projection_scratch_bytes(void* handle, int rows, bool accumulate,
    size_t* bytes, char* error, size_t capacity) noexcept {
    try { *bytes = static_cast<Projection*>(handle)->prepare(rows,accumulate).scratch_bytes; return 0; }
    catch (const std::exception& e) { error_message(error,capacity,e.what()); }
    catch (...) { error_message(error,capacity,"unknown oneDNN exception"); }
    return 1;
}
int decision_projection_geglu_bytes(void* handle, int rows, size_t* bytes, char* error, size_t capacity) noexcept {
    try { *bytes=static_cast<Projection*>(handle)->prepare_geglu(rows).required_bytes; return 0; }
    catch (const std::exception& e) {error_message(error,capacity,e.what());}
    catch (...) {error_message(error,capacity,"unknown oneDNN exception");}
    return 1;
}
int decision_projection_geglu(void* handle, const float* projected, int rows, float* result,
    void* workspace, size_t workspace_capacity, char* error, size_t capacity) noexcept {
    try {static_cast<Projection*>(handle)->geglu(projected,rows,result,workspace,workspace_capacity);return 0;}
    catch (const std::exception& e) {error_message(error,capacity,e.what());}
    catch (...) {error_message(error,capacity,"unknown oneDNN exception");}
    return 1;
}
void decision_projection_destroy(void* handle) noexcept {
    delete static_cast<Projection*>(handle);
}
size_t decision_projection_bytes(const void* handle) noexcept {
    return static_cast<const Projection*>(handle)->retained_bytes;
}
}
