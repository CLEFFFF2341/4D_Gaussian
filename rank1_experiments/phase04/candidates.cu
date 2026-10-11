// Independent read-only export of MoRGS 5c62ae8 geometry/bin/image buffers.
// Buffer layout and alpha*T traversal follow its Inria research renderer.
// No changes to the original RGB forward/backward implementation or ABI.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>

template<class T> T* take(char*& chunk, size_t n) {
    auto off=(reinterpret_cast<uintptr_t>(chunk)+127)&~uintptr_t(127);
    auto ptr=reinterpret_cast<T*>(off); chunk=reinterpret_cast<char*>(ptr+n); return ptr;
}

__global__ void candidates(const float2* xy,const float4* conic,const uint32_t* list,
 const uint2* ranges,const bool* alive,int H,int W,int K,int64_t* ids,float* weights,float* mass) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=H*W) return;
    int x=p%W,y=p/W;
    uint2 r=ranges[(y/16)*((W+15)/16)+x/16];
    float T=1.f,all=0.f,matched=0.f;
    float best[64]; int rows[64];
    for(int j=0;j<K;j++){best[j]=0.f;rows[j]=-1;}
    for(uint32_t i=r.x;i<r.y;i++){
        int row=list[i]; float2 v=xy[row]; float4 c=conic[row];
        float dx=v.x-x,dy=v.y-y;
        float power=-.5f*(c.x*dx*dx+c.z*dy*dy)-c.y*dx*dy;
        if(power>0.f) continue;
        float alpha=fminf(.99f,c.w*expf(power));
        if(alpha<1.f/255.f)continue;
        float next=T*(1.f-alpha);
        if(next<.0001f)break;
        float w=alpha*T;all+=w;
        if(alive[row]){
            matched+=w;
            // Strict greater retains native depth order for equal weights.
            for(int j=0;j<K;j++) if(w>best[j]){
                for(int l=K-1;l>j;l--){best[l]=best[l-1];rows[l]=rows[l-1];}
                best[j]=w;rows[j]=row;break;
            }
        }
        T=next;
    }
    for(int j=0;j<K;j++){ids[p*K+j]=rows[j];weights[p*K+j]=best[j];}
    mass[p*2]=all;mass[p*2+1]=matched;
}

std::vector<torch::Tensor> export_candidates(torch::Tensor geom,torch::Tensor bin,
 torch::Tensor image,torch::Tensor alive,int64_t P,int64_t R,int64_t H,int64_t W,int64_t K,bool debug) {
    TORCH_CHECK(geom.is_cuda()&&bin.is_cuda()&&image.is_cuda()&&alive.is_cuda(),"CUDA required");
    TORCH_CHECK(K>0&&K<=64&&P>0&&H>0&&W>0&&R>=0,"invalid dimensions");
    TORCH_CHECK(alive.scalar_type()==torch::kBool&&alive.numel()==P&&alive.is_contiguous(),"alive bool contiguous [P]");
    TORCH_CHECK(geom.scalar_type()==torch::kUInt8&&bin.scalar_type()==torch::kUInt8&&image.scalar_type()==torch::kUInt8,"buffer dtype");
    const c10::cuda::CUDAGuard guard(geom.device());
    TORCH_CHECK(alive.device()==geom.device()&&bin.device()==geom.device()&&image.device()==geom.device(),"device mismatch");
    char* g=reinterpret_cast<char*>(geom.data_ptr<uint8_t>());
    take<float>(g,P);take<bool>(g,P*3);take<int>(g,P);
    auto xy=take<float2>(g,P);take<float>(g,P*2);take<float>(g,P*6);
    auto conic=take<float4>(g,P);
    char* b=reinterpret_cast<char*>(bin.data_ptr<uint8_t>());
    auto list=take<uint32_t>(b,R);
    char* im=reinterpret_cast<char*>(image.data_ptr<uint8_t>());
    take<float>(im,H*W);take<uint32_t>(im,H*W);auto ranges=take<uint2>(im,H*W);
    auto opts=geom.options().dtype(torch::kFloat32);
    auto ids=torch::empty({H,W,K},opts.dtype(torch::kInt64));
    auto weights=torch::empty({H,W,K},opts);auto mass=torch::empty({H,W,2},opts);
    auto stream=at::cuda::getCurrentCUDAStream();
    candidates<<<(H*W+127)/128,128,0,stream>>>(xy,conic,list,ranges,alive.data_ptr<bool>(),H,W,K,ids.data_ptr<int64_t>(),weights.data_ptr<float>(),mass.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    std::vector<torch::Tensor> result={ids,weights,mass};
    if(debug){
        auto xo=torch::empty({P,2},opts),co=torch::empty({P,4},opts);
        auto li=torch::empty({R},opts.dtype(torch::kInt32));
        auto ra=torch::empty({((H+15)/16)*((W+15)/16),2},opts.dtype(torch::kInt32));
        C10_CUDA_CHECK(cudaMemcpyAsync(xo.data_ptr(),xy,P*sizeof(float2),cudaMemcpyDeviceToDevice,stream));
        C10_CUDA_CHECK(cudaMemcpyAsync(co.data_ptr(),conic,P*sizeof(float4),cudaMemcpyDeviceToDevice,stream));
        C10_CUDA_CHECK(cudaMemcpyAsync(li.data_ptr(),list,R*sizeof(uint32_t),cudaMemcpyDeviceToDevice,stream));
        C10_CUDA_CHECK(cudaMemcpyAsync(ra.data_ptr(),ranges,ra.numel()*sizeof(uint32_t),cudaMemcpyDeviceToDevice,stream));
        result.insert(result.end(),{xo,co,li,ra});
    }
    return result;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("export_candidates",&export_candidates);}
