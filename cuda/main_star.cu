// hough_cuda_gpu_scan.cu
// CUDA gradient-guided circle Hough with GPU-side scan optimization
// Compile:
// nvcc -O3 -arch=sm_60 main_star_gpu_scan.cu -o hough_gpu_scan pkg-config --cflags --libs opencv4 --compiler-options '-fPIC'
//
// Run:
// ./hough_gpu_scan star_8k.png --mode circle
// ./hough_gpu_scan star_8k.png --mode line

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
using namespace std;
using namespace cv;
using namespace chrono;

#define CUDA_CHECK(call) do{ cudaError_t e=(call); if(e!=cudaSuccess){ fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

struct Params{ 
    string input, output="output.png", mode="circle"; 
    int rmin=100,rmax=200,rstep=5; 
    int canny_low=80,canny_high=180; 
    int theta_step_deg=1, theta_window_deg=4; 
    int threads=256; 
};

Params parse_args(int argc,char**argv){
    Params p; 
    if(argc<2){ cerr<<"Usage\n"; exit(1);} 
    p.input=argv[1];
    for(int i=2;i<argc;i++){ 
        string a=argv[i];
        if(a=="-o"&&i+1<argc) p.output=argv[++i];
        else if(a=="--mode"&&i+1<argc) p.mode=argv[++i];
        else if(a=="--rmin"&&i+1<argc) p.rmin=stoi(argv[++i]);
        else if(a=="--rmax"&&i+1<argc) p.rmax=stoi(argv[++i]);
        else if(a=="--rstep"&&i+1<argc) p.rstep=stoi(argv[++i]);
        else if(a=="--tstep"&&i+1<argc) p.theta_step_deg=stoi(argv[++i]);
        else if(a=="--twin"&&i+1<argc) p.theta_window_deg=stoi(argv[++i]);
        else if(a=="--threads"&&i+1<argc) p.threads=stoi(argv[++i]);
    }

    if (p.mode == "circle")
        p.output = "out_circle_gpu_scan.png";
    else if (p.mode == "line")
        p.output = "out_line_gpu_scan.png";
    return p;
}

struct Edge { 
    int x,y; 
    float nx,ny; 
    float ori; 
};

struct CircleResult {
    int cx, cy, r, votes;
};

// GPU 端 Scan：掃描單個累加器找最大值
_global_ void find_max_single_radius_kernel(
    const int* d_acc, 
    int sz, 
    int W,
    int radius_idx,
    CircleResult* d_results
) {
    int local_max_votes = 0;
    int local_max_idx = 0;
    
    // 每個執行緒掃描一部分
    for(int i = threadIdx.x; i < sz; i += blockDim.x){
        if(d_acc[i] > local_max_votes){
            local_max_votes = d_acc[i];
            local_max_idx = i;
        }
    }
    
    // Block 內 reduce
    _shared_ int shared_max_votes;
    _shared_ int shared_max_idx;
    
    if(threadIdx.x == 0) {
        shared_max_votes = 0;
        shared_max_idx = 0;
    }
    __syncthreads();
    
    if(local_max_votes > 0) {
        atomicMax(&shared_max_votes, local_max_votes);
    }
    __syncthreads();
    
    // 所有執行緒重新尋找匹配的索引
    for(int i = threadIdx.x; i < sz; i += blockDim.x){
        if(d_acc[i] == shared_max_votes && threadIdx.x == 0){
            shared_max_idx = i;
            break;
        }
    }
    __syncthreads();
    
    // Thread 0 寫回結果
    if(threadIdx.x == 0) {
        d_results[radius_idx].votes = shared_max_votes;
        d_results[radius_idx].cx = shared_max_idx % W;
        d_results[radius_idx].cy = shared_max_idx / W;
    }
}

// GPU 端優化 Scan：掃描所有 21 個累加器（單一核心）
_global_ void find_best_all_radii_optimized_kernel(
    const int* d_acc_persistent,
    int W, int H, int n_radii,
    int rmin, int rstep,
    CircleResult* d_results
) {
    // 每個 block 處理一個半徑
    int radius_idx = blockIdx.x;
    if(radius_idx >= n_radii) return;
    
    const int* acc = d_acc_persistent + radius_idx * (size_t)(W*H);
    int sz = W*H;
    
    int local_max_votes = 0;
    int local_max_idx = 0;
    
    // 每個執行緒掃描一部分累加器
    for(int i = threadIdx.x; i < sz; i += blockDim.x) {
        if(acc[i] > local_max_votes) {
            local_max_votes = acc[i];
            local_max_idx = i;
        }
    }
    
    // Block 內共享記憶體 reduce
    _shared_ int shared_max_votes;
    _shared_ int shared_max_idx;
    
    if(threadIdx.x == 0) {
        shared_max_votes = 0;
        shared_max_idx = 0;
    }
    __syncthreads();
    
    // 更新共享最大值
    atomicMax(&shared_max_votes, local_max_votes);
    __syncthreads();
    
    // 所有線程同步後，再找索引
    if(local_max_votes == shared_max_votes && threadIdx.x == 0) {
        shared_max_idx = local_max_idx;
    }
    __syncthreads();
    
    // Thread 0 寫回結果
    if(threadIdx.x == 0) {
        int r = rmin + radius_idx * rstep;
        d_results[radius_idx].votes = shared_max_votes;
        d_results[radius_idx].cx = shared_max_idx % W;
        d_results[radius_idx].cy = shared_max_idx / W;
        d_results[radius_idx].r = r;
    }
}

_global_ void circle_vote_kernel(const Edge* d_edges, int n, int W, int H, int r, int* d_acc){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=n) return;
    Edge e = d_edges[idx];
    int cx = (int)roundf(e.x + r * e.nx);
    int cy = (int)roundf(e.y + r * e.ny);
    if(cx>=0 && cx < W && cy>=0 && cy < H) atomicAdd(&d_acc[cy*W + cx], 1);
    int cx2 = (int)roundf(e.x - r * e.nx);
    int cy2 = (int)roundf(e.y - r * e.ny);
    if(cx2>=0 && cx2 < W && cy2>=0 && cy2 < H) atomicAdd(&d_acc[cy2*W + cx2], 1);
}

_global_ void line_vote_kernel(const Edge* d_edges, int n, double rho_min, double rho_max, int nrho, int rho_off, double* d_cos, double* d_sin, int T, int theta_step, int window_deg, int* d_acc){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=n) return;
    Edge e = d_edges[idx];
    float base = e.ori + M_PI/2.0f;
    int base_deg = (int)roundf(base * 180.0f / M_PI) % 180; 
    if(base_deg<0) base_deg+=180;
    for(int d=-window_deg; d<=window_deg; ++d){
        int deg = base_deg + d;
        if(deg<0) deg+=180; 
        if(deg>=180) deg-=180;
        if(deg % theta_step != 0) continue;
        int ti = deg / theta_step;
        double theta = deg * M_PI / 180.0;
        double rho = e.x * cos(theta) + e.y * sin(theta);
        int ri = (int)round(rho) + rho_off;
        if(ri>=0 && ri<nrho){
            int pos = ti * nrho + ri;
            atomicAdd(&d_acc[pos], 1);
        }
    }
}

int main(int argc,char**argv){
    auto t_total_start = high_resolution_clock::now();
    Params p = parse_args(argc,argv);
    
    // Step 1: Image I/O
    auto t_io_start = high_resolution_clock::now();
    Mat img = imread(p.input);
    if(img.empty()){ cerr<<"Cannot open image\n"; return -1; }
    Mat gray; 
    cvtColor(img,gray,COLOR_BGR2GRAY);
    auto t_io_end = high_resolution_clock::now();
    double io_ms = duration<double,milli>(t_io_end - t_io_start).count();

    // Canny
    auto t0 = high_resolution_clock::now();
    Mat edges; 
    Canny(gray, edges, p.canny_low, p.canny_high);
    auto t1 = high_resolution_clock::now();
    double canny_ms = duration<double,milli>(t1-t0).count();

    // Sobel gradients
    t0 = high_resolution_clock::now();
    Mat gx, gy; 
    Sobel(gray,gx,CV_32F,1,0,3); 
    Sobel(gray,gy,CV_32F,0,1,3);
    vector<Edge> edgelist; 
    edgelist.reserve(1000000);
    for(int y=0;y<edges.rows;y++){
        const uchar* er = edges.ptr<uchar>(y);
        const float* gxr = gx.ptr<float>(y);
        const float* gyr = gy.ptr<float>(y);
        for(int x=0;x<edges.cols;x++){
            if(!er[x]) continue;
            float dx=gxr[x], dy=gyr[x];
            float mag = sqrtf(dx*dx + dy*dy);
            if(mag < 1e-6f) continue;
            edgelist.push_back({x,y,dx/mag,dy/mag,(float)atan2f(dy,dx)});
        }
    }
    auto t2 = high_resolution_clock::now();
    double grad_ms = duration<double,milli>(t2-t0).count();

    int W = gray.cols, H = gray.rows;
    int n_edges = (int)edgelist.size();
    if(n_edges==0){ cerr<<"No edges\n"; return 0; }

    // Copy edges to device
    auto t_h2d_edges_start = high_resolution_clock::now();
    Edge* d_edges = nullptr;
    CUDA_CHECK(cudaMalloc(&d_edges, sizeof(Edge) * n_edges));
    CUDA_CHECK(cudaMemcpy(d_edges, edgelist.data(), sizeof(Edge) * n_edges, cudaMemcpyHostToDevice));
    auto t_h2d_edges_end = high_resolution_clock::now();
    double h2d_edges_ms = duration<double,milli>(t_h2d_edges_end - t_h2d_edges_start).count();

    int threads = p.threads;
    int blocks = (n_edges + threads - 1) / threads;

    if(p.mode=="line"){
        int theta_step = p.theta_step_deg;
        int T = 180 / theta_step;
        double diag = sqrt((double)W*W + (double)H*H);
        int nrho = (int)diag*2 + 1;
        int rho_off = nrho/2;
        size_t acc_elems = (size_t)nrho * T;
        int* d_acc = nullptr;
        CUDA_CHECK(cudaMalloc(&d_acc, sizeof(int) * acc_elems));
        CUDA_CHECK(cudaMemset(d_acc, 0, sizeof(int) * acc_elems));
        
        cudaEvent_t start, stop; 
        CUDA_CHECK(cudaEventCreate(&start)); 
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        line_vote_kernel<<<blocks, threads>>>(d_edges, n_edges, -diag, diag, nrho, rho_off, nullptr, nullptr, T, theta_step, p.theta_window_deg, d_acc);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop)); 
        CUDA_CHECK(cudaEventSynchronize(stop));
        float kernel_ms=0; 
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
        cout<<"GPU kernel (line) time: "<<kernel_ms<<" ms\n";
        
        // Copy back and find best
        vector<int> h_acc(acc_elems);
        CUDA_CHECK(cudaMemcpy(h_acc.data(), d_acc, sizeof(int) * acc_elems, cudaMemcpyDeviceToHost));
        
        int best_votes=0, bt=0, br=0;
        for(int ti=0; ti<T; ++ti) {
            for(int ri=0; ri<nrho; ++ri) { 
                int v=h_acc[ti*nrho + ri]; 
                if(v>best_votes){ best_votes=v; bt=ti; br=ri; } 
            }
        }
        float best_theta = bt * theta_step * M_PI / 180.0f;
        float best_rho = br - rho_off;
        cout<<"Best line: rho="<<best_rho<<" theta(deg)="<<(best_theta*180.0f/M_PI)<<" votes="<<best_votes<<"\n";
        Mat out = img.clone();
        double a=cos(best_theta), b=sin(best_theta); 
        double x0=a*best_rho, y0=b*best_rho;
        Point p1(cvRound(x0+2000*(-b)), cvRound(y0+2000*(a))), 
               p2(cvRound(x0-2000*(-b)), cvRound(y0-2000*(a)));
        line(out,p1,p2,Scalar(0,0,255),3); 
        imwrite(p.output,out);
        CUDA_CHECK(cudaFree(d_acc));
        CUDA_CHECK(cudaEventDestroy(start)); 
        CUDA_CHECK(cudaEventDestroy(stop));
    } else {
        // Circle detection with GPU-side scan
        int rmin=p.rmin, rmax=p.rmax, rstep=p.rstep;
        int n_radii = (rmax - rmin) / rstep + 1;
        
        // 計算記憶體大小
        size_t acc_total_size = (size_t)W * H * n_radii;
        size_t acc_bytes = sizeof(int) * acc_total_size;
        
        cout<<"Memory allocation: "<<(acc_bytes / (1024.0*1024.0))<<" MB for accumulator\n";
        
        // Step 2: GPU Memory allocation
        auto t_mem_alloc_start = high_resolution_clock::now();
        int* d_acc_persistent = nullptr;
        CUDA_CHECK(cudaMalloc(&d_acc_persistent, acc_bytes));
        CUDA_CHECK(cudaMemset(d_acc_persistent, 0, acc_bytes));
        
        CircleResult* d_results = nullptr;
        CUDA_CHECK(cudaMalloc(&d_results, sizeof(CircleResult) * n_radii));
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t_mem_alloc_end = high_resolution_clock::now();
        double mem_alloc_ms = duration<double,milli>(t_mem_alloc_end - t_mem_alloc_start).count();
        
        // Timing events
        cudaEvent_t t_start, t_stop, t_scan_start, t_scan_stop;
        CUDA_CHECK(cudaEventCreate(&t_start));
        CUDA_CHECK(cudaEventCreate(&t_stop));
        CUDA_CHECK(cudaEventCreate(&t_scan_start));
        CUDA_CHECK(cudaEventCreate(&t_scan_stop));
        
        // Step 3: Host to Device transfer
        auto t_h2d_start = high_resolution_clock::now();
        // (edges copy done earlier)
        auto t_h2d_end = high_resolution_clock::now();
        
        // Step 4: GPU voting kernels
        CUDA_CHECK(cudaEventRecord(t_start));
        int radius_idx = 0;
        for (int r = rmin; r <= rmax; r += rstep) {
            int* acc_ptr = d_acc_persistent + (radius_idx * (size_t)(W*H));
            circle_vote_kernel<<<blocks, threads>>>(d_edges, n_edges, W, H, r, acc_ptr);
            CUDA_CHECK(cudaGetLastError());
            radius_idx++;
        }
        CUDA_CHECK(cudaEventRecord(t_stop));
        CUDA_CHECK(cudaEventSynchronize(t_stop));
        float voting_ms=0;
        CUDA_CHECK(cudaEventElapsedTime(&voting_ms, t_start, t_stop));
        
        // Step 5: GPU-side scan
        CUDA_CHECK(cudaEventRecord(t_scan_start));
        find_best_all_radii_optimized_kernel<<<n_radii, 256>>>(
            d_acc_persistent, W, H, n_radii,
            rmin, rstep,
            d_results
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(t_scan_stop));
        CUDA_CHECK(cudaEventSynchronize(t_scan_stop));
        float scan_ms=0;
        CUDA_CHECK(cudaEventElapsedTime(&scan_ms, t_scan_start, t_scan_stop));
        
        // Step 6: Device to Host transfer
        auto t_d2h_start = high_resolution_clock::now();
        vector<CircleResult> h_results(n_radii);
        CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, sizeof(CircleResult) * n_radii, cudaMemcpyDeviceToHost));
        auto t_d2h_end = high_resolution_clock::now();
        double d2h_ms = duration<double,milli>(t_d2h_end - t_d2h_start).count();
        
        // Step 7: CPU-side results processing
        auto t_cpu_start = high_resolution_clock::now();
        int best_votes=0, best_cx=0, best_cy=0, best_r=0;
        for (int ri = 0; ri < n_radii; ri++) {
            if (h_results[ri].votes > best_votes) {
                best_votes = h_results[ri].votes;
                best_cx = h_results[ri].cx;
                best_cy = h_results[ri].cy;
                best_r = h_results[ri].r;
            }
        }
        auto t_cpu_end = high_resolution_clock::now();
        double cpu_results_ms = duration<double,milli>(t_cpu_end - t_cpu_start).count();
        
        // Step 8: Image I/O
        auto t_img_out_start = high_resolution_clock::now();
        Mat out=img.clone(); 
        if(best_votes>0) {
            circle(out,Point(best_cx,best_cy),best_r,Scalar(0,255,0),3);
            circle(out,Point(best_cx,best_cy),3,Scalar(0,0,255),-1);
        }
        imwrite(p.output,out);
        auto t_img_out_end = high_resolution_clock::now();
        double img_output_ms = duration<double,milli>(t_img_out_end - t_img_out_start).count();
        
        cout<<"GPU voting time: "<<voting_ms<<" ms\n";
        cout<<"GPU scan time: "<<scan_ms<<" ms\n";
        cout<<"Best circle: cx="<<best_cx<<" cy="<<best_cy<<" r="<<best_r<<" votes="<<best_votes<<"\n";
        
        cout<<"\n╔════════════════════════════════════════════════════════════════╗\n";
        cout<<"║         CUDA GPU Scan - Detailed Time Breakdown              ║\n";
        cout<<"╚════════════════════════════════════════════════════════════════╝\n\n";
        
        cout<<"Image I/O (imread+cvtColor):     "<<io_ms<<" ms\n";
        cout<<"Canny edge detection:            "<<canny_ms<<" ms\n";
        cout<<"Sobel + Edge list creation:      "<<grad_ms<<" ms\n";
        cout<<"GPU memory allocation:           "<<mem_alloc_ms<<" ms\n";
        cout<<"H2D transfer (edges):            "<<h2d_edges_ms<<" ms\n";
        cout<<"GPU voting kernels:              "<<voting_ms<<" ms\n";
        cout<<"GPU scan (reduce):               "<<scan_ms<<" ms\n";
        cout<<"D2H transfer (results):          "<<d2h_ms<<" ms\n";
        cout<<"CPU results processing:          "<<cpu_results_ms<<" ms\n";
        cout<<"Image output (clone+draw+write): "<<img_output_ms<<" ms\n";
        cout<<"\n───────────────────────────────────────────────────────────────\n";
        
        CUDA_CHECK(cudaEventDestroy(t_start));
        CUDA_CHECK(cudaEventDestroy(t_stop));
        CUDA_CHECK(cudaEventDestroy(t_scan_start));
        CUDA_CHECK(cudaEventDestroy(t_scan_stop));
        CUDA_CHECK(cudaFree(d_acc_persistent));
        CUDA_CHECK(cudaFree(d_results));
    }

    CUDA_CHECK(cudaFree(d_edges));
    auto t_total_end = high_resolution_clock::now();
    double total_ms = duration<double,milli>(t_total_end - t_total_start).count();
    
    cout<<"Total: "<<total_ms<<" ms\n";
    
    cout<<"\n╔════════════════════════════════════════════════════════════════╗\n";
    cout<<"║         GPU Scan Optimized Hough Transform                   ║\n";
    cout<<"╚════════════════════════════════════════════════════════════════╝\n\n";
    cout<<"Total time: "<<total_ms<<" ms\n";
    
    return 0;
}