// hough_cuda.cu
// CUDA gradient-guided line & circle Hough
// Compile:
// nvcc -O3 hough_cuda.cu -o hough_cuda `pkg-config --cflags --libs opencv4` --compiler-options '-fPIC'
//
// Run:
// ./hough_cuda input.jpg --mode circle --rmin 20 --rmax 200 --rstep 2 -o out_cuda_circle.png
// ./hough_cuda input.jpg --mode line --tstep 1 --twin 4 -o out_cuda_line.png

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

struct Params{ string input, output="output.png", mode="circle"; int rmin=20,rmax=120,rstep=1; int canny_low=50,canny_high=150; int theta_step_deg=1, theta_window_deg=4; int threads=256; };

Params parse_args(int argc,char**argv){
    Params p; if(argc<2){ cerr<<"Usage\n"; exit(1);} p.input=argv[1];
    for(int i=2;i<argc;i++){ string a=argv[i];
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
        p.output = "out_circle_cuda.png";
    else if (p.mode == "line")
        p.output = "out_line_cuda.png";
    return p;
}

struct Edge { int x,y; float nx,ny; float ori; };

__global__ void line_vote_kernel(const Edge* d_edges, int n, double rho_min, double rho_max, int nrho, int rho_off, double* d_cos, double* d_sin, int T, int theta_step, int window_deg, int* d_acc){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=n) return;
    Edge e = d_edges[idx];
    // compute base degree
    float base = e.ori + M_PI/2.0f;
    int base_deg = (int)roundf(base * 180.0f / M_PI) % 180; if(base_deg<0) base_deg+=180;
    for(int d=-window_deg; d<=window_deg; ++d){
        int deg = base_deg + d;
        if(deg<0) deg+=180; if(deg>=180) deg-=180;
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

__global__ void circle_vote_kernel(const Edge* d_edges, int n, int W, int H, int r, int* d_acc){
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

int main(int argc,char**argv){
    auto t_total_start = high_resolution_clock::now();
    Params p = parse_args(argc,argv);
    Mat img = imread(p.input);
    if(img.empty()){ cerr<<"Cannot open image\n"; return -1; }
    Mat gray; cvtColor(img,gray,COLOR_BGR2GRAY);

    // Canny
    auto t0 = high_resolution_clock::now();
    Mat edges; Canny(gray, edges, p.canny_low, p.canny_high);
    auto t1 = high_resolution_clock::now();
    double canny_ms = duration<double,milli>(t1-t0).count();

    // Gradients
    t0 = high_resolution_clock::now();
    Mat gx, gy; Sobel(gray,gx,CV_32F,1,0,3); Sobel(gray,gy,CV_32F,0,1,3);
    vector<Edge> edgelist; edgelist.reserve(1000000);
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

    // copy edges to device
    Edge* d_edges = nullptr;
    CUDA_CHECK(cudaMalloc(&d_edges, sizeof(Edge) * n_edges));
    CUDA_CHECK(cudaMemcpy(d_edges, edgelist.data(), sizeof(Edge) * n_edges, cudaMemcpyHostToDevice));

    int threads = p.threads;
    int blocks = (n_edges + threads - 1) / threads;

    if(p.mode=="line"){
        int theta_step = p.theta_step_deg;
        int T = 180 / theta_step;
        double diag = sqrt((double)W*W + (double)H*H);
        int nrho = (int)diag*2 + 1;
        int rho_off = nrho/2;
        // allocate cos/sin not necessary; kernel computes cos/sin inline
        size_t acc_elems = (size_t)nrho * T;
        int* d_acc = nullptr;
        CUDA_CHECK(cudaMalloc(&d_acc, sizeof(int) * acc_elems));
        CUDA_CHECK(cudaMemset(d_acc, 0, sizeof(int) * acc_elems));
        // kernel timing
        cudaEvent_t start, stop; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        line_vote_kernel<<<blocks, threads>>>(d_edges, n_edges, -diag, diag, nrho, rho_off, nullptr, nullptr, T, theta_step, p.theta_window_deg, d_acc);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
        float kernel_ms=0; CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
        cout<<"CUDA kernel (line) time: "<<kernel_ms<<" ms\n";
        // copy back
        vector<int> h_acc(acc_elems);
        CUDA_CHECK(cudaMemcpy(h_acc.data(), d_acc, sizeof(int) * acc_elems, cudaMemcpyDeviceToHost));
        // find best
        int best_votes=0, bt=0, br=0;
        for(int ti=0; ti<T; ++ti) for(int ri=0; ri<nrho; ++ri){ int v=h_acc[ti*nrho + ri]; if(v>best_votes){ best_votes=v; bt=ti; br=ri; } }
        float best_theta = bt * theta_step * M_PI / 180.0f;
        float best_rho = br - rho_off;
        cout<<"Best line (GPU): rho="<<best_rho<<" theta(deg)="<<(best_theta*180.0f/M_PI)<<" votes="<<best_votes<<"\n";
        // Mat out = img.clone();
        // double a=cos(best_theta), b=sin(best_theta); double x0=a*best_rho, y0=b*best_rho;
        // Point p1(cvRound(x0+2000*(-b)), cvRound(y0+2000*(a))), p2(cvRound(x0-2000*(-b)), cvRound(y0-2000*(a)));
        // line(out,p1,p2,Scalar(0,0,255),3); imwrite(p.output,out);
        CUDA_CHECK(cudaFree(d_acc));
        CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    } else {
        int rmin=p.rmin, rmax=p.rmax, rstep=p.rstep;
        int best_votes=0,best_cx=0,best_cy=0,best_r=0;
        // per-radius allocate device accumulator
        float total_kernel_ms = 0;
        for (int r = rmin; r <= rmax; r += rstep) {
            int sz = W*H;
            int* d_acc = nullptr;
            CUDA_CHECK(cudaMalloc(&d_acc, sizeof(int) * (size_t)sz));
            CUDA_CHECK(cudaMemset(d_acc, 0, sizeof(int) * (size_t)sz));
            // kernel
            cudaEvent_t start, stop; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
            CUDA_CHECK(cudaEventRecord(start));
            circle_vote_kernel<<<blocks, threads>>>(d_edges, n_edges, W, H, r, d_acc);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
            float kernel_ms=0; CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
            total_kernel_ms += kernel_ms;
            // copy
            vector<int> h_acc((size_t)sz);
            CUDA_CHECK(cudaMemcpy(h_acc.data(), d_acc, sizeof(int) * (size_t)sz, cudaMemcpyDeviceToHost));
            // find best
            int local_best=0, lc=0, lr=0;
            for(size_t i=0;i<h_acc.size();++i){ int v=h_acc[i]; if(v>local_best){ local_best=v; lc=i%W; lr=i/W; } }
            if(local_best>best_votes){ best_votes=local_best; best_cx=lc; best_cy=lr; best_r=r; }
            CUDA_CHECK(cudaFree(d_acc));
            CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
        }
        cout<<"CUDA kernel (circle) time: "<<total_kernel_ms<<" ms\n";
        cout<<"Best circle (GPU): cx="<<best_cx<<" cy="<<best_cy<<" r="<<best_r<<" votes="<<best_votes<<"\n";
        // Mat out=img.clone(); if(best_votes>0) circle(out,Point(best_cx,best_cy),best_r,Scalar(0,255,0),3); imwrite(p.output,out);
    }

    CUDA_CHECK(cudaFree(d_edges));
    auto t_total_end = high_resolution_clock::now();
    cout<<"Timing summary (ms): Canny="<<canny_ms<<" Grad="<<grad_ms<<" Total="<<duration<double,milli>(t_total_end - t_total_start).count()<<"\n";
    return 0;
}
