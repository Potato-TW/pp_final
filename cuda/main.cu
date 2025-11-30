// main.cu
// CUDA + OpenCV Hough Transform (lines + naive circles demo)
// Compile with provided Makefile
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

// ----------------- utility macros -----------------
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                             \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

// ----------------- params -----------------
struct Params {
    string input_image;
    string output_image = "out_cuda.png";
    int device = 0;
    int threads_per_block = 256;
    int theta_bins = 180;  // number of theta bins
    double rho_res = 1.0;  // rho resolution (pixels)
    int canny_low = 50;
    int canny_high = 150;
    int peak_threshold = 120;  // votes threshold to consider a line
    int top_k = 20;
    string mode = "line";  // "line" or "circle"
    // circle params (naive)
    int r_min = 20;
    int r_max = 100;
    int r_step = 1;
    int circle_theta_step_deg = 12;  // coarse sampling for naive CHT
};

Params parse_args(int argc, char** argv) {
    Params p;
    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if (a == "-i" && i + 1 < argc)
            p.input_image = argv[++i];
        else if (a == "-o" && i + 1 < argc)
            p.output_image = argv[++i];
        else if (a == "-d" && i + 1 < argc)
            p.device = stoi(argv[++i]);
        else if (a == "-t" && i + 1 < argc)
            p.threads_per_block = stoi(argv[++i]);
        else if (a == "--theta" && i + 1 < argc)
            p.theta_bins = stoi(argv[++i]);
        else if (a == "--rho" && i + 1 < argc)
            p.rho_res = stod(argv[++i]);
        else if (a == "--canny-low" && i + 1 < argc)
            p.canny_low = stoi(argv[++i]);
        else if (a == "--canny-high" && i + 1 < argc)
            p.canny_high = stoi(argv[++i]);
        else if (a == "--peak" && i + 1 < argc)
            p.peak_threshold = stoi(argv[++i]);
        else if (a == "--topk" && i + 1 < argc)
            p.top_k = stoi(argv[++i]);
        else if (a == "--mode" && i + 1 < argc)
            p.mode = argv[++i];
        else if (a == "--rmin" && i + 1 < argc)
            p.r_min = stoi(argv[++i]);
        else if (a == "--rmax" && i + 1 < argc)
            p.r_max = stoi(argv[++i]);
        else if (a == "--rstep" && i + 1 < argc)
            p.r_step = stoi(argv[++i]);
        else if (a == "--circle-theta-step" && i + 1 < argc)
            p.circle_theta_step_deg = stoi(argv[++i]);
        else {
            cerr << "Unknown arg: " << a << "\n";
        }
    }
    if (p.input_image.empty()) {
        cerr << "Usage: -i input.jpg [-o out.png] [--mode line|circle] [--theta N] [--rho R] ...\n";
        exit(1);
    }
    return p;
}

// ----------------- device kernels -----------------

// Precomputed sin/cos table: we will allocate and copy an array of doubles to device
// Kernel: each thread processes one edge (point). For theta = 0..T-1 compute rho and atomicAdd to
// accumulator. accumulator layout: acc[rho_bin * T + theta]
__global__ void sht_vote_kernel(const int2* d_edges, int n_edges, const double* d_cos,
                                const double* d_sin, int T, double rho_min, double rho_max,
                                int rho_bins, int* d_acc) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_edges) return;
    int2 p = d_edges[idx];
    double x = (double)p.x;
    double y = (double)p.y;
    double rho_span = rho_max - rho_min;
    // loop over theta
    for (int t = 0; t < T; ++t) {
        double rho = x * d_cos[t] + y * d_sin[t];
        // map to bin
        double relative = (rho - rho_min) / rho_span;            // in [0,1]
        int rbin = (int)floor(relative * (rho_bins - 1) + 0.5);  // rounding
        if (rbin < 0 || rbin >= rho_bins) continue;
        int acc_idx = rbin * T + t;
        // atomic increment
        atomicAdd(&d_acc[acc_idx], 1);
    }
}

// Naive CHT kernel (very expensive): each thread handles one edge, loops radii & theta sample,
// computes center (cx,cy) and atomicAdd into acc ((cy*W+cx)*rcount + rindex).
__global__ void cht_vote_kernel(const int2* d_edges, int n_edges, int width, int height, int rmin,
                                int rstep, int rcount, int theta_step_deg, int* d_acc) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_edges) return;
    int2 p = d_edges[idx];
    int x = p.x;
    int y = p.y;
    for (int r = rmin; r < rmin + rcount * rstep; r += rstep) {
        int rindex = (r - rmin) / rstep;
        for (int deg = 0; deg < 360; deg += theta_step_deg) {
            double th = deg * M_PI / 180.0;
            int cx = (int)round(x - r * cos(th));
            int cy = (int)round(y - r * sin(th));
            if (cx < 0 || cx >= width || cy < 0 || cy >= height) continue;
            long long flat = ((long long)cy * width + cx) * (long long)rcount + rindex;
            // atomic add (flat fits into 32-bit index if accumulator size small enough)
            atomicAdd(&d_acc[flat], 1);
        }
    }
}

// ----------------- host helpers -----------------

struct Peak {
    int rbin;
    int tbin;
    int votes;
};
bool peak_cmp(const Peak& a, const Peak& b) { return a.votes > b.votes; }

int main(int argc, char** argv) {
    Params p = parse_args(argc, argv);

    // pick device
    CUDA_CHECK(cudaSetDevice(p.device));

    // load image
    Mat img = imread(p.input_image, IMREAD_COLOR);
    if (img.empty()) {
        cerr << "Cannot open image: " << p.input_image << "\n";
        return -1;
    }
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // edge detection
    Mat edges_img;
    Canny(gray, edges_img, p.canny_low, p.canny_high);
    vector<Point> edges_host;
    edges_host.reserve(100000);
    for (int y = 0; y < edges_img.rows; ++y) {
        const uchar* row = edges_img.ptr<uchar>(y);
        for (int x = 0; x < edges_img.cols; ++x) {
            if (row[x]) edges_host.emplace_back(x, y);
        }
    }
    int n_edges = (int)edges_host.size();
    cout << "Detected edges: " << n_edges << "\n";
    if (n_edges == 0) {
        cerr << "No edges found. Try lowering Canny thresholds.\n";
        return -1;
    }

    if (p.mode == "line") {
        int W = gray.cols, H = gray.rows;
        double diag = sqrt((double)W * W + (double)H * H);
        double rho_min = -diag;
        double rho_max = diag;
        int rho_bins = (int)ceil((rho_max - rho_min) / p.rho_res);
        int T = p.theta_bins;
        cout << "SHT: width=" << W << " height=" << H << " diag=" << diag << "\n";
        cout << "rho_bins=" << rho_bins << " theta_bins=" << T << "\n";

        // build sin/cos table on host
        vector<double> h_cos(T), h_sin(T);
        for (int t = 0; t < T; ++t) {
            double theta = (double)t * M_PI / (double)T;  // [0, pi)
            h_cos[t] = cos(theta);
            h_sin[t] = sin(theta);
        }

        // copy edges to device
        int2* d_edges = nullptr;
        CUDA_CHECK(cudaMalloc(&d_edges, sizeof(int2) * n_edges));
        vector<int2> tmp_edges(n_edges);
        for (int i = 0; i < n_edges; ++i) {
            tmp_edges[i].x = edges_host[i].x;
            tmp_edges[i].y = edges_host[i].y;
        }
        CUDA_CHECK(
            cudaMemcpy(d_edges, tmp_edges.data(), sizeof(int2) * n_edges, cudaMemcpyHostToDevice));

        // copy cos/sin to device
        double* d_cos = nullptr;
        double* d_sin = nullptr;
        CUDA_CHECK(cudaMalloc(&d_cos, sizeof(double) * T));
        CUDA_CHECK(cudaMalloc(&d_sin, sizeof(double) * T));
        CUDA_CHECK(cudaMemcpy(d_cos, h_cos.data(), sizeof(double) * T, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sin, h_sin.data(), sizeof(double) * T, cudaMemcpyHostToDevice));

        // allocate accumulator on device: size = rho_bins * T
        long long acc_size = (long long)rho_bins * (long long)T;
        if (acc_size > (1LL << 31) - 1) {
            cerr << "Accumulator size too large: " << acc_size << " elements\n";
            CUDA_CHECK(cudaFree(d_edges));
            CUDA_CHECK(cudaFree(d_cos));
            CUDA_CHECK(cudaFree(d_sin));
            return -1;
        }
        int* d_acc = nullptr;
        CUDA_CHECK(cudaMalloc(&d_acc, sizeof(int) * (size_t)acc_size));
        CUDA_CHECK(cudaMemset(d_acc, 0, sizeof(int) * (size_t)acc_size));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        // launch kernel: each thread handles one edge and loops theta
        int threads = p.threads_per_block;
        int blocks = (n_edges + threads - 1) / threads;
        cout << "Launching kernel blocks=" << blocks << " threads=" << threads << "\n";
        sht_vote_kernel<<<blocks, threads>>>(d_edges, n_edges, d_cos, d_sin, T, rho_min, rho_max,
                                             rho_bins, d_acc);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        cout << "Runtime (CUDA kernel voting phase only): " << ms << " ms" << endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // copy accumulator back
        vector<int> h_acc((size_t)acc_size);
        CUDA_CHECK(cudaMemcpy(h_acc.data(), d_acc, sizeof(int) * (size_t)acc_size,
                              cudaMemcpyDeviceToHost));

        // find peaks
        vector<Peak> peaks;
        for (int r = 0; r < rho_bins; ++r) {
            for (int t = 0; t < T; ++t) {
                int v = h_acc[r * T + t];
                if (v >= p.peak_threshold) peaks.push_back({r, t, v});
            }
        }
        sort(peaks.begin(), peaks.end(), peak_cmp);
        if ((int)peaks.size() > p.top_k) peaks.resize(p.top_k);
        cout << "Found candidate lines: " << peaks.size() << "\n";

        // draw lines
        Mat out = img.clone();
        for (auto& pr : peaks) {
            double rho = rho_min + (double)pr.rbin * (rho_max - rho_min) / (double)(rho_bins - 1);
            double theta = (double)pr.tbin * M_PI / (double)T;
            double a = cos(theta), b = sin(theta);
            double x0 = a * rho, y0 = b * rho;
            Point pt1, pt2;
            pt1.x = cvRound(x0 + 1000 * (-b));
            pt1.y = cvRound(y0 + 1000 * (a));
            pt2.x = cvRound(x0 - 1000 * (-b));
            pt2.y = cvRound(y0 - 1000 * (a));
            line(out, pt1, pt2, Scalar(0, 0, 255), 2);
        }
        imwrite(p.output_image, out);
        cout << "Saved " << p.output_image << "\n";

        // cleanup
        CUDA_CHECK(cudaFree(d_edges));
        CUDA_CHECK(cudaFree(d_cos));
        CUDA_CHECK(cudaFree(d_sin));
        CUDA_CHECK(cudaFree(d_acc));
    } else if (p.mode == "circle") {
        // Naive CHT on GPU: WARNING — may be extremely slow / memory-heavy for large images
        int W = gray.cols, H = gray.rows;
        int rmin = p.r_min, rmax = p.r_max, rstep = p.r_step;
        int rcount = (rmax - rmin) / rstep + 1;
        long long flatten = (long long)W * (long long)H * (long long)rcount;
        cout << "CHT naive: W=" << W << " H=" << H << " rcount=" << rcount << " flatten=" << flatten
             << "\n";
        if (flatten > (1LL << 28)) {
            cerr << "Accumulator too large for naive CHT. Reduce r-range or image size.\n";
            return -1;
        }

        // copy edges to device
        int2* d_edges = nullptr;
        CUDA_CHECK(cudaMalloc(&d_edges, sizeof(int2) * n_edges));
        vector<int2> tmp_edges(n_edges);
        for (int i = 0; i < n_edges; ++i) {
            tmp_edges[i].x = edges_host[i].x;
            tmp_edges[i].y = edges_host[i].y;
        }
        CUDA_CHECK(
            cudaMemcpy(d_edges, tmp_edges.data(), sizeof(int2) * n_edges, cudaMemcpyHostToDevice));

        // allocate accumulator on device
        int* d_acc = nullptr;
        CUDA_CHECK(cudaMalloc(&d_acc, sizeof(int) * (size_t)flatten));
        CUDA_CHECK(cudaMemset(d_acc, 0, sizeof(int) * (size_t)flatten));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        // launch naive kernel
        int threads = p.threads_per_block;
        int blocks = (n_edges + threads - 1) / threads;
        cout << "Launching CHT kernel blocks=" << blocks << " threads=" << threads << "\n";
        cht_vote_kernel<<<blocks, threads>>>(d_edges, n_edges, W, H, rmin, rstep, rcount,
                                             p.circle_theta_step_deg, d_acc);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        cout << "Runtime (CUDA kernel voting phase only): " << ms << " ms" << endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // copy accumulator back
        vector<int> h_acc((size_t)flatten);
        CUDA_CHECK(
            cudaMemcpy(h_acc.data(), d_acc, sizeof(int) * (size_t)flatten, cudaMemcpyDeviceToHost));

        // find best circle
        int best_votes = 0;
        int best_cx = 0, best_cy = 0, best_r = 0;
        for (int cy = 0; cy < H; ++cy) {
            for (int cx = 0; cx < W; ++cx) {
                long long base = ((long long)cy * W + cx) * (long long)rcount;
                for (int ri = 0; ri < rcount; ++ri) {
                    int v = h_acc[base + ri];
                    if (v > best_votes) {
                        best_votes = v;
                        best_cx = cx;
                        best_cy = cy;
                        best_r = rmin + ri * rstep;
                    }
                }
            }
        }
        cout << "Best circle center=(" << best_cx << "," << best_cy << ") r=" << best_r
             << " votes=" << best_votes << "\n";
        Mat out = img.clone();
        if (best_votes > 0) circle(out, Point(best_cx, best_cy), best_r, Scalar(0, 255, 0), 2);
        imwrite(p.output_image, out);
        cout << "Saved " << p.output_image << "\n";

        CUDA_CHECK(cudaFree(d_edges));
        CUDA_CHECK(cudaFree(d_acc));
    } else {
        cerr << "Unknown mode: " << p.mode << "\n";
        return -1;
    }

    return 0;
}
