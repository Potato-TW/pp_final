// circle_grad_pthread.cpp
// Pthreads gradient-guided CHT (per-radius 2D accumulator, threads created per-radius)
// Compile: g++ circle_grad_pthread.cpp -O3 -std=c++17 `pkg-config --cflags --libs opencv4` -pthread -o circle_grad_pthread

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <pthread.h>

using namespace std;
using namespace cv;
using namespace std::chrono;

struct Params {
    string input;
    string output = "out_circle_pt.png";
    int rmin = 20;
    int rmax = 200;
    int rstep = 1;
    int canny_low = 50;
    int canny_high = 150;
    int threads = 8;
    bool vote_both_dirs = true;
};

Params parse_args(int argc, char** argv) {
    Params p;
    if (argc < 2) { cerr << "Usage: " << argv[0] << " input.jpg [options]\n"; exit(1); }
    p.input = argv[1];
    for (int i=2;i<argc;i++){
        string a = argv[i];
        if (a=="-o" && i+1<argc) p.output = argv[++i];
        else if (a=="--rmin" && i+1<argc) p.rmin = stoi(argv[++i]);
        else if (a=="--rmax" && i+1<argc) p.rmax = stoi(argv[++i]);
        else if (a=="--rstep" && i+1<argc) p.rstep = stoi(argv[++i]);
        else if (a=="--canny-low" && i+1<argc) p.canny_low = stoi(argv[++i]);
        else if (a=="--canny-high" && i+1<argc) p.canny_high = stoi(argv[++i]);
        else if (a=="-t" && i+1<argc) p.threads = stoi(argv[++i]);
        else if (a=="--no-both") p.vote_both_dirs = false;
        else { cerr << "Unknown arg: " << a << "\n"; }
    }
    return p;
}

struct Edge { int x,y; float nx, ny; };
struct ThreadArg {
    int tid;
    const vector<Edge>* edges;
    int start_idx, end_idx;
    int r;
    int W, H;
    bool vote_both;
    vector<int>* acc; // shared global acc
};

static inline void atomic_inc_int(int* ptr) {
    __sync_fetch_and_add(ptr, 1);
}

void* thread_worker(void* arg) {
    ThreadArg* ta = (ThreadArg*)arg;
    const vector<Edge>& edges = *ta->edges;
    for (int i = ta->start_idx; i < ta->end_idx; ++i) {
        const Edge &e = edges[i];
        int cx = int(round(e.x + ta->r * e.nx));
        int cy = int(round(e.y + ta->r * e.ny));
        if (cx >= 0 && cx < ta->W && cy >= 0 && cy < ta->H) {
            size_t idx = (size_t)cy * ta->W + cx;
            atomic_inc_int((&(*ta->acc)[idx]));
        }
        if (ta->vote_both) {
            int cx2 = int(round(e.x - ta->r * e.nx));
            int cy2 = int(round(e.y - ta->r * e.ny));
            if (cx2 >= 0 && cx2 < ta->W && cy2 >= 0 && cy2 < ta->H) {
                size_t idx2 = (size_t)cy2 * ta->W + cx2;
                atomic_inc_int((&(*ta->acc)[idx2]));
            }
        }
    }
    return nullptr;
}

int main(int argc, char** argv) {
    Params p = parse_args(argc, argv);
    cout << "Input: " << p.input << " Output: " << p.output << "\n";
    cout << "R range: " << p.rmin << " .. " << p.rmax << " step " << p.rstep << "\n";
    cout << "Threads: " << p.threads << " vote_both=" << p.vote_both_dirs << "\n";

    auto t_total_start = high_resolution_clock::now();

    Mat img = imread(p.input, IMREAD_COLOR);
    if (img.empty()) { cerr << "Cannot open image\n"; return -1; }
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    auto t0 = high_resolution_clock::now();
    Mat edges;
    Canny(gray, edges, p.canny_low, p.canny_high);
    auto t1 = high_resolution_clock::now();
    double canny_ms = duration<double,milli>(t1-t0).count();

    // gradient
    t0 = high_resolution_clock::now();
    Mat gx, gy;
    Sobel(gray, gx, CV_32F, 1, 0, 3);
    Sobel(gray, gy, CV_32F, 0, 1, 3);
    vector<Edge> edges_list;
    for (int y=0;y<edges.rows;y++){
        const uchar* row = edges.ptr<uchar>(y);
        const float* gx_row = gx.ptr<float>(y);
        const float* gy_row = gy.ptr<float>(y);
        for (int x=0;x<edges.cols;x++){
            if (!row[x]) continue;
            float dx = gx_row[x];
            float dy = gy_row[x];
            float mag = sqrt(dx*dx + dy*dy);
            if (mag < 1e-6f) continue;
            edges_list.push_back({x,y,dx/mag, dy/mag});
        }
    }
    auto t2 = high_resolution_clock::now();
    double grad_ms = duration<double,milli>(t2-t0).count();
    cout << "Edges: " << edges_list.size() << "\n";

    int W = gray.cols, H = gray.rows;
    int rmin = p.rmin, rmax = p.rmax, rstep = p.rstep;

    int best_votes = 0; int best_cx=0,best_cy=0,best_r=0;

    auto t_vote_start = high_resolution_clock::now();

    // For each radius, create threads to process edges chunk.
    vector<pthread_t> tids(p.threads);
    vector<ThreadArg> targs(p.threads);

    for (int r = rmin; r <= rmax; r += rstep) {
        // allocate accumulator for this radius
        vector<int> acc((size_t)W * H);
        fill(acc.begin(), acc.end(), 0);

        // prepare thread args: partition edges into roughly equal chunks
        int N = (int)edges_list.size();
        int per = (N + p.threads - 1) / p.threads;
        for (int ti=0; ti<p.threads; ++ti) {
            int s = ti * per;
            int e = min(N, s + per);
            targs[ti].tid = ti;
            targs[ti].edges = &edges_list;
            targs[ti].start_idx = s;
            targs[ti].end_idx = e;
            targs[ti].r = r;
            targs[ti].W = W;
            targs[ti].H = H;
            targs[ti].vote_both = p.vote_both_dirs;
            targs[ti].acc = &acc;
            if (s >= e) {
                // create idle thread that immediately returns
                pthread_create(&tids[ti], nullptr, [](void* a)->void*{return nullptr;}, nullptr);
            } else {
                pthread_create(&tids[ti], nullptr, thread_worker, &targs[ti]);
            }
        }

        // join
        for (int ti=0; ti<p.threads; ++ti) pthread_join(tids[ti], nullptr);

        // scan acc for local best
        int local_best = 0; int local_cx=0, local_cy=0;
        for (size_t i=0;i<acc.size();++i){
            int v = acc[i];
            if (v > local_best) {
                local_best = v;
                local_cx = (int)(i % W);
                local_cy = (int)(i / W);
            }
        }
        if (local_best > best_votes) {
            best_votes = local_best;
            best_cx = local_cx; best_cy = local_cy; best_r = r;
        }
        // optional progress
        int percent = (int)((double)(r - rmin) / (double)(rmax - rmin) * 100.0);
        cout << "\rr="<<r<<" best_votes="<<best_votes<<" best_r="<<best_r<<" ("<<percent<<"%)" << flush;
    }

    auto t_vote_end = high_resolution_clock::now();
    double vote_ms = duration<double,milli>(t_vote_end - t_vote_start).count();
    cout << "\nVoting total: " << vote_ms << " ms\n";
    cout << "Best circle: cx="<<best_cx<<" cy="<<best_cy<<" r="<<best_r<<" votes="<<best_votes<<"\n";

    Mat out = img.clone();
    if (best_votes > 0) {
        circle(out, Point(best_cx,best_cy), best_r, Scalar(0,255,0), 3);
        circle(out, Point(best_cx,best_cy), 3, Scalar(0,0,255), -1);
    }
    imwrite(p.output, out);

    auto t_total_end = high_resolution_clock::now();
    double total_ms = duration<double,milli>(t_total_end - t_total_start).count();
    cout << "Timing summary (ms): Canny="<<canny_ms<<" Grad="<<grad_ms<<" Vote="<<vote_ms<<" Total="<<total_ms<<"\n";

    return 0;
}
