#include <immintrin.h>
#include <pthread.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;
using namespace chrono;

struct Params {
    string input, output = "output_localavx.png", mode = "circle";
    int rmin = 100, rmax = 200, rstep = 1;
    int canny_low = 80, canny_high = 180;
    int theta_step_deg = 1, theta_window_deg = 4;
    int threads = 8;
};

Params parse_args(int argc, char** argv) {
    Params p;
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " input.png [options]\n";
        exit(1);
    }
    p.input = argv[1];
    for (int i = 2; i < argc; i++) {
        string a = argv[i];
        if (a == "-o" && i + 1 < argc)
            p.output = argv[++i];
        else if (a == "--mode" && i + 1 < argc)
            p.mode = argv[++i];
        else if (a == "--rmin" && i + 1 < argc)
            p.rmin = stoi(argv[++i]);
        else if (a == "--rmax" && i + 1 < argc)
            p.rmax = stoi(argv[++i]);
        else if (a == "--rstep" && i + 1 < argc)
            p.rstep = stoi(argv[++i]);
        else if (a == "--tstep" && i + 1 < argc)
            p.theta_step_deg = stoi(argv[++i]);
        else if (a == "--twin" && i + 1 < argc)
            p.theta_window_deg = stoi(argv[++i]);
        else if (a == "-t" && i + 1 < argc)
            p.threads = stoi(argv[++i]);
    }
    if (p.mode == "circle")
        p.output = "out_circle_pthreads_2d.png";
    else if (p.mode == "line")
        p.output = "out_line_pthreads_2d.png";
    return p;
}

struct Edge {
    int x, y;
    float nx, ny;
};

struct ThreadArg {
    const vector<Edge>* edges;
    size_t start, end;
    int W, H;
    int rmin, rmax, rstep;
    int* local_acc;  // aligned memory for AVX2
    double elapsed_ms;
};

void* vote_circle_thread(void* arg) {
    auto t0 = high_resolution_clock::now();
    ThreadArg* ta = (ThreadArg*)arg;
    const vector<Edge>& edges = *ta->edges;
    int W = ta->W, H = ta->H;

    for (size_t i = ta->start; i < ta->end; i++) {
        const Edge& e = edges[i];
        for (int r = ta->rmin; r <= ta->rmax; r += ta->rstep) {
            // 正方向
            int cx = int(round(e.x + r * e.nx));
            int cy = int(round(e.y + r * e.ny));
            if ((unsigned)cx < (unsigned)W && (unsigned)cy < (unsigned)H)
                ta->local_acc[cy * W + cx]++;
            // 反方向
            int cx2 = int(round(e.x - r * e.nx));
            int cy2 = int(round(e.y - r * e.ny));
            if ((unsigned)cx2 < (unsigned)W && (unsigned)cy2 < (unsigned)H)
                ta->local_acc[cy2 * W + cx2]++;
        }
    }

    auto t1 = high_resolution_clock::now();
    ta->elapsed_ms = duration<double, milli>(t1 - t0).count();
    return nullptr;
}

// AVX2 合併 local_acc
void merge_acc_avx(int* acc, const vector<int*>& locals, int N, int T) {
    size_t i = 0;
    for (; i + 8 <= (size_t)N; i += 8) {
        __m256i vsum = _mm256_setzero_si256();
        for (int t = 0; t < T; t++) {
            __m256i v = _mm256_load_si256((__m256i*)(locals[t] + i));
            vsum = _mm256_add_epi32(vsum, v);
        }
        _mm256_store_si256((__m256i*)(acc + i), vsum);
    }
    // tail
    for (; i < (size_t)N; i++) {
        int s = 0;
        for (int t = 0; t < T; t++) s += locals[t][i];
        acc[i] = s;
    }
}

int main(int argc, char** argv) {
    Params p = parse_args(argc, argv);

    Mat img = imread(p.input);
    if (img.empty()) {
        cerr << "Cannot open image\n";
        return -1;
    }

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    Mat edges;
    Canny(gray, edges, p.canny_low, p.canny_high);

    Mat gx, gy;
    Sobel(gray, gx, CV_32F, 1, 0, 3);
    Sobel(gray, gy, CV_32F, 0, 1, 3);

    vector<Edge> edgelist;
    for (int y = 0; y < edges.rows; y++) {
        const uchar* er = edges.ptr<uchar>(y);
        const float* gxr = gx.ptr<float>(y);
        const float* gyr = gy.ptr<float>(y);
        for (int x = 0; x < edges.cols; x++) {
            if (!er[x]) continue;
            float dx = gxr[x], dy = gyr[x];
            float mag = sqrt(dx * dx + dy * dy);
            if (mag < 1e-6) continue;
            edgelist.push_back({x, y, dx / mag, dy / mag});
        }
    }

    cout << "Edges: " << edgelist.size() << "\n";

    int W = gray.cols, H = gray.rows;
    int N = W * H;

    int T = p.threads;
    vector<int*> local_acc(T);
    for (int t = 0; t < T; t++) local_acc[t] = (int*)_mm_malloc(sizeof(int) * N, 32);

    // init
    for (int t = 0; t < T; t++) fill(local_acc[t], local_acc[t] + N, 0);

    // thread pool
    vector<pthread_t> tids(T);
    vector<ThreadArg> targs(T);
    size_t per = (edgelist.size() + T - 1) / T;

    vector<float> time_vec(p.threads, 0);

    auto t_vote_start = high_resolution_clock::now();
    for (int t = 0; t < T; t++) {
        size_t s = t * per;
        size_t e = min(edgelist.size(), s + per);
        targs[t] = {&edgelist, s, e, W, H, p.rmin, p.rmax, p.rstep, local_acc[t], 0.0};
        if (s < e) pthread_create(&tids[t], nullptr, vote_circle_thread, &targs[t]);
    }

    for (int t = 0; t < T; t++) pthread_join(tids[t], nullptr);


    // merge
    int* acc = (int*)_mm_malloc(sizeof(int) * N, 32);
    merge_acc_avx(acc, local_acc, N, T);

    auto t_vote_end = high_resolution_clock::now();
    auto vote_ms = duration<double, milli>(t_vote_end - t_vote_start).count();
    cout << "Voting: " << vote_ms << " ms\n";
    // find best
    int best_votes = 0, best_cx = 0, best_cy = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            if (acc[y * W + x] > best_votes) {
                best_votes = acc[y * W + x];
                best_cx = x;
                best_cy = y;
            }

    cout << "Best circle center: (" << best_cx << "," << best_cy << ") votes=" << best_votes
         << "\n";

    for (int t = 0; t < T; t++) _mm_free(local_acc[t]);
    _mm_free(acc);

    for (int t = 0; t < T; t++)
        cout << "Thread " << t << " time: " << targs[t].elapsed_ms << " ms\n";

    return 0;
}
