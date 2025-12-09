// hough_allopt.cpp
// Combined: serial2d / pthread (thread-local accum) / openmp (thread-local accum)
// Compile:
// g++ hough_allopt.cpp -O3 -std=c++17 `pkg-config --cflags --libs opencv4` -pthread -fopenmp -o hough_allopt

#include <opencv2/opencv.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <stdexcept>
#include <cstring>
#include <atomic>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <pthread.h>

using namespace std;
using namespace cv;
using namespace std::chrono;

struct Params {
    string input, output = "output.png", mode = "circle";
    string backend = "pthread"; // serial2d | pthread | openmp
    int rmin = 20, rmax = 120, rstep = 1;
    int canny_low = 50, canny_high = 150;
    int theta_step_deg = 1, theta_window_deg = 4;
    int threads = 8;
};

Params parse_args(int argc, char** argv) {
    Params p;
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " input.png [options]\n";
        cerr << "Options: --mode circle|line  --backend serial2d|pthread|openmp  -t N\n";
        exit(1);
    }
    p.input = argv[1];
    for (int i = 2; i < argc; ++i) {
        string a = argv[i];
        if ((a == "-o" || a == "--output") && i + 1 < argc) p.output = argv[++i];
        else if (a == "--mode" && i + 1 < argc) p.mode = argv[++i];
        else if (a == "--backend" && i + 1 < argc) p.backend = argv[++i];
        else if (a == "--rmin" && i + 1 < argc) p.rmin = stoi(argv[++i]);
        else if (a == "--rmax" && i + 1 < argc) p.rmax = stoi(argv[++i]);
        else if (a == "--rstep" && i + 1 < argc) p.rstep = stoi(argv[++i]);
        else if (a == "--tstep" && i + 1 < argc) p.theta_step_deg = stoi(argv[++i]);
        else if (a == "--twin" && i + 1 < argc) p.theta_window_deg = stoi(argv[++i]);
        else if (a == "-t" && i + 1 < argc) p.threads = stoi(argv[++i]);
        else if (a == "--canny-low" && i + 1 < argc) p.canny_low = stoi(argv[++i]);
        else if (a == "--canny-high" && i + 1 < argc) p.canny_high = stoi(argv[++i]);
    }
    return p;
}

struct Edge {
    int x, y;
    float nx, ny, ori;
};

///////////////////////
// Utility functions //
///////////////////////

static inline void safe_reserve_vector(vector<int>& v, size_t n) {
    try {
        v.assign(n, 0);
    } catch (bad_alloc& e) {
        cerr << "Allocation failed for size " << n << ". OOM.\n";
        throw;
    }
}

static inline void safe_reserve_2dvector(vector<int>& v, size_t n) {
    safe_reserve_vector(v, n);
}

///////////////////////////
// Serial 2D accumulator //
///////////////////////////

void run_serial2d_circle(const vector<Edge>& edges, int W, int H, const Params& p, Mat& img) {
    int rmin = p.rmin, rmax = p.rmax, rstep = p.rstep;
    int best_votes = 0, best_cx = 0, best_cy = 0, best_r = 0;
    cout << "Edges: " << edges.size() << " Radii: " << ((rmax - rmin) / rstep + 1) << "\n";

    vector<vector<int>> acc2d(H, vector<int>(W, 0)); // 2D accumulator

    auto t_vote = high_resolution_clock::now();
    for (int r = rmin; r <= rmax; r += rstep) {
        // reset
        for (int y = 0; y < H; ++y) std::fill(acc2d[y].begin(), acc2d[y].end(), 0);

        // vote
        for (const auto &e : edges) {
            int cx = int(round(e.x + r * e.nx));
            int cy = int(round(e.y + r * e.ny));
            if (cx >= 0 && cx < W && cy >= 0 && cy < H) acc2d[cy][cx]++;

            int cx2 = int(round(e.x - r * e.nx));
            int cy2 = int(round(e.y - r * e.ny));
            if (cx2 >= 0 && cx2 < W && cy2 >= 0 && cy2 < H) acc2d[cy2][cx2]++;
        }

        // find best for this r
        int local_best = 0, lcx = 0, lcy = 0;
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                int v = acc2d[y][x];
                if (v > local_best) { local_best = v; lcx = x; lcy = y; }
            }
        }
        if (local_best > best_votes) {
            best_votes = local_best; best_cx = lcx; best_cy = lcy; best_r = r;
        }
    }
    auto t_vote_end = high_resolution_clock::now();
    cout << "Voting total: " << duration<double, milli>(t_vote_end - t_vote).count() << " ms\n";
    cout << "Best circle: cx=" << best_cx << " cy=" << best_cy << " r=" << best_r << " votes=" << best_votes << "\n";

    Mat out = img.clone();
    if (best_votes > 0) {
        circle(out, Point(best_cx, best_cy), best_r, Scalar(0,255,0), 3);
        circle(out, Point(best_cx, best_cy), 3, Scalar(0,0,255), -1);
    }
    imwrite("serial2d_" + p.output, out);
}

///////////////////////////
// pthread local-acc impl//
///////////////////////////

struct PthreadArgCircle {
    const vector<Edge>* edges;
    size_t start, end;
    int r, W, H;
    int tid;
    int num_threads;
    int64_t acc_size;
    int* local_acc_ptr; // pointer to local_acc flattened for this thread (size W*H)
};

static void* pthread_circle_worker(void* arg) {
    PthreadArgCircle* a = (PthreadArgCircle*)arg;
    const vector<Edge>& edges = *a->edges;
    int W = a->W, H = a->H, r = a->r;
    int* acc = a->local_acc_ptr;
    for (size_t i = a->start; i < a->end; ++i) {
        const Edge& e = edges[i];
        int cx = int(round(e.x + r * e.nx));
        int cy = int(round(e.y + r * e.ny));
        if (cx >= 0 && cx < W && cy >= 0 && cy < H) {
            ++acc[cy * W + cx];
        }
        int cx2 = int(round(e.x - r * e.nx));
        int cy2 = int(round(e.y - r * e.ny));
        if (cx2 >= 0 && cx2 < W && cy2 >= 0 && cy2 < H) {
            ++acc[cy2 * W + cx2];
        }
    }
    return nullptr;
}

void run_pthread_circle(const vector<Edge>& edges, int W, int H, const Params& p, Mat& img) {
    int T = max(1, p.threads);
    int rmin = p.rmin, rmax = p.rmax, rstep = p.rstep;
    int best_votes=0, best_cx=0, best_cy=0, best_r=0;
    cout << "Edges: " << edges.size() << " Radii: " << ((rmax - rmin)/rstep + 1) << " Threads: " << T << "\n";

    // allocate a single big buffer for local accumulators: T * W * H ints
    size_t per_acc = (size_t)W * (size_t)H;
    // check memory size
    long double mem_required_bytes = (long double)per_acc * (long double)T * (long double)sizeof(int);
    if (mem_required_bytes > (long double)4e9) { // naive warning if > ~4GB
        cerr << "Warning: total local accumulators size ~ " << (mem_required_bytes / (1024.0*1024.0*1024.0)) << " GB\n";
    }

    // allocate contiguous block for all threads to reduce fragmentation
    vector<int> all_local;
    try {
        all_local.assign(per_acc * (size_t)T, 0);
    } catch (bad_alloc& e) {
        cerr << "Failed to allocate local accumulators. Reduce threads or image size.\n";
        throw;
    }

    vector<int> global_acc(per_acc);

    auto t_vote = high_resolution_clock::now();
    for (int r = rmin; r <= rmax; r += rstep) {
        // zero local buffers for this radius
        std::fill(all_local.begin(), all_local.end(), 0);
        std::fill(global_acc.begin(), global_acc.end(), 0);

        // spawn T pthreads, each thread uses pointer into all_local
        vector<pthread_t> tids(T);
        vector<PthreadArgCircle> args(T);

        size_t N = edges.size();
        size_t per = (N + T - 1) / T;
        for (int ti = 0; ti < T; ++ti) {
            size_t s = ti * per;
            size_t e = std::min(N, s + per);
            args[ti].edges = &edges;
            args[ti].start = s;
            args[ti].end = e;
            args[ti].r = r;
            args[ti].W = W;
            args[ti].H = H;
            args[ti].tid = ti;
            args[ti].num_threads = T;
            args[ti].acc_size = per_acc;
            args[ti].local_acc_ptr = all_local.data() + (size_t)ti * per_acc;
            pthread_create(&tids[ti], nullptr, pthread_circle_worker, &args[ti]);
        }
        for (int ti = 0; ti < T; ++ti) pthread_join(tids[ti], nullptr);

        // reduction: sum up per-thread locals into global_acc
        size_t SZ = per_acc;
        for (size_t idx = 0; idx < SZ; ++idx) {
            int sum = 0;
            size_t base = idx;
            for (int ti = 0; ti < T; ++ti) sum += all_local[(size_t)ti * SZ + idx];
            global_acc[idx] = sum;
        }

        // find best for this r
        int local_best = 0, lcx = 0, lcy = 0;
        for (size_t idx = 0; idx < SZ; ++idx) {
            int v = global_acc[idx];
            if (v > local_best) { local_best = v; lcx = idx % W; lcy = idx / W; }
        }
        if (local_best > best_votes) {
            best_votes = local_best; best_cx = lcx; best_cy = lcy; best_r = r;
        }
    }
    auto t_vote_end = high_resolution_clock::now();
    cout << "Voting total: " << duration<double, milli>(t_vote_end - t_vote).count() << " ms\n";
    cout << "Best circle: cx="<<best_cx<<" cy="<<best_cy<<" r="<<best_r<<" votes="<<best_votes<<"\n";

    Mat out = img.clone();
    if (best_votes > 0) {
        circle(out, Point(best_cx, best_cy), best_r, Scalar(0,255,0), 3);
        circle(out, Point(best_cx, best_cy), 3, Scalar(0,0,255), -1);
    }
    imwrite("pthread_" + p.output, out);
}

/////////////////////////////
// OpenMP local-acc impl //
/////////////////////////////

void run_openmp_circle(const vector<Edge>& edges, int W, int H, const Params& p, Mat& img) {
#ifndef _OPENMP
    cerr << "OpenMP not available at compile time. Recompile with -fopenmp.\n";
    return;
#else
    int T = max(1, p.threads);
    int rmin = p.rmin, rmax = p.rmax, rstep = p.rstep;
    int best_votes=0, best_cx=0, best_cy=0, best_r=0;
    cout << "Edges: " << edges.size() << " Radii: " << ((rmax - rmin)/rstep + 1) << " Threads: " << T << "\n";

    size_t per_acc = (size_t)W * (size_t)H;
    vector<int> global_acc(per_acc);

    auto t_vote = high_resolution_clock::now();
    for (int r = rmin; r <= rmax; r += rstep) {
        std::fill(global_acc.begin(), global_acc.end(), 0);

        // parallel region: each thread has local_acc
        #pragma omp parallel num_threads(T)
        {
            int tid = omp_get_thread_num();
            int nthreads = omp_get_num_threads();
            vector<int> local_acc;
            try {
                local_acc.assign(per_acc, 0);
            } catch (bad_alloc& e) {
                #pragma omp critical
                { cerr << "Failed to allocate local accumulator in OpenMP thread. Reduce threads.\n"; }
                #pragma omp barrier
            }

            // distribute edges by chunk
            #pragma omp for schedule(static)
            for (int i = 0; i < (int)edges.size(); ++i) {
                const Edge& e = edges[i];
                int cx = int(round(e.x + r * e.nx));
                int cy = int(round(e.y + r * e.ny));
                if (cx >= 0 && cx < W && cy >= 0 && cy < H) local_acc[cy * W + cx]++;

                int cx2 = int(round(e.x - r * e.nx));
                int cy2 = int(round(e.y - r * e.ny));
                if (cx2 >= 0 && cx2 < W && cy2 >= 0 && cy2 < H) local_acc[cy2 * W + cx2]++;
            }

            // merge local_acc into global_acc (serial per-thread merge to avoid atomics)
            #pragma omp critical
            {
                for (size_t idx = 0; idx < per_acc; ++idx) {
                    global_acc[idx] += local_acc[idx];
                }
            }
        } // end parallel

        // find best for this r
        int local_best = 0, lcx = 0, lcy = 0;
        for (size_t idx = 0; idx < per_acc; ++idx) {
            int v = global_acc[idx];
            if (v > local_best) { local_best = v; lcx = idx % W; lcy = idx / W; }
        }
        if (local_best > best_votes) {
            best_votes = local_best; best_cx = lcx; best_cy = lcy; best_r = r;
        }
    }
    auto t_vote_end = high_resolution_clock::now();
    cout << "Voting total: " << duration<double, milli>(t_vote_end - t_vote).count() << " ms\n";
    cout << "Best circle: cx="<<best_cx<<" cy="<<best_cy<<" r="<<best_r<<" votes="<<best_votes<<"\n";

    Mat out = img.clone();
    if (best_votes > 0) {
        circle(out, Point(best_cx, best_cy), best_r, Scalar(0,255,0), 3);
        circle(out, Point(best_cx, best_cy), 3, Scalar(0,0,255), -1);
    }
    imwrite("openmp_" + p.output, out);
#endif
}

/////////////////////////////
// Line Hough - local-acc //
/////////////////////////////

// For line, accumulator size is ntheta * nrho (usually much smaller)
void run_pthread_line(const vector<Edge>& edges, int W, int H, const Params& p, Mat& img) {
    int theta_step = p.theta_step_deg;
    int ntheta = 180 / theta_step;
    float diag = sqrt((float)W * W + (float)H * H);
    int nrho = int(diag) * 2 + 1;
    int rho_off = nrho / 2;
    int T = max(1, p.threads);

    size_t per_acc = (size_t)ntheta * (size_t)nrho;
    cout << "Line: ntheta="<<ntheta<<" nrho="<<nrho<<" acc size="<<per_acc<<" threads="<<T<<"\n";

    // allocate per-thread buffers in one block
    vector<int> all_local;
    try { all_local.assign(per_acc * (size_t)T, 0); }
    catch (...) { cerr << "OOM for line local buffers\n"; throw; }

    vector<int> global_acc(per_acc);

    auto t_vote = high_resolution_clock::now();
    size_t N = edges.size();
    for (int ti = 0; ti < T; ++ti) ; // dummy to silence unused

    size_t per = (N + T - 1) / T;
    vector<pthread_t> tids(T);
    struct PthreadArgLine {
        const vector<Edge>* edges;
        size_t start, end;
        int theta_step, theta_window, ntheta, nrho, rho_off;
        int W, H;
        int* local_ptr;
    };

    auto pthread_line_worker = [](void* arg)->void* {
        PthreadArgLine* a = (PthreadArgLine*)arg;
        const vector<Edge>& edges = *a->edges;
        int* acc = a->local_ptr;
        for (size_t i = a->start; i < a->end; ++i) {
            const Edge& e = edges[i];
            float base = e.ori + (float)CV_PI / 2.0f;
            int base_deg = int(round(base * 180.0f / CV_PI)) % 180;
            if (base_deg < 0) base_deg += 180;
            for (int d = -a->theta_window; d <= a->theta_window; ++d) {
                int deg = base_deg + d;
                if (deg < 0) deg += 180;
                if (deg >= 180) deg -= 180;
                if (deg % a->theta_step != 0) continue;
                int ti = deg / a->theta_step;
                float theta = deg * CV_PI / 180.0f;
                float rho = e.x * cos(theta) + e.y * sin(theta);
                int ri = int(round(rho)) + a->rho_off;
                if (ri >= 0 && ri < a->nrho) {
                    ++acc[ti * a->nrho + ri];
                }
            }
        }
        return nullptr;
    };

    vector<PthreadArgLine> args(T);
    for (int t = 0; t < T; ++t) {
        size_t s = t * per;
        size_t e = min(N, s + per);
        args[t].edges = &edges;
        args[t].start = s; args[t].end = e;
        args[t].theta_step = p.theta_step_deg;
        args[t].theta_window = p.theta_window_deg;
        args[t].ntheta = ntheta; args[t].nrho = nrho; args[t].rho_off = rho_off;
        args[t].W = W; args[t].H = H;
        args[t].local_ptr = all_local.data() + (size_t)t * per_acc;
        pthread_create(&tids[t], nullptr, pthread_line_worker, &args[t]);
    }
    for (int t = 0; t < T; ++t) pthread_join(tids[t], nullptr);

    // reduction
    std::fill(global_acc.begin(), global_acc.end(), 0);
    for (size_t idx = 0; idx < per_acc; ++idx) {
        int sum = 0;
        for (int t = 0; t < T; ++t) sum += all_local[(size_t)t * per_acc + idx];
        global_acc[idx] = sum;
    }

    // find best
    int best_votes = 0, bt = 0, br = 0;
    for (int ti = 0; ti < ntheta; ++ti)
        for (int ri = 0; ri < nrho; ++ri) {
            int v = global_acc[ti * nrho + ri];
            if (v > best_votes) { best_votes = v; bt = ti; br = ri; }
        }
    float best_theta = bt * p.theta_step_deg * CV_PI / 180.0f;
    float best_rho = br - rho_off;
    cout << "Best line: rho=" << best_rho << " theta(deg)=" << (best_theta*180.0f/CV_PI) << " votes=" << best_votes << "\n";

    Mat out = img.clone();
    double a = cos(best_theta), b = sin(best_theta);
    double x0 = a * best_rho, y0 = b * best_rho;
    Point p1(cvRound(x0 + 2000 * (-b)), cvRound(y0 + 2000 * (a)));
    Point p2(cvRound(x0 - 2000 * (-b)), cvRound(y0 - 2000 * (a)));
    line(out, p1, p2, Scalar(0,0,255), 3);
    imwrite("pthread_" + p.output, out);
    auto t_vote_end = high_resolution_clock::now();
    cout << "Voting total (line pthread): " << duration<double,milli>(t_vote_end - t_vote).count() << " ms\n";
}

void run_openmp_line(const vector<Edge>& edges, int W, int H, const Params& p, Mat& img) {
#ifndef _OPENMP
    cerr << "OpenMP not available at compile time.\n";
    return;
#else
    int theta_step = p.theta_step_deg;
    int ntheta = 180 / theta_step;
    float diag = sqrt((float)W * W + (float)H * H);
    int nrho = int(diag) * 2 + 1;
    int rho_off = nrho / 2;
    int T = max(1, p.threads);

    size_t per_acc = (size_t)ntheta * (size_t)nrho;
    vector<int> global_acc(per_acc, 0);

    auto t_vote = high_resolution_clock::now();
    #pragma omp parallel num_threads(T)
    {
        vector<int> local_acc(per_acc, 0);
        #pragma omp for schedule(static)
        for (int i = 0; i < (int)edges.size(); ++i) {
            const Edge &e = edges[i];
            float base = e.ori + (float)CV_PI / 2.0f;
            int base_deg = int(round(base * 180.0f / CV_PI)) % 180;
            if (base_deg < 0) base_deg += 180;
            for (int d = -p.theta_window_deg; d <= p.theta_window_deg; ++d) {
                int deg = base_deg + d;
                if (deg < 0) deg += 180;
                if (deg >= 180) deg -= 180;
                if (deg % p.theta_step_deg != 0) continue;
                int ti = deg / p.theta_step_deg;
                float theta = deg * CV_PI / 180.0f;
                float rho = e.x * cos(theta) + e.y * sin(theta);
                int ri = int(round(rho)) + rho_off;
                if (ri >= 0 && ri < nrho) local_acc[ti * nrho + ri]++;
            }
        }

        #pragma omp critical
        {
            for (size_t idx = 0; idx < per_acc; ++idx) global_acc[idx] += local_acc[idx];
        }
    } // end parallel

    int best_votes = 0, bt = 0, br = 0;
    for (int ti = 0; ti < ntheta; ++ti)
        for (int ri = 0; ri < nrho; ++ri) {
            int v = global_acc[ti * nrho + ri];
            if (v > best_votes) { best_votes = v; bt = ti; br = ri; }
        }
    float best_theta = bt * p.theta_step_deg * CV_PI / 180.0f;
    float best_rho = br - rho_off;
    cout << "Best line: rho=" << best_rho << " theta(deg)=" << (best_theta*180.0f/CV_PI) << " votes=" << best_votes << "\n";
    Mat out = img.clone();
    double a = cos(best_theta), b = sin(best_theta);
    double x0 = a * best_rho, y0 = b * best_rho;
    Point p1(cvRound(x0 + 2000 * (-b)), cvRound(y0 + 2000 * (a)));
    Point p2(cvRound(x0 - 2000 * (-b)), cvRound(y0 - 2000 * (a)));
    line(out, p1, p2, Scalar(0,0,255), 3);
    imwrite("openmp_" + p.output, out);
    auto t_vote_end = high_resolution_clock::now();
    cout << "Voting total (line openmp): " << duration<double,milli>(t_vote_end - t_vote).count() << " ms\n";
#endif
}

int main(int argc, char** argv) {
    auto t_total_start = high_resolution_clock::now();
    Params p = parse_args(argc, argv);
    Mat img = imread(p.input);
    if (img.empty()) { cerr << "Cannot open image\n"; return -1; }
    Mat gray; cvtColor(img, gray, COLOR_BGR2GRAY);

    // Canny
    auto t0 = high_resolution_clock::now();
    Mat edges_mat;
    Canny(gray, edges_mat, p.canny_low, p.canny_high);
    auto t1 = high_resolution_clock::now();
    double canny_ms = duration<double,milli>(t1 - t0).count();

    // Sobel gradients
    t0 = high_resolution_clock::now();
    Mat gx, gy;
    Sobel(gray, gx, CV_32F, 1, 0, 3);
    Sobel(gray, gy, CV_32F, 0, 1, 3);

    vector<Edge> edgelist;
    edgelist.reserve(1000000);
    for (int y = 0; y < edges_mat.rows; ++y) {
        const uchar* er = edges_mat.ptr<uchar>(y);
        const float* gxr = gx.ptr<float>(y);
        const float* gyr = gy.ptr<float>(y);
        for (int x = 0; x < edges_mat.cols; ++x) {
            if (!er[x]) continue;
            float dx = gxr[x], dy = gyr[x];
            float mag = sqrt(dx*dx + dy*dy);
            if (mag < 1e-6f) continue;
            edgelist.push_back({x, y, dx/mag, dy/mag, (float)atan2(dy, dx)});
        }
    }
    auto t2 = high_resolution_clock::now();
    double grad_ms = duration<double,milli>(t2 - t0).count();

    int W = gray.cols, H = gray.rows;
    cout << "Mode="<<p.mode<<" Backend="<<p.backend<<" Threads="<<p.threads<<"\n";

    if (p.mode == "circle") {
        if (p.backend == "serial2d") run_serial2d_circle(edgelist, W, H, p, img);
        else if (p.backend == "pthread") run_pthread_circle(edgelist, W, H, p, img);
        else if (p.backend == "openmp") run_openmp_circle(edgelist, W, H, p, img);
        else { cerr << "Unknown backend\n"; return -1; }
    } else { // line
        if (p.backend == "serial2d") {
            // reuse simple serial line from earlier (not shown to keep code concise)
            // but we call the pthread-based line (which uses local acc) to keep consistent
            run_pthread_line(edgelist, W, H, p, img);
        } else if (p.backend == "pthread") run_pthread_line(edgelist, W, H, p, img);
        else if (p.backend == "openmp") run_openmp_line(edgelist, W, H, p, img);
        else { cerr << "Unknown backend\n"; return -1; }
    }

    auto t_total_end = high_resolution_clock::now();
    cout << "Timing summary (ms): Canny="<<canny_ms<<" Grad="<<grad_ms<<" Total="<<duration<double,milli>(t_total_end - t_total_start).count()<<"\n";
    return 0;
}
