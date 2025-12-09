#include <pthread.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;
using namespace chrono;

struct Params {
    string input, output = "output.png", mode = "circle";
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

static inline void atomic_inc_int(int* ptr) { __sync_fetch_and_add(ptr, 1); }

struct Edge {
    int x, y;
    float nx, ny, ori;
};

// ---------------- Circle pthread ----------------
struct ThreadArgCircle2D {
    const vector<Edge>* edges;
    size_t start, end;
    int r, W, H;
    vector<vector<int>>* acc2d;
    double elapsed_ms;
};

void* vote_circle_thread_2d(void* arg) {
    ThreadArgCircle2D* ta = (ThreadArgCircle2D*)arg;
    const vector<Edge>& edges = *ta->edges;

    auto t0 = high_resolution_clock::now();
    for (size_t i = ta->start; i < ta->end; i++) {
        const Edge& e = edges[i];
        int cx = int(round(e.x + ta->r * e.nx));
        int cy = int(round(e.y + ta->r * e.ny));
        if (cx >= 0 && cx < ta->W && cy >= 0 && cy < ta->H) atomic_inc_int(&(*ta->acc2d)[cy][cx]);
        int cx2 = int(round(e.x - ta->r * e.nx));
        int cy2 = int(round(e.y - ta->r * e.ny));
        if (cx2 >= 0 && cx2 < ta->W && cy2 >= 0 && cy2 < ta->H)
            atomic_inc_int(&(*ta->acc2d)[cy2][cx2]);
    }

    auto t1 = high_resolution_clock::now();
    ta->elapsed_ms = duration<double, milli>(t1 - t0).count();

    return nullptr;
}

// ---------------- Line pthread ----------------
struct ThreadArgLine2D {
    const vector<Edge>* edges;
    size_t start, end;
    int theta_step, theta_window, ntheta, nrho, rho_off;
    int W, H;
    vector<vector<int>>* acc2d;
    double elapsed_ms;
};

void* vote_line_thread_2d(void* arg) {
    ThreadArgLine2D* ta = (ThreadArgLine2D*)arg;
    const vector<Edge>& edges = *ta->edges;

    auto t0 = high_resolution_clock::now();
    for (size_t i = ta->start; i < ta->end; i++) {
        const Edge& e = edges[i];
        float base = e.ori + (float)CV_PI / 2.0f;
        int base_deg = (int)round(base * 180.0f / CV_PI) % 180;
        if (base_deg < 0) base_deg += 180;
        for (int d = -ta->theta_window; d <= ta->theta_window; ++d) {
            int deg = base_deg + d;
            if (deg < 0) deg += 180;
            if (deg >= 180) deg -= 180;
            if (deg % ta->theta_step != 0) continue;
            int ti = deg / ta->theta_step;
            float theta = deg * CV_PI / 180.0f;
            float rho = e.x * cos(theta) + e.y * sin(theta);
            int ri = int(round(rho)) + ta->rho_off;
            if (ri >= 0 && ri < ta->nrho) atomic_inc_int(&(*ta->acc2d)[ti][ri]);
        }
    }
    auto t1 = high_resolution_clock::now();
    ta->elapsed_ms = duration<double, milli>(t1 - t0).count();

    return nullptr;
}

// ---------------- Main ----------------
int main(int argc, char** argv) {
    auto t_total_start = high_resolution_clock::now();
    auto t_readimg = high_resolution_clock::now();
    Params p = parse_args(argc, argv);

    Mat img = imread(p.input);
    if (img.empty()) {
        cerr << "Cannot open image\n";
        return -1;
    }

    auto t_readimg_end = high_resolution_clock::now();
    double readimg_ms = duration<double, milli>(t_readimg_end - t_readimg).count();
    cout << "Image read time: " << readimg_ms << " ms\n";

    auto t_cvtcolor = high_resolution_clock::now();
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    auto t_cvtcolor_end = high_resolution_clock::now();
    double cvtcolor_ms = duration<double, milli>(t_cvtcolor_end - t_cvtcolor).count();
    cout << "Color conversion time: " << cvtcolor_ms << " ms\n";

    auto t0 = high_resolution_clock::now();
    Mat edges;
    Canny(gray, edges, p.canny_low, p.canny_high);
    auto t1 = high_resolution_clock::now();
    double canny_ms = duration<double, milli>(t1 - t0).count();
    cout << "Canny: " << canny_ms << " ms\n";

    // Sobel
    auto t_sobel_start = high_resolution_clock::now();

    Mat gx, gy;
    Sobel(gray, gx, CV_32F, 1, 0, 3);
    Sobel(gray, gy, CV_32F, 0, 1, 3);

    auto t_sobel_end = high_resolution_clock::now();
    double sobel_ms = duration<double, milli>(t_sobel_end - t_sobel_start).count();
    cout << "Sobel time: " << sobel_ms << " ms\n";

    auto t_grad_start = high_resolution_clock::now();
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
            edgelist.push_back({x, y, dx / mag, dy / mag, (float)atan2(dy, dx)});
        }
    }
    auto t_grad_end = high_resolution_clock::now();
    double grad_ms = duration<double, milli>(t_grad_end - t_grad_start).count();
    cout << "Grad: " << grad_ms << " ms\n";

    int W = gray.cols, H = gray.rows;

    if (p.mode == "circle") {
        int rmin = p.rmin, rmax = p.rmax, rstep = p.rstep;
        int best_votes = 0, best_cx = 0, best_cy = 0, best_r = 0;

        cout << "Edges: " << edgelist.size() << "\n";

        vector<vector<int>> acc2d(H, vector<int>(W, 0));

        struct timer {
            int cnt;
            float time;
        };
        vector<timer> time_vec(p.threads);
        for (auto& i : time_vec) {
            i.cnt = 0;
            i.time = 0.0f;
        }

        auto t_vote_start = high_resolution_clock::now();
        for (int r = rmin; r <= rmax; r += rstep) {
            for (int y = 0; y < H; y++) fill(acc2d[y].begin(), acc2d[y].end(), 0);

            int T = p.threads;
            vector<pthread_t> tids(T);
            vector<ThreadArgCircle2D> targs(T);
            size_t N = edgelist.size();
            size_t per = (N + T - 1) / T;
            for (int ti = 0; ti < T; ++ti) {
                size_t s = ti * per;
                size_t e = min(N, s + per);
                targs[ti] = {&edgelist, s, e, r, W, H, &acc2d};
                if (s < e) pthread_create(&tids[ti], nullptr, vote_circle_thread_2d, &targs[ti]);
                // else
                //     pthread_create(
                //         &tids[ti], nullptr, [](void*) -> void* { return nullptr; }, nullptr);
            }
            for (int ti = 0; ti < T; ++ti) pthread_join(tids[ti], nullptr);

            // int local_best = 0, lcx = 0, lcy = 0;
            // for (int y = 0; y < H; y++)
            //     for (int x = 0; x < W; x++)
            //         if (acc2d[y][x] > local_best) {
            //             local_best = acc2d[y][x];
            //             lcx = x;
            //             lcy = y;
            //         }
            // if (local_best > best_votes) {
            //     best_votes = local_best;
            //     best_cx = lcx;
            //     best_cy = lcy;
            //     best_r = r;
            // }

            for (int ti = 0; ti < time_vec.size(); ti++) {
                time_vec[ti].cnt++;
                time_vec[ti].time += targs[ti].elapsed_ms;
            }
        }

        for (int t = 0; t < time_vec.size(); t++)
            cout << "Thread " << t << " time: " << time_vec[t].time << " ms\n";

        auto t_vote_end = high_resolution_clock::now();
        cout << "Voting: " << duration<double, milli>(t_vote_end - t_vote_start).count() << " ms\n";

    } else {  // line
        int theta_step = p.theta_step_deg;
        int ntheta = 180 / theta_step;
        float diag = sqrt((float)W * W + (float)H * H);
        int nrho = int(diag) * 2 + 1;
        int rho_off = nrho / 2;

        vector<vector<int>> acc2d(ntheta, vector<int>(nrho, 0));

        int T = p.threads;
        vector<pthread_t> tids(T);
        vector<ThreadArgLine2D> targs(T);
        size_t N = edgelist.size();
        size_t per = (N + T - 1) / T;

        auto t_vote_start = high_resolution_clock::now();

        for (int ti = 0; ti < T; ++ti) {
            size_t s = ti * per;
            size_t e = min(N, s + per);
            targs[ti] = {&edgelist, s, e, theta_step, p.theta_window_deg, ntheta, nrho,
                         rho_off,   W, H, &acc2d};
            if (s < e) pthread_create(&tids[ti], nullptr, vote_line_thread_2d, &targs[ti]);
            // else
            //     pthread_create(&tids[ti], nullptr, [](void*) -> void* { return nullptr; },
            //     nullptr);
        }
        for (int ti = 0; ti < T; ++ti) pthread_join(tids[ti], nullptr);

        auto t_vote_end = high_resolution_clock::now();
        cout << "Voting: " << duration<double, milli>(t_vote_end - t_vote_start).count() << " ms\n";

        // int best_votes = 0, bt = 0, br = 0;
        // for (int ti = 0; ti < ntheta; ++ti)
        //     for (int ri = 0; ri < nrho; ++ri) {
        //         int v = acc2d[ti][ri];
        //         if (v > best_votes) {
        //             best_votes = v;
        //             bt = ti;
        //             br = ri;
        //         }
        //     }
        // float best_theta = bt * theta_step * CV_PI / 180.0f;
        // float best_rho = br - rho_off;
        // cout << "Best line: rho=" << best_rho << " theta(deg)=" << (best_theta * 180.0f / CV_PI)
        //      << " votes=" << best_votes << "\n";
        for (int t = 0; t < p.threads; t++)
            cout << "Thread " << t << " time: " << targs[t].elapsed_ms << " ms\n";
    }

    // for (int t = 0; t < T; t++)
    //     cout << "Thread " << t << " time: " << targs[t].elapsed_ms << " ms\n";

    auto t_total_end = high_resolution_clock::now();
    cout << " Total=" << duration<double, milli>(t_total_end - t_total_start).count() << "\n";
    return 0;
}
