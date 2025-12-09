/*
Threads divide block rows
use 2D edge list
1D global acc array
no atomic
*/

#include <pthread.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;
using namespace cv;
using namespace chrono;

struct Params {
    string input, output = "output_blockrow.png", mode = "circle";
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
        p.output = "out_circle_rowbloc.png";
    else if (p.mode == "line")
        p.output = "out_line_rowblock.png";
    return p;
}

// ---------------- Edge ----------------
struct Edge {
    int x, y;
    float nx, ny, ori;
};

// ---------------- Thread Arguments ----------------
struct ThreadArgCircle {
    const vector<vector<Edge>>* row_edges;
    int start_row, end_row;
    int r, W, H;
    vector<int>* global_acc;
    double elapsed_ms;
};
struct ThreadArgLine {
    const vector<vector<Edge>>* row_edges;
    int start_row, end_row;
    int theta_step, theta_window, ntheta, nrho, rho_off;
    vector<int>* global_acc;
    double elapsed_ms;
};

// ---------------- Worker Threads ----------------
void* vote_circle_worker(void* arg) {
    ThreadArgCircle* ta = (ThreadArgCircle*)arg;

    auto t0 = high_resolution_clock::now();

    for (int y = ta->start_row; y < ta->end_row; y++) {
        for (const Edge& e : (*ta->row_edges)[y]) {
            int cx = int(round(e.x + ta->r * e.nx));
            int cy = int(round(e.y + ta->r * e.ny));
            if (cx >= 0 && cx < ta->W && cy >= 0 && cy < ta->H)
                (*ta->global_acc)[cy * ta->W + cx]++;
            int cx2 = int(round(e.x - ta->r * e.nx));
            int cy2 = int(round(e.y - ta->r * e.ny));
            if (cx2 >= 0 && cx2 < ta->W && cy2 >= 0 && cy2 < ta->H)
                (*ta->global_acc)[cy2 * ta->W + cx2]++;
        }
    }

    auto t1 = high_resolution_clock::now();
    ta->elapsed_ms = duration<double, milli>(t1 - t0).count();
    return nullptr;
}

void* vote_line_worker(void* arg) {
    ThreadArgLine* ta = (ThreadArgLine*)arg;

    auto t0 = high_resolution_clock::now();

    for (int y = ta->start_row; y < ta->end_row; y++) {
        for (const Edge& e : (*ta->row_edges)[y]) {
            float base = e.ori + (float)CV_PI / 2.0f;
            int base_deg = int(round(base * 180.0 / CV_PI)) % 180;
            if (base_deg < 0) base_deg += 180;
            for (int d = -ta->theta_window; d <= ta->theta_window; d++) {
                int deg = base_deg + d;
                if (deg < 0) deg += 180;
                if (deg >= 180) deg -= 180;
                if (deg % ta->theta_step != 0) continue;
                int ti = deg / ta->theta_step;
                float theta = deg * CV_PI / 180.0f;
                float rho = e.x * cos(theta) + e.y * sin(theta);
                int ri = int(round(rho)) + ta->rho_off;
                if (ri >= 0 && ri < ta->nrho) (*ta->global_acc)[ti * ta->nrho + ri]++;
            }
        }
    }

    auto t1 = high_resolution_clock::now();
    ta->elapsed_ms = duration<double, milli>(t1 - t0).count();

    return nullptr;
}

// ---------------- Main ----------------
int main(int argc, char** argv) {
    auto t_total_start = high_resolution_clock::now();
    auto t_readimg_start = high_resolution_clock::now();

    Params p = parse_args(argc, argv);

    Mat img = imread(p.input);
    if (img.empty()) {
        cerr << "Cannot open image\n";
        return -1;
    }

    auto t_readimg_end = high_resolution_clock::now();
    double readimg_ms = duration<double, milli>(t_readimg_end - t_readimg_start).count();
    cout << "Image read time: " << readimg_ms << " ms\n";

    auto t_cvtcolor = high_resolution_clock::now();

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    auto t_cvtcolor_end = high_resolution_clock::now();
    double cvtcolor_ms = duration<double, milli>(t_cvtcolor_end - t_cvtcolor).count();
    cout << "Color conversion time: " << cvtcolor_ms << " ms\n";

    // Canny
    auto t_canny_start = high_resolution_clock::now();

    Mat edges;
    Canny(gray, edges, p.canny_low, p.canny_high);

    auto t_canny_end = high_resolution_clock::now();
    double canny_ms = duration<double, milli>(t_canny_end - t_canny_start).count();
    cout << "Canny time: " << canny_ms << " ms\n";

    // Sobel
    auto t_sobel_start = high_resolution_clock::now();

    Mat gx, gy;
    Sobel(gray, gx, CV_32F, 1, 0, 3);
    Sobel(gray, gy, CV_32F, 0, 1, 3);

    auto t_sobel_end = high_resolution_clock::now();
    double sobel_ms = duration<double, milli>(t_sobel_end - t_sobel_start).count();
    cout << "Sobel time: " << sobel_ms << " ms\n";

    auto t_grad_start = high_resolution_clock::now();
    int H = gray.rows, W = gray.cols;
    vector<vector<Edge>> row_edges(H);
    for (int y = 0; y < H; y++) {
        const uchar* er = edges.ptr<uchar>(y);
        const float* gxr = gx.ptr<float>(y);
        const float* gyr = gy.ptr<float>(y);
        for (int x = 0; x < W; x++) {
            if (!er[x]) continue;
            float dx = gxr[x], dy = gyr[x];
            float mag = sqrt(dx * dx + dy * dy);
            if (mag < 1e-6) continue;
            row_edges[y].push_back({x, y, dx / mag, dy / mag, (float)atan2(dy, dx)});
        }
    }

    int total_edgesize = 0;
    for (int y = 0; y < H; y++) total_edgesize += (int)row_edges[y].size();

    auto t_grad_end = high_resolution_clock::now();
    double grad_ms = duration<double, milli>(t_grad_end - t_grad_start).count();
    cout << "Grad: " << grad_ms << " ms\n";

    if (p.mode == "circle") {
        vector<int> global_acc((size_t)W * H);
        for (auto& a : global_acc) a = 0;

        int best_votes = 0, best_cx = 0, best_cy = 0, best_r = 0;
        cout << "Edges: " << total_edgesize << "\n";

        struct timer {
            int cnt;
            float time;
        };

        vector<timer> time_vec(p.threads);
        for (auto& i : time_vec) {
            i.cnt = 0;
            i.time = 0.0f;
        }

        auto t_vote = high_resolution_clock::now();
        for (int r = p.rmin; r <= p.rmax; r += p.rstep) {
            for (auto& a : global_acc) a = 0;

            vector<pthread_t> threads(p.threads);
            vector<ThreadArgCircle> targs(p.threads);
            int rows_per_block = (H + p.threads - 1) / p.threads;

            for (int ti = 0; ti < p.threads; ti++) {
                int s = ti * rows_per_block, e = min(H, s + rows_per_block);
                targs[ti] = {&row_edges, s, e, r, W, H, &global_acc};
                pthread_create(&threads[ti], nullptr, vote_circle_worker, &targs[ti]);
            }
            for (int ti = 0; ti < p.threads; ti++) {
                pthread_join(threads[ti], nullptr);
            }

            // // find best circle
            // int local_best = 0, lcx = 0, lcy = 0;
            // for (size_t i = 0; i < global_acc.size(); i++) {
            //     int v = global_acc[i];
            //     if (v > local_best) {
            //         local_best = v;
            //         lcx = i % W;
            //         lcy = i / W;
            //     }
            // }
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

        auto t_vote_end = high_resolution_clock::now();
        cout << "Voting: " << duration<double, milli>(t_vote_end - t_vote).count() << " ms\n";

        for (int t = 0; t < time_vec.size(); t++)
            cout << "Thread " << t << " time: " << time_vec[t].time << " ms\n";

    } else {
        int theta_step = p.theta_step_deg;
        int ntheta = 180 / theta_step;
        float diag = sqrt(float(W * W + H * H));
        int nrho = int(diag) * 2 + 1, rho_off = nrho / 2;

        vector<int> global_acc((size_t)nrho * ntheta);
        for (auto& a : global_acc) a = 0;

        vector<pthread_t> threads(p.threads);
        vector<ThreadArgLine> targs(p.threads);
        int rows_per_block = (H + p.threads - 1) / p.threads;

        auto t_vote = high_resolution_clock::now();

        for (int ti = 0; ti < p.threads; ti++) {
            int s = ti * rows_per_block, e = min(H, s + rows_per_block);
            targs[ti] = {&row_edges, s,    e,       theta_step, p.theta_window_deg,
                         ntheta,     nrho, rho_off, &global_acc};
            pthread_create(&threads[ti], nullptr, vote_line_worker, &targs[ti]);
        }
        for (int ti = 0; ti < p.threads; ti++) pthread_join(threads[ti], nullptr);

        auto t_vote_end = high_resolution_clock::now();
        cout << "Voting total: " << duration<double, milli>(t_vote_end - t_vote).count() << " ms\n";

        // auto t_best_start = high_resolution_clock::now();
        // int best_votes = 0, bt = 0, br = 0;
        // for (int ti = 0; ti < ntheta; ti++)
        //     for (int ri = 0; ri < nrho; ri++) {
        //         int v = global_acc[ti * nrho + ri];
        //         if (v > best_votes) {
        //             best_votes = v;
        //             bt = ti;
        //             br = ri;
        //         }
        //     }
        // auto t_best_end = high_resolution_clock::now();
        // cout << "Best line search time: "
        //      << duration<double, milli>(t_best_end - t_best_start).count() << " ms\n";
        // float best_theta = bt * theta_step * CV_PI / 180.0f;
        // float best_rho = br - rho_off;
        // cout << "Best line: rho=" << best_rho << " theta(deg)=" << (best_theta * 180.0 / CV_PI)
        //      << " votes=" << best_votes << "\n";

        for (int t = 0; t < p.threads; t++)
            cout << "Thread " << t << " time: " << targs[t].elapsed_ms << " ms\n";
    }

    auto t_total_end = high_resolution_clock::now();
    cout << "Total=" << duration<double, milli>(t_total_end - t_total_start).count() << "\n";

    return 0;
}
