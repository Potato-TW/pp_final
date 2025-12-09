/*
divide rsteps into #threads
1D edge list
2D acc arr
atomic global acc
*/

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
    string input, output = "output_divide.png", mode = "circle";
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
        p.output = "out_circle_pthreads.png";
    else if (p.mode == "line")
        p.output = "out_line_pthreads.png";
    return p;
}

static inline void atomic_inc_int(int* ptr) { __sync_fetch_and_add(ptr, 1); }

struct Edge {
    int x, y;
    float nx, ny, ori;
};

// --- Circle multi-radius thread ---
struct ThreadArgCircleR {
    const vector<Edge>* edges;
    int r_start, r_end;  // radius range for this thread
    int W, H;
    vector<vector<int>>* acc_list;  // acc per radius
    double elapsed_ms;
};

void* vote_circle_r_thread(void* arg) {
    ThreadArgCircleR* ta = (ThreadArgCircleR*)arg;
    const vector<Edge>& edges = *ta->edges;

    auto t0 = high_resolution_clock::now();
    for (int r = ta->r_start; r <= ta->r_end; ++r) {
        vector<int>& acc = (*ta->acc_list)[r - ta->r_start];
        fill(acc.begin(), acc.end(), 0);

        for (const Edge& e : edges) {
            int cx1 = int(round(e.x + r * e.nx));
            int cy1 = int(round(e.y + r * e.ny));
            if (cx1 >= 0 && cx1 < ta->W && cy1 >= 0 && cy1 < ta->H)
                atomic_inc_int(&acc[cy1 * ta->W + cx1]);

            int cx2 = int(round(e.x - r * e.nx));
            int cy2 = int(round(e.y - r * e.ny));
            if (cx2 >= 0 && cx2 < ta->W && cy2 >= 0 && cy2 < ta->H)
                atomic_inc_int(&acc[cy2 * ta->W + cx2]);
        }
    }

    auto t1 = high_resolution_clock::now();
    double worker_ms = duration<double, milli>(t1 - t0).count();
    ta->elapsed_ms = duration<double, milli>(t1 - t0).count();
    return nullptr;
}

// --- Line thread remains the same ---
struct ThreadArgLine {
    const vector<Edge>* edges;
    size_t start, end;
    int theta_step, theta_window, ntheta, nrho, rho_off;
    int W, H;
    vector<int>* acc;
    double elapsed_ms;
};

void* vote_line_thread(void* arg) {
    ThreadArgLine* ta = (ThreadArgLine*)arg;
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
            if (ri >= 0 && ri < ta->nrho) {
                atomic_inc_int(&(*ta->acc)[ti * ta->nrho + ri]);
            }
        }
    }
    auto t1 = high_resolution_clock::now();
    double worker_ms = duration<double, milli>(t1 - t0).count();
    ta->elapsed_ms = duration<double, milli>(t1 - t0).count();
    return nullptr;
}

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

    // --- Canny + gradient ---
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
        cout << "Edges: " << edgelist.size() << "\n";

        int R = (p.rmax - p.rmin) / p.rstep + 1;
        int T = p.threads;
        vector<pthread_t> tids(T);
        vector<ThreadArgCircleR> targs(T);

        int r_per_thread = (R + T - 1) / T;
        vector<vector<vector<int>>> acc_lists(T);  // (#thread, steps for 1 thread, W*H)

        auto t_vote_start = high_resolution_clock::now();
        for (int ti = 0; ti < T; ++ti) {
            int r_s = p.rmin + ti * r_per_thread * p.rstep;
            int r_e = min(p.rmax, r_s + r_per_thread * p.rstep - p.rstep);
            int r_count = (r_e - r_s) / p.rstep + 1;
            acc_lists[ti].resize(r_count);
            for (int ri = 0; ri < r_count; ++ri) acc_lists[ti][ri].resize((size_t)W * H);

            targs[ti] = {&edgelist, r_s, r_e, W, H, &acc_lists[ti]};
            pthread_create(&tids[ti], nullptr, vote_circle_r_thread, &targs[ti]);
        }
        for (int ti = 0; ti < T; ++ti) pthread_join(tids[ti], nullptr);
        auto t_vote_end = high_resolution_clock::now();
        cout << "Voting: " << duration<double, milli>(t_vote_end - t_vote_start).count() << " ms\n";

        // int nums_best = 5;
        // struct Circle {
        //     int cx, cy, r, votes;
        // };
        // vector<Circle> circles;
        // for (int ti = 0; ti < T; ++ti) {
        //     int r_s = targs[ti].r_start;
        //     int r_e = targs[ti].r_end;
        //     int r_count = (r_e - r_s) / p.rstep + 1;
        //     for (int ri = 0; ri < r_count; ++ri) {
        //         vector<int>& acc = acc_lists[ti][ri];
        //         int best_votes = 0, bx = 0, by = 0;
        //         for (int i = 0; i < W * H; ++i) {
        //             if (acc[i] > best_votes) {
        //                 best_votes = acc[i];
        //                 bx = i % W;
        //                 by = i / W;
        //             }
        //         }
        //         if (best_votes > 0) {
        //             if (circles.size() < 40)
        //                 circles.push_back({bx, by, r_s + ri * p.rstep, best_votes});
        //             else
        //                 circles[circles.size() - 1] = {bx, by, r_s + ri * p.rstep, best_votes};
        //         }
        //         sort(circles.begin(), circles.end(),
        //              [](const Circle& a, const Circle& b) { return a.votes > b.votes; });
        //     }
        // }
        // circles.resize(nums_best);

        // Mat out_gray;
        // cvtColor(gray, out_gray, COLOR_GRAY2BGR);
        // for (const auto& c : circles) {
        //     circle(out_gray, Point(c.cx, c.cy), c.r, Scalar(0, 255, 0), 7);  // 粗綠
        //     circle(out_gray, Point(c.cx, c.cy), 4, Scalar(0, 0, 255), -1);   // center red
        // }
        // imwrite(p.output, out_gray);
        // cout << "Saved: " << p.output << "\n";
    } else {
        // --- line code remains the same ---
        int theta_step = p.theta_step_deg;
        int ntheta = 180 / theta_step;
        float diag = sqrt((float)W * W + (float)H * H);
        int nrho = int(diag) * 2 + 1;
        int rho_off = nrho / 2;

        vector<int> acc((size_t)nrho * ntheta, 0);
        int T = p.threads;

        vector<pthread_t> tids(T);
        vector<ThreadArgLine> targs(T);
        size_t N = edgelist.size();
        size_t per = (N + T - 1) / T;

        auto t_vote_start = high_resolution_clock::now();

        for (int ti = 0; ti < T; ++ti) {
            size_t s = ti * per, e = min(N, s + per);
            targs[ti] = {&edgelist, s, e, theta_step, p.theta_window_deg, ntheta, nrho,
                         rho_off,   W, H, &acc};
            pthread_create(&tids[ti], nullptr, vote_line_thread, &targs[ti]);
        }
        for (int ti = 0; ti < T; ++ti) pthread_join(tids[ti], nullptr);

        auto t_vote_end = high_resolution_clock::now();
        cout << "Voting: " << duration<double, milli>(t_vote_end - t_vote_start).count() << " ms\n";

        // int best_lines_count = 20;
        // struct Line {
        //     int ti;  // theta index
        //     int ri;  // rho index
        //     int votes;
        // };
        // vector<Line> lines;
        // for (int ti = 0; ti < ntheta; ++ti)
        //     for (int ri = 0; ri < nrho; ++ri) {
        //         int v = acc[ti * nrho + ri];
        //         if (v > 0)
        //             if (lines.size() < best_lines_count)
        //                 lines.push_back({ti, ri, v});
        //             else
        //                 lines[lines.size() - 1] = {ti, ri, v};
        //         sort(lines.begin(), lines.end(),
        //              [](const Line& a, const Line& b) { return a.votes > b.votes; });
        //     }

        // // 畫出前 40 條線
        // Mat out_gray;
        // cvtColor(gray, out_gray, COLOR_GRAY2BGR);
        // for (const auto& l : lines) {
        //     float theta = l.ti * theta_step * CV_PI / 180.0f;
        //     float rho = l.ri - rho_off;
        //     double a = cos(theta), b = sin(theta);
        //     double x0 = a * rho, y0 = b * rho;
        //     Point p1(cvRound(x0 + 2000 * (-b)), cvRound(y0 + 2000 * (a)));
        //     Point p2(cvRound(x0 - 2000 * (-b)), cvRound(y0 - 2000 * (a)));
        //     line(out_gray, p1, p2, Scalar(0, 0, 255), 3);  // 紅色粗線
        // }

        // imwrite(p.output, out_gray);
        // cout << "Saved: " << p.output << "\n";
    }
    
    for (int t = 0; t < T; t++)
        cout << "Thread " << t << " time: " << targs[t].elapsed_ms << " ms\n";

    auto t_total_end = high_resolution_clock::now();
    cout << "Total=" << duration<double, milli>(t_total_end - t_total_start).count() << "\n";

    return 0;
}
