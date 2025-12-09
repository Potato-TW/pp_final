#include <omp.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;
using namespace cv;
using namespace chrono;

struct Params {
    string input, output = "output_star.png", mode = "circle";
    int rmin = 100, rmax = 200, rstep = 1;
    int canny_low = 80, canny_high = 180;
    int theta_step_deg = 1, theta_window_deg = 4;
    int threads = 8;
};

Params parse_args(int argc, char** argv) {
    Params p;
    if (argc < 2) {
        cerr << "Usage\n";
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
        p.output = "out_circle_omp.png";
    else if (p.mode == "line")
        p.output = "out_line_omp.png";
    return p;
}

struct Edge {
    int x, y;
    float nx, ny, ori;
};

int main(int argc, char** argv) {
    auto t_total_start = high_resolution_clock::now();
    Params p = parse_args(argc, argv);
    omp_set_num_threads(p.threads);

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
            edgelist.push_back({x, y, dx / mag, dy / mag, atan2(dy, dx)});
        }
    }

    int W = gray.cols, H = gray.rows;

    if (p.mode == "circle") {
        vector<int> acc((size_t)W * H, 0);
        int best_votes = 0, best_cx = 0, best_cy = 0, best_r = 0;

        struct timer {
            int cnt;
            float time;
        };

        vector<timer> thread_times(p.threads);

        for (auto& i : thread_times) {
            i.cnt = 0;
            i.time = 0;
        }

        auto t_vote_start = high_resolution_clock::now();

        for (int r = p.rmin; r <= p.rmax; r += p.rstep) {
            fill(acc.begin(), acc.end(), 0);

#pragma omp parallel
            {
                int tid = omp_get_thread_num();
                auto t0 = high_resolution_clock::now();

#pragma omp for schedule(dynamic, 1024)
                for (size_t i = 0; i < edgelist.size(); i++) {
                    const Edge& e = edgelist[i];
                    int cx1 = int(round(e.x + r * e.nx));
                    int cy1 = int(round(e.y + r * e.ny));
                    if (cx1 >= 0 && cx1 < W && cy1 >= 0 && cy1 < H)
#pragma omp atomic
                        acc[cy1 * W + cx1]++;

                    int cx2 = int(round(e.x - r * e.nx));
                    int cy2 = int(round(e.y - r * e.ny));
                    if (cx2 >= 0 && cx2 < W && cy2 >= 0 && cy2 < H)
#pragma omp atomic
                        acc[cy2 * W + cx2]++;
                }

                auto t1 = high_resolution_clock::now();
                thread_times[tid].time += duration<double, milli>(t1 - t0).count();
                thread_times[tid].cnt++;
            }

            // int local_best = 0, lcx = 0, lcy = 0;
            // for (size_t i = 0; i < acc.size(); i++)
            //     if (acc[i] > local_best) {
            //         local_best = acc[i];
            //         lcx = i % W;
            //         lcy = i / W;
            //     }
            // if (local_best > best_votes) {
            //     best_votes = local_best;
            //     best_cx = lcx;
            //     best_cy = lcy;
            //     best_r = r;
            // }

            // cout << "Radius " << r << " thread times: ";
        }

        auto t_vote_end = high_resolution_clock::now();
        cout << "Voting: " << duration<double, milli>(t_vote_end - t_vote_start).count() << " ms\n";

        for (int i = 0; i < thread_times.size(); ++i) {
            cout << "thread: " << i << " time: " << thread_times[i].time
                 << " ms\n";
        }

        cout << "Best circle center: (" << best_cx << "," << best_cy << ") r=" << best_r
             << " votes=" << best_votes << "\n";

    } else {  // line
        int theta_step = p.theta_step_deg;
        int ntheta = 180 / theta_step;
        float diag = sqrt((float)W * W + (float)H * H);
        int nrho = int(diag) * 2 + 1;
        int rho_off = nrho / 2;

        vector<int> acc((size_t)nrho * ntheta, 0);

        vector<double> thread_times(p.threads, 0.0);

        auto t_vote_start = high_resolution_clock::now();

#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto t0 = high_resolution_clock::now();

#pragma omp for schedule(dynamic, 1024)
            for (size_t i = 0; i < edgelist.size(); i++) {
                const Edge& e = edgelist[i];
                float base_deg = e.ori * 180.0f / CV_PI + 90.0f;
                if (base_deg < 0) base_deg += 180.0f;
                int base_i = int(round(base_deg));

                for (int d = -p.theta_window_deg; d <= p.theta_window_deg; d++) {
                    int deg = (base_i + d + 180) % 180;
                    if (deg % theta_step != 0) continue;
                    int ti = deg / theta_step;
                    float theta = deg * CV_PI / 180.0f;
                    float rho = e.x * cos(theta) + e.y * sin(theta);
                    int ri = int(round(rho)) + rho_off;
                    if (ri >= 0 && ri < nrho)
#pragma omp atomic
                        acc[ti * nrho + ri]++;
                }
            }

            auto t1 = high_resolution_clock::now();
            thread_times[tid] = duration<double, milli>(t1 - t0).count();
        }

        auto t_vote_end = high_resolution_clock::now();
        cout << "Voting: " << duration<double, milli>(t_vote_end - t_vote_start).count() << " ms\n";

        int best_votes = 0, bt = 0, br = 0;
        for (int ti = 0; ti < ntheta; ++ti)
            for (int ri = 0; ri < nrho; ++ri) {
                int v = acc[ti * nrho + ri];
                if (v > best_votes) {
                    best_votes = v;
                    bt = ti;
                    br = ri;
                }
            }
        float best_theta = bt * theta_step * CV_PI / 180.0f;
        float best_rho = br - rho_off;
        cout << "Best line: rho=" << best_rho << " theta(deg)=" << (best_theta * 180.0f / CV_PI)
             << " votes=" << best_votes << "\n";

        for (int i = 0; i < thread_times.size(); ++i) {
            cout << "thread: " << i << " time: " << thread_times[i] << " ms\n";
        }

        cout << "ms\n";
    }

    auto t_total_end = high_resolution_clock::now();
    cout << "Total time: " << duration<double, milli>(t_total_end - t_total_start).count()
         << " ms\n";
    return 0;
}
