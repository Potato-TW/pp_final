// circle_line_serial.cpp
// Serial gradient-guided line & circle Hough (no threading)
// Compile:
// g++ circle_line_serial.cpp -O3 -std=c++17 `pkg-config --cflags --libs opencv4` -o hough_serial

#include <chrono>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;
using namespace cv;
using namespace std::chrono;

struct Params {
    string input, output = "output.png", mode = "circle";
    int rmin = 20, rmax = 120, rstep = 1;
    int canny_low = 50, canny_high = 150;
    int theta_step_deg = 1, theta_window_deg = 4;
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
        else if (a == "--canny-low" && i + 1 < argc)
            p.canny_low = stoi(argv[++i]);
        else if (a == "--canny-high" && i + 1 < argc)
            p.canny_high = stoi(argv[++i]);
        else if (a == "--tstep" && i + 1 < argc)
            p.theta_step_deg = stoi(argv[++i]);
        else if (a == "--twin" && i + 1 < argc)
            p.theta_window_deg = stoi(argv[++i]);
    }

    if (p.mode == "circle")
        p.output = "out_circle_serial.png";
    else if (p.mode == "line")
        p.output = "out_line_serial.png";
    return p;
}

int main(int argc, char** argv) {
    auto t_total = high_resolution_clock::now();
    Params p = parse_args(argc, argv);
    Mat img = imread(p.input);
    if (img.empty()) {
        cerr << "Cannot open image\n";
        return -1;
    }
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // Canny
    auto t0 = high_resolution_clock::now();
    Mat edges;
    Canny(gray, edges, p.canny_low, p.canny_high);
    auto t1 = high_resolution_clock::now();
    double canny_ms = duration<double, milli>(t1 - t0).count();

    // Sobel gradients (float)
    t0 = high_resolution_clock::now();
    Mat gx, gy;
    Sobel(gray, gx, CV_32F, 1, 0, 3);
    Sobel(gray, gy, CV_32F, 0, 1, 3);
    struct Edge {
        int x, y;
        float nx, ny;
        float ori;
    };
    vector<Edge> edgelist;
    edgelist.reserve(1000000);
    for (int y = 0; y < edges.rows; y++) {
        const uchar* er = edges.ptr<uchar>(y);
        const float* gxr = gx.ptr<float>(y);
        const float* gyr = gy.ptr<float>(y);
        for (int x = 0; x < edges.cols; x++) {
            if (!er[x]) continue;
            float dx = gxr[x], dy = gyr[x];
            float mag = sqrt(dx * dx + dy * dy);
            if (mag < 1e-6) continue;
            float nx = dx / mag, ny = dy / mag;
            float ang = atan2(dy, dx);
            edgelist.push_back({x, y, nx, ny, ang});
        }
    }
    auto t2 = high_resolution_clock::now();
    double grad_ms = duration<double, milli>(t2 - t0).count();

    int W = gray.cols, H = gray.rows;
    if (p.mode == "circle") {
        int rmin = p.rmin, rmax = p.rmax, rstep = p.rstep;
        int rcount = (rmax - rmin) / rstep + 1;
        cout << "Edges: " << edgelist.size() << " Radii: " << rcount << "\n";
        int best_votes = 0, best_cx = 0, best_cy = 0, best_r = 0;
        
        auto t_vote = high_resolution_clock::now();
        // per-radius accumulator to save memory
        vector<int> acc((size_t)W * H);
        for (int r = rmin; r <= rmax; r += rstep) {
            fill(acc.begin(), acc.end(), 0);
            // vote
            for (const auto& e : edgelist) {
                int cx = int(round(e.x + r * e.nx));
                int cy = int(round(e.y + r * e.ny));
                if (cx >= 0 && cx < W && cy >= 0 && cy < H) acc[cy * W + cx]++;
                int cx2 = int(round(e.x - r * e.nx));
                int cy2 = int(round(e.y - r * e.ny));
                if (cx2 >= 0 && cx2 < W && cy2 >= 0 && cy2 < H) acc[cy2 * W + cx2]++;
            }
            // scan best
            int local_best = 0, local_cx = 0, local_cy = 0;
            for (size_t i = 0; i < acc.size(); ++i) {
                if (acc[i] > local_best) {
                    local_best = acc[i];
                    local_cx = i % W;
                    local_cy = i / W;
                }
            }
            if (local_best > best_votes) {
                best_votes = local_best;
                best_cx = local_cx;
                best_cy = local_cy;
                best_r = r;
            }
        }
        auto t_vote_end = high_resolution_clock::now();
        double vote_ms = duration<double, milli>(t_vote_end - t_vote).count();
        cout << "Voting total: " << vote_ms << " ms\n";
        cout << "Best circle: cx=" << best_cx << " cy=" << best_cy << " r=" << best_r
             << " votes=" << best_votes << "\n";
        Mat out = img.clone();
        if (best_votes > 0) {
            circle(out, Point(best_cx, best_cy), best_r, Scalar(0, 255, 0), 3);
            circle(out, Point(best_cx, best_cy), 3, Scalar(0, 0, 255), -1);
        }
        imwrite(p.output, out);
    } else {  // line
        int theta_step = p.theta_step_deg;
        int ntheta = 180 / theta_step;
        float diag = sqrt((float)W * W + (float)H * H);
        int nrho = int(diag) * 2 + 1;
        int rho_off = nrho / 2;
        vector<int> acc(nrho * ntheta);
        fill(acc.begin(), acc.end(), 0);
        auto t_vote = high_resolution_clock::now();
        int twin = p.theta_window_deg;
        for (const auto& e : edgelist) {
            float base = e.ori + (float)CV_PI / 2.0f;
            int base_deg = int(base * 180.0f / CV_PI) % 180;
            if (base_deg < 0) base_deg += 180;
            for (int d = -twin; d <= twin; ++d) {
                int deg = base_deg + d;
                if (deg < 0) deg += 180;
                if (deg >= 180) deg -= 180;
                if (deg % theta_step != 0) continue;
                int ti = deg / theta_step;
                float theta = deg * CV_PI / 180.0f;
                float rho = e.x * cos(theta) + e.y * sin(theta);
                int ri = int(round(rho)) + rho_off;
                if (ri >= 0 && ri < nrho) acc[ti * nrho + ri]++;
            }
        }
        auto t_vote_end = high_resolution_clock::now();
        double vote_ms = duration<double, milli>(t_vote_end - t_vote).count();
        cout << "Voting total: " << vote_ms << " ms\n";
        int best_votes = 0, bt = 0, br = 0;
        for (int ti = 0; ti < ntheta; ++ti) {
            for (int ri = 0; ri < nrho; ++ri) {
                int v = acc[ti * nrho + ri];
                if (v > best_votes) {
                    best_votes = v;
                    bt = ti;
                    br = ri;
                }
            }
        }
        float best_theta = bt * theta_step * CV_PI / 180.0f;
        float best_rho = br - rho_off;
        cout << "Best line: rho=" << best_rho << " theta(deg)=" << (best_theta * 180.0f / CV_PI)
             << " votes=" << best_votes << "\n";
        Mat out = img.clone();
        double a = cos(best_theta), b = sin(best_theta);
        double x0 = a * best_rho, y0 = b * best_rho;
        Point p1(cvRound(x0 + 2000 * (-b)), cvRound(y0 + 2000 * (a)));
        Point p2(cvRound(x0 - 2000 * (-b)), cvRound(y0 - 2000 * (a)));
        line(out, p1, p2, Scalar(0, 0, 255), 3);
        imwrite(p.output, out);
    }

    auto t_total_end = high_resolution_clock::now();
    cout << "Timing summary (ms): Canny=" << canny_ms << " Grad=" << grad_ms
         << " Total=" << duration<double, milli>(t_total_end - t_total).count() << "\n";
    return 0;
}
