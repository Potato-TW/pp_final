// line_grad_omp.cpp
// Gradient-guided Hough Line Transform + OpenMP
// Compile: g++ line_grad_omp.cpp -O3 -std=c++17 -fopenmp `pkg-config --cflags --libs opencv4` -o line_grad_omp

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace cv;
using namespace std::chrono;

struct Params {
    string input;
    string output = "out_line_omp.png";
    int canny_low = 50;
    int canny_high = 150;
    int threads = 8;
    int theta_step_deg = 1;      // discretization of theta
    int theta_window_deg = 4;    // ± range around gradient-based angle
};

Params parse_args(int argc, char** argv) {
    Params p;
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " input.jpg [options]\n";
        exit(1);
    }
    p.input = argv[1];

    for (int i=2;i<argc;i++){
        string a = argv[i];
        if (a=="-o" && i+1<argc) p.output = argv[++i];
        else if (a=="--canny-low" && i+1<argc) p.canny_low = stoi(argv[++i]);
        else if (a=="--canny-high" && i+1<argc) p.canny_high = stoi(argv[++i]);
        else if (a=="-t" && i+1<argc) p.threads = stoi(argv[++i]);
        else if (a=="--tstep" && i+1<argc) p.theta_step_deg = stoi(argv[++i]);
        else if (a=="--twin" && i+1<argc) p.theta_window_deg = stoi(argv[++i]);
        else cerr << "Unknown arg: " << a << "\n";
    }
    return p;
}

static inline void atomic_inc_int(int* ptr) {
    __sync_fetch_and_add(ptr, 1);
}

int main(int argc, char** argv) {
    Params p = parse_args(argc, argv);
    cout << "Input: " << p.input << " Output: " << p.output << "\n";
    cout << "Threads: " << p.threads << "\n";

    auto t_total_start = high_resolution_clock::now();

    Mat img = imread(p.input);
    if (img.empty()) { cerr << "Cannot open image\n"; return -1; }

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // Canny
    auto t0 = high_resolution_clock::now();
    Mat edges;
    Canny(gray, edges, p.canny_low, p.canny_high);
    auto t1 = high_resolution_clock::now();
    double canny_ms = duration<double,milli>(t1-t0).count();

    // Gradient: orientation = atan2(dy, dx)
    t0 = high_resolution_clock::now();
    Mat gx, gy;
    Sobel(gray, gx, CV_32F, 1, 0, 3);
    Sobel(gray, gy, CV_32F, 0, 1, 3);

    struct Edge { int x,y; float ori; };
    vector<Edge> edgelist;
    edgelist.reserve(2000000);

    for (int y=0; y<edges.rows; y++){
        const uchar* row = edges.ptr<uchar>(y);
        const float* gx_row = gx.ptr<float>(y);
        const float* gy_row = gy.ptr<float>(y);

        for (int x=0; x<edges.cols; x++){
            if (!row[x]) continue;
            float dx = gx_row[x];
            float dy = gy_row[x];
            float ang = atan2(dy, dx);  // gradient direction
            float line_ang = ang + CV_PI/2; // line orientation = grad+90 deg
            edgelist.push_back({x, y, line_ang});
        }
    }
    auto t2 = high_resolution_clock::now();
    double grad_ms = duration<double,milli>(t2-t0).count();

    int W = gray.cols, H = gray.rows;
    float diag = sqrt(W*W + H*H);
    int nrho = int(diag) * 2;  // ρ from [-diag, diag]
    int nrho_offset = nrho/2;

    int theta_step = p.theta_step_deg;
    int ntheta = 180 / theta_step;

    vector<int> acc(nrho * ntheta);
    fill(acc.begin(), acc.end(), 0);

    omp_set_num_threads(p.threads);

    // voting
    auto t_vote_start = high_resolution_clock::now();
    #pragma omp parallel for schedule(dynamic, 10000)
    for (size_t i=0; i<edgelist.size(); ++i) {
        const Edge &e = edgelist[i];

        float base_theta = e.ori;  // predicted line direction
        int base_deg = int(base_theta * 180.0 / CV_PI);
        if (base_deg < 0) base_deg += 180;

        for (int d=-p.theta_window_deg; d<=p.theta_window_deg; d++){
            int deg = base_deg + d;
            if (deg < 0) deg += 180;
            if (deg >= 180) deg -= 180;

            if (deg % theta_step != 0) continue;

            int ti = deg / theta_step;
            float theta = deg * CV_PI / 180.0;

            float rho = e.x * cos(theta) + e.y * sin(theta);
            int ri = int(rho) + nrho_offset;

            if (ri >= 0 && ri < nrho) {
                atomic_inc_int(&acc[ti * nrho + ri]);
            }
        }
    }
    auto t_vote_end = high_resolution_clock::now();
    double vote_ms = duration<double,milli>(t_vote_end - t_vote_start).count();

    // find best line
    int best_votes = 0;
    int best_ti = 0, best_ri = 0;
    for (int ti=0; ti<ntheta; ti++){
        for (int ri=0; ri<nrho; ri++){
            int v = acc[ti*nrho + ri];
            if (v > best_votes){
                best_votes = v;
                best_ti = ti;
                best_ri = ri;
            }
        }
    }

    float best_theta = best_ti * theta_step * CV_PI / 180.0;
    float best_rho = best_ri - nrho_offset;

    cout << "Best line: rho=" << best_rho << " theta=" << best_theta
         << " votes=" << best_votes << "\n";

    // draw line
    Mat out = img.clone();
    double a = cos(best_theta), b = sin(best_theta);
    double x0 = a * best_rho, y0 = b * best_rho;
    Point p1(cvRound(x0 + 2000*(-b)), cvRound(y0 + 2000*( a)));
    Point p2(cvRound(x0 - 2000*(-b)), cvRound(y0 - 2000*( a)));
    line(out, p1, p2, Scalar(0,0,255), 3);

    imwrite(p.output, out);

    auto t_total_end = high_resolution_clock::now();
    double total_ms = duration<double,milli>(t_total_end - t_total_start).count();

    cout << "Timing (ms): Canny="<<canny_ms<<" Gradient="<<grad_ms<<" Vote="<<vote_ms
         <<" Total="<<total_ms<<"\n";

    return 0;
}
