// circle_grad_omp.cpp
// OpenMP gradient-guided CHT (per-radius 2D accumulator)
// Compile: g++ circle_grad_omp.cpp -O3 -fopenmp `pkg-config --cflags --libs opencv4` -o circle_grad_omp

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <atomic>

#include <omp.h>

using namespace std;
using namespace cv;
using namespace std::chrono;

struct Params {
    string input;
    string output = "out_circle_omp.png";
    int rmin = 20;
    int rmax = 200;
    int rstep = 1;
    int circle_theta_step_deg = 6; // not used here (we use gradient)
    int canny_low = 50;
    int canny_high = 150;
    int threads = 8;
    bool vote_both_dirs = true; // vote both +grad and -grad
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

int main(int argc, char** argv) {
    Params p = parse_args(argc, argv);
    cout << "Input: " << p.input << " Output: " << p.output << "\n";
    cout << "R range: " << p.rmin << " .. " << p.rmax << " step " << p.rstep << "\n";
    cout << "Threads: " << p.threads << " vote_both_dirs=" << p.vote_both_dirs << "\n";

    auto t_total_start = high_resolution_clock::now();

    Mat img = imread(p.input, IMREAD_COLOR);
    if (img.empty()) { cerr << "Cannot open image\n"; return -1; }
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // Canny
    auto t0 = high_resolution_clock::now();
    Mat edges;
    Canny(gray, edges, p.canny_low, p.canny_high);
    auto t1 = high_resolution_clock::now();
    double canny_ms = duration<double,milli>(t1-t0).count();

    // gradient (Sobel)
    t0 = high_resolution_clock::now();
    Mat gx, gy;
    Sobel(gray, gx, CV_32F, 1, 0, 3);
    Sobel(gray, gy, CV_32F, 0, 1, 3);
    // build edge list and normalized gradient for each edge pixel
    struct Edge { int x,y; float nx, ny; };
    vector<Edge> edges_list;
    edges_list.reserve(1000000);
    for (int y=0;y<edges.rows;y++){
        const uchar* row = edges.ptr<uchar>(y);
        const float* gx_row = gx.ptr<float>(y);
        const float* gy_row = gy.ptr<float>(y);
        for (int x=0;x<edges.cols;x++){
            if (!row[x]) continue;
            float dx = gx_row[x];
            float dy = gy_row[x];
            float mag = sqrt(dx*dx + dy*dy);
            if (mag < 1e-6f) {
                // skip very weak gradient; fallback to angle unknown -> skip
                continue;
            }
            // normalize
            float nx = dx / mag;
            float ny = dy / mag;
            // In practice gradient direction may point toward or away from center;
            // we'll vote in both directions if enabled
            edges_list.push_back({x,y,nx,ny});
        }
    }
    auto t2 = high_resolution_clock::now();
    double grad_ms = duration<double,milli>(t2-t0).count();
    cout << "Edges after Canny+grad: " << edges_list.size() << "\n";

    int W = gray.cols, H = gray.rows;
    int rmin = p.rmin, rmax = p.rmax, rstep = p.rstep;
    int rcount = (rmax - rmin) / rstep + 1;

    // prepare final best
    int best_votes = 0;
    int best_cx=0, best_cy=0, best_r=0;

    // set OpenMP threads
    omp_set_num_threads(p.threads);

    auto t_vote_start = high_resolution_clock::now();

    // iterate radii sequentially, maintain per-radius 2D accumulator
    // accumulate into single vector<int> size W*H
    vector<int> acc;
    acc.assign((size_t)W * H, 0);

    // We'll also print progress occasionally
    int printed = 0;
    for (int ri = 0, r = rmin; r <= rmax; r += rstep, ++ri) {
        fill(acc.begin(), acc.end(), 0);

        // parallel over edges, each edge votes for center at (x +- r*nx, y +- r*ny)
        #pragma omp parallel for schedule(dynamic, 1024)
        for (size_t ei=0; ei<edges_list.size(); ++ei) {
            const Edge &e = edges_list[ei];
            // direction 1
            int cx = int(round(e.x + r * e.nx));
            int cy = int(round(e.y + r * e.ny));
            if (cx >= 0 && cx < W && cy >= 0 && cy < H) {
                size_t idx = (size_t)cy * W + cx;
                #pragma omp atomic
                acc[idx]++;
            }
            if (p.vote_both_dirs) {
                int cx2 = int(round(e.x - r * e.nx));
                int cy2 = int(round(e.y - r * e.ny));
                if (cx2 >= 0 && cx2 < W && cy2 >= 0 && cy2 < H) {
                    size_t idx2 = (size_t)cy2 * W + cx2;
                    #pragma omp atomic
                    acc[idx2]++;
                }
            }
        }

        // find best in this radius
        int local_best_votes = 0;
        int local_cx = 0, local_cy = 0;
        // single-threaded scan
        for (size_t i=0;i<acc.size();++i){
            int v = acc[i];
            if (v > local_best_votes) {
                local_best_votes = v;
                local_cx = (int)(i % W);
                local_cy = (int)(i / W);
            }
        }

        if (local_best_votes > best_votes) {
            best_votes = local_best_votes;
            best_cx = local_cx;
            best_cy = local_cy;
            best_r = r;
        }

        // optional progress print
        if ((ri * 100 / rcount) != printed) {
            printed = ri * 100 / rcount;
            cout << "\rProgress: " << printed << "%  best_votes=" << best_votes << " r=" << best_r << "    " << flush;
        }
    }

    auto t_vote_end = high_resolution_clock::now();
    double vote_ms = duration<double,milli>(t_vote_end - t_vote_start).count();
    cout << "\nVoting total: " << vote_ms << " ms\n";
    cout << "Best circle: cx="<<best_cx<<" cy="<<best_cy<<" r="<<best_r<<" votes="<<best_votes<<"\n";

    // draw on image
    Mat out = img.clone();
    if (best_votes > 0) {
        circle(out, Point(best_cx, best_cy), best_r, Scalar(0,255,0), 3);
        circle(out, Point(best_cx, best_cy), 3, Scalar(0,0,255), -1);
    }

    imwrite(p.output, out);

    auto t_total_end = high_resolution_clock::now();
    double total_ms = duration<double,milli>(t_total_end - t_total_start).count();
    cout << "Timing summary (ms): Canny="<<canny_ms<<" Grad="<<grad_ms<<" Vote="<<vote_ms<<" Total="<<total_ms<<"\n";

    return 0;
}
