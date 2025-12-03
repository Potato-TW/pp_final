// // main.cpp
// // Compile: see Makefile below
// #include <pthread.h>

// #include <atomic>
// #include <chrono>
// #include <cmath>
// #include <cstring>
// #include <iostream>
// #include <opencv2/opencv.hpp>
// #include <sstream>
// #include <string>
// #include <vector>

// using namespace cv;
// using namespace std;

// // -------- Config / structs --------
// struct Params {
//     string input_image;
//     string output_image = "out_pthread.png";
//     int num_threads = 8;
//     int chunk_size = 1024;  // edges per chunk
//     // SHT params
//     double rho_res = 1.0;  // pixels
//     int theta_bins = 180;
//     // CHT params
//     int r_min = 10;
//     int r_max = 200;
//     int r_step = 1;
//     string mode = "line";  // "line" or "circle"
//     int canny_low = 50;
//     int canny_high = 150;
//     int peak_threshold = 100;  // for line accumulator
//     int top_k = 20;            // output top-k lines
// };

// struct WorkerArgs {
//     int id;
//     const vector<Point>* edges;
//     atomic<int>* global_idx;
//     Params* p;
//     // For SHT
//     int rho_bins;
//     double rho_max;
//     double rho_min;
//     double theta_step;
//     vector<int>* local_acc;  // flattened rho_bins * theta_bins
//     // For CHT
//     int width;
//     int height;
//     // pointer to per-thread circ accumulator (optional)
//     vector<int>* local_circ;  // flattened (cx*cy*rad)
// };

// // Helper: parse args (very simple)
// Params parse_args(int argc, char** argv) {
//     Params p;
//     for (int i = 1; i < argc; i++) {
//         string a = argv[i];
//         if (a == "-i" && i + 1 < argc)
//             p.input_image = argv[++i];
//         else if (a == "-o" && i + 1 < argc)
//             p.output_image = argv[++i];
//         else if (a == "-t" && i + 1 < argc)
//             p.num_threads = stoi(argv[++i]);
//         else if (a == "-c" && i + 1 < argc)
//             p.chunk_size = stoi(argv[++i]);
//         else if (a == "--mode" && i + 1 < argc)
//             p.mode = argv[++i];
//         else if (a == "--rho" && i + 1 < argc)
//             p.rho_res = stod(argv[++i]);
//         else if (a == "--theta" && i + 1 < argc)
//             p.theta_bins = stoi(argv[++i]);
//         else if (a == "--rmin" && i + 1 < argc)
//             p.r_min = stoi(argv[++i]);
//         else if (a == "--rmax" && i + 1 < argc)
//             p.r_max = stoi(argv[++i]);
//         else if (a == "--canny-low" && i + 1 < argc)
//             p.canny_low = stoi(argv[++i]);
//         else if (a == "--canny-high" && i + 1 < argc)
//             p.canny_high = stoi(argv[++i]);
//         else if (a == "--peak" && i + 1 < argc)
//             p.peak_threshold = stoi(argv[++i]);
//         else if (a == "--topk" && i + 1 < argc)
//             p.top_k = stoi(argv[++i]);
//         else {
//             cerr << "Unknown arg: " << a << endl;
//         }
//     }
//     if (p.input_image.empty()) {
//         cerr << "Usage: -i input.jpg [-o out.png] [-t threads] [--mode line|circle] ...\n";
//         exit(1);
//     }
//     return p;
// }

// // -------- Worker functions --------
// // SHT worker: processes chunks of edges, writes into local_acc
// void* sht_worker(void* a) {
//     WorkerArgs* wa = (WorkerArgs*)a;
//     const vector<Point>& edges = *wa->edges;
//     atomic<int>& idx = *wa->global_idx;
//     Params* p = wa->p;
//     int rho_bins = wa->rho_bins;
//     double rho_min = wa->rho_min;
//     double rho_step = wa->theta_step * 0 + wa->rho_max * 0;  // dummy to silence warning
//     // compute cos/sin table for theta
//     int T = p->theta_bins;
//     vector<double> cos_t(T), sin_t(T);
//     double theta_step = (M_PI) / T;
//     for (int t = 0; t < T; t++) {
//         double th = t * theta_step;
//         cos_t[t] = cos(th);
//         sin_t[t] = sin(th);
//     }
//     // local accumulator pointer
//     vector<int>& local = *wa->local_acc;
//     while (true) {
//         int start = idx.fetch_add(wa->p->chunk_size);
//         if (start >= (int)edges.size()) break;
//         int end = min((int)edges.size(), start + wa->p->chunk_size);
//         for (int ei = start; ei < end; ++ei) {
//             const Point& pt = edges[ei];
//             double x = pt.x;
//             double y = pt.y;
//             for (int t = 0; t < T; t++) {
//                 double rho = x * cos_t[t] + y * sin_t[t];
//                 int rbin = int(round((rho - wa->rho_min) / wa->rho_max *
//                                      (rho_bins - 1)));  // WRONG placeholder
//                 // The above line is placeholder; compute properly below
//             }
//         }
//     }
//     // The above loop used placeholders to compute faster; redo accurate loop to avoid subtle bugs
//     // We'll implement a second pass with correct rho mapping (the cost is minor compared to
//     // complexity). Reset idx for reuse? No, just return; correct implementation does not need
//     // double-pass. But above attempt had placeholder; to keep correctness, redo properly from start
//     // with correct mapping.

//     // Reset local to zero (safe)
//     std::fill(local.begin(), local.end(), 0);

//     // real loop
//     while (true) {
//         int start = idx.fetch_add(wa->p->chunk_size);
//         if (start >= (int)edges.size()) break;
//         int end = min((int)edges.size(), start + wa->p->chunk_size);
//         for (int ei = start; ei < end; ++ei) {
//             const Point& pt = edges[ei];
//             double x = pt.x;
//             double y = pt.y;
//             for (int t = 0; t < T; t++) {
//                 double rho = x * cos_t[t] + y * sin_t[t];
//                 // map rho in [rho_min, rho_max] to bin [0, rho_bins-1]
//                 int rbin =
//                     int(round((rho - wa->rho_min) / (wa->rho_max - wa->rho_min) * (rho_bins - 1)));
//                 if (rbin < 0 || rbin >= rho_bins) continue;
//                 int idx_flat = rbin * T + t;
//                 local[idx_flat]++;
//             }
//         }
//     }

//     pthread_exit(nullptr);
//     return nullptr;
// }

// // CHT worker: naive circle accumulator (center x,y and radius)
// void* cht_worker(void* a) {
//     WorkerArgs* wa = (WorkerArgs*)a;
//     const vector<Point>& edges = *wa->edges;
//     atomic<int>& idx = *wa->global_idx;
//     Params* p = wa->p;
//     int width = wa->width;
//     int height = wa->height;
//     int rmin = p->r_min;
//     int rmax = p->r_max;
//     int rcount = (rmax - rmin) / p->r_step + 1;
//     vector<int>& local = *wa->local_circ;  // indexed as ((cy*width)+cx)*rcount + r_index

//     while (true) {
//         int start = idx.fetch_add(wa->p->chunk_size);
//         if (start >= (int)edges.size()) break;
//         int end = min((int)edges.size(), start + wa->p->chunk_size);
//         for (int ei = start; ei < end; ++ei) {
//             const Point& pt = edges[ei];
//             int x = pt.x;
//             int y = pt.y;
//             // for each possible center (cx,cy) in some stride? naive is O(W*H*R) per edge -> too
//             // slow. Instead iterate radii and compute candidate centers using circle paramization:
//             // cx = x - r*cos(theta), cy = y - r*sin(theta) for theta in 0..360 steps.
//             // For simplicity we'll iterate theta in 0..359 with step 6 degrees to reduce cost.
//             for (int r = rmin; r <= rmax; r += p->r_step) {
//                 int rindex = (r - rmin) / p->r_step;
//                 // sample theta coarsely
//                 for (int deg = 0; deg < 360; deg += 6) {
//                     double th = deg * M_PI / 180.0;
//                     int cx = int(round(x - r * cos(th)));
//                     int cy = int(round(y - r * sin(th)));
//                     if (cx < 0 || cx >= width || cy < 0 || cy >= height) continue;
//                     int flat = ((cy * width) + cx) * rcount + rindex;
//                     local[flat]++;
//                 }
//             }
//         }
//     }

//     pthread_exit(nullptr);
//     return nullptr;
// }

// // -------- Utilities: build edge list with Canny --------
// vector<Point> build_edge_list(const Mat& gray, int canny_low, int canny_high) {
//     Mat edges;
//     Canny(gray, edges, canny_low, canny_high);
//     vector<Point> edge_list;
//     for (int y = 0; y < edges.rows; y++) {
//         const uchar* row = edges.ptr<uchar>(y);
//         for (int x = 0; x < edges.cols; x++) {
//             if (row[x]) edge_list.emplace_back(x, y);
//         }
//     }
//     return edge_list;
// }

// // -------- Peak detection for SHT --------
// struct LineRecord {
//     int rbin;
//     int tbin;
//     int votes;
// };
// bool cmp_line(const LineRecord& a, const LineRecord& b) { return a.votes > b.votes; }

// // -------- Main --------
// int main(int argc, char** argv) {
//     Params p = parse_args(argc, argv);
//     cout << "Input: " << p.input_image << " mode=" << p.mode << " threads=" << p.num_threads
//          << "\n";
//     Mat img = imread(p.input_image, IMREAD_COLOR);
//     if (img.empty()) {
//         cerr << "Cannot open image: " << p.input_image << "\n";
//         return -1;
//     }
//     Mat gray;
//     cvtColor(img, gray, COLOR_BGR2GRAY);

//     // Edge detection
//     auto edges = build_edge_list(gray, p.canny_low, p.canny_high);
//     cout << "Detected edges: " << edges.size() << "\n";
//     if (edges.empty()) {
//         cerr << "No edges found. Try lowering Canny thresholds.\n";
//         return -1;
//     }

//     if (p.mode == "line") {
//         // SHT setup
//         int width = gray.cols, height = gray.rows;
//         double diag = sqrt(width * width + height * height);
//         double rho_min = -diag;
//         double rho_max = diag;
//         int rho_bins = int(ceil((rho_max - rho_min) / p.rho_res));
//         int T = p.theta_bins;
//         cout << "SHT: rho_bins=" << rho_bins << " theta_bins=" << T << "\n";

//         // per-thread local accumulators
//         int threads = p.num_threads;
//         vector<vector<int>> local_accs(threads);
//         for (int i = 0; i < threads; i++) local_accs[i].assign(rho_bins * T, 0);

//         // prepare WorkerArgs
//         vector<WorkerArgs> wargs(threads);
//         atomic<int> global_idx(0);
//         pthread_t* tids = new pthread_t[threads];

//         for (int i = 0; i < threads; i++) {
//             wargs[i].id = i;
//             wargs[i].edges = &edges;
//             wargs[i].global_idx = &global_idx;
//             wargs[i].p = &p;
//             wargs[i].rho_bins = rho_bins;
//             wargs[i].rho_min = rho_min;
//             wargs[i].rho_max = rho_max;
//             wargs[i].theta_step = (M_PI) / T;
//             wargs[i].local_acc = &local_accs[i];
//         }

//         auto t_start = std::chrono::high_resolution_clock::now();

//         // create threads
//         for (int i = 0; i < threads; i++) {
//             int rc = pthread_create(&tids[i], nullptr, sht_worker, &wargs[i]);
//             if (rc) {
//                 cerr << "pthread_create failed\n";
//                 return -1;
//             }
//         }
//         // wait
//         for (int i = 0; i < threads; i++) pthread_join(tids[i], nullptr);

//         auto t_end = std::chrono::high_resolution_clock::now();
//         double elapsed = std::chrono::duration<double, std::milli>(t_end - t_start).count();
//         cout << "Runtime (Pthreads voting phase only): " << elapsed << " ms" << endl;

//         // merge local_accs into global (serial merge)
//         vector<int> global_acc(rho_bins * T, 0);
//         for (int i = 0; i < threads; i++) {
//             auto& la = local_accs[i];
//             for (size_t k = 0; k < la.size(); k++) global_acc[k] += la[k];
//         }

//         // find peaks
//         vector<LineRecord> recs;
//         for (int r = 0; r < rho_bins; r++) {
//             for (int t = 0; t < T; t++) {
//                 int v = global_acc[r * T + t];
//                 if (v >= p.peak_threshold) recs.push_back({r, t, v});
//             }
//         }
//         sort(recs.begin(), recs.end(), cmp_line);
//         if (recs.size() > (size_t)p.top_k) recs.resize(p.top_k);
//         cout << "Found " << recs.size() << " candidate lines.\n";

//         // Draw top lines
//         Mat out = img.clone();
//         for (auto& lr : recs) {
//             double rho = rho_min + (double)lr.rbin * (rho_max - rho_min) / (rho_bins - 1);
//             double theta = lr.tbin * (M_PI) / T;
//             // line in image: rho = x*cos + y*sin
//             // compute two points for drawing
//             double a = cos(theta), b = sin(theta);
//             double x0 = a * rho, y0 = b * rho;
//             Point pt1, pt2;
//             pt1.x = cvRound(x0 + 1000 * (-b));
//             pt1.y = cvRound(y0 + 1000 * (a));
//             pt2.x = cvRound(x0 - 1000 * (-b));
//             pt2.y = cvRound(y0 - 1000 * (a));
//             line(out, pt1, pt2, Scalar(0, 0, 255), 2);
//         }
//         imwrite(p.output_image, out);
//         cout << "Output saved to " << p.output_image << "\n";
//         delete[] tids;
//     } else if (p.mode == "circle") {
//         // CHT naive implementation (note: expensive)
//         int width = gray.cols, height = gray.rows;
//         int rmin = p.r_min, rmax = p.r_max;
//         int rcount = (rmax - rmin) / p.r_step + 1;
//         cout << "CHT: width=" << width << " height=" << height << " rcount=" << rcount << "\n";

//         int threads = p.num_threads;
//         vector<vector<int>> local_circ(threads);
//         // flattened size = width * height * rcount  => could be huge. Warn if too big.
//         long long flatten_size = (long long)width * height * rcount;
//         if (flatten_size > (1LL << 28)) {
//             cerr << "CHT accumulator too large (" << flatten_size
//                  << "). Reduce r range or image size.\n";
//             return -1;
//         }
//         for (int i = 0; i < threads; i++) local_circ[i].assign((size_t)flatten_size, 0);

//         vector<WorkerArgs> wargs(threads);
//         atomic<int> global_idx(0);
//         pthread_t* tids = new pthread_t[threads];

//         for (int i = 0; i < threads; i++) {
//             wargs[i].id = i;
//             wargs[i].edges = &edges;
//             wargs[i].global_idx = &global_idx;
//             wargs[i].p = &p;
//             wargs[i].width = width;
//             wargs[i].height = height;
//             wargs[i].local_circ = &local_circ[i];
//         }

//         auto t_start = std::chrono::high_resolution_clock::now();

//         for (int i = 0; i < threads; i++) {
//             int rc = pthread_create(&tids[i], nullptr, cht_worker, &wargs[i]);
//             if (rc) {
//                 cerr << "pthread_create failed\n";
//                 return -1;
//             }
//         }
//         for (int i = 0; i < threads; i++) pthread_join(tids[i], nullptr);

//         auto t_end = std::chrono::high_resolution_clock::now();
//         double elapsed = std::chrono::duration<double, std::milli>(t_end - t_start).count();
//         cout << "Runtime (Pthreads voting phase only): " << elapsed << " ms" << endl;

//         // merge
//         vector<int> global_acc((size_t)flatten_size, 0);
//         for (int i = 0; i < threads; i++) {
//             auto& lc = local_circ[i];
//             for (size_t k = 0; k < lc.size(); k++) global_acc[k] += lc[k];
//         }

//         // find best circle (brute)
//         int best_votes = 0;
//         int best_cx = 0, best_cy = 0, best_r = 0;
//         for (int cy = 0; cy < height; ++cy) {
//             for (int cx = 0; cx < width; ++cx) {
//                 long long base = ((long long)cy * width + cx) * rcount;
//                 for (int ri = 0; ri < rcount; ++ri) {
//                     int v = global_acc[base + ri];
//                     if (v > best_votes) {
//                         best_votes = v;
//                         best_cx = cx;
//                         best_cy = cy;
//                         best_r = rmin + ri * p.r_step;
//                     }
//                 }
//             }
//         }
//         cout << "Best circle: center=(" << best_cx << "," << best_cy << ") r=" << best_r
//              << " votes=" << best_votes << "\n";
//         Mat out = img.clone();
//         circle(out, Point(best_cx, best_cy), best_r, Scalar(0, 255, 0), 2);
//         imwrite(p.output_image, out);
//         cout << "Output saved to " << p.output_image << "\n";
//         delete[] tids;
//     } else {
//         cerr << "Unknown mode: " << p.mode << "\n";
//         return -1;
//     }

//     return 0;
// }

// hough_pthreads_fixed.cpp
// Compile: see Makefile below
#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <atomic>
#include <vector>
#include <cmath>
#include <cstring>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

// ---------- params ----------
struct Params {
    string input_image;
    string output_image = "out_pthreads.png";
    int num_threads = 8;
    int chunk_size = 1024;
    // SHT
    double rho_res = 1.0;
    int theta_bins = 180;
    int peak_threshold = 120;
    int top_k = 20;
    int canny_low = 50;
    int canny_high = 150;
    string mode = "line"; // "line" or "circle"
    // CHT
    int r_min = 20;
    int r_max = 120;
    int r_step = 1;
    int circle_theta_step_deg = 6;
};

Params parse_args(int argc, char** argv) {
    Params p;
    for (int i=1;i<argc;i++){
        string a = argv[i];
        if (a == "-i" && i+1<argc) p.input_image = argv[++i];
        else if (a == "-o" && i+1<argc) p.output_image = argv[++i];
        else if (a == "-t" && i+1<argc) p.num_threads = stoi(argv[++i]);
        else if (a == "-c" && i+1<argc) p.chunk_size = stoi(argv[++i]);
        else if (a == "--mode" && i+1<argc) p.mode = argv[++i];
        else if (a == "--rho" && i+1<argc) p.rho_res = stod(argv[++i]);
        else if (a == "--theta" && i+1<argc) p.theta_bins = stoi(argv[++i]);
        else if (a == "--peak" && i+1<argc) p.peak_threshold = stoi(argv[++i]);
        else if (a == "--topk" && i+1<argc) p.top_k = stoi(argv[++i]);
        else if (a == "--rmin" && i+1<argc) p.r_min = stoi(argv[++i]);
        else if (a == "--rmax" && i+1<argc) p.r_max = stoi(argv[++i]);
        else if (a == "--rstep" && i+1<argc) p.r_step = stoi(argv[++i]);
        else if (a == "--circle-theta-step" && i+1<argc) p.circle_theta_step_deg = stoi(argv[++i]);
        else {
            cerr << "Unknown arg: " << a << endl;
        }
    }
    if (p.input_image.empty()) {
        cerr << "Usage: -i input.jpg [-o out.png] [-t threads] [--mode line|circle] ...\n";
        exit(1);
    }
    return p;
}

// ---------- worker args ----------
struct WorkerArgs {
    int id;
    const vector<Point>* edges;
    atomic<int>* global_idx;
    Params* p;

    // SHT
    int rho_bins;
    double rho_min;
    double rho_max;
    const vector<double>* cos_table;
    const vector<double>* sin_table;
    vector<int>* local_acc; // flattened rho_bins * theta_bins

    // CHT
    int width;
    int height;
    int rcount;
    vector<int>* local_circ; // flattened W * H * rcount
};

// ---------- SHT worker ----------
void* sht_worker(void* arg) {
    WorkerArgs* wa = (WorkerArgs*)arg;
    const vector<Point>& edges = *wa->edges;
    atomic<int>& idx = *wa->global_idx;
    Params* p = wa->p;
    int T = p->theta_bins;
    int rho_bins = wa->rho_bins;
    double rho_min = wa->rho_min;
    double rho_max = wa->rho_max;
    const vector<double>& cos_t = *wa->cos_table;
    const vector<double>& sin_t = *wa->sin_table;
    vector<int>& local = *wa->local_acc;
    int chunk = p->chunk_size;

    while (true) {
        int start = idx.fetch_add(chunk);
        if (start >= (int)edges.size()) break;
        int end = min((int)edges.size(), start + chunk);
        for (int ei = start; ei < end; ++ei) {
            const Point &pt = edges[ei];
            double x = pt.x;
            double y = pt.y;
            for (int t = 0; t < T; ++t) {
                double rho = x * cos_t[t] + y * sin_t[t];
                // map rho to bin using rho_res
                int rbin = int( floor( (rho - rho_min) / (wa->rho_max - wa->rho_min) * (rho_bins - 1) + 0.5 ) );
                // alternatively simpler: int rbin = int(round((rho - rho_min)/p->rho_res));
                // But using rho_bins ensures consistent mapping:
                if (rbin < 0 || rbin >= rho_bins) continue;
                local[rbin * T + t] ++;
            }
        }
    }
    return nullptr;
}

// ---------- CHT worker (naive) ----------
void* cht_worker(void* arg) {
    WorkerArgs* wa = (WorkerArgs*)arg;
    const vector<Point>& edges = *wa->edges;
    atomic<int>& idx = *wa->global_idx;
    Params* p = wa->p;
    int W = wa->width;
    int H = wa->height;
    int rmin = p->r_min;
    int rstep = p->r_step;
    int rcount = wa->rcount;
    vector<int>& local = *wa->local_circ;
    int theta_step = p->circle_theta_step_deg;
    int chunk = p->chunk_size;

    while (true) {
        int start = idx.fetch_add(chunk);
        if (start >= (int)edges.size()) break;
        int end = min((int)edges.size(), start + chunk);
        for (int ei = start; ei < end; ++ei) {
            const Point &pt = edges[ei];
            int x = pt.x, y = pt.y;
            for (int r = rmin, ri = 0; r <= p->r_max; r += rstep, ++ri) {
                for (int deg = 0; deg < 360; deg += theta_step) {
                    double th = deg * M_PI / 180.0;
                    int cx = int(round(x - r * cos(th)));
                    int cy = int(round(y - r * sin(th)));
                    if (cx < 0 || cx >= W || cy < 0 || cy >= H) continue;
                    long long idx_flat = ((long long)cy * W + cx) * (long long)rcount + ri;
                    local[(size_t)idx_flat] ++;
                }
            }
        }
    }
    return nullptr;
}

// ---------- helper: build edge list ----------
vector<Point> build_edge_list(const Mat& gray, int canny_low, int canny_high) {
    Mat edges;
    Canny(gray, edges, canny_low, canny_high);
    vector<Point> el;
    for (int y=0;y<edges.rows;y++){
        const uchar* row = edges.ptr<uchar>(y);
        for (int x=0;x<edges.cols;x++){
            if (row[x]) el.emplace_back(x,y);
        }
    }
    return el;
}

// ---------- main ----------
int main(int argc, char** argv) {
    Params p = parse_args(argc, argv);
    cout << "Input: " << p.input_image << " mode="<<p.mode<<" threads="<<p.num_threads<<"\n";

    Mat img = imread(p.input_image, IMREAD_COLOR);
    if (img.empty()) {
        cerr << "Cannot open image: " << p.input_image << "\n";
        return -1;
    }
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    vector<Point> edges = build_edge_list(gray, p.canny_low, p.canny_high);
    cout << "Detected edges: " << edges.size() << "\n";
    if (edges.empty()) {
        cerr << "No edges found. Try lowering Canny thresholds.\n";
        return -1;
    }

    if (p.mode == "line") {
        int W = gray.cols, H = gray.rows;
        double diag = sqrt((double)W*W + (double)H*H);
        double rho_min = -diag;
        double rho_max = diag;
        int rho_bins = (int)ceil((rho_max - rho_min) / p.rho_res);
        int T = p.theta_bins;
        cout << "SHT: rho_bins="<<rho_bins<<" theta_bins="<<T<<"\n";

        // precompute trig tables (shared read-only)
        vector<double> cos_t(T), sin_t(T);
        for (int t=0;t<T;t++){
            double theta = (double)t * M_PI / (double)T;
            cos_t[t] = cos(theta);
            sin_t[t] = sin(theta);
        }

        // per-thread local accumulators
        int threads = p.num_threads;
        vector<vector<int>> local_accs(threads, vector<int>( (size_t)rho_bins * T, 0 ));

        // prepare worker args
        vector<WorkerArgs> wargs(threads);
        vector<pthread_t> tids(threads);
        atomic<int> global_idx(0);

        for (int i=0;i<threads;i++){
            wargs[i].id = i;
            wargs[i].edges = &edges;
            wargs[i].global_idx = &global_idx;
            wargs[i].p = &p;
            wargs[i].rho_bins = rho_bins;
            wargs[i].rho_min = rho_min;
            wargs[i].rho_max = rho_max;
            wargs[i].cos_table = &cos_t;
            wargs[i].sin_table = &sin_t;
            wargs[i].local_acc = &local_accs[i];
        }

        // timing: voting phase
        auto t_start = chrono::high_resolution_clock::now();
        for (int i=0;i<threads;i++){
            int rc = pthread_create(&tids[i], nullptr, sht_worker, &wargs[i]);
            if (rc) { cerr << "pthread_create failed\n"; return -1; }
        }
        for (int i=0;i<threads;i++) pthread_join(tids[i], nullptr);
        auto t_end = chrono::high_resolution_clock::now();
        double voting_ms = chrono::duration<double, milli>(t_end - t_start).count();
        cout << "Voting (threads) time: " << voting_ms << " ms\n";

        // serial merge
        vector<int> global_acc((size_t)rho_bins * T, 0);
        for (int i=0;i<threads;i++){
            auto &la = local_accs[i];
            for (size_t k=0;k<la.size();k++) global_acc[k] += la[k];
        }

        // peak detection
        struct Peak { int r, t, v; };
        vector<Peak> peaks;
        for (int r=0;r<rho_bins;r++){
            for (int t=0;t<T;t++){
                int v = global_acc[r*T + t];
                if (v >= p.peak_threshold) peaks.push_back({r,t,v});
            }
        }
        sort(peaks.begin(), peaks.end(), [](const Peak& a, const Peak& b){ return a.v > b.v; });
        if (peaks.size() > (size_t)p.top_k) peaks.resize(p.top_k);
        cout << "Found " << peaks.size() << " peaks.\n";

        // draw
        Mat out = img.clone();
        for (auto &pk : peaks){
            double rho = rho_min + (double)pk.r * (rho_max - rho_min) / (double)(rho_bins - 1);
            double theta = (double)pk.t * M_PI / (double)T;
            double a = cos(theta), b = sin(theta);
            double x0 = a*rho, y0 = b*rho;
            Point pt1(cvRound(x0 + 2000 * (-b)), cvRound(y0 + 2000 * a));
            Point pt2(cvRound(x0 - 2000 * (-b)), cvRound(y0 - 2000 * a));
            line(out, pt1, pt2, Scalar(0,0,255), 2, LINE_AA);
        }
        imwrite(p.output_image, out);
        cout << "Saved " << p.output_image << endl;
    }
    else if (p.mode == "circle") {
        int W = gray.cols, H = gray.rows;
        int rmin = p.r_min, rmax = p.r_max, rstep = p.r_step;
        int rcount = (rmax - rmin) / rstep + 1;
        long long flatten_size = 1LL * W * H * rcount;
        cout << "CHT naive: W="<<W<<" H="<<H<<" rcount="<<rcount<<" flatten="<<flatten_size<<"\n";
        if (flatten_size > (1LL<<28)) {
            cerr << "Accumulator too large ("<<flatten_size<<"). Reduce r range or image size.\n";
            return -1;
        }

        int threads = p.num_threads;
        vector<vector<int>> local_circs(threads, vector<int>((size_t)flatten_size, 0));
        vector<WorkerArgs> wargs(threads);
        vector<pthread_t> tids(threads);
        atomic<int> global_idx(0);

        for (int i=0;i<threads;i++){
            wargs[i].id = i;
            wargs[i].edges = &edges;
            wargs[i].global_idx = &global_idx;
            wargs[i].p = &p;
            wargs[i].width = W;
            wargs[i].height = H;
            wargs[i].rcount = rcount;
            wargs[i].local_circ = &local_circs[i];
        }

        auto t_start = chrono::high_resolution_clock::now();
        for (int i=0;i<threads;i++){
            int rc = pthread_create(&tids[i], nullptr, cht_worker, &wargs[i]);
            if (rc) { cerr << "pthread_create failed\n"; return -1; }
        }
        for (int i=0;i<threads;i++) pthread_join(tids[i], nullptr);
        auto t_end = chrono::high_resolution_clock::now();
        double voting_ms = chrono::duration<double, milli>(t_end - t_start).count();
        cout << "Voting (circle, threads) time: " << voting_ms << " ms\n";

        // merge
        vector<int> global_acc((size_t)flatten_size, 0);
        for (int i=0;i<threads;i++){
            auto &lc = local_circs[i];
            for (size_t k=0;k<lc.size();k++) global_acc[k] += lc[k];
        }

        // find best circle
        int best_votes = 0;
        int best_cx=0, best_cy=0, best_r=0;
        for (int cy=0; cy<H; ++cy){
            for (int cx=0; cx<W; ++cx){
                long long base = ((long long)cy * W + cx) * (long long)rcount;
                for (int ri=0; ri<rcount; ++ri){
                    int v = global_acc[base + ri];
                    if (v > best_votes) {
                        best_votes = v;
                        best_cx = cx; best_cy = cy;
                        best_r = rmin + ri * rstep;
                    }
                }
            }
        }
        cout << "Best circle: center=("<<best_cx<<","<<best_cy<<") r="<<best_r<<" votes="<<best_votes<<"\n";
        Mat out = img.clone();
        if (best_votes > 0) circle(out, Point(best_cx,best_cy), best_r, Scalar(0,255,0), 2);
        imwrite(p.output_image, out);
        cout << "Saved " << p.output_image << endl;
    }
    else {
        cerr << "Unknown mode: " << p.mode << endl;
        return -1;
    }

    return 0;
}
