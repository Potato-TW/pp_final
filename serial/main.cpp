#include <chrono>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;
using namespace cv;

struct Params {
    string input_image;
    string output_image = "out_serial.png";
    string mode = "line";  // "line" or "circle"
    int theta_bins = 180;
    double rho_res = 1.0;
    int canny_low = 50;
    int canny_high = 150;
    int peak_threshold = 120;
    int top_k = 20;

    // circle params
    int r_min = 20;
    int r_max = 120;
    int r_step = 1;
    int circle_theta_step_deg = 6;  // angular sampling
};

Params parse_args(int argc, char** argv) {
    Params p;
    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if (a == "-i")
            p.input_image = argv[++i];
        else if (a == "-o")
            p.output_image = argv[++i];
        else if (a == "--mode")
            p.mode = argv[++i];
        else if (a == "--theta")
            p.theta_bins = stoi(argv[++i]);
        else if (a == "--rho")
            p.rho_res = stod(argv[++i]);
        else if (a == "--canny-low")
            p.canny_low = stoi(argv[++i]);
        else if (a == "--canny-high")
            p.canny_high = stoi(argv[++i]);
        else if (a == "--peak")
            p.peak_threshold = stoi(argv[++i]);
        else if (a == "--topk")
            p.top_k = stoi(argv[++i]);
        else if (a == "--rmin")
            p.r_min = stoi(argv[++i]);
        else if (a == "--rmax")
            p.r_max = stoi(argv[++i]);
        else if (a == "--rstep")
            p.r_step = stoi(argv[++i]);
        else if (a == "--circle-theta-step")
            p.circle_theta_step_deg = stoi(argv[++i]);
    }
    if (p.input_image.empty()) {
        cerr << "Usage: -i input.jpg [--mode line|circle] ..." << endl;
        exit(1);
    }
    return p;
}

struct Peak {
    int rbin, tbin, votes;
};
bool cmp_peak(const Peak& a, const Peak& b) { return a.votes > b.votes; }

int main(int argc, char** argv) {
    Params p = parse_args(argc, argv);

    Mat img = imread(p.input_image, IMREAD_COLOR);
    if (img.empty()) {
        cerr << "Cannot open image\n";
        return -1;
    }

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // Edge detection
    Mat edges;
    Canny(gray, edges, p.canny_low, p.canny_high);

    vector<Point> edge_list;
    for (int y = 0; y < edges.rows; y++)
        for (int x = 0; x < edges.cols; x++)
            if (edges.at<uchar>(y, x)) edge_list.emplace_back(x, y);

    cout << "Edges: " << edge_list.size() << endl;
    if (edge_list.empty()) return 0;

    // -------------------------------------------------------
    // LINE MODE — SHT
    // -------------------------------------------------------
    if (p.mode == "line") {
        int W = img.cols, H = img.rows;
        double diag = sqrt(W * W + H * H);
        double rho_min = -diag, rho_max = diag;
        int T = p.theta_bins;
        int rho_bins = (int)ceil((rho_max - rho_min) / p.rho_res);

        cout << "Line mode: rho_bins=" << rho_bins << " theta_bins=" << T << endl;

        vector<double> cos_table(T), sin_table(T);
        for (int t = 0; t < T; ++t) {
            double theta = t * M_PI / T;
            cos_table[t] = cos(theta);
            sin_table[t] = sin(theta);
        }

        vector<int> acc(rho_bins * T, 0);

        auto t0 = chrono::high_resolution_clock::now();
        for (auto& pt : edge_list) {
            double x = pt.x, y = pt.y;
            for (int t = 0; t < T; ++t) {
                double rho = x * cos_table[t] + y * sin_table[t];
                int rbin = int((rho - rho_min) / (rho_max - rho_min) * (rho_bins - 1) + 0.5);
                if (rbin >= 0 && rbin < rho_bins) acc[rbin * T + t]++;
            }
        }
        auto t1 = chrono::high_resolution_clock::now();
        cout << "Runtime (Serial Line voting only): "
             << chrono::duration<double, milli>(t1 - t0).count() << " ms" << endl;

        vector<Peak> peaks;
        for (int r = 0; r < rho_bins; r++)
            for (int t = 0; t < T; t++) {
                int v = acc[r * T + t];
                if (v >= p.peak_threshold) peaks.push_back({r, t, v});
            }

        sort(peaks.begin(), peaks.end(), cmp_peak);
        if ((int)peaks.size() > p.top_k) peaks.resize(p.top_k);

        Mat out = img.clone();
        for (auto& pk : peaks) {
            double rho = rho_min + pk.rbin * (rho_max - rho_min) / (rho_bins - 1);
            double theta = pk.tbin * M_PI / T;
            double a = cos(theta), b = sin(theta);
            double x0 = a * rho, y0 = b * rho;
            Point pt1(cvRound(x0 + 2000 * (-b)), cvRound(y0 + 2000 * a));
            Point pt2(cvRound(x0 - 2000 * (-b)), cvRound(y0 - 2000 * a));
            line(out, pt1, pt2, Scalar(0, 0, 255), 2);
        }

        imwrite(p.output_image, out);
        cout << "Saved " << p.output_image << endl;
        return 0;
    }

    // -------------------------------------------------------
    // CIRCLE MODE — CHT
    // -------------------------------------------------------
    if (p.mode == "circle") {
        int W = img.cols, H = img.rows;
        int rmin = p.r_min, rmax = p.r_max, rstep = p.r_step;
        int rcount = (rmax - rmin) / rstep + 1;
        long long acc_size = 1LL * W * H * rcount;

        cout << "Circle mode: W=" << W << " H=" << H << " rcount=" << rcount << endl;
        if (acc_size > (1LL << 28)) {
            cerr << "Accumulator too large — reduce r range or image size!" << endl;
            return -1;
        }

        vector<int> acc(acc_size, 0);

        auto t0 = chrono::high_resolution_clock::now();
        for (auto& pt : edge_list) {
            int x = pt.x, y = pt.y;
            for (int r = rmin; r <= rmax; r += rstep) {
                int rindex = (r - rmin) / rstep;
                for (int deg = 0; deg < 360; deg += p.circle_theta_step_deg) {
                    double th = deg * M_PI / 180.0;
                    int cx = int(round(x - r * cos(th)));
                    int cy = int(round(y - r * sin(th)));
                    if (cx < 0 || cx >= W || cy < 0 || cy >= H) continue;
                    long long idx = ((long long)cy * W + cx) * rcount + rindex;
                    acc[idx]++;
                }
            }
        }
        auto t1 = chrono::high_resolution_clock::now();
        cout << "Runtime (Serial Circle voting only): "
             << chrono::duration<double, milli>(t1 - t0).count() << " ms" << endl;

        // find best circle
        int best_votes = 0;
        int best_cx = 0, best_cy = 0, best_r = 0;
        for (int cy = 0; cy < H; cy++)
            for (int cx = 0; cx < W; cx++) {
                long long base = ((long long)cy * W + cx) * rcount;
                for (int ri = 0; ri < rcount; ri++) {
                    int v = acc[base + ri];
                    if (v > best_votes) {
                        best_votes = v;
                        best_cx = cx;
                        best_cy = cy;
                        best_r = rmin + ri * rstep;
                    }
                }
            }

        cout << "Detected circle: center=(" << best_cx << "," << best_cy << ") r=" << best_r
             << " votes=" << best_votes << endl;

        Mat out = img.clone();
        if (best_votes > 0) circle(out, Point(best_cx, best_cy), best_r, Scalar(0, 255, 0), 2);

        imwrite(p.output_image, out);
        cout << "Saved " << p.output_image << endl;
        return 0;
    }

    cerr << "Unknown mode" << endl;
    return 0;
}
