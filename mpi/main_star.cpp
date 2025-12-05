// hough_mpi.cpp
// MPI gradient-guided line & circle Hough
// Compile:
// mpicxx hough_mpi.cpp -O3 -std=c++17 `pkg-config --cflags --libs opencv4` -o hough_mpi
//
// Run:
// mpirun -np 4 ./hough_mpi input.jpg --mode circle --rmin 20 --rmax 200 --rstep 2 -o
// out_mpi_circle.png

#include <mpi.h>

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
    int rmin = 20, rmax = 120, rstep = 1;
    int canny_low = 50, canny_high = 150;
    int theta_step_deg = 1, theta_window_deg = 4;
};

Params parse_args(int argc, char** argv, int rank) {
    Params p;
    if (rank == 0) {
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
        }
    }

    if (p.mode == "circle")
        p.output = "out_circle_mpi.png";
    else if (p.mode == "line")
        p.output = "out_line_mpi.png";
    return p;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    Params p = parse_args(argc, argv, rank);
    // rank0 loads image and builds edgelist
    vector<int> dims(2, 0);
    vector<unsigned char> edges_buf;
    int W = 0, H = 0;
    vector<float>
        gradbuf;  // store gx,gy per pixel? but we will build sparse edgelist then broadcast
    struct ERec {
        int x, y;
        float nx, ny, ori;
    };
    vector<ERec> edgelist;
    if (rank == 0) {
        Mat img = imread(p.input);
        if (img.empty()) {
            cerr << "Cannot open image\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);
        W = gray.cols;
        H = gray.rows;
        Mat edges;
        Canny(gray, edges, p.canny_low, p.canny_high);
        Mat gx, gy;
        Sobel(gray, gx, CV_32F, 1, 0, 3);
        Sobel(gray, gy, CV_32F, 0, 1, 3);
        for (int y = 0; y < H; y++) {
            const uchar* er = edges.ptr<uchar>(y);
            const float* gxr = gx.ptr<float>(y);
            const float* gyr = gy.ptr<float>(y);
            for (int x = 0; x < W; x++) {
                if (!er[x]) continue;
                float dx = gxr[x], dy = gyr[x];
                float mag = sqrt(dx * dx + dy * dy);
                if (mag < 1e-6) continue;
                edgelist.push_back({x, y, dx / mag, dy / mag, (float)atan2(dy, dx)});
            }
        }
        cout << "Rank0 edges: " << edgelist.size() << "\n";
    }
    // broadcast W,H and edgelist size
    MPI_Bcast(&W, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&H, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int N = edgelist.size();
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // serialize edgelist to floats for broadcast
    vector<float> sendbuf;
    if (rank == 0) {
        sendbuf.resize(N * 5);
        for (int i = 0; i < N; i++) {
            sendbuf[i * 5 + 0] = edgelist[i].x;
            sendbuf[i * 5 + 1] = edgelist[i].y;
            sendbuf[i * 5 + 2] = edgelist[i].nx;
            sendbuf[i * 5 + 3] = edgelist[i].ny;
            sendbuf[i * 5 + 4] = edgelist[i].ori;
        }
    }
    if (rank != 0) sendbuf.resize(N * 5);
    MPI_Bcast(sendbuf.data(), N * 5, MPI_FLOAT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        edgelist.resize(N);
        for (int i = 0; i < N; i++) {
            edgelist[i].x = int(sendbuf[i * 5 + 0]);
            edgelist[i].y = int(sendbuf[i * 5 + 1]);
            edgelist[i].nx = sendbuf[i * 5 + 2];
            edgelist[i].ny = sendbuf[i * 5 + 3];
            edgelist[i].ori = sendbuf[i * 5 + 4];
        }
    }
    // now every rank has edgelist
    // perform either circle or line: each rank computes local accumulator and reduce
    if (p.mode == "circle") {
        int rmin = p.rmin, rmax = p.rmax, rstep = p.rstep;
        int W2 = W, H2 = H;
        int best_votes_global = 0, best_cx_global = 0, best_cy_global = 0, best_r_global = 0;
        for (int r = rmin; r <= rmax; r += rstep) {
            // each rank compute local acc W*H
            vector<int> local((size_t)W2 * H2, 0);
            // scatter edges among ranks by simple striding to balance
            for (int i = rank; i < N; i += world) {
                auto& e = edgelist[i];
                int cx = int(round(e.x + r * e.nx));
                int cy = int(round(e.y + r * e.ny));
                if (cx >= 0 && cx < W2 && cy >= 0 && cy < H2) local[cy * W2 + cx]++;
                int cx2 = int(round(e.x - r * e.nx)), cy2 = int(round(e.y - r * e.ny));
                if (cx2 >= 0 && cx2 < W2 && cy2 >= 0 && cy2 < H2) local[cy2 * W2 + cx2]++;
            }
            // reduce to global accumulator (sum)
            vector<int> global((size_t)W2 * H2, 0);
            MPI_Reduce(local.data(), global.data(), W2 * H2, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                // find best for this radius
                int best_votes = 0, best_cx = 0, best_cy = 0;
                for (size_t i = 0; i < global.size(); ++i) {
                    int v = global[i];
                    if (v > best_votes) {
                        best_votes = v;
                        best_cx = i % W2;
                        best_cy = i / W2;
                    }
                }
                if (best_votes > best_votes_global) {
                    best_votes_global = best_votes;
                    best_cx_global = best_cx;
                    best_cy_global = best_cy;
                    best_r_global = r;
                }
            }
        }
        if (rank == 0) {
            cout << "Best circle: cx=" << best_cx_global << " cy=" << best_cy_global
                 << " r=" << best_r_global << " votes=" << best_votes_global << "\n";
            Mat img = imread(p.input);
            Mat out = img.clone();
            if (best_votes_global > 0)
                circle(out, Point(best_cx_global, best_cy_global), best_r_global, Scalar(0, 255, 0),
                       3);
            imwrite(p.output, out);
        }
    } else {
        int theta_step = p.theta_step_deg;
        int ntheta = 180 / theta_step;
        float diag = sqrt((float)W * W + (float)H * H);
        int nrho = int(diag) * 2 + 1;
        int rho_off = nrho / 2;
        // each rank compute local acc (rho x theta)
        vector<int> local((size_t)nrho * ntheta, 0);
        for (int i = rank; i < N; i += world) {
            auto& e = edgelist[i];
            float base = e.ori + (float)CV_PI / 2.0f;
            int base_deg = (int)round(base * 180.0f / CV_PI) % 180;
            if (base_deg < 0) base_deg += 180;
            int twin = p.theta_window_deg;
            for (int d = -twin; d <= twin; ++d) {
                int deg = base_deg + d;
                if (deg < 0) deg += 180;
                if (deg >= 180) deg -= 180;
                if (deg % theta_step != 0) continue;
                int ti = deg / theta_step;
                float theta = deg * CV_PI / 180.0f;
                float rho = e.x * cos(theta) + e.y * sin(theta);
                int ri = int(round(rho)) + rho_off;
                if (ri >= 0 && ri < nrho) local[ti * nrho + ri]++;
            }
        }
        vector<int> global((size_t)nrho * ntheta, 0);
        MPI_Reduce(local.data(), global.data(), nrho * ntheta, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            int best_votes = 0, bt = 0, br = 0;
            for (int ti = 0; ti < ntheta; ++ti)
                for (int ri = 0; ri < nrho; ++ri) {
                    int v = global[ti * nrho + ri];
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
            Mat img = imread(p.input);
            Mat out = img.clone();
            double a = cos(best_theta), b = sin(best_theta);
            double x0 = a * best_rho, y0 = b * best_rho;
            Point p1(cvRound(x0 + 2000 * (-b)), cvRound(y0 + 2000 * (a)));
            Point p2(cvRound(x0 - 2000 * (-b)), cvRound(y0 - 2000 * (a)));
            line(out, p1, p2, Scalar(0, 0, 255), 3);
            imwrite(p.output, out);
        }
    }

    MPI_Finalize();
    return 0;
}
