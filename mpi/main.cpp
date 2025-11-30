#include <mpi.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;
using namespace cv;

struct Params {
    string input_image;
    string output_image = "out_mpi.png";
    string mode = "line";
    int theta_bins = 180;
    double rho_res = 1.0;
    int canny_low = 50;
    int canny_high = 150;
    int peak_threshold = 120;
    int top_k = 20;
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
    }
    if (p.input_image.empty()) {
        cerr << "Usage: -i input.jpg --mode line ..." << endl;
        exit(1);
    }
    return p;
}

struct Peak {
    int rbin, tbin, votes;
};
bool cmp_peak(const Peak& a, const Peak& b) { return a.votes > b.votes; }

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    Params p = parse_args(argc, argv);

    Mat img, gray, edges;
    vector<Point> edge_list;
    int W = 0, H = 0;

    // ---------- Rank 0: read image & create edge list ----------
    if (rank == 0) {
        img = imread(p.input_image, IMREAD_COLOR);
        if (img.empty()) {
            cerr << "Cannot open image\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        cvtColor(img, gray, COLOR_BGR2GRAY);
        Canny(gray, edges, p.canny_low, p.canny_high);

        for (int y = 0; y < edges.rows; y++)
            for (int x = 0; x < edges.cols; x++)
                if (edges.at<uchar>(y, x)) edge_list.emplace_back(x, y);

        W = img.cols;
        H = img.rows;
        cout << "Edges = " << edge_list.size() << endl;
    }

    // ---------- broadcast resolution & edge count ------------
    int N = edge_list.size();
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&W, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&H, 1, MPI_INT, 0, MPI_COMM_WORLD);

    edge_list.resize(N);
    MPI_Bcast(edge_list.data(), N * sizeof(Point), MPI_BYTE, 0, MPI_COMM_WORLD);

    // ---------- SHT parameters ----------
    double diag = sqrt(W * W + H * H);
    double rho_min = -diag, rho_max = diag;
    int T = p.theta_bins;
    int rho_bins = int(ceil((rho_max - rho_min) / p.rho_res));

    vector<double> cos_table(T), sin_table(T);
    for (int t = 0; t < T; ++t) {
        double theta = t * M_PI / T;
        cos_table[t] = cos(theta);
        sin_table[t] = sin(theta);
    }

    // ---------- local accumulator ----------
    vector<int> local_acc(rho_bins * T, 0);
    vector<int> global_acc(rho_bins * T, 0);

    int begin = rank * N / world;
    int end = (rank + 1) * N / world;

    MPI_Barrier(MPI_COMM_WORLD);
    auto t0 = chrono::high_resolution_clock::now();

    // ---------- voting ----------
    for (int k = begin; k < end; k++) {
        double x = edge_list[k].x;
        double y = edge_list[k].y;
        for (int t = 0; t < T; t++) {
            double rho = x * cos_table[t] + y * sin_table[t];
            int rbin = int((rho - rho_min) / (rho_max - rho_min) * (rho_bins - 1) + 0.5);
            if (rbin >= 0 && rbin < rho_bins) local_acc[rbin * T + t]++;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t1 = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration<double, milli>(t1 - t0).count();

    cout << "Rank " << rank << " voting time = " << elapsed << " ms" << endl;

    // ---------- reduce accumulators ----------
    MPI_Reduce(local_acc.data(), global_acc.data(), rho_bins * T, MPI_INT, MPI_SUM, 0,
               MPI_COMM_WORLD);

    // ---------- rank 0: peak detection + draw ----------
    if (rank == 0) {
        vector<Peak> peaks;
        for (int r = 0; r < rho_bins; r++)
            for (int t = 0; t < T; t++) {
                int v = global_acc[r * T + t];
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
    }

    MPI_Finalize();
    return 0;
}
