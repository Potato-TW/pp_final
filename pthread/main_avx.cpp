#include <immintrin.h>
#include <pthread.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;
using namespace cv;
using namespace chrono;

// ------------------ Parameters ------------------
struct Params {
    string input, output = "output.png", mode = "circle";
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
        else if (a == "-t" && i + 1 < argc)
            p.threads = stoi(argv[++i]);
    }
    return p;
}

// ------------------ Edge ------------------
struct Edge {
    int x, y;
    float nx, ny;  // only used in circle
};

// ------------------ Thread Args ------------------
struct ThreadArgCircle {
    const vector<Edge>* edges;
    size_t start, end;
    int W, H;
    int rmin, rmax, rstep;
    vector<int>* local_acc;
    double thread_time;
};

// ------------------ Circle vote thread ------------------
void* vote_circle_thread_avx(void* arg) {
    auto t0 = high_resolution_clock::now();
    ThreadArgCircle* ta = (ThreadArgCircle*)arg;
    const auto& edges = *ta->edges;
    int W = ta->W, H = ta->H;
    vector<int>& acc = *ta->local_acc;

    for (size_t i = ta->start; i < ta->end; i += 8) {
        int rem = min((size_t)8, ta->end - i);
        float xs[8] = {0}, ys[8] = {0};
        float nx[8] = {0}, ny[8] = {0};
        for (int k = 0; k < rem; k++) {
            xs[k] = (float)edges[i + k].x;
            ys[k] = (float)edges[i + k].y;
            nx[k] = edges[i + k].nx;
            ny[k] = edges[i + k].ny;
        }

        __m256 vx = _mm256_loadu_ps(xs);
        __m256 vy = _mm256_loadu_ps(ys);
        __m256 vnx = _mm256_loadu_ps(nx);
        __m256 vny = _mm256_loadu_ps(ny);

        for (int r = ta->rmin; r <= ta->rmax; r += ta->rstep) {
            __m256 vr = _mm256_set1_ps((float)r);
            // 正方向
            __m256 vcx = _mm256_add_ps(vx, _mm256_mul_ps(vr, vnx));
            __m256 vcy = _mm256_add_ps(vy, _mm256_mul_ps(vr, vny));
            int icx[8], icy[8];
            _mm256_storeu_si256((__m256i*)icx, _mm256_cvtps_epi32(vcx));
            _mm256_storeu_si256((__m256i*)icy, _mm256_cvtps_epi32(vcy));
            for (int k = 0; k < rem; k++)
                if ((unsigned)icx[k] < (unsigned)W && (unsigned)icy[k] < (unsigned)H)
                    acc[icy[k] * W + icx[k]]++;

            // 反方向
            __m256 vcx2 = _mm256_sub_ps(vx, _mm256_mul_ps(vr, vnx));
            __m256 vcy2 = _mm256_sub_ps(vy, _mm256_mul_ps(vr, vny));
            int icx2[8], icy2[8];
            _mm256_storeu_si256((__m256i*)icx2, _mm256_cvtps_epi32(vcx2));
            _mm256_storeu_si256((__m256i*)icy2, _mm256_cvtps_epi32(vcy2));
            for (int k = 0; k < rem; k++)
                if ((unsigned)icx2[k] < (unsigned)W && (unsigned)icy2[k] < (unsigned)H)
                    acc[icy2[k] * W + icx2[k]]++;
        }
    }
    ta->thread_time = duration<double, milli>(high_resolution_clock::now() - t0).count();
    return nullptr;
}

// ------------------ Thread Arg Line ------------------
struct ThreadArgLine {
    const vector<Edge>* edges;
    size_t start, end;
    int theta_step, theta_window, ntheta, nrho, rho_off;
    int W, H;
    vector<int>* local_acc;  // local accumulator
    double thread_time;
};

// ------------------ Line vote thread AVX2 ------------------
void* vote_line_thread_avx(void* arg) {
    auto t0 = high_resolution_clock::now();
    ThreadArgLine* ta = (ThreadArgLine*)arg;
    const vector<Edge>& edges = *ta->edges;
    int ntheta = ta->ntheta;
    int nrho = ta->nrho;
    int rho_off = ta->rho_off;
    int theta_step = ta->theta_step;
    int theta_window = ta->theta_window;

    vector<int>& acc = *ta->local_acc;

    // 預計算 cos/sin table
    vector<float> cos_table(ntheta), sin_table(ntheta);
    for (int ti = 0; ti < ntheta; ++ti) {
        float theta = ti * theta_step * CV_PI / 180.0f;
        cos_table[ti] = cos(theta);
        sin_table[ti] = sin(theta);
    }

    for (size_t i = ta->start; i < ta->end; i++) {
        const Edge& e = edges[i];
        // 計算大約角度 ± window
        float base_deg = atan2(e.ny, e.nx) * 180.0f / CV_PI + 90.0f;
        if (base_deg < 0) base_deg += 180.0f;
        int tstart = max(0, (int)(base_deg - theta_window));
        int tend = min(179, (int)(base_deg + theta_window));

        for (int deg = tstart; deg <= tend; deg += theta_step) {
            int ti = deg / theta_step;
            float cos_t = cos_table[ti];
            float sin_t = sin_table[ti];
            float rho_f = e.x * cos_t + e.y * sin_t;
            int ri = (int)round(rho_f) + rho_off;
            if (ri >= 0 && ri < nrho) {
                acc[ti * nrho + ri]++;
            }
        }
    }

    ta->thread_time = duration<double, milli>(high_resolution_clock::now() - t0).count();
    return nullptr;
}

// ------------------ Merge AVX2 for line ------------------
void merge_acc_line_avx(int* acc_global, const vector<vector<int>>& locals, int ntheta, int nrho,
                        int T) {
    size_t N = ntheta * nrho;
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        __m256i vsum = _mm256_setzero_si256();
        for (int t = 0; t < T; t++) {
            __m256i v = _mm256_loadu_si256((__m256i*)(locals[t].data() + i));
            vsum = _mm256_add_epi32(vsum, v);
        }
        _mm256_storeu_si256((__m256i*)(acc_global + i), vsum);
    }
    for (; i < N; i++) {
        int s = 0;
        for (int t = 0; t < T; t++) s += locals[t][i];
        acc_global[i] = s;
    }
}

// ------------------ AVX2 merge ------------------
void merge_acc_avx(int* acc, const vector<vector<int>>& locals, int N, int T) {
    size_t i = 0;
    for (; i + 8 <= (size_t)N; i += 8) {
        __m256i vsum = _mm256_setzero_si256();
        for (int t = 0; t < T; t++) {
            __m256i v = _mm256_loadu_si256((__m256i*)(locals[t].data() + i));
            vsum = _mm256_add_epi32(vsum, v);
        }
        _mm256_storeu_si256((__m256i*)(acc + i), vsum);
    }
    for (; i < (size_t)N; i++) {
        int s = 0;
        for (int t = 0; t < T; t++) s += locals[t][i];
        acc[i] = s;
    }
}

// ------------------ Main ------------------
int main(int argc, char** argv) {
    auto t_total_start = high_resolution_clock::now();
    Params p = parse_args(argc, argv);

    Mat img = imread(p.input);
    if (img.empty()) {
        cerr << "Cannot open image\n";
        return -1;
    }

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Mat edges_mat;
    Canny(gray, edges_mat, p.canny_low, p.canny_high);
    Mat gx, gy;
    Sobel(gray, gx, CV_32F, 1, 0, 3);
    Sobel(gray, gy, CV_32F, 0, 1, 3);

    vector<Edge> edgelist;
    for (int y = 0; y < edges_mat.rows; y++) {
        const uchar* er = edges_mat.ptr<uchar>(y);
        const float* gxr = gx.ptr<float>(y);
        const float* gyr = gy.ptr<float>(y);
        for (int x = 0; x < edges_mat.cols; x++) {
            if (!er[x]) continue;
            float dx = gxr[x], dy = gyr[x];
            float mag = sqrt(dx * dx + dy * dy);
            if (mag < 1e-6) continue;
            edgelist.push_back({x, y, dx / mag, dy / mag});
        }
    }

    int W = gray.cols, H = gray.rows;
    int T = p.threads;
    size_t per = (edgelist.size() + T - 1) / T;

    vector<pthread_t> tids(T);
    double vote_ms = 0.0;

    if (p.mode == "circle") {
        vector<vector<int>> local_acc(T, vector<int>(W * H, 0));
        vector<ThreadArgCircle> targs(T);
        auto t_vote_start = high_resolution_clock::now();
        for (int ti = 0; ti < T; ti++) {
            size_t s = ti * per;
            size_t e = min(edgelist.size(), s + per);
            targs[ti] = {&edgelist, s, e, W, H, p.rmin, p.rmax, p.rstep, &local_acc[ti], 0.0};
            if (s < e) pthread_create(&tids[ti], nullptr, vote_circle_thread_avx, &targs[ti]);
        }
        for (int ti = 0; ti < T; ti++) pthread_join(tids[ti], nullptr);

        int* acc = (int*)_mm_malloc(sizeof(int) * W * H, 32);
        merge_acc_avx(acc, local_acc, W * H, T);

        auto t_vote_end = high_resolution_clock::now();
        vote_ms = duration<double, milli>(t_vote_end - t_vote_start).count();
        cout << "Voting time: " << duration<double, milli>(t_vote_end - t_vote_start).count()
             << " ms\n";

        int best_votes = 0, best_cx = 0, best_cy = 0;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                if (acc[y * W + x] > best_votes) {
                    best_votes = acc[y * W + x];
                    best_cx = x;
                    best_cy = y;
                }

        cout << "Best circle center: (" << best_cx << "," << best_cy << ") votes=" << best_votes
             << "\n";
        _mm_free(acc);
        for (int ti = 0; ti < T; ti++)
            cout << "Thread " << ti << " time: " << targs[ti].thread_time << " ms\n";

    } else if (p.mode == "line") {
        int theta_step = p.theta_step_deg;
        int ntheta = 180 / theta_step;
        float diag = sqrt((float)W * W + (float)H * H);
        int nrho = int(diag) * 2 + 1;
        int rho_off = nrho / 2;

        vector<vector<int>> local_acc(T, vector<int>(ntheta * nrho, 0));
        vector<ThreadArgLine> targs(T);
        vector<pthread_t> tids(T);

        size_t per = (edgelist.size() + T - 1) / T;

        auto t_vote_start = high_resolution_clock::now();

        for (int ti = 0; ti < T; ti++) {
            size_t s = ti * per, e = min(edgelist.size(), s + per);
            targs[ti] = {&edgelist, s,       e, theta_step, p.theta_window_deg, ntheta,
                         nrho,      rho_off, W, H,          &local_acc[ti],     0.0};
            if (s < e) pthread_create(&tids[ti], nullptr, vote_line_thread_avx, &targs[ti]);
        }

        for (int ti = 0; ti < T; ti++) pthread_join(tids[ti], nullptr);

        int* acc_global = (int*)_mm_malloc(sizeof(int) * ntheta * nrho, 32);
        merge_acc_line_avx(acc_global, local_acc, ntheta, nrho, T);

        auto t_vote_end = high_resolution_clock::now();
        cout << "Voting time: " << duration<double, milli>(t_vote_end - t_vote_start).count()
             << " ms\n";

        // 找最大 votes
        int best_votes = 0, best_ti = 0, best_ri = 0;
        for (int ti = 0; ti < ntheta; ti++)
            for (int ri = 0; ri < nrho; ri++) {
                int v = acc_global[ti * nrho + ri];
                if (v > best_votes) {
                    best_votes = v;
                    best_ti = ti;
                    best_ri = ri;
                }
            }
        float best_theta = best_ti * theta_step * CV_PI / 180.0f;
        float best_rho = best_ri - rho_off;
        cout << "Best line: rho=" << best_rho << " theta(deg)=" << best_theta * 180.0 / CV_PI
             << " votes=" << best_votes << "\n";

        _mm_free(acc_global);
        for (int ti = 0; ti < T; ti++)
            cout << "Thread " << ti << " time: " << targs[ti].thread_time << " ms\n";
    }

    // cout << "Voting time: " << vote_ms << " ms\n";
    cout << "Total time: "
         << duration<double, milli>(high_resolution_clock::now() - t_total_start).count()
         << " ms\n";

    return 0;
}
