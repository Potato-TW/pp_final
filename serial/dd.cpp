#include <chrono>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
using namespace std;
using namespace cv;
using namespace std::chrono;

struct Circle {
    int cx, cy, r, votes;
};

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " input.png\n";
        return -1;
    }
    string input = argv[1];
    Mat img = imread(input);
    if (img.empty()) {
        cerr << "Cannot open image\n";
        return -1;
    }

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // Canny edges
    Mat edges;
    Canny(gray, edges, 80, 180);

    // Sobel
    Mat gx, gy;
    Sobel(gray, gx, CV_32F, 1, 0, 3);
    Sobel(gray, gy, CV_32F, 0, 1, 3);

    struct Edge { int x, y; float nx, ny; };
    vector<Edge> edgelist;
    for (int y = 0; y < edges.rows; y++) {
        const uchar* er = edges.ptr<uchar>(y);
        const float* gxr = gx.ptr<float>(y);
        const float* gyr = gy.ptr<float>(y);
        for (int x = 0; x < edges.cols; x++) {
            if (!er[x]) continue;
            float dx = gxr[x], dy = gyr[x];
            float mag = sqrt(dx*dx + dy*dy);
            if (mag < 1e-6) continue;
            edgelist.push_back({x, y, dx/mag, dy/mag});
        }
    }

    int W = gray.cols, H = gray.rows;
    int rmin = 100, rmax = 200, rstep = 1;
    vector<Circle> best20; // 最多20個
    vector<int> acc((size_t)W * H);

    for (int r = rmin; r <= rmax; r += rstep) {
        fill(acc.begin(), acc.end(), 0);
        // 投票
        for (const auto& e : edgelist) {
            int cx1 = int(round(e.x + r*e.nx));
            int cy1 = int(round(e.y + r*e.ny));
            if (cx1>=0 && cx1<W && cy1>=0 && cy1<H) acc[cy1*W + cx1]++;
            int cx2 = int(round(e.x - r*e.nx));
            int cy2 = int(round(e.y - r*e.ny));
            if (cx2>=0 && cx2<W && cy2>=0 && cy2<H) acc[cy2*W + cx2]++;
        }
        // 找每個半徑的局部最佳點
        int local_best = 0, lcx = 0, lcy = 0;
        for (int y=0; y<H; y++)
            for (int x=0; x<W; x++) {
                int v = acc[y*W + x];
                if (v > local_best) { local_best=v; lcx=x; lcy=y; }
            }
        if (local_best==0) continue;
        // 插入到 best20 vector
        if (best20.size() < 20) {
            best20.push_back({lcx, lcy, r, local_best});
        } else {
            // 找到最小票數，若本輪更大就替換
            auto it = min_element(best20.begin(), best20.end(),
                                  [](const Circle& a, const Circle& b){ return a.votes < b.votes; });
            if (local_best > it->votes) *it = {lcx, lcy, r, local_best};
        }
    }

    // 按票數排序
    sort(best20.begin(), best20.end(),
         [](const Circle& a, const Circle& b){ return a.votes > b.votes; });

    Mat out = img.clone();
    for (const auto& c : best20) {
        circle(out, Point(c.cx, c.cy), c.r, Scalar(0,255,0), 5);
        circle(out, Point(c.cx, c.cy), 3, Scalar(0,0,255), -1);
        cout << "Circle: cx=" << c.cx << " cy=" << c.cy << " r=" << c.r << " votes=" << c.votes << "\n";
    }

    imwrite("out_circle_top20.png", out);
    cout << "Saved out_circle_top20.png\n";
    return 0;
}
