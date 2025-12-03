#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
using namespace std;
using namespace cv;
using namespace std::chrono;

int main(int argc, char** argv) {
    if (argc < 4) {
        cout << "Usage:\n";
        cout << "  ./opencv_ht line   input.png  output.png\n";
        cout << "  ./opencv_ht circle input.png  output.png\n";
        return -1;
    }

    string mode = argv[1];
    string input = argv[2];
    string output = argv[3];

    auto total_start = high_resolution_clock::now();

    Mat img = imread(input);
    if (img.empty()) {
        cerr << "Error: cannot load image.\n";
        return -1;
    }

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // --------------------------
    // Canny time
    // --------------------------
    auto canny_start = high_resolution_clock::now();
    Mat edges;
    Canny(gray, edges, 50, 150);
    auto canny_end = high_resolution_clock::now();
    double canny_ms = duration<double, milli>(canny_end - canny_start).count();

    Mat out = img.clone();

    // ============================================================
    // LINE DETECTION (HoughLines)
    // ============================================================
    if (mode == "line") {
        vector<Vec2f> lines;

        auto hough_start = high_resolution_clock::now();
        HoughLines(edges, lines, 1, CV_PI/180, 120);
        auto hough_end = high_resolution_clock::now();
        double hough_ms = duration<double, milli>(hough_end - hough_start).count();

        cout << "Detected lines: " << lines.size() << endl;

        for (size_t i = 0; i < lines.size(); i++) {
            float rho = lines[i][0];
            float theta = lines[i][1];

            double a = cos(theta), b = sin(theta);
            double x0 = a * rho, y0 = b * rho;

            Point pt1, pt2;
            pt1.x = cvRound(x0 + 2000 * (-b));
            pt1.y = cvRound(y0 + 2000 * ( a));
            pt2.x = cvRound(x0 - 2000 * (-b));
            pt2.y = cvRound(y0 - 2000 * ( a));

            line(out, pt1, pt2, Scalar(0, 0, 255), 2, LINE_AA);
        }

        auto total_end = high_resolution_clock::now();
        double total_ms = duration<double, milli>(total_end - total_start).count();

        cout << "===== OpenCV Line Detection Timing =====\n";
        cout << "Canny:        " << canny_ms << " ms\n";
        cout << "HoughLines:   " << hough_ms << " ms\n";
        cout << "Total:        " << total_ms << " ms\n";
    }

    // ============================================================
    // CIRCLE DETECTION (HoughCircles)
    // ============================================================
    else if (mode == "circle") {
        vector<Vec3f> circles;

        // Gaussian blur for circle detection
        Mat blur_img;
        GaussianBlur(gray, blur_img, Size(9, 9), 2, 2);

        auto hough_start = high_resolution_clock::now();
        HoughCircles(blur_img, circles, HOUGH_GRADIENT, 1.2, 50,
                     100, 30, 10, 200);
        auto hough_end = high_resolution_clock::now();
        double hough_ms = duration<double, milli>(hough_end - hough_start).count();

        cout << "Detected circles: " << circles.size() << endl;

        for (size_t i = 0; i < circles.size(); i++) {
            int cx = cvRound(circles[i][0]);
            int cy = cvRound(circles[i][1]);
            int r  = cvRound(circles[i][2]);

            // green circle
            circle(out, Point(cx, cy), r, Scalar(0, 255, 0), 2);
            // red center dot
            circle(out, Point(cx, cy), 3, Scalar(0, 0, 255), -1);
        }

        auto total_end = high_resolution_clock::now();
        double total_ms = duration<double, milli>(total_end - total_start).count();

        cout << "===== OpenCV Circle Detection Timing =====\n";
        cout << "Canny:         " << canny_ms << " ms\n";
        cout << "HoughCircles:  " << hough_ms << " ms\n";
        cout << "Total:         " << total_ms << " ms\n";
    }

    else {
        cerr << "Error: mode must be 'line' or 'circle'\n";
        return -1;
    }

    imwrite(output, out);
    cout << "Saved output to: " << output << endl;

    return 0;
}
