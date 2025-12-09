// hough_pthread_1d.cpp
// Pthreads 1D gradient-guided line & circle Hough (no false sharing)
// Compile:
// g++ hough_pthread_1d.cpp -O3 -std=c++17 `pkg-config --cflags --libs opencv4` -pthread -o hough_pthread_1d
//
// Run examples:
// ./hough_pthread_1d input.jpg --mode circle -t 8 --rmin 20 --rmax 200 --rstep 2 -o out_pt_circle.png
// ./hough_pthread_1d input.jpg --mode line -t 8 --tstep 1 --twin 4 -o out_pt_line.png

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <pthread.h>
#include <chrono>
#include <string>
#include <algorithm>

using namespace std;
using namespace cv;
using namespace std::chrono;

struct Params {
    string input, output = "output.png", mode = "circle";
    int rmin = 20, rmax = 120, rstep = 1;
    int canny_low = 50, canny_high = 150;
    int theta_step_deg = 1, theta_window_deg = 4;
    int threads = 8;
};

Params parse_args(int argc, char** argv) {
    Params p;
    if (argc < 2) { cerr << "Usage: " << argv[0] << " input.png [options]\n"; exit(1); }
    p.input = argv[1];
    for (int i = 2; i < argc; i++) {
        string a = argv[i];
        if (a == "-o" && i+1<argc) p.output = argv[++i];
        else if (a == "--mode" && i+1<argc) p.mode = argv[++i];
        else if (a == "--rmin" && i+1<argc) p.rmin = stoi(argv[++i]);
        else if (a == "--rmax" && i+1<argc) p.rmax = stoi(argv[++i]);
        else if (a == "--rstep" && i+1<argc) p.rstep = stoi(argv[++i]);
        else if (a == "--tstep" && i+1<argc) p.theta_step_deg = stoi(argv[++i]);
        else if (a == "--twin" && i+1<argc) p.theta_window_deg = stoi(argv[++i]);
        else if (a == "-t" && i+1<argc) p.threads = stoi(argv[++i]);
    }
    if (p.mode=="circle") p.output="out_circle_pthreads_1d.png";
    else if(p.mode=="line") p.output="out_line_pthreads_1d.png";
    return p;
}

// Edge struct
struct Edge { int x,y; float nx,ny; float ori; };

// Circle Hough thread argument
struct ThreadArgCircle {
    const vector<Edge>* edges;
    size_t start,end;
    int r,W,H;
    int* local_acc;
};

void* vote_circle_thread(void* arg) {
    ThreadArgCircle* ta = (ThreadArgCircle*)arg;
    int W=ta->W,H=ta->H,r=ta->r;
    int* acc = ta->local_acc;
    for(size_t i=ta->start;i<ta->end;i++){
        const Edge& e = (*ta->edges)[i];
        int cx = int(round(e.x + r*e.nx));
        int cy = int(round(e.y + r*e.ny));
        if(cx>=0 && cx<W && cy>=0 && cy<H) acc[cy*W+cx]++;
        int cx2 = int(round(e.x - r*e.nx));
        int cy2 = int(round(e.y - r*e.ny));
        if(cx2>=0 && cx2<W && cy2>=0 && cy2<H) acc[cy2*W+cx2]++;
    }
    return nullptr;
}

// Line Hough thread argument
struct ThreadArgLine {
    const vector<Edge>* edges;
    size_t start,end;
    int theta_step, theta_window, ntheta, nrho, rho_off;
    int W,H;
    int* local_acc;
};

void* vote_line_thread(void* arg){
    ThreadArgLine* ta = (ThreadArgLine*)arg;
    int W=ta->W,H=ta->H;
    for(size_t i=ta->start;i<ta->end;i++){
        const Edge& e=(*ta->edges)[i];
        float base = e.ori + (float)CV_PI/2.0f;
        int base_deg = int(round(base*180.0/CV_PI))%180;
        if(base_deg<0) base_deg+=180;
        for(int d=-ta->theta_window; d<=ta->theta_window; ++d){
            int deg=base_deg+d;
            if(deg<0) deg+=180;
            if(deg>=180) deg-=180;
            if(deg%ta->theta_step!=0) continue;
            int ti=deg/ta->theta_step;
            float theta = deg*CV_PI/180.0f;
            float rho = e.x*cos(theta) + e.y*sin(theta);
            int ri = int(round(rho))+ta->rho_off;
            if(ri>=0 && ri<ta->nrho) ta->local_acc[ti*ta->nrho+ri]++;
        }
    }
    return nullptr;
}

int main(int argc, char** argv){
    auto t_total_start=high_resolution_clock::now();
    Params p = parse_args(argc,argv);

    Mat img = imread(p.input);
    if(img.empty()){ cerr<<"Cannot open image\n"; return -1;}
    Mat gray;
    cvtColor(img,gray,COLOR_BGR2GRAY);

    // Canny
    auto t0=high_resolution_clock::now();
    Mat edges;
    Canny(gray,edges,p.canny_low,p.canny_high);
    auto t1=high_resolution_clock::now();
    double canny_ms = duration<double,milli>(t1-t0).count();

    // Sobel
    t0=high_resolution_clock::now();
    Mat gx,gy;
    Sobel(gray,gx,CV_32F,1,0,3);
    Sobel(gray,gy,CV_32F,0,1,3);
    vector<Edge> edgelist;
    edgelist.reserve(1000000);
    for(int y=0;y<edges.rows;y++){
        const uchar* er = edges.ptr<uchar>(y);
        const float* gxr = gx.ptr<float>(y);
        const float* gyr = gy.ptr<float>(y);
        for(int x=0;x<edges.cols;x++){
            if(!er[x]) continue;
            float dx=gxr[x],dy=gyr[x];
            float mag=sqrt(dx*dx+dy*dy);
            if(mag<1e-6) continue;
            edgelist.push_back({x,y,dx/mag,dy/mag,(float)atan2(dy,dx)});
        }
    }
    auto t2=high_resolution_clock::now();
    double grad_ms = duration<double,milli>(t2-t0).count();

    int W=gray.cols,H=gray.rows;

    if(p.mode=="circle"){
        int best_votes=0,best_cx=0,best_cy=0,best_r=0;
        size_t per_acc = W*H;
        vector<int> all_local(per_acc*p.threads,0);
        vector<int> global_acc(per_acc,0);

        auto t_vote_start=high_resolution_clock::now();
        for(int r=p.rmin;r<=p.rmax;r+=p.rstep){
            fill(all_local.begin(),all_local.end(),0);
            fill(global_acc.begin(),global_acc.end(),0);

            size_t N=edgelist.size();
            size_t chunk = (N+p.threads-1)/p.threads;
            vector<pthread_t> tids(p.threads);
            vector<ThreadArgCircle> args(p.threads);

            for(int t=0;t<p.threads;t++){
                size_t s = t*chunk;
                size_t e = min(N,s+chunk);
                args[t]={&edgelist,s,e,r,W,H,all_local.data()+t*per_acc};
                pthread_create(&tids[t],nullptr,vote_circle_thread,&args[t]);
            }
            for(int t=0;t<p.threads;t++) pthread_join(tids[t],nullptr);

            // reduction
            for(size_t idx=0;idx<per_acc;idx++){
                int sum=0;
                for(int t=0;t<p.threads;t++) sum+=all_local[t*per_acc+idx];
                global_acc[idx]=sum;
            }

            // find best
            for(int y=0;y<H;y++){
                for(int x=0;x<W;x++){
                    int v = global_acc[y*W+x];
                    if(v>best_votes){ best_votes=v; best_cx=x; best_cy=y; best_r=r;}
                }
            }
        }
        auto t_vote_end=high_resolution_clock::now();
        cout<<"Voting total: "<<duration<double,milli>(t_vote_end-t_vote_start).count()<<" ms\n";
        cout<<"Best circle: cx="<<best_cx<<" cy="<<best_cy<<" r="<<best_r<<" votes="<<best_votes<<"\n";

        Mat out=img.clone();
        if(best_votes>0){
            circle(out,Point(best_cx,best_cy),best_r,Scalar(0,255,0),3);
            circle(out,Point(best_cx,best_cy),3,Scalar(0,0,255),-1);
        }
        imwrite(p.output,out);

    } else { // line
        int theta_step=p.theta_step_deg;
        int ntheta = 180/theta_step;
        float diag = sqrt((float)W*W + (float)H*H);
        int nrho = int(diag)*2 +1;
        int rho_off = nrho/2;
        int twin = p.theta_window_deg;

        vector<int> all_local(nrho*ntheta*p.threads,0);
        vector<int> global_acc(nrho*ntheta,0);

        auto t_vote_start=high_resolution_clock::now();
        size_t N=edgelist.size();
        size_t chunk = (N+p.threads-1)/p.threads;
        vector<pthread_t> tids(p.threads);
        vector<ThreadArgLine> args(p.threads);

        for(int t=0;t<p.threads;t++){
            size_t s=t*chunk;
            size_t e=min(N,s+chunk);
            args[t]={&edgelist,s,e,theta_step,twin,ntheta,nrho,rho_off,W,H,all_local.data()+t*nrho*ntheta};
            pthread_create(&tids[t],nullptr,vote_line_thread,&args[t]);
        }
        for(int t=0;t<p.threads;t++) pthread_join(tids[t],nullptr);

        // reduction
        for(int idx=0;idx<nrho*ntheta;idx++){
            int sum=0;
            for(int t=0;t<p.threads;t++) sum+=all_local[t*nrho*ntheta + idx];
            global_acc[idx]=sum;
        }

        // find best
        int best_votes=0,bt=0,br=0;
        for(int ti=0;ti<ntheta;ti++)
            for(int ri=0;ri<nrho;ri++){
                int v = global_acc[ti*nrho+ri];
                if(v>best_votes){ best_votes=v; bt=ti; br=ri; }
            }
        float best_theta = bt*theta_step*CV_PI/180.0f;
        float best_rho = br - rho_off;
        cout<<"Voting total: "<<duration<double,milli>(high_resolution_clock::now()-t_vote_start).count()<<" ms\n";
        cout<<"Best line: rho="<<best_rho<<" theta(deg)="<<(best_theta*180.0/CV_PI)<<" votes="<<best_votes<<"\n";

        Mat out=img.clone();
        double a=cos(best_theta),b=sin(best_theta);
        double x0 = a*best_rho, y0=b*best_rho;
        Point p1(cvRound(x0+2000*(-b)),cvRound(y0+2000*(a)));
        Point p2(cvRound(x0-2000*(-b)),cvRound(y0-2000*(a)));
        line(out,p1,p2,Scalar(0,0,255),3);
        imwrite(p.output,out);
    }

    auto t_total_end=high_resolution_clock::now();
    cout<<"Timing summary (ms): Canny="<<canny_ms<<" Grad="<<grad_ms
        <<" Total="<<duration<double,milli>(t_total_end-t_total_start).count()<<"\n";

    return 0;
}
