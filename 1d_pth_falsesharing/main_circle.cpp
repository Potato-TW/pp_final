#include <pthread.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <cmath>
#include <chrono>
using namespace std;
using namespace cv;
using namespace chrono;

struct Params {
    string input, output="out_circle_pthread_fixed.png";
    int rmin=20, rmax=120, rstep=1;
    int canny_low=50, canny_high=150;
    int threads=8;
};

struct Edge { int x,y; float nx,ny; };

struct CircleVote {
    int cx,cy,r,votes;
};

struct ThreadArg {
    const vector<Edge>* edges;
    size_t start,end;
    int W,H;
    int rmin,rmax,rstep;
    vector<CircleVote>* local_best; // store each thread's local best per radius
};

void* vote_circle_thread(void* arg){
    ThreadArg* ta = (ThreadArg*)arg;
    const vector<Edge>& edges = *ta->edges;
    int W = ta->W, H = ta->H;
    int rmin = ta->rmin, rmax = ta->rmax, rstep = ta->rstep;

    // store best circle per radius
    vector<CircleVote> best_per_radius;

    for(int r=rmin; r<=rmax; r+=rstep){
        vector<int> acc((size_t)W*H,0); // thread-local accumulator
        for(size_t i=ta->start; i<ta->end; i++){
            const Edge& e = edges[i];
            int cx1 = int(round(e.x + r*e.nx));
            int cy1 = int(round(e.y + r*e.ny));
            if(cx1>=0 && cx1<W && cy1>=0 && cy1<H) acc[cy1*W+cx1]++;
            int cx2 = int(round(e.x - r*e.nx));
            int cy2 = int(round(e.y - r*e.ny));
            if(cx2>=0 && cx2<W && cy2>=0 && cy2<H) acc[cy2*W+cx2]++;
        }
        // find local best
        CircleVote cb{0,0,r,0};
        for(size_t i=0;i<acc.size();i++){
            if(acc[i]>cb.votes){
                cb.votes=acc[i];
                cb.cx=i%W;
                cb.cy=i/W;
            }
        }
        best_per_radius.push_back(cb);
    }

    *ta->local_best = best_per_radius;
    return nullptr;
}

Params parse_args(int argc,char** argv){
    Params p;
    if(argc<2){ cerr<<"Usage\n"; exit(1);}
    p.input=argv[1];
    for(int i=2;i<argc;i++){
        string a=argv[i];
        if(a=="-o" && i+1<argc) p.output=argv[++i];
        else if(a=="--rmin" && i+1<argc) p.rmin=stoi(argv[++i]);
        else if(a=="--rmax" && i+1<argc) p.rmax=stoi(argv[++i]);
        else if(a=="--rstep" && i+1<argc) p.rstep=stoi(argv[++i]);
        else if(a=="-t" && i+1<argc) p.threads=stoi(argv[++i]);
    }
    return p;
}

int main(int argc,char** argv){
    auto t_total=high_resolution_clock::now();
    Params p = parse_args(argc,argv);
    Mat img = imread(p.input);
    if(img.empty()){ cerr<<"Cannot open image\n"; return -1;}
    Mat gray; cvtColor(img,gray,COLOR_BGR2GRAY);

    // Canny
    Mat edges; Canny(gray,edges,p.canny_low,p.canny_high);

    // Sobel
    Mat gx,gy; Sobel(gray,gx,CV_32F,1,0,3); Sobel(gray,gy,CV_32F,0,1,3);
    vector<Edge> edgelist;
    for(int y=0;y<edges.rows;y++){
        const uchar* er = edges.ptr<uchar>(y);
        const float* gxr = gx.ptr<float>(y);
        const float* gyr = gy.ptr<float>(y);
        for(int x=0;x<edges.cols;x++){
            if(!er[x]) continue;
            float dx=gxr[x], dy=gyr[x];
            float mag=sqrt(dx*dx+dy*dy);
            if(mag<1e-6) continue;
            edgelist.push_back({x,y,dx/mag,dy/mag});
        }
    }
    int W=gray.cols,H=gray.rows;

    int T = p.threads;
    vector<pthread_t> tids(T);
    vector< vector<CircleVote> > local_bests(T);
    vector<ThreadArg> targs(T);

    size_t N=edgelist.size();
    size_t per=(N+T-1)/T;
    for(int ti=0;ti<T;ti++){
        size_t s=ti*per, e=min(N,s+per);
        targs[ti] = {&edgelist,s,e,W,H,p.rmin,p.rmax,p.rstep,&local_bests[ti]};
        pthread_create(&tids[ti],nullptr,vote_circle_thread,&targs[ti]);
    }
    for(int ti=0;ti<T;ti++) pthread_join(tids[ti],nullptr);

    // reduction: find global best
    CircleVote best{0,0,0,0};
    for(int ti=0;ti<T;ti++){
        for(auto& cb: local_bests[ti]){
            if(cb.votes>best.votes) best=cb;
        }
    }

    cout<<"Best circle: cx="<<best.cx<<" cy="<<best.cy<<" r="<<best.r
        <<" votes="<<best.votes<<"\n";

    Mat out=img.clone();
    if(best.votes>0){
        circle(out,Point(best.cx,best.cy),best.r,Scalar(0,255,0),3);
        circle(out,Point(best.cx,best.cy),3,Scalar(0,0,255),-1);
    }
    imwrite(p.output,out);

    auto t_end=high_resolution_clock::now();
    cout<<"Total time(ms): "<<duration<double,milli>(t_end-t_total).count()<<"\n";
}
