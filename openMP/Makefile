# g++ circle_grad_omp.cpp -O3 -fopenmp `pkg-config --cflags --libs opencv4` -o circle_grad_omp

CXX = g++
CXXFLAGS = -O3 -std=c++17 -fopenmp
LDFLAGS = `pkg-config --cflags --libs opencv4`

TARGET_LINE = hough_line_omp.out
TARGET_CIRCLE = hough_circle_omp.out

TARGET_STAR = hough_star_omp.out

all: line circle star

line: line.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET_LINE) line.cpp $(LDFLAGS) 

circle: circle.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET_CIRCLE) circle.cpp $(LDFLAGS) 

star: main_star.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET_STAR) main_star.cpp $(LDFLAGS)

clean:
	rm -f $(TARGET_LINE) $(TARGET_CIRCLE) $(TARGET_STAR)