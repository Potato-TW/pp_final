# g++ circle_grad_omp.cpp -O3 -fopenmp `pkg-config --cflags --libs opencv4` -o circle_grad_omp

CXX = g++
CXXFLAGS = -O3 -std=c++17 -fopenmp
LDFLAGS = `pkg-config --cflags --libs opencv4`
AVX2=-mavx2 -mfma


TARGET_STAR_STATIC = hough_star_static_omp.out
TARGET_STAR_DYNAMIC = hough_star_dynamic_omp.out
TARGET_AVX = hough_avx_omp.out

MAIN_AVX = main_avx.cpp
MAIN_STAR_DYNAMIC = main_star_dynamic.cpp
MAIN_STAR_STATIC = main_star_static.cpp

all: $(TARGET_AVX) $(TARGET_STAR_DYNAMIC) $(TARGET_STAR_STATIC)

$(TARGET_AVX): $(MAIN_AVX)
	$(CXX) $(CXXFLAGS) -o $(TARGET_AVX) $(MAIN_AVX) $(LDFLAGS) $(AVX2)

$(TARGET_STAR_STATIC): $(MAIN_STAR_STATIC)
	$(CXX) $(CXXFLAGS) -o $(TARGET_STAR_STATIC) $(MAIN_STAR_STATIC) $(LDFLAGS) $(AVX2)

$(TARGET_STAR_DYNAMIC): $(MAIN_STAR_DYNAMIC)
	$(CXX) $(CXXFLAGS) -o $(TARGET_STAR_DYNAMIC) $(MAIN_STAR_DYNAMIC) $(LDFLAGS) $(AVX2)

clean:
	rm -f *.out

cleanall: clean
	rm -f *.png