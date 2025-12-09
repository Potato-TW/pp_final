CXX = g++
# CXXFLAGS = -O3 -std=c++17 `pkg-config --cflags opencv4` -pthread -I ../opencv/include/
CXXFLAGS = -O3 -std=c++17 `pkg-config --cflags --libs opencv4` -pthread 
LDFLAGS = `pkg-config --libs opencv4`

# TARGET = hough_pthreads.out
TARGET_STAR_LOCALACC = hough_star_localacc_pthreads.out
TARGET_STAR = hough_star_pthreads.out
TARGET_STAR_2D = hough_star_2d_pthreads.out
TARGET_BLOCK = hough_block_pthreads.out
# TARGET_STEP_DIVIDE = hough_step_pthreads.out
TARGET_AVX = hough_avx_pthreads.out

# MAIN = main.cpp
MAIN_BLOCK = main_block.cpp
MAIN_STAR = main_star.cpp
MAIN_STAR_LOCALACC = main_star_localacc.cpp
MAIN_STAR_2D = main_star_2d.cpp
# MAIN_STEP_DIVIDE = main_step_divide.cpp
MAIN_AVX = main_avx.cpp

all: $(TARGET_BLOCK) $(TARGET_STAR) $(TARGET_STAR_LOCALACC) $(TARGET_STAR_2D) $(TARGET_AVX)


$(TARGET_BLOCK): $(MAIN_BLOCK)
	$(CXX) $(CXXFLAGS) -o $(TARGET_BLOCK) $(MAIN_BLOCK) $(LDFLAGS)

$(TARGET_STAR_LOCALACC): $(MAIN_STAR_LOCALACC)
	$(CXX) $(CXXFLAGS) -mavx2 -mfma -o $(TARGET_STAR_LOCALACC) $(MAIN_STAR_LOCALACC) $(LDFLAGS)

$(TARGET_STAR): $(MAIN_STAR)
	$(CXX) $(CXXFLAGS) -o $(TARGET_STAR) $(MAIN_STAR) $(LDFLAGS)

$(TARGET_STAR_2D): $(MAIN_STAR_2D)
	$(CXX) $(CXXFLAGS) -o $(TARGET_STAR_2D) $(MAIN_STAR_2D) $(LDFLAGS)

$(TARGET_AVX): $(MAIN_AVX)
	$(CXX) $(CXXFLAGS) -mavx2 -mfma -o $(TARGET_AVX) $(MAIN_AVX) $(LDFLAGS)


# $(TARGET_STEP_DIVIDE): $(MAIN_STEP_DIVIDE)
# 	$(CXX) $(CXXFLAGS) -o $(TARGET_STEP_DIVIDE) $(MAIN_STEP_DIVIDE) $(LDFLAGS)

clean:
	rm -f *.out

cleanall: clean
	rm -f *.png