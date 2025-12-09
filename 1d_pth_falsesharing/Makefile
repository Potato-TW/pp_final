CXX = g++
# CXXFLAGS = -O3 -std=c++17 `pkg-config --cflags opencv4` -pthread -I ../opencv/include/
CXXFLAGS = -O3 -std=c++17 `pkg-config --cflags --libs opencv4` -pthread 

LDFLAGS = `pkg-config --libs opencv4`

TARGET = hough_pthreads.out
TARGET_CIRCLE = hough_circle_pthreads.out
TARGET_STAR = hough_star_pthreads.out

all: $(TARGET) circle $(TARGET_STAR)

$(TARGET): main.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) main.cpp $(LDFLAGS)

circle: main_circle.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET_CIRCLE) main_circle.cpp $(LDFLAGS)

$(TARGET_STAR): main_star.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET_STAR) main_star.cpp $(LDFLAGS)

clean:
	rm -f $(TARGET) $(TARGET_CIRCLE) $(TARGET_STAR)

