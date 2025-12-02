CXX = g++
# CXXFLAGS = -O3 -std=c++17 `pkg-config --cflags opencv4` -pthread -I ../opencv/include/
CXXFLAGS = -O3 -std=c++17 `pkg-config --cflags --libs opencv4` -pthread 

LDFLAGS = `pkg-config --libs opencv4`

TARGET = hough_pthreads.out

all: $(TARGET)

$(TARGET): main.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) main.cpp $(LDFLAGS)

clean:
	rm -f $(TARGET)

