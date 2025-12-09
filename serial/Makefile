CXX = g++
CXXFLAGS = -O3 -std=c++17 `pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv`
LDFLAGS = `pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv`

TARGET = hough_serial.out
TARGET_STAR = hough_star_serial.out

all: $(TARGET) $(TARGET_STAR)

$(TARGET): main.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) main.cpp $(LDFLAGS)

$(TARGET_STAR): main_star.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET_STAR) main_star.cpp $(LDFLAGS)

clean:
	rm -f $(TARGET) $(TARGET_STAR)
	

