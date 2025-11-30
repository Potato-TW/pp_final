CXX = g++
CXXFLAGS = -O3 -std=c++17 `pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv`
LDFLAGS = `pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv`

TARGET = serial 

all: $(TARGET)

$(TARGET): main.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) main.cpp $(LDFLAGS)

clean:
	rm -f $(TARGET)

