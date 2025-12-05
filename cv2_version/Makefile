CXX = g++
CXXFLAG = -O3 `pkg-config --cflags --libs opencv4`

TARGET = hough_opencv.out

all: $(TARGET)

$(TARGET): main.cpp
	$(CXX) main.cpp -o $(TARGET) $(CXXFLAG)

clean:
	rm -f $(TARGET)

# g++ main.cpp -o opencv_ht -O3 `pkg-config --cflags --libs opencv4`
# ./hough_opencv.out line ../lena.png out_line.png
# ./hough_opencv.out circle ../lena.png out_circle.png