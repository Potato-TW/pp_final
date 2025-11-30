MPICXX = mpicxx
CXXFLAGS = -O3 -std=c++17 `pkg-config --cflags --libs opencv4`
TARGET = hough_mpi

all: $(TARGET)

$(TARGET): main.cpp
	$(MPICXX) main.cpp $(CXXFLAGS) -o $(TARGET) 

clean:
	rm -f $(TARGET)

# mpicxx hough_mpi.cpp -O3 -std=c++17 `pkg-config --cflags --libs opencv4` -o hough_mpi


# mpirun -np 4 ./hough_mpi -i ../lena.png -o mpi_out.png --mode line
