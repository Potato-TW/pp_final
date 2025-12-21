Download opencv and mpi first
```bash
sudo apt update
sudo apt install libopencv-dev 
sudo apt install openmpi-bin libopenmpi-dev

下載銀河圖
wget https://assets.science.nasa.gov/content/dam/science/missions/webb/science/2022/07/STScI-01GA6KKWG229B16K4Q38CH3BXS.png
```

Run code 
```bash
make clean ; make
./<exe file name> -i ../lena.png --mode line(circle)
有些不一樣 先打./<exe file name> 就能看到怎麼寫 在加 --mode line或circle
```