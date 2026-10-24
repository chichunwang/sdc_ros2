# ubuntu常用指令

開啟遠程ssh(不需要了)
```bash
sudo systemctl status ssh
```
開啟docker
```bash
docker start my_docker
docker start "docker name"
```
進入docker
```bash
docker exec -it my_docker bash
docker exec -it "docker name" bash
docker exec -it midterm01_track3 bash
```
離開docker
```bash
exit
```
正在執行
```bash
docker ps
```
所有容器
```bash
docker ps -a
```
關機
```bash
sudo shutdown -h now
```
開啟nvidia驅動顯示
```bash
watch -n 0.2 nvidia-smi
```

