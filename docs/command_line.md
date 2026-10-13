# ubuntu常用指令
開啟遠程ssh不需要了
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
```
離開docker
```bash
exit
```
