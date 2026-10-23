# ubuntu常用指令

Midterm01 GPU 的 Track3 v2 full 手動命令、可選短段、歷史 legacy 命令與圖表指令，見 [50｜指令表](../midterm01_gpu/doc/50_COMMAND_REFERENCE.md)；scale 意義、模型假設與 v1 full 失敗紀錄見 [52｜Track3 因果 Kalman 與逐幀尺度配準](../midterm01_gpu/doc/52_TRACK3_CAUSAL_KALMAN_SCALE_20261023.md)，v2 初始化修正及合成／機制測試界限見 [53｜Track3 初始化修正與驗證](../midterm01_gpu/doc/53_TRACK3_INITIALIZATION_REPAIR_20261023.md)。目前沒有真實 v2 定位或新分數；完整定位由使用者手動啟動。

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
正在執行
```bash
docker ps
```
所以容器
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

