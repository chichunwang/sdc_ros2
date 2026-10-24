# ubuntu常用指令

Track3 現行前向 map-only runner 為 `run_track3_map_only.py`，method=`track3_forward_gpu_structural_ndt_icp_reacquisition_v2`；僅用前鏡頭，不接 IMU。現行手動 full 與 200 幀診斷命令見 [50｜指令表](../midterm01_gpu/doc/50_COMMAND_REFERENCE.md)。200 幀 run `track03_test_2610232248` 完整結束，map 182 採用／18 拒絕、lost 0、pair 198/199 採用、19 次重捕獲嘗試中 14 次成功，最長無 map 0.107871264 秒；40 項機制／I/O 檢查、CPU/CUDA weighted-fit parity 及 5／200 幀矩陣與 CSV 前綴逐 byte 核對通過。這只支持該段完整性與機制結果，不證明正式定位精度；目前沒有 full 或正式新分數。`--vram-budget-gib 4` 是 Torch allocator cap，不含 driver／NVRTC 全部 VRAM。詳細方法見 [51｜Track3 架構、演進與診斷](../midterm01_gpu/doc/51_TRACK3_ARCHITECTURE_AND_DEVELOPMENT_20261023.md)。

歷史 v1 full `track03_full_2610231912` 僅處理 1040/3142 幀後以 return code 1 結束，`output_complete_and_finite=false`、`validation_errors=[]`；map 98 採用／942 拒絕，idx125最後接受、idx181起失追859幀，最長無 map 32.893173258秒。終止原因未記錄，不推定；原 run 保留，不推薦原樣重跑。同一 CLI 現在執行 v2，不能重現該 v1；舊 source snapshot 保留。分析見 [51｜Track3 架構、演進與診斷](../midterm01_gpu/doc/51_TRACK3_ARCHITECTURE_AND_DEVELOPMENT_20261023.md)。

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

