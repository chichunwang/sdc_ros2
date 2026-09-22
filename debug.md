為什麼會發生這個狀況？

你現在其實已經走到最後一步：

GTX 1050 Ti
    ↓
Ubuntu 有偵測到             ✓
    ↓
nvidia-driver-580 已安裝     ✓
    ↓
kernel 6.8.0-139 對應模組    ✓
    ↓
NVIDIA DKMS module 編譯      ✓
    ↓
module 有數位簽章            ✓
    ↓
Secure Boot 信任這把 MOK     ✗ ← 現在卡這裡
    ↓
載入 nvidia module
    ↓
nvidia-smi

而你的：

signer: ricky-OMEN... Secure Boot Module Signature key

其實是很好的訊號——不是驅動沒有簽名，而只是 Secure Boot 還不認識這把簽名 key。

Ubuntu 官方也特別說明，DKMS 在 Secure Boot 環境中會使用機器自己的 MOK 簽署 module；如果 MOK 尚未 enroll，就需要在下一次開機透過 MokManager 加入。

所以現在不要重裝 580、不要換 535，也先不要關 Secure Boot。直接從：

sudo update-secureboot-policy --enroll-key

開始即可。