# FedKWS-UI
Source Code for Interspeech 2022 paper: Avoid Overfitting User Specific Information in Federated Keyword Spotting

## Dataset
  * Download Speech Commands dataset: [Torchaudio SC](https://pytorch.org/tutorials/intermediate/speech_command_recognition_with_torchaudio.html)
  * Preprocess: `preprocess.py`
  * Load Data: `speech_commands_data.py` and `speech_commands_feddata.py`

## Running Tips
  * Run FedAvg, FedMMD, FedProx, FedOpt, FedKWS-UI (Ours) with `train_xxx.py`.
  * Algorithm code: `fedavg.py`, `fedkwsui.py`, etc.

## Citation
  * Xin-Chun Li, Jin-Lin Tang, Shaoming Song, Bingshuai Li, Yinchuan Li, Yunfeng Shao, Le Gan, De-Chuan Zhan. Avoid Overfitting User Specific Information in Federated Keyword Spotting. In: Proceedings of the 23rd INTERSPEECH Conference (INTERSPEECH'2022), online conference, Songdo ConvensiA, Incheon, Korea, 2022.
  * \[[Bibtex](https://dblp.org/pid/246/2947.html)\]

