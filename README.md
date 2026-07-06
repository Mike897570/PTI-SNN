# PTI-SNN: A Spatio-Temporal Attention Spiking Neural Network for EEG Emotion Recognition

[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-red)](https://pytorch.org/)
[![SpikingJelly](https://img.shields.io/badge/SpikingJelly-SNN-blue)](https://github.com/fangwei123456/spikingjelly)

[English](#english) | [中文](#中文)

---

## English

This repository contains the official PyTorch and SpikingJelly implementation of **PTI-SNN**. 

**PTI-SNN** is a bio-plausible and energy-efficient Spiking Neural Network framework for EEG emotion recognition. By bridging neurophysiology and affective computing, it introduces two core biophysical mechanisms into spatiotemporal attention modeling:

PTI-STA: Modulates spatial attention via macroscopic neuronal spike rates (SGSA) and guides temporal attention using continuous subthreshold membrane potentials (MMTA).

FAEDE & SC-STF: Utilizes a frequency-aware event-driven encoder for multi-band spike generation, and an event-driven gating mechanism (SC-STF) for robust feature fusion with linear complexity.

Our model achieves state-of-the-art LOSO cross-subject accuracy on SEED (76.44%) and SEED-IV (61.76%), while reducing theoretical energy consumption by 93.57% compared to traditional ANNs.

---

## 中文

本项目是论文 **《A Spatio-Temporal Attention Spiking Neural Network with Population Coding and Temporal Integration for EEG Emotion Recognition》** 的官方代码实现。

**PTI-SNN** 框架创造性地将宏观空间群体编码（神经元发放率）与微观阈下时间积分（连续膜电位动态）融入注意力机制（PTI-STA），解决了传统 SNN 偏离生物物理本质的问题。结合频率感知事件驱动编码器（FAEDE）与线性复杂度的脉冲一致性时空融合模块（SC-STF），模型在 SEED (76.44%) 和 SEED-IV (61.76%) 数据集的跨被试评估中达到 SOTA 性能，并比人工神经网络节约了 93.57% 的理论能耗。
