# VLA Tokenization Assessments

We are currently focusing on mechanistic interpretability research analyzing how models trained with the FAST action tokenization scheme use different frequency components of action data. The goal is to gain insights into token representation efficiency, attention distribution, and generalization ability.

## **Key Features**

- **Frequency Band Analysis:** Investigate how OpenVLA processes low, mid, and high-frequency components of action data.
- **Efficient Tokenization:** Evaluate the impact of the FAST tokenization scheme on generalization and efficiency.
- **Attention Attribution:** Analyze attention distribution across frequency bands.
- **Comparative Benchmarking:** Compare FAST tokenization against naive approaches to assess performance improvements.

## **Project Structure**

```
OpenVLA/
│-- analyses/
│-- datas/
│-- results/
│-- scripts/
│-- README.md
```

---

### **Prerequisites**

Required dependencies:

```
brew install python3 wget awscli gdown
pip install numpy requests tqdm torch transformers
```

### **Datasets**

- **DROID Dataset:** Multi-task robot manipulation data.
- **Libero Dataset:** Robotic benchmark dataset.
- **Table Bussing Dataset:** High-frequency tasks.
- **Laundry Folding Dataset:** Complex dexterous tasks.

`scripts/download_data.sh` for automated downloads.

---

## **Contact**

ida.caspary24[at]imperial.ac.uk
