# Setup Pretrained OOD Detection Models

In [method_module.py](../lib/method_module.py), we include support for popular methods such as max logit, energy score, and PEBAL. Here are the steps to set up these models with pretrained weights.

## OOD Methods Without Additional Training (Max Logit, Energy Score)

Many OOD detection methods leverage existing segmentation models and apply a different scoring strategy without the need for retraining the network.
To conduct experiments on these methods, please download the DeepLab v3+ models pretrained on the Cityscapes dataset provide by [NVIDIA](https://github.com/NVIDIA/semantic-segmentation/tree/sdcnet).
- **DeepLab v3+ WideResnet38**: [Download here](https://drive.google.com/file/d/1P4kPaMY-SmQ3yPJQTJ7xMGAB_Su-1zTl/view)
- **DeepLab v3+ ResNet101**: [Download here](https://drive.google.com/file/d/1Rqty9pRhGdfhkfqlWbFUFgdFp0DvfORN/view)

After downloading, specify the weight path in the `build_model` function within [utils.py](../lib/utils/utils.py).
The configuration file `exp/atta.yaml` can be adjusted to switch between different backbone architectures.

## OOD Method Requiring Additional Training (PEBAL)

[PEBAL](https://arxiv.org/pdf/2111.12264.pdf) is an advanced OOD detection approach that necessitates additional training. To conduct experiments on PEBAL, download the pretrained model provided by their [official code repo](https://github.com/tianyu0207/PEBAL).

- **PEBAL Pretrained Model**: [Download here](https://drive.google.com/file/d/12CebI1TlgF724-xvI3vihjbIPPn5Icpm/view)

Once downloaded, specify its path in the `build_pebal_model` function in [utils.py](../lib/utils/utils.py).

## Custom OOD Methods

For researchers looking to experiment with other OOD detection strategies, our framework is designed to be flexible. Here's a template to guide you in integrating a custom method:

```python
class YourCustomMethodName:
    def __init__(self):
        self.model = build_model()

    def getscore_from_logit(self, logit):
        # Implement your logic for scoring based on logit here.

    def anomaly_score(self, image, ret_logit=False):
        logit = self.model(image)
        anomaly_score = self.getscore_from_logit(logit)

        if ret_logit:
            return anomaly_score, logit
        return anomaly_score
```