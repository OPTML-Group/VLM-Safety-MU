<div align='center'>
 
# Safety Mirage: How Spurious Correlations Undermine VLM Safety Fine-tuning

</div>

<table align="center">
  <tr>
    <td align="center"> 
      <img src="./images/VLM_unlearn_teasor.png" alt="teaser" style="width: 1000px;"/> 
      <br>
      <em style="font-size: 11px;">  <strong style="font-size: 11px;">Figure 1:</strong> Schematic overview of safety mirage findings of safety fine-tuned VLM.</em>
    </td>
  </tr>
</table>

This is the official code repository for the paper [Safety Mirage: How Spurious Correlations Undermine VLM Safety Fine-tuning](https://github.com/OPTML-Group/VLM-Safety-MU).

## Abstract

Recent vision-language models (VLMs) have made remarkable strides in generative modeling with multimodal inputs, particularly text and images. However, their susceptibility to generating harmful content when exposed to unsafe queries raises critical safety concerns.
While current alignment strategies primarily rely on supervised safety fine-tuning with curated datasets, we identify a fundamental limitation we call the "safety mirage" where supervised fine-tuning inadvertently reinforces spurious correlations between superficial textual patterns and safety responses, rather than fostering deep, intrinsic mitigation of harm.
We show that these spurious correlations leave fine-tuned VLMs vulnerable even to a simple one-word modification-based attack, where substituting a single word in text queries with a spurious correlation-inducing alternative can effectively bypass safeguards.
Additionally, these correlations contribute to the over prudence, causing fine-tuned VLMs to refuse benign queries unnecessarily.
To address this issue, we show machine unlearning (MU) as a powerful alternative to supervised safety fine-tuning as it avoids biased feature-label mappings and directly removes harmful knowledge from VLMs while preserving their general capabilities.
Extensive evaluations across safety benchmarks show that under one-word attacks, MU-based alignment reduces the attack success rate by up to 60.17% and cuts unnecessary rejections by over 84.20%.
**WARNING: There exist AI generations that may be offensive in nature.**

## Code

We will release our data, code and model soon.

## Contributors
* [Yiwei Chen](https://www.linkedin.com/in/yiwei-melody-chen/)
* [Yuguang Yao](https://www.cse.msu.edu/~yaoyugua/)

## Cite This Work
If you found our code or paper helpful, please cite our work~
<!-- ```
@article{fan2024simplicity,
  title={Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning},
  author={Fan, Chongyu and Liu, Jiancheng and Lin, Licong and Jia, Jinghan and Zhang, Ruiqi and Mei, Song and Liu, Sijia},
  journal={arXiv preprint arXiv:2410.07163},
  year={2024}
}
``` -->