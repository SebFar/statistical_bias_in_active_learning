# On Statistial Bias In Active Learning: How And When to Fix It
Code to accompany the paper "On Statistical Bias In Active Learning: How and When to Fix It" at ICLR 2021.

Code to manage the active learning loop with the estimators $R_{pure}$ and $R_{lure}$ are implemented in the `dataloader.py` file.
In all cases, we store the active learning statistics alongside the data and update things as more data are acquired to shift indices from the pool to the acquired dataset.
This builds off code in the `alternative_mnist.py` file which loads the underlying data themselves.

The main experiment loops are located in `main.py` which runs the vanilla version of the active learning experiment used in Figure 3; `main_with_no_fitting.py` which contains an Active Testing (Kossen et al. 2021) style evaluation; and `overfitting-bias.py` which is used for Figure 4.

Helpful files are `active_learning_utils.py` and `models.py` which merely store functions and classes used by other parts of the code.
You can also find a YAML file that is designed to ask for the right requirements in `minimal-active.yml` as well as `torch1x.yml` which contains a snapshot of a configuration where I believe everything definitely worked.

# User note
There are some slight naming inconsistencies because the names of these methods changed at different points in development.
In the code itself `naive` means $R_{pure}$ while `refined` means $R_{lure}$.
Sometimes I also refer to $R_{sure}$ which is what $R_{lure}$ was called before reviewers suggested a change of name to avoid confusion with Stein's unbiased risk estimate.

# Citing this work
If you use this code or its derivatives please consider citing

```
@article{farquhar_statistical_2020,
                    title={On Statistical Bias In Active Learning: How and When to Fix It},
                    author={Farquhar, Sebastian and Gal, Yarin and Rainforth, Tom},
                    journal={International Conference on Learning Representations},
                    year={2021}
                  }
```
