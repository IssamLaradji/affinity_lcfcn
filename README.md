
# Weakly Supervised Fish Segmentation

### JCU fish

- https://www.dropbox.com/sh/b2jlua76ogyr5rk/AABsJVljG7v2BOunE1k4f_XTa?dl=0

## semseg Weakly supervised for JCU fish

```
python trainval.py -e weakly_JCUfish -sb <savedir_base> -d <datadir> -r 1
```
## affinity Weakly supervised for JCU fish

```
python trainval.py -e weakly_JCUfish_aff -sb <savedir_base> -d <datadir> -r 1
```

# Citation

```
@misc{laradji2020affinity,
      title={Affinity LCFCN: Learning to Segment Fish with Weak Supervision}, 
      author={Issam Laradji and Alzayat Saleh and Pau Rodriguez and 
                  Derek Nowrouzezahrai and Mostafa Rahimi Azghadi and David Vazquez},
      year={2020},
      eprint={2011.03149},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
