
# Weak Supervision
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

## Pascal with Point Supervision

```
python trainval.py -e pascal_weakly -sb <savedir_base> -d <datadir> -r 1
```

```
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "pascal_weakly",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/trainval.py",
            "console": "integratedTerminal",
            "args":[
                "-e", "pascal_weakly",
                "-sb", "/mnt/home/results/weak_supervision/",
                "-d", "/mnt/home/datasets/",
                "-r", "1"
        ],
        },
    ]
}
```
