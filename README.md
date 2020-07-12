# Dukang
模型蒸馏工具，以酒神杜康为名

基于Opennmt-tf 2.1分支开发的模型蒸馏工具。能够蒸馏Opennmt-tf定义的模型（或者遵从Opennmt-tf model API开发的模型）。

## Usage

### 安装:

```
pip install --upgrade pip
pip install tensorflow_addons
```

下载[Opennmt-tf](https://github.com/OpenNMT/OpenNMT-tf)切换到tf2.1分支。

### 训练:

```bash
python -m dukang.bin.main --model_type BaseDistill \
          --config=config/config.yml --auto_config \
          train --with_eval
```
### 模型导出:

```bash
python -m dukang.bin.main --model_type BaseDistill \
          --config=config/config.yml --auto_config \
          export --export_dir models/export
```

