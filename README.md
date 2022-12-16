# Feature encoders








**Note:** Run the following commands below in SageMaker Studio `System terminal` to install `git-lfs`. This is to enable working with large model files as part of this repository.

```bash
$ curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.rpm.sh | sudo bash
$ sudo yum install git-lfs -y
$ git lfs install
$ cd feature-encoders
$ git lfs track "*.bin"
$ git add text/data/bert-model/pytorch_model.bin
$ git commit -m 'Adding BERT base model'
$ git push
```
